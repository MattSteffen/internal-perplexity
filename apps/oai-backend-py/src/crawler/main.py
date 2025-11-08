import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from .chunker import Chunker, ChunkingConfig
from .converter import Converter, ConverterConfig, create_converter
from .converter.markitdown import MarkItDownConfig
from .converter.pymupdf4llm import PyMuPDF4LLMConfig
from .converter.types import DocumentInput

# When run as part of the crawler package
from .document import Document
from .extractor.extractor import (
    MetadataExtractionResult,
    MetadataExtractor,
    MetadataExtractorConfig,
)
from .llm.embeddings import Embedder, EmbedderConfig, get_embedder
from .llm.llm import LLM, LLMConfig, get_llm
from .vector_db import (
    DatabaseBenchmark,
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    get_db,
    get_db_benchmark,
)

# # Reserved keys that should not appear in metadata to avoid conflicts with database schema
# RESERVED = {
#     "default_document_id",
#     "default_chunk_index",
#     "default_source",
#     "default_text",
#     "default_text_embedding",
#     "default_text_sparse_embedding",
#     "default_metadata",
#     "default_metadata_sparse_embedding",
# }


def sanitize_metadata(md: dict, schema: dict | None = None) -> dict:
    """
    Sanitize metadata by validating against a JSON schema.

    With the new prefixed field naming strategy, reserved key conflicts are no longer possible,
    so we only perform schema validation if provided.

    When validation fails, missing required fields are filled with default values instead
    of returning an empty dict.

    Args:
        md: Metadata dictionary to sanitize
        schema: Optional JSON schema to validate against

    Returns:
        Sanitized metadata dictionary with defaults for missing required fields
    """
    if not isinstance(md, dict):
        return {}

    # With prefixed fields, no need to remove reserved keys - conflicts are impossible
    sanitized = md.copy()

    # Optional JSON schema validation
    if schema:
        try:
            import jsonschema

            jsonschema.validate(instance=sanitized, schema=schema)
        except jsonschema.ValidationError:
            pass

        except Exception:
            pass

    return sanitized


class CrawlerConfig(BaseModel):
    """Configuration for the document crawler with Pydantic validation.

    This class provides type-safe configuration management for the crawler system,
    with automatic validation and serialization capabilities.
    """

    embeddings: EmbedderConfig = Field(..., description="Configuration for the embedding model")
    llm: LLMConfig = Field(..., description="Configuration for the main LLM used for metadata extraction")
    vision_llm: LLMConfig = Field(..., description="Configuration for the vision LLM used for image processing")
    database: DatabaseClientConfig = Field(..., description="Configuration for the vector database")
    converter: ConverterConfig = Field(
        default_factory=lambda: PyMuPDF4LLMConfig(type="pymupdf4llm"),
        description="Configuration for document conversion to markdown",
    )
    extractor: MetadataExtractorConfig = Field(
        default_factory=lambda: MetadataExtractorConfig(type="basic", llm=LLMConfig.ollama(model_name="llama3.2:3b")),
        description="Configuration for metadata extraction",
    )
    chunking: ChunkingConfig = Field(
        default_factory=lambda: ChunkingConfig.create(chunk_size=10000),
        description="Configuration for text chunking",
    )
    metadata_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for metadata validation")
    temp_dir: str = Field(
        default="tmp/",
        min_length=1,
        description="Temporary directory for caching processed documents",
    )
    benchmark: bool = Field(default=False, description="Whether to run benchmarking after crawling")
    generate_benchmark_questions: bool = Field(
        default=False,
        description="Generate benchmark questions during metadata extraction",
    )
    num_benchmark_questions: int = Field(
        default=3,
        gt=0,
        description="Number of benchmark questions to generate per document",
    )
    security_groups: list[str] | None = Field(
        default=None,
        description="List of security groups for RBAC access control. If provided, the user must have this role to see the documents.",
    )
    model_config = {"validate_assignment": True}

    @classmethod
    def create(
        cls,
        embeddings: EmbedderConfig,
        llm: LLMConfig,
        vision_llm: LLMConfig,
        database: DatabaseClientConfig,
        converter: ConverterConfig | None = None,
        extractor: MetadataExtractorConfig | None = None,
        chunking: ChunkingConfig | None = None,
        metadata_schema: dict[str, Any] | None = None,
        temp_dir: str = "tmp/",
        benchmark: bool = False,
        generate_benchmark_questions: bool = False,
        num_benchmark_questions: int = 3,
        security_groups: list[str] | None = None,
    ) -> CrawlerConfig:
        """Create a CrawlerConfig with type-safe parameters."""
        return cls(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter or PyMuPDF4LLMConfig(type="pymupdf4llm"),
            extractor=extractor or MetadataExtractorConfig(type="basic", llm=llm),
            chunking=chunking or ChunkingConfig.create(chunk_size=10000),
            metadata_schema=metadata_schema or {},
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            security_groups=security_groups,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CrawlerConfig:
        """Create a CrawlerConfig from a dictionary configuration.

        This method provides backward compatibility with dictionary-based configurations
        while leveraging Pydantic's validation capabilities.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Validated CrawlerConfig instance

        Example:
            >>> config = CrawlerConfig.from_dict({
            ...     "embeddings": {"provider": "ollama", "model": "all-minilm:v2", ...},
            ...     "llm": {"model_name": "llama3.2:3b", ...},
            ...     "vision_llm": {"model_name": "llava:latest", ...},
            ...     "database": {"provider": "milvus", "collection": "docs", ...},
            ... })
        """
        # Extract nested configs and convert them to Pydantic models
        embeddings_dict = config_dict.get("embeddings", {})
        embeddings = EmbedderConfig(**embeddings_dict)

        llm_dict = config_dict.get("llm", {})
        llm = LLMConfig(**llm_dict)

        vision_llm_dict = config_dict.get("vision_llm", {})
        vision_llm = LLMConfig(**vision_llm_dict)

        database_dict = config_dict.get("database", {})
        database = DatabaseClientConfig(**database_dict)

        # Handle converter config
        converter_dict = config_dict.get("converter", {"type": "pymupdf4llm"})
        converter_type = converter_dict.get("type", "pymupdf4llm")
        if converter_type == "markitdown":
            converter: ConverterConfig = MarkItDownConfig(**converter_dict)
        else:
            converter = PyMuPDF4LLMConfig(**converter_dict)

        # Handle extractor config
        extractor_dict = config_dict.get("extractor", {})
        if extractor_dict:
            # If extractor has its own llm config, convert it
            if "llm" in extractor_dict and isinstance(extractor_dict["llm"], dict):
                extractor_dict["llm"] = LLMConfig(**extractor_dict["llm"])
            extractor = MetadataExtractorConfig(**extractor_dict)
        else:
            extractor = MetadataExtractorConfig(type="basic", llm=llm)

        # Handle chunking config
        chunking_dict = config_dict.get("chunking", {})
        if chunking_dict:
            chunking = ChunkingConfig(**chunking_dict)
        else:
            # Use legacy chunk_size if no chunking config provided
            chunk_size = config_dict.get("chunk_size", 10000)
            chunking = ChunkingConfig.create(chunk_size=chunk_size)

        # Extract utility parameters
        utils = config_dict.get("utils", {})
        chunk_size = utils.get("chunk_size", 10000)
        temp_dir = utils.get("temp_dir", "tmp/")
        benchmark = utils.get("benchmark", False)

        # Extract other top-level parameters
        metadata_schema = config_dict.get("metadata_schema", {})
        generate_benchmark_questions = config_dict.get("generate_benchmark_questions", False)
        num_benchmark_questions = config_dict.get("num_benchmark_questions", 3)
        security_groups = config_dict.get("security_groups", None)
        return cls(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter,
            extractor=extractor,
            chunking=chunking,
            chunk_size=chunk_size,
            metadata_schema=metadata_schema,
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            security_groups=security_groups,
        )

    @classmethod
    def from_collection_description(
        cls,
        description_json: dict[str, Any],
        collection_name: str,
        database_config: DatabaseClientConfig | None = None,
    ) -> CrawlerConfig:
        """Create a CrawlerConfig from a collection description JSON.

        Parses the pipeline_config stored in the collection description to reconstruct
        the CrawlerConfig used to create the collection.

        Args:
            description_json: Parsed JSON from collection description field
            collection_name: Name of the collection (for database config)
            database_config: Optional database config override

        Returns:
            CrawlerConfig instance reconstructed from description

        Raises:
            ValueError: If pipeline_config is missing or invalid
        """
        pipeline_config = description_json.get("pipeline_config")
        if not pipeline_config:
            raise ValueError("Collection description does not contain pipeline_config")

        # Handle predefined pipeline with overrides
        if "pipeline_name" in pipeline_config:
            from src.endpoints.document_pipelines import ConfigOverrides, _override_config
            from src.endpoints.pipeline_registry import get_registry

            pipeline_name = pipeline_config["pipeline_name"]
            registry = get_registry()

            if not registry.has_pipeline(pipeline_name):
                raise ValueError(f"Pipeline '{pipeline_name}' not found in registry")

            base_config = registry.get_config(pipeline_name)

            # Apply overrides if present
            overrides_dict = pipeline_config.get("overrides", {})
            if overrides_dict:
                overrides = ConfigOverrides(**overrides_dict)
                config = _override_config(base_config, overrides)
            else:
                config = base_config

        # Handle custom full config
        elif "full_config" in pipeline_config:
            config = cls.from_dict(pipeline_config["full_config"])
        else:
            raise ValueError("pipeline_config must contain either 'pipeline_name' or 'full_config'")

        # Override collection name in database config
        if collection_name:
            original_db_config = config.database
            config.database = DatabaseClientConfig(
                provider=original_db_config.provider,
                collection=collection_name,
                host=original_db_config.host,
                port=original_db_config.port,
                username=original_db_config.username,
                password=original_db_config.password,
                partition=original_db_config.partition,
                recreate=False,
                collection_description=original_db_config.collection_description,
            )

        # Override with provided database config if given
        if database_config:
            config.database = database_config

        return config


class Crawler:
    def __init__(
        self,
        config: CrawlerConfig,
        converter: Converter | None = None,
        extractor: MetadataExtractor | None = None,
        vector_db: DatabaseClient | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db
        self.chunker = chunker

        self.benchmarker: DatabaseBenchmark | None = None

        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        # initialize defaults if provided as none
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        if self.extractor is None:
            self.extractor = MetadataExtractor(config=self.config.extractor, llm=self.llm)

        if self.vector_db is None:
            # TODO: This must have properly formatted collection description.
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config.metadata_schema,
                self.config.extractor.context,
            )
            if self.config.benchmark:
                self.benchmarker = get_db_benchmark(self.config.database, self.config.embeddings)

        if self.converter is None:
            self.converter = create_converter(self.config.converter)

        if self.chunker is None:
            self.chunker = Chunker(self.config.chunking)

    def _get_filepaths(self, path: str | list[str]) -> list[str]:
        """
        Returns a list of file paths based on the input path.

        Args:
            path: Either a file path or a directory path

        Returns:
            List[str]:
                - If path is a file: a list containing only that file path
                - If path is a directory: a list of all file paths in that directory (including subdirectories)
                - If path is neither: an empty list
        """
        if isinstance(path, list):
            return path

        # Check if the path exists
        if not os.path.exists(path):
            return []

        # If path is a file, return a list with just that file
        if os.path.isfile(path):
            return [path]

        # If path is a directory, collect all files within it
        elif os.path.isdir(path):
            file_paths = []
            total_size = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass

            return file_paths

        # If path is neither a file nor a directory
        else:
            return []

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human readable format."""
        value: float = float(bytes_value)
        for unit in ["B", "KB", "MB", "GB"]:
            if value < 1024.0:
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f}B"

    def _cache_document(self, temp_filepath: Path | None, document: Document) -> None:
        """Cache document to temporary file if temp_filepath is provided."""
        if temp_filepath:
            try:
                document.save(str(temp_filepath))
            except Exception:
                pass

    def crawl(self, path: str | list[str]) -> None:
        """
        Crawl the given path(s) and process the documents.
        """
        filepaths = self._get_filepaths(path)

        if not filepaths:
            return

        temp_dir = Path(self.config.temp_dir)
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        stats: dict[str, Any] = {
            "total_files": len(filepaths),
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "total_file_size": 0,
        }

        # Main processing loop with progress bar
        with tqdm(total=len(filepaths), desc="Processing files", unit="file") as pbar:
            for filepath in filepaths:
                file_start_time = time.time()
                file_size = 0

                try:
                    file_size = os.path.getsize(filepath)
                    stats["total_file_size"] += file_size
                except OSError:
                    pass

                pbar.set_postfix_str(f"Current: {os.path.basename(filepath)}")

                # Check for duplicates
                if self.vector_db is None:
                    raise RuntimeError("Vector database not initialized")
                if self.vector_db.check_duplicate(filepath, 0):
                    stats["skipped_files"] += 1
                    pbar.update(1)
                    continue

                document = Document.create(
                    source=filepath,
                    security_group=self.config.security_groups,
                )

                # Check for cached processing results
                temp_filepath = None
                if temp_dir:
                    filename = os.path.splitext(os.path.basename(filepath))[0] + ".json"
                    temp_filepath = temp_dir / filename
                    if temp_filepath.exists():
                        try:
                            document.load(str(temp_filepath))
                        except Exception:
                            pass

                # Convert document if not cached
                if not document.is_converted():
                    try:
                        if self.converter is None:
                            raise RuntimeError("Converter not initialized")
                        doc_input = DocumentInput.from_document(document)
                        converted = self.converter.convert(doc_input)
                        document.markdown = converted.markdown
                        self._cache_document(temp_filepath, document)
                    except Exception:
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                # Extract metadata
                if not document.is_extracted():
                    if self.extractor is None:
                        raise RuntimeError("Metadata extractor not initialized")
                    try:
                        extraction_result: MetadataExtractionResult = self.extractor.run(document)
                        document.metadata = extraction_result.metadata
                        document.benchmark_questions = extraction_result.benchmark_questions
                        # Sanitize metadata before creating database document
                        document.metadata = (
                            sanitize_metadata(
                                document.metadata or {},
                                self.config.metadata_schema,
                            )
                            or {}
                        )
                        self._cache_document(temp_filepath, document)
                    except Exception:
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                # Chunk the text
                if not document.is_chunked():
                    if self.chunker is None:
                        raise RuntimeError("Chunker not initialized")
                    document.chunks = self.chunker.chunk_text(document)
                    self._cache_document(temp_filepath, document)

                # Generate embeddings for chunks
                if document.chunks is None:
                    continue
                if self.embedder is None:
                    raise RuntimeError("Embedder not initialized")
                entities: list[DatabaseDocument] = []
                embeddings = self.embedder.embed_batch(document.chunks)
                for i, chunk in enumerate(document.chunks):
                    doc = DatabaseDocument(
                        default_document_id=document.document_id,
                        default_minio=document.minio_url or "",
                        default_text=chunk,
                        default_text_embedding=embeddings[i],
                        default_chunk_index=i,
                        default_source=document.source,
                        metadata=document.metadata,
                        security_group=document.security_group,
                    )
                    entities.append(doc)

                # Store in database
                if entities:
                    if self.vector_db is None:
                        raise RuntimeError("Vector database not initialized")
                    try:
                        self.vector_db.insert_data(entities)
                        stats["total_chunks"] += len(entities)
                    except Exception:
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue
                else:
                    stats["failed_files"] += 1
                    pbar.update(1)
                    continue

                # Log file completion
                file_time = time.time() - file_start_time
                stats["processed_files"] += 1
                stats["total_processing_time"] += file_time

                pbar.update(1)

        # Final statistics calculation
        # total_time = time.time() - crawl_start_time

    def benchmark(self, generate_queries: bool = False) -> None:
        if self.benchmarker:
            results = self.benchmarker.run_benchmark(generate_queries)
            self.benchmarker.plot_results(results, "benchmark_results")
            self.benchmarker.save_results(results, "benchmark_results/results.json")
