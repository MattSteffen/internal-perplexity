import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from tqdm import tqdm

from .chunker import Chunker, ChunkingConfig
from .converter import Converter, ConverterConfig, create_converter

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
    CollectionDescription,
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    get_db,
    get_db_benchmark,
)
from .vector_db.milvus_utils import extract_collection_description

# # Reserved keys that should not appear in metadata to avoid conflicts with database schema
# TODO: Change the prefix to be `_` instead of `default_`
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


def sanitize_metadata(md: dict, schema: dict = None) -> dict:
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
        ...,
        description="Configuration for document conversion to markdown",
    )
    extractor: MetadataExtractorConfig = Field(
        ...,
        description="Configuration for metadata extraction",
    )
    chunking: ChunkingConfig = Field(
        ...,
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
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig with type-safe parameters."""
        return cls(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter,
            extractor=extractor,
            chunking=chunking,
            metadata_schema=metadata_schema or {},
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            security_groups=security_groups,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CrawlerConfig":
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
        # Create a copy to avoid mutating the input
        processed_dict = config_dict.copy()

        # Handle backward compatibility: map "utils" dict to top-level fields
        if "utils" in processed_dict:
            utils = processed_dict.pop("utils")
            if isinstance(utils, dict):
                # Map utils keys to top-level fields if not already present
                if "temp_dir" not in processed_dict and "temp_dir" in utils:
                    processed_dict["temp_dir"] = utils["temp_dir"]
                if "benchmark" not in processed_dict and "benchmark" in utils:
                    processed_dict["benchmark"] = utils["benchmark"]

        # Handle extractor's nested llm config if present (needs special handling)
        if "extractor" in processed_dict and isinstance(processed_dict["extractor"], dict):
            extractor_dict = processed_dict["extractor"]
            if "llm" in extractor_dict and isinstance(extractor_dict["llm"], dict):
                # Convert nested llm dict to LLMConfig instance
                extractor_dict = extractor_dict.copy()
                extractor_dict["llm"] = LLMConfig(**extractor_dict["llm"])
                processed_dict["extractor"] = extractor_dict

        # Use Pydantic's model_validate to handle nested configs automatically
        # This will validate and convert nested dicts to their respective Pydantic models
        try:
            return cls.model_validate(processed_dict)
        except Exception as e:
            # Provide more context for validation errors
            raise ValueError(f"Failed to create CrawlerConfig from dictionary: {str(e)}") from e

    @classmethod
    def from_collection_description(
        cls,
        description: CollectionDescription,
        database_config: DatabaseClientConfig,
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig from a CollectionDescription.

        Args:
            description: CollectionDescription instance containing the config
            database_config: Database configuration (collection name will be used)

        Returns:
            CrawlerConfig instance restored from the collection description

        Raises:
            ValueError: If collection_config_json is None or invalid
        """
        return description.to_crawler_config(database_config)


class Crawler:
    def __init__(
        self,
        config: CrawlerConfig,
        converter: Converter = None,
        extractor: MetadataExtractor = None,
        vector_db: DatabaseClient = None,
        embedder: Embedder = None,
        llm: LLM = None,
        chunker: Chunker = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db
        self.chunker = chunker

        self.benchmarker = None

        self._initialize_defaults()

    def _restore_config_from_collection(self, database_config: DatabaseClientConfig) -> CrawlerConfig | None:
        """
        Restore CrawlerConfig from an existing collection if it exists.

        Args:
            database_config: Database configuration to check

        Returns:
            Restored CrawlerConfig if collection exists and has valid config, None otherwise
        """
        # Only restore for Milvus collections
        if database_config.provider != "milvus":
            return None

        # Skip if recreate is True (we want to overwrite)
        if database_config.recreate:
            return None

        try:
            # Connect to Milvus to check if collection exists
            client = MilvusClient(uri=database_config.uri, token=database_config.token)
            if not client.has_collection(database_config.collection):
                return None

            # Get collection description
            collection_info = client.describe_collection(database_config.collection)
            description_str = extract_collection_description(collection_info)

            if not description_str:
                return None

            # Parse and restore config from collection
            description_obj = CollectionDescription.from_json(description_str)
            if not description_obj or not description_obj.collection_config_json:
                return None

            restored_config = CrawlerConfig.from_collection_description(description_obj, database_config)

            # Validate that restored config matches current config
            # Compare key fields that should match (excluding database config)
            current_dict = self.config.model_dump(exclude={"database"})
            restored_dict = restored_config.model_dump(exclude={"database"})

            # Check if configs match (allowing for some differences in database config)
            if current_dict != restored_dict:
                # Create a detailed error message
                differences = []
                for key in set(current_dict.keys()) | set(restored_dict.keys()):
                    if current_dict.get(key) != restored_dict.get(key):
                        differences.append(f"  - {key}: current={current_dict.get(key)}, restored={restored_dict.get(key)}")
                raise ValueError(
                    f"Restored config from collection '{database_config.collection}' does not match provided config. "
                    f"Differences:\n" + "\n".join(differences) + "\n"
                    "Set recreate=True to overwrite the existing collection, or update your config to match."
                )

            # Configs match, return restored config
            return restored_config

        except Exception as e:
            # If restoration fails, log but return None to allow continuation with original config
            # This allows the system to work even if description parsing fails
            import warnings

            warnings.warn(
                f"Failed to restore config from collection '{database_config.collection}': {str(e)}. " f"Using provided config instead.",
                UserWarning,
            )
            return None

    def _initialize_defaults(self) -> None:
        """Initialize default components if not provided, and restore config from collection if applicable."""
        # Restore config from collection if it exists and recreate=False
        if self.vector_db is None:
            restored_config = self._restore_config_from_collection(self.config.database)
            if restored_config is not None:
                self.config = restored_config

            # Initialize embedder first (needed for vector_db dimension)
            if self.embedder is None:
                self.embedder = get_embedder(self.config.embeddings)

            # Pass the crawler config as a dict to be stored in collection description
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config.metadata_schema,
                self.config.extractor.context,
                collection_config_json=self.config.model_dump(),
            )
            if self.config.benchmark:
                self.benchmarker = get_db_benchmark(self.config.database, self.config.embeddings)

        # Initialize LLM (needed for extractor)
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

        # Initialize embedder if not already done
        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        # Initialize extractor (requires LLM)
        if self.extractor is None:
            self.extractor = MetadataExtractor(config=self.config.extractor, llm=self.llm)

        # Initialize converter
        if self.converter is None:
            self.converter = create_converter(self.config.converter)

        # Initialize chunker
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
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}B"

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
        # crawl_start_time = time.time()

        filepaths = self._get_filepaths(path)

        if not filepaths:
            return

        temp_dir = Path(self.config.temp_dir)
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        stats = {
            "total_files": len(filepaths),
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0,
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
                        self.converter.convert(document)
                        self._cache_document(temp_filepath, document)
                    except Exception:
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                # Extract metadata
                if not document.is_extracted():
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
                    document.chunks = self.chunker.chunk_text(document)
                    self._cache_document(temp_filepath, document)

                # Generate embeddings for chunks
                if not document.text_embeddings:
                    try:
                        document.text_embeddings = self.embedder.embed_batch(document.chunks)
                        self._cache_document(temp_filepath, document)
                    except Exception:
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                # Create database entities from document
                try:
                    entities = document.to_database_entities()
                except ValueError:
                    stats["failed_files"] += 1
                    pbar.update(1)
                    continue

                # Store in database
                if entities:
                    try:
                        # Convert entities to DatabaseDocument objects for insertion
                        db_docs = [DatabaseDocument(**entity) for entity in entities]
                        self.vector_db.insert_data(db_docs)
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
