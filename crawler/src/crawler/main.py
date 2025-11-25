import json
import os
import re
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
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    get_db,
    get_db_benchmark,
)

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
        converter_dict = config_dict.get("converter")
        if not converter_dict:
            raise ValueError("converter configuration is required")
        converter = ConverterConfig(**converter_dict)

        # Handle extractor config
        extractor_dict = config_dict.get("extractor")
        if not extractor_dict:
            raise ValueError("extractor configuration is required")
        # If extractor has its own llm config, convert it
        if "llm" in extractor_dict and isinstance(extractor_dict["llm"], dict):
            extractor_dict["llm"] = LLMConfig(**extractor_dict["llm"])
        extractor = MetadataExtractorConfig(**extractor_dict)

        # Handle chunking config
        chunking_dict = config_dict.get("chunking")
        if not chunking_dict:
            raise ValueError("chunking configuration is required")
        chunking = ChunkingConfig(**chunking_dict)

        # Extract utility parameters
        utils = config_dict.get("utils", {})
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
            metadata_schema=metadata_schema,
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            security_groups=security_groups,
        )

    @classmethod
    def from_collection(
        cls,
        text: str,
        database_config: DatabaseClientConfig,
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig from a collection by parsing natural language.

        Parses natural language like "add <doc> to <collection X>" to extract
        the collection name, then loads the config from that collection's description.

        Args:
            text: Natural language text containing collection name (e.g., "add doc.pdf to collection my_collection")
            database_config: Database configuration for connecting to Milvus

        Returns:
            CrawlerConfig instance loaded from the collection description

        Raises:
            ValueError: If collection name cannot be extracted or collection not found
            RuntimeError: If connection to Milvus fails or description is invalid

        Example:
            >>> db_config = DatabaseClientConfig.milvus(collection="temp", host="localhost")
            >>> config = CrawlerConfig.from_collection("add doc.pdf to collection my_collection", db_config)
        """
        # Parse collection name from natural language
        # Try various patterns: "to collection X", "collection X", "to X", etc.
        collection_name = None

        # Pattern 1: "to collection <name>" or "to <name>"
        match = re.search(r"to\s+(?:collection\s+)?([a-zA-Z0-9_]+)", text, re.IGNORECASE)
        if match:
            collection_name = match.group(1)

        # Pattern 2: "collection <name>"
        if not collection_name:
            match = re.search(r"collection\s+([a-zA-Z0-9_]+)", text, re.IGNORECASE)
            if match:
                collection_name = match.group(1)

        # Pattern 3: Just look for a word that might be a collection name
        # (fallback - less reliable)
        if not collection_name:
            # Try to find a word that looks like a collection name
            words = re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]*)\b", text)
            # Filter out common words
            common_words = {"add", "to", "the", "a", "an", "this", "that", "doc", "document", "file"}
            candidates = [w for w in words if w.lower() not in common_words]
            if candidates:
                collection_name = candidates[-1]  # Take the last candidate

        if not collection_name:
            raise ValueError(f"Could not extract collection name from text: {text}")

        # Connect to Milvus
        try:
            client = MilvusClient(uri=database_config.uri, token=database_config.token)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Milvus: {str(e)}") from e

        # Check if collection exists
        try:
            collections = client.list_collections()
            if collection_name not in collections:
                raise ValueError(f"Collection '{collection_name}' not found. Available collections: {collections}")
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {str(e)}") from e

        # Get collection description
        try:
            collection_info = client.describe_collection(collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to describe collection '{collection_name}': {str(e)}") from e

        # Extract description string
        description_str = None
        if isinstance(collection_info, dict):
            schema = collection_info.get("schema")
            if schema and isinstance(schema, dict):
                description_str = schema.get("description")
            elif "description" in collection_info:
                description_str = collection_info["description"]
        elif hasattr(collection_info, "schema"):
            schema = collection_info.schema
            if hasattr(schema, "description"):
                description_str = schema.description
        elif hasattr(collection_info, "description"):
            description_str = collection_info.description

        if not description_str:
            raise ValueError(f"Collection '{collection_name}' does not have a description")

        # Parse description JSON
        try:
            description_dict = json.loads(description_str) if isinstance(description_str, str) else description_str
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse collection description as JSON: {str(e)}") from e

        # Extract collection_config_json
        if not isinstance(description_dict, dict):
            raise ValueError(f"Collection description is not a dictionary: {type(description_dict)}")

        collection_config_dict = description_dict.get("collection_config_json")
        if not collection_config_dict:
            raise ValueError("Collection description does not contain 'collection_config_json'")

        # Create CrawlerConfig from the config dictionary
        try:
            config = cls.from_dict(collection_config_dict)
        except Exception as e:
            raise ValueError(f"Failed to create CrawlerConfig from collection config: {str(e)}") from e

        # Override collection name in database config
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

        return config


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

    def _initialize_defaults(self) -> None:
        # initialize defaults if provided as none
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        if self.extractor is None:
            self.extractor = MetadataExtractor(config=self.config.extractor, llm=self.llm)

        if self.vector_db is None:
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
