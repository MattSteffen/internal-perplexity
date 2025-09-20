from dataclasses import dataclass, field
import os
import uuid
import time
import logging
from typing import Dict, Union, List, Any, Optional
import json
from pathlib import Path
from tqdm import tqdm


# Configure logging for the entire crawler system
def setup_crawler_logging(
    log_level: str = "INFO", log_file: str = None
) -> logging.Logger:
    """Setup comprehensive logging for the crawler system and all processing modules.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # List of all logger names used in the crawler system
    logger_names = [
        "Crawler",
        "OllamaLLM",
        "Extractor",
        "MultiSchemaExtractor",
        "OllamaEmbedder",
        "MarkItDownConverter",
        "DoclingConverter",
        "DoclingVLMConverter",
        "PyMuPDFConverter",
    ]

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure all loggers
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent propagation to root logger to avoid duplicates
        logger.propagate = False

        # Clear any existing handlers and set up fresh ones
        logger.handlers.clear()

        # Add all handlers to this logger
        for handler in handlers:
            logger.addHandler(handler)

        logger.debug(f"{logger_name} logging handlers initialized")

    # Return the main crawler logger
    return logging.getLogger("Crawler")


from .processing import (
    Embedder,
    EmbedderConfig,
    get_embedder,
    Extractor,
    ExtractorConfig,
    LLM,
    LLMConfig,
    get_llm,
    Converter,
    ConverterConfig,
    create_converter,
    create_extractor,
)
from .storage import (
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


def sanitize_metadata(
    md: dict, schema: dict = None, logger: logging.Logger = None
) -> dict:
    """
    Sanitize metadata by validating against a JSON schema.

    With the new prefixed field naming strategy, reserved key conflicts are no longer possible,
    so we only perform schema validation if provided.

    When validation fails, missing required fields are filled with default values instead
    of returning an empty dict.

    Args:
        md: Metadata dictionary to sanitize
        schema: Optional JSON schema to validate against
        logger: Optional logger for validation errors

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
            if logger:
                logger.debug("Metadata validation passed")
        except jsonschema.ValidationError as e:
            if logger:
                logger.warning(
                    f"Metadata validation failed: {e.message}, adding defaults for missing required fields"
                )

        except Exception as e:
            if logger:
                logger.warning(
                    f"Error during metadata validation: {e}, adding defaults for missing required fields"
                )

    return sanitized


"""
Crawler takes a directory or list of filepaths, extracts the markdown and metadata from each, chunks them, and stores them in a vector database.
The crawler class is a base class.


# TODO: Update this to the new config with the types, not json

Example config:
{
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "llm": {
        "model": "llama3.2:3b",  # or use "model_name": "llama3.2:3b"
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "vision_llm": {
        "model": "llava:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "Milvus",
        "collection": "documents",
        "recreate": False,
    },
    "converter": {
        "type": "markitdown",
        "vision_llm": {
            "model": "llava:latest",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        }
    },
    "extractor": {},
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 512},
            "author": {"type": "string", "maxLength": 256},
        }
    },
    "utils": {
        "chunk_size": 10000,
        "temp_dir": "tmp/",
        "benchmark": False,
    }
}
"""


@dataclass
class CrawlerConfig:
    """Configuration for the document crawler."""

    embeddings: EmbedderConfig
    llm: LLMConfig
    vision_llm: LLMConfig
    database: DatabaseClientConfig
    converter: ConverterConfig = field(default_factory=lambda: ConverterConfig())
    extractor: ExtractorConfig = field(default_factory=lambda: ExtractorConfig())
    chunk_size: int = 10000  # treated as maximum if using semantic chunking
    metadata_schema: Dict[str, Any] = field(default_factory=dict)
    temp_dir: str = "tmp/"
    benchmark: bool = False
    generate_benchmark_questions: bool = (
        False  # Generate benchmark questions during metadata extraction
    )
    num_benchmark_questions: int = 3  # Number of benchmark questions to generate
    log_level: str = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file: Optional[str] = None  # Optional log file path

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.num_benchmark_questions <= 0:
            raise ValueError("Number of benchmark questions must be positive")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")

    @classmethod
    def create(
        cls,
        embeddings: EmbedderConfig,
        llm: LLMConfig,
        vision_llm: LLMConfig,
        database: DatabaseClientConfig,
        converter: Optional[ConverterConfig] = None,
        extractor: Optional[ExtractorConfig] = None,
        chunk_size: int = 10000,
        metadata_schema: Optional[Dict[str, Any]] = None,
        temp_dir: str = "tmp/",
        benchmark: bool = False,
        generate_benchmark_questions: bool = False,
        num_benchmark_questions: int = 3,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig with type-safe parameters."""
        return cls(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter or ConverterConfig(),
            extractor=extractor or ExtractorConfig(),
            chunk_size=chunk_size,
            metadata_schema=metadata_schema or {},
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            log_level=log_level,
            log_file=log_file,
        )

    @classmethod
    def default_ollama(
        cls,
        collection: str = "documents",
        embed_model: str = "all-minilm:v2",
        llm_model: str = "gpt-oss:20b",
        vision_model: str = "granite-3.2vision:latest",
        base_url: str = "http://localhost:11434",
        host: str = "localhost",
        port: int = 19530,
        **kwargs,
    ) -> "CrawlerConfig":
        """Create a default configuration using Ollama models."""
        embeddings = EmbedderConfig.ollama(model=embed_model, base_url=base_url)
        llm = LLMConfig.ollama(
            model_name=llm_model, base_url=base_url, structured_output="tools"
        )
        vision_llm = LLMConfig.ollama(model_name=vision_model, base_url=base_url)
        database = DatabaseClientConfig.milvus(
            collection=collection, host=host, port=port
        )

        return cls.create(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            **kwargs,
        )


class Crawler:
    def __init__(
        self,
        config: CrawlerConfig,
        converter: Converter = None,
        extractor: Extractor = None,
        vector_db: DatabaseClient = None,
        embedder: Embedder = None,
        llm: LLM = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db

        self.benchmarker = None

        # Setup logging
        self.logger = setup_crawler_logging(config.log_level, config.log_file)
        self.logger.info("Initializing Crawler with configuration")
        self.logger.debug(f"Config: {config}")

        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        # initialize defaults if provided as none
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        if self.extractor is None:
            self.extractor = create_extractor(self.config.extractor, self.llm)

        if self.vector_db is None:
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config.metadata_schema,
            )
            if self.config.benchmark:
                self.benchmarker = get_db_benchmark(
                    self.config.database, self.config.embeddings
                )

        if self.converter is None:
            # Use the converter config directly
            self.converter = create_converter(
                self.config.converter.type, self.config.converter
            )

    def _get_filepaths(self, path: Union[str, list[str]]) -> List[str]:
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
            self.logger.info(f"Processing {len(path)} files from provided list")
            return path

        # Check if the path exists
        if not os.path.exists(path):
            self.logger.error(f"Path {path} does not exist, returning empty list")
            return []

        # If path is a file, return a list with just that file
        if os.path.isfile(path):
            file_size = os.path.getsize(path)
            self.logger.info(
                f"Processing single file: {path} ({self._format_bytes(file_size)})"
            )
            return [path]

        # If path is a directory, collect all files within it
        elif os.path.isdir(path):
            self.logger.info(f"Scanning directory: {path}")
            file_paths = []
            total_size = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        self.logger.warning(f"Could not get size for file: {file_path}")

            self.logger.info(
                f"Found {len(file_paths)} files in directory ({self._format_bytes(total_size)} total)"
            )
            return file_paths

        # If path is neither a file nor a directory
        else:
            self.logger.error(
                f"Path {path} is neither a file nor a directory, returning empty list"
            )
            return []

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}B"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}B"

    def crawl(self, path: Union[str, List[str]]) -> None:
        """
        Crawl the given path(s) and process the documents with comprehensive logging.
        """
        crawl_start_time = time.time()
        self.logger.info("=== Starting document crawling process ===")

        filepaths = self._get_filepaths(path)

        if not filepaths:
            self.logger.warning("No files to process")
            return

        temp_dir = Path(self.config.temp_dir)
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Using temp directory: {temp_dir.absolute()}")

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
                    self.logger.warning(f"Could not get size for file: {filepath}")

                pbar.set_postfix_str(f"Current: {os.path.basename(filepath)}")

                # Check for duplicates
                if self.vector_db.check_duplicate(filepath, 0):
                    self.logger.info(
                        f"⏭️  Skipping duplicate document: {os.path.basename(filepath)}"
                    )
                    stats["skipped_files"] += 1
                    pbar.update(1)
                    continue

                self.logger.info(
                    f"📄 Processing: {os.path.basename(filepath)} ({self._format_bytes(file_size)})"
                )

                markdown = None
                metadata = None
                temp_filepath = None

                # Check for cached processing results
                if temp_dir:
                    filename = os.path.splitext(os.path.basename(filepath))[0] + ".json"
                    temp_filepath = temp_dir / filename
                    if temp_filepath.exists():
                        self.logger.info(
                            f"📋 Loading cached document from {temp_filepath}"
                        )
                        with open(temp_filepath, "r") as f:
                            data = json.load(f)
                            markdown = data["text"]
                            metadata = data["metadata"]

                # Convert document if not cached
                if markdown is None or metadata is None:
                    convert_start = time.time()
                    self.logger.info("🔄 Converting document to markdown...")
                    try:
                        markdown = self.converter.convert(filepath)
                        self.logger.info(
                            f"✅ Conversion completed in {time.time() - convert_start:.2f}s"
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Conversion failed for {filepath}: {e}")
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                    # Extract metadata
                    extract_start = time.time()
                    self.logger.info("🧠 Extracting metadata...")
                    try:
                        metadata = self.extractor.extract_metadata(markdown)
                        self.logger.info(
                            f"✅ Metadata extraction completed in {time.time() - extract_start:.2f}s"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"❌ Metadata extraction failed for {filepath}: {e}"
                        )
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue

                    # Cache results
                    if temp_filepath:
                        try:
                            with open(temp_filepath, "w") as f:
                                json.dump({"text": markdown, "metadata": metadata}, f)
                            self.logger.debug(
                                f"💾 Cached processed document to {temp_filepath}"
                            )
                        except Exception as e:
                            self.logger.warning(f"⚠️  Failed to cache document: {e}")

                # Chunk the text
                chunk_start = time.time()
                self.logger.info("✂️  Chunking text...")
                chunks = self.extractor.chunk_text(markdown, self.config.chunk_size)
                chunk_time = time.time() - chunk_start
                self.logger.info(
                    f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s (avg {len(chunks)/chunk_time:.1f} chunks/sec)"
                )

                # Generate embeddings for chunks
                embed_start = time.time()
                self.logger.info("🧮 Generating embeddings...")
                entities: List[DatabaseDocument] = []

                # Process chunks with progress tracking
                with tqdm(
                    total=len(chunks),
                    desc="Embedding chunks",
                    unit="chunk",
                    leave=False,
                ) as chunk_pbar:
                    for i, chunk in enumerate(chunks):
                        chunk_embed_start = time.time()
                        doc_id = str(uuid.uuid4())
                        try:
                            embeddings = self.embedder.embed(chunk)
                            chunk_embed_time = time.time() - chunk_embed_start

                            # Sanitize metadata before creating database document
                            sanitized_metadata = sanitize_metadata(
                                metadata, self.config.metadata_schema, self.logger
                            )

                            # Create entities for database
                            doc = DatabaseDocument(
                                default_document_id=doc_id,
                                default_minio=filepath,
                                default_text=chunk,
                                default_text_embedding=embeddings,
                                default_chunk_index=i,
                                default_source=os.path.basename(filepath),
                                metadata=sanitized_metadata,
                            )

                            entities.append(doc)

                            chunk_pbar.set_postfix_str(
                                f"Chunk {i+1}/{len(chunks)} ({chunk_embed_time:.3f}s)"
                            )
                            chunk_pbar.update(1)

                        except Exception as e:
                            self.logger.error(f"❌ Failed to embed chunk {i}: {e}")
                            continue

                embed_time = time.time() - embed_start
                self.logger.info(
                    f"✅ Generated embeddings for {len(entities)} chunks in {embed_time:.2f}s (avg {len(entities)/embed_time:.1f} chunks/sec)"
                )

                # Store in database
                if entities:
                    store_start = time.time()
                    self.logger.info("💾 Storing documents in database...")
                    try:
                        self.vector_db.insert_data(entities)
                        store_time = time.time() - store_start
                        self.logger.info(
                            f"✅ Successfully stored {len(entities)} documents in {store_time:.2f}s"
                        )
                        stats["total_chunks"] += len(entities)
                    except Exception as e:
                        self.logger.error(f"❌ Failed to store documents: {e}")
                        stats["failed_files"] += 1
                        pbar.update(1)
                        continue
                else:
                    self.logger.warning("⚠️  No entities to store")
                    stats["failed_files"] += 1
                    pbar.update(1)
                    continue

                # Log file completion
                file_time = time.time() - file_start_time
                stats["processed_files"] += 1
                stats["total_processing_time"] += file_time

                self.logger.info(
                    f"🎉 Completed processing {os.path.basename(filepath)} in {file_time:.2f}s"
                )
                self.logger.debug(f"File metadata: {metadata}")

                pbar.update(1)

        # Log final statistics
        total_time = time.time() - crawl_start_time
        self.logger.info("=== Crawling process completed ===")
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   • Total files found: {stats['total_files']}")
        self.logger.info(f"   • Files processed: {stats['processed_files']}")
        self.logger.info(f"   • Files skipped (duplicates): {stats['skipped_files']}")
        self.logger.info(f"   • Files failed: {stats['failed_files']}")
        self.logger.info(f"   • Total chunks created: {stats['total_chunks']}")
        self.logger.info(
            f"   • Total data processed: {self._format_bytes(stats['total_file_size'])}"
        )
        self.logger.info(f"   • Total processing time: {total_time:.2f}s")
        self.logger.info(
            f"   • Average time per file: {stats['total_processing_time']/max(stats['processed_files'], 1):.2f}s"
        )
        self.logger.info(
            f"   • Processing rate: {stats['processed_files']/total_time:.2f} files/sec"
        )

        if stats["failed_files"] > 0:
            self.logger.warning(
                f"⚠️  {stats['failed_files']} files failed to process - check logs above for details"
            )

    def benchmark(self, generate_queries: bool = False) -> None:
        if self.benchmarker:
            results = self.benchmarker.run_benchmark(generate_queries)
            self.benchmarker.plot_results(results, "benchmark_results")
            self.benchmarker.save_results(results, "benchmark_results/results.json")
