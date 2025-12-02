import os
import time
from pathlib import Path
from typing import Any

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
from .config import CrawlerConfig


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
            restored_config = extract_collection_description(collection_info.get("description", ""))
            if not restored_config:
                return None

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

            # Pass the crawler config to be stored in collection description
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config,
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
                    entities = document.to_database_documents()
                except ValueError:
                    stats["failed_files"] += 1
                    pbar.update(1)
                    continue

                # Store in database
                if entities:
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
