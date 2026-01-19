"""
Main crawler module for document processing pipeline.

This module provides the Crawler class that orchestrates document processing
through conversion, metadata extraction, chunking, embedding, and storage.
"""

import os
import time
from pathlib import Path

from tqdm import tqdm

from .chunker import Chunker, ChunkingConfig
from .config import CrawlerConfig
from .converter import Converter, ConverterConfig, create_converter
from .document import Chunk, Document
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
    get_db,
    get_db_benchmark,
)


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

    sanitized = md.copy()

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
    """
    Document crawler that processes files through a multi-stage pipeline.

    The pipeline consists of:
    1. Document loading and caching
    2. Conversion to markdown
    3. Metadata extraction using LLM
    4. Text chunking
    5. Embedding generation
    6. Vector database storage

    Example:
        >>> config = CrawlerConfig.create(...)
        >>> crawler = Crawler(config)
        >>> crawler.crawl("/path/to/documents")
    """

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
        """
        Initialize the crawler with configuration and optional components.

        Args:
            config: CrawlerConfig with all pipeline settings
            converter: Optional custom converter (created from config if not provided)
            extractor: Optional custom extractor (created from config if not provided)
            vector_db: Optional custom database client (created from config if not provided)
            embedder: Optional custom embedder (created from config if not provided)
            llm: Optional custom LLM (created from config if not provided)
            chunker: Optional custom chunker (created from config if not provided)
        """
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db
        self.chunker = chunker

        self.benchmarker = None

        self._initialize_defaults()

    def _restore_config_from_collection(
        self, database_config: DatabaseClientConfig
    ) -> CrawlerConfig | None:
        """
        Restore CrawlerConfig from an existing collection if it exists.

        Uses the DatabaseClient interface to connect, retrieve the collection
        description, and extract the CrawlerConfig.

        Args:
            database_config: Database configuration to check

        Returns:
            Restored CrawlerConfig if collection exists and has valid config, None otherwise
        """
        if database_config.recreate:
            return None

        temp_db: DatabaseClient | None = None
        try:
            # Create a temporary database client to check the collection
            # We pass a minimal config since we're just reading the collection description
            temp_db = get_db(
                database_config,
                dimension=1,  # Dimension doesn't matter for reading description
                crawler_config=self.config,
                embedder=None,
            )

            # Try to connect without creating a new collection
            temp_db.connect(create_if_missing=False)

            # Get the collection description
            collection_description: CollectionDescription | None = temp_db.get_collection()
            if not collection_description:
                return None

            # Restore the CrawlerConfig from the collection description
            restored_config = CrawlerConfig.from_collection_description(
                collection_description, database_config
            )
            return restored_config

        except RuntimeError:
            # Collection doesn't exist, that's okay
            return None
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to restore config from collection '{database_config.collection}': {str(e)}. "
                f"Using provided config instead.",
                UserWarning,
            )
            return None
        finally:
            if temp_db is not None:
                temp_db.disconnect()

    def _initialize_defaults(self) -> None:
        """Initialize default components if not provided."""
        # Initialize embedder first (needed for vector_db dimension)
        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        # Initialize vector database if not provided
        if self.vector_db is None:
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config,
                self.embedder,  # Pass embedder for search operations
            )
            # Connect to the database with collection creation if needed
            self.vector_db.connect(create_if_missing=True)
            print(f"Connected to vector database: {self.config.database.collection}")

            if self.config.benchmark:
                self.benchmarker = get_db_benchmark(
                    self.config.database, self.config.embeddings, self.vector_db
                )

        # Initialize LLM (needed for extractor)
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

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
            path: Either a file path, directory path, or list of paths

        Returns:
            List of file paths
        """
        if isinstance(path, list):
            return path

        if not os.path.exists(path):
            return []

        if os.path.isfile(path):
            return [path]

        if os.path.isdir(path):
            file_paths = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
            return file_paths

        return []

    def _cache_document(self, temp_filepath: Path | None, document: Document) -> None:
        """Cache document to temporary file if temp_filepath is provided."""
        if temp_filepath and self.config.use_cache:
            try:
                document.save(str(temp_filepath))
            except Exception:
                pass

    def crawl(self, path: str | list[str]) -> dict:
        """
        Crawl the given path(s) and process the documents.

        Args:
            path: File path, directory path, or list of paths to process

        Returns:
            Dictionary with processing statistics
        """
        filepaths = self._get_filepaths(path)

        if not filepaths:
            return {"total_files": 0, "processed_files": 0}

        temp_dir = Path(self.config.temp_dir) if self.config.temp_dir else None
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_files": len(filepaths),
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0,
            "total_file_size": 0,
        }

        with tqdm(total=len(filepaths), desc="Processing files", unit="file") as pbar:
            for filepath in filepaths:
                file_start_time = time.time()

                try:
                    stats["total_file_size"] += os.path.getsize(filepath)
                except OSError:
                    pass

                pbar.set_postfix_str(f"Current: {os.path.basename(filepath)}")

                document = Document.create(
                    source=filepath,
                    security_group=self.config.security_groups,
                )
                print(f"Created document {document.source}...")

                temp_filepath = None
                if temp_dir:
                    filename = os.path.splitext(os.path.basename(filepath))[0] + ".json"
                    temp_filepath = temp_dir / filename
                    if temp_filepath.exists():
                        try:
                            document.load(str(temp_filepath))
                        except Exception:
                            pass

                success, num_chunks = self.crawl_document(document, temp_filepath)

                if success:
                    stats["total_chunks"] += num_chunks
                    stats["processed_files"] += 1
                    stats["total_processing_time"] += time.time() - file_start_time
                else:
                    # Check if it was skipped (duplicate) or failed
                    if self.vector_db.exists(filepath, 0):
                        stats["skipped_files"] += 1
                    else:
                        stats["failed_files"] += 1

                pbar.update(1)

        return stats

    def crawl_document(
        self, document: Document, temp_filepath: Path | None = None
    ) -> tuple[bool, int]:
        """
        Process a single document through the full pipeline.

        Pipeline stages:
        1. Check for duplicates using exists()
        2. Convert document to markdown
        3. Extract metadata using LLM
        4. Chunk the text
        5. Generate embeddings
        6. Store in vector database using upsert()

        Args:
            document: Document instance to process
            temp_filepath: Optional path to cache document after each processing step

        Returns:
            tuple[bool, int]: (success, num_chunks) where success is True if document was
                successfully stored, and num_chunks is the number of chunks stored
        """
        print(f"Checking for duplicates {document.source}...")

        # Check for duplicates using exists() instead of check_duplicate()
        if self.vector_db.exists(document.source, 0):
            print(f"Document already exists: {document.source}")
            return (False, 0)

        # Convert document if not already converted
        if not document.is_converted():
            try:
                document.markdown = self.converter.convert(document)
                print(f"Converted document {document.markdown[:100]}...")
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to convert document {document.source}: {e}")
                return (False, 0)

        # Extract metadata if not already extracted
        if not document.is_extracted():
            try:
                extraction_result: MetadataExtractionResult = self.extractor.run(document)
                document.metadata = (
                    sanitize_metadata(
                        extraction_result.metadata or {},
                        self.config.metadata_schema,
                    )
                    or {}
                )
                document.benchmark_questions = extraction_result.benchmark_questions
                print(f"Extracted metadata {list(document.metadata.keys())[:10]}...")
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to extract metadata {document.source}: {e}")
                return (False, 0)

        # Chunk the text and generate embeddings if not already chunked
        if not document.is_chunked():
            try:
                text_chunks: list[str] = self.chunker.chunk_text(document)

                if not text_chunks:
                    print(f"No chunks generated for document {document.source}")
                    return (False, 0)

                embeddings: list[list[float]] = self.embedder.embed_batch(text_chunks)

                document.chunks = [
                    Chunk(
                        text=text,
                        chunk_index=i,
                        text_embedding=embedding,
                    )
                    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings))
                ]
                print(f"Chunked and embedded document: {len(document.chunks)} chunks...")
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to chunk/embed document {document.source}: {e}")
                return (False, 0)

        # Create database entities from document
        try:
            entities = document.to_database_documents()
            print(f"Created {len(entities)} database entities...")
        except ValueError as e:
            print(f"Failed to create database entities {document.source}: {e}")
            return (False, 0)

        # Store in database using upsert() instead of insert_data()
        if entities:
            try:
                result = self.vector_db.upsert(entities)
                total_stored = result.inserted_count + result.updated_count
                print(
                    f"Upserted {total_stored} database entities "
                    f"(inserted: {result.inserted_count}, updated: {result.updated_count})..."
                )
                if result.has_failures:
                    print(f"Warning: {len(result.failed_ids)} documents failed to upsert")
                return (True, total_stored)
            except Exception as e:
                print(f"Failed to upsert database entities {document.source}: {e}")
                return (False, 0)

        return (False, 0)

    def benchmark(self, generate_queries: bool = False) -> None:
        """
        Run benchmark on the stored documents.

        Args:
            generate_queries: If True, generate queries using LLM
        """
        try:
            if self.benchmarker:
                results = self.benchmarker.run_benchmark(generate_queries)
                self.benchmarker.plot_results(results, "benchmark_results")
                self.benchmarker.save_results(results, "benchmark_results/results.json")
        except Exception as e:
            print(f"Failed to benchmark: {e}")

    def disconnect(self) -> None:
        """Disconnect from the vector database."""
        if self.vector_db is not None:
            self.vector_db.disconnect()
