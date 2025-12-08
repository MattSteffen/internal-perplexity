import os
import time
from pathlib import Path

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
    get_db,
    get_db_benchmark,
)
from .vector_db.milvus_client import MilvusDB
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
        print("Restoring config from collection...")
        print("TYPE OF DATABASE CONFIG ->", type(database_config))
        print("DATABASE CONFIG ->", database_config)
        # Only restore for Milvus collections
        if database_config.provider != "milvus":
            return None
        print("Got provider")

        # Skip if recreate is True (we want to overwrite)
        if database_config.recreate:
            return None

        try:
            # Get collection description using the database client's static method
            collection_description = MilvusDB.get_collection_description(database_config)
            if not collection_description:
                return None

            # Convert CollectionDescription to CrawlerConfig using the canonical method
            restored_config = CrawlerConfig.from_collection_description(
                collection_description, database_config
            )
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
        # Restore config from collection if it exists and recreate=False, then merge with provided config
        if self.vector_db is None:
            # TODO: This is not working, so we're not using it for now.
            # restored_config = self._restore_config_from_collection(self.config.database)
            # if restored_config is not None:
            #     # Merge: provided config overrides stored config
            #     self.config = restored_config.merge_with(self.config)
            #     print("Merged config ->", self.config.database)
            # Initialize embedder first (needed for vector_db dimension)
            if self.embedder is None:
                self.embedder = get_embedder(self.config.embeddings)

            # Pass the crawler config to be stored in collection description
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config,
            )
            print("Vector db ->", self.vector_db)
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

    def _cache_document(self, temp_filepath: Path | None, document: Document) -> None:
        """Cache document to temporary file if temp_filepath is provided."""
        if temp_filepath and self.config.use_cache:
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

        temp_dir = Path(self.config.temp_dir) if self.config.temp_dir else None
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

                # Create document
                document = Document.create(
                    source=filepath,
                    security_group=self.config.security_groups,
                )
                print(f"Created document {document.source}...")

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

                # Process document through pipeline
                success, num_chunks = self.crawl_document(document, temp_filepath)

                if success:
                    stats["total_chunks"] += num_chunks
                    file_time = time.time() - file_start_time
                    stats["processed_files"] += 1
                    stats["total_processing_time"] += file_time
                else:
                    # Check if it was skipped (duplicate) or failed
                    if self.vector_db.check_duplicate(filepath, 0):
                        stats["skipped_files"] += 1
                    else:
                        stats["failed_files"] += 1

                pbar.update(1)

    def crawl_document(self, document: Document, temp_filepath: Path | None = None) -> tuple[bool, int]:
        """
        Process a single document through the full pipeline: check duplicates, convert,
        extract, chunk, embed, and insert into the database.

        Args:
            document: Document instance to process
            temp_filepath: Optional path to cache document after each processing step

        Returns:
            tuple[bool, int]: (success, num_chunks) where success is True if document was
                successfully inserted, and num_chunks is the number of chunks inserted (0 if failed)
        """
        print(f"Checking for duplicates {document.source}...")
        # Check for duplicates
        if self.vector_db.check_duplicate(document.source, 0):
            return (False, 0)

        # Convert document if not already converted
        if not document.is_converted():
            try:
                self.converter.convert(document)
                print(f"Converted document {document.markdown[:100]}...")
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to convert document {document.source}: {e}")
                return (False, 0)

        # Extract metadata if not already extracted
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
                print(f"Extracted metadata {list(document.metadata.keys())[:10]}...")
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to extract metadata {document.source}: {e}")
                return (False, 0)

        # Chunk the text if not already chunked
        if not document.is_chunked():
            document.chunks = self.chunker.chunk_text(document)
            print(f"Chunked document {len(document.chunks)} chunks...")
            self._cache_document(temp_filepath, document)

        # Generate embeddings for chunks if not already embedded
        if not document.text_embeddings:
            try:
                document.text_embeddings = self.embedder.embed_batch(document.chunks)
                self._cache_document(temp_filepath, document)
            except Exception as e:
                print(f"Failed to chunk document {document.source}: {e}")
                return (False, 0)

        # Create database entities from document
        try:
            entities = document.to_database_documents()
            print(f"Created {len(entities)} database entities...")
        except ValueError as e:
            print(f"Failed to create database entities {document.source}: {e}")
            return (False, 0)

        # Store in database
        if entities:
            try:
                self.vector_db.insert_data(entities)
                print(f"Inserted {len(entities)} database entities...")
                return (True, len(entities))
            except Exception as e:
                print(f"Failed to insert database entities {document.source}: {e}")
                return (False, 0)

        return (False, 0)

    def with_llm(self, llm: LLMConfig | LLM) -> "Crawler":
        """
        Override the LLM configuration or instance and return a new Crawler instance.
        
        Args:
            llm: Either an LLMConfig object or an LLM instance
            
        Returns:
            New Crawler instance with updated LLM
        """
        if isinstance(llm, LLMConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"llm": llm})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=None,  # Will be recreated with new LLM
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=None,  # Will be recreated with new config
                chunker=self.chunker,
            )
        elif isinstance(llm, LLM):
            # Instance provided - use directly, extract config if possible
            # Try to get config from instance, otherwise keep existing config
            llm_config = getattr(llm, "config", None) if hasattr(llm, "config") else self.config.llm
            new_config = self.config.model_copy(update={"llm": llm_config})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=None,  # Will be recreated with new LLM
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=llm,  # Use provided instance directly
                chunker=self.chunker,
            )
        else:
            raise TypeError(f"Expected LLMConfig or LLM instance, got {type(llm)}")

    def with_embedder(self, embedder: EmbedderConfig | Embedder) -> "Crawler":
        """
        Override the embedder configuration or instance and return a new Crawler instance.
        
        Args:
            embedder: Either an EmbedderConfig object or an Embedder instance
            
        Returns:
            New Crawler instance with updated embedder
        """
        if isinstance(embedder, EmbedderConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"embeddings": embedder})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=None,  # Will be recreated with new embedder dimension
                embedder=None,  # Will be recreated with new config
                llm=self.llm,
                chunker=self.chunker,
            )
        elif isinstance(embedder, Embedder):
            # Instance provided - use directly, extract config if possible
            embedder_config = getattr(embedder, "config", None) if hasattr(embedder, "config") else self.config.embeddings
            new_config = self.config.model_copy(update={"embeddings": embedder_config})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=None,  # Will be recreated with new embedder dimension
                embedder=embedder,  # Use provided instance directly
                llm=self.llm,
                chunker=self.chunker,
            )
        else:
            raise TypeError(f"Expected EmbedderConfig or Embedder instance, got {type(embedder)}")

    def with_converter(self, converter: ConverterConfig | Converter) -> "Crawler":
        """
        Override the converter configuration or instance and return a new Crawler instance.
        
        Args:
            converter: Either a ConverterConfig object or a Converter instance
            
        Returns:
            New Crawler instance with updated converter
        """
        if isinstance(converter, ConverterConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"converter": converter})
            return Crawler(
                config=new_config,
                converter=None,  # Will be recreated with new config
                extractor=self.extractor,
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        elif isinstance(converter, Converter):
            # Instance provided - use directly, extract config if possible
            converter_config = getattr(converter, "config", None) if hasattr(converter, "config") else self.config.converter
            new_config = self.config.model_copy(update={"converter": converter_config})
            return Crawler(
                config=new_config,
                converter=converter,  # Use provided instance directly
                extractor=self.extractor,
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        else:
            raise TypeError(f"Expected ConverterConfig or Converter instance, got {type(converter)}")

    def with_extractor(self, extractor: MetadataExtractorConfig | MetadataExtractor) -> "Crawler":
        """
        Override the extractor configuration or instance and return a new Crawler instance.
        
        Args:
            extractor: Either a MetadataExtractorConfig object or a MetadataExtractor instance
            
        Returns:
            New Crawler instance with updated extractor
        """
        if isinstance(extractor, MetadataExtractorConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"extractor": extractor})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=None,  # Will be recreated with new config
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        elif isinstance(extractor, MetadataExtractor):
            # Instance provided - use directly, extract config if possible
            extractor_config = getattr(extractor, "config", None) if hasattr(extractor, "config") else self.config.extractor
            new_config = self.config.model_copy(update={"extractor": extractor_config})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=extractor,  # Use provided instance directly
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        else:
            raise TypeError(f"Expected MetadataExtractorConfig or MetadataExtractor instance, got {type(extractor)}")

    def with_chunking(self, chunking: ChunkingConfig | Chunker) -> "Crawler":
        """
        Override the chunking configuration or instance and return a new Crawler instance.
        
        Args:
            chunking: Either a ChunkingConfig object or a Chunker instance
            
        Returns:
            New Crawler instance with updated chunker
        """
        if isinstance(chunking, ChunkingConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"chunking": chunking})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=None,  # Will be recreated with new config
            )
        elif isinstance(chunking, Chunker):
            # Instance provided - use directly, extract config if possible
            chunking_config = getattr(chunking, "config", None) if hasattr(chunking, "config") else self.config.chunking
            new_config = self.config.model_copy(update={"chunking": chunking_config})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self.llm,
                chunker=chunking,  # Use provided instance directly
            )
        else:
            raise TypeError(f"Expected ChunkingConfig or Chunker instance, got {type(chunking)}")

    def with_database(self, database: DatabaseClientConfig | DatabaseClient) -> "Crawler":
        """
        Override the database configuration or instance and return a new Crawler instance.
        
        Args:
            database: Either a DatabaseClientConfig object or a DatabaseClient instance
            
        Returns:
            New Crawler instance with updated database client
        """
        if isinstance(database, DatabaseClientConfig):
            # Config provided - update config and recreate instance
            new_config = self.config.model_copy(update={"database": database})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=None,  # Will be recreated with new database config
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        elif isinstance(database, DatabaseClient):
            # Instance provided - use directly, extract config if possible
            database_config = getattr(database, "config", None) if hasattr(database, "config") else self.config.database
            new_config = self.config.model_copy(update={"database": database_config})
            return Crawler(
                config=new_config,
                converter=self.converter,
                extractor=self.extractor,
                vector_db=database,  # Use provided instance directly
                embedder=self.embedder,
                llm=self.llm,
                chunker=self.chunker,
            )
        else:
            raise TypeError(f"Expected DatabaseClientConfig or DatabaseClient instance, got {type(database)}")

    def benchmark(self, generate_queries: bool = False) -> None:
        try:
            if self.benchmarker:
                results = self.benchmarker.run_benchmark(generate_queries)
                self.benchmarker.plot_results(results, "benchmark_results")
                self.benchmarker.save_results(results, "benchmark_results/results.json")
        except Exception as e:
            print(f"Failed to benchmark: {e}")