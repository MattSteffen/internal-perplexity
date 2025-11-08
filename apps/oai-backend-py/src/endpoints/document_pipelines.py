"""Document pipeline upload endpoint handler."""

import json
import uuid
from json import dumps, loads
from pathlib import Path
from typing import Any

from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.auth_utils import verify_token
from src.crawler import Crawler, CrawlerConfig
from src.crawler.llm.embeddings import EmbedderConfig
from src.crawler.llm.llm import LLMConfig
from src.crawler.vector_db.database_client import DatabaseClientConfig
from src.endpoints.pipeline_registry import get_registry


class ConfigOverrides(BaseModel):
    """Configuration overrides for pipeline processing."""

    embedding_model: str | None = None
    llm_model: str | None = None
    vision_model: str | None = None
    security_groups: list[str] | None = None


class ProcessedDocument(BaseModel):
    """Response model for document processing endpoint."""

    metadata: dict[str, Any]
    file_name: str
    file_size: int | None = None


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    message: str
    pipeline_name: str
    document_id: str | None = None
    chunks_created: int = 0
    processing_time_sec: float = 0.0


def _override_config(
    base_config: CrawlerConfig,
    overrides: ConfigOverrides,
) -> CrawlerConfig:
    """Apply configuration overrides to base pipeline config.

    Args:
        base_config: Base pipeline configuration
        overrides: Configuration overrides from request

    Returns:
        Modified CrawlerConfig with overrides applied
    """
    # Start with base config
    config_dict = base_config.model_dump()

    # Override LLM models if provided
    if overrides.embedding_model:
        embeddings = EmbedderConfig.ollama(
            model=overrides.embedding_model,
            base_url=base_config.embeddings.base_url,
        )
        config_dict["embeddings"] = embeddings

    if overrides.llm_model:
        llm = LLMConfig.ollama(
            model_name=overrides.llm_model,
            base_url=base_config.llm.base_url,
            structured_output=base_config.llm.structured_output,
        )
        config_dict["llm"] = llm
        # Also update extractor's LLM config
        if config_dict.get("extractor"):
            config_dict["extractor"]["llm"] = llm.model_dump()

    if overrides.vision_model:
        vision_llm = LLMConfig.ollama(
            model_name=overrides.vision_model,
            base_url=base_config.vision_llm.base_url,
        )
        config_dict["vision_llm"] = vision_llm
        # Update converter's VLM config if it exists
        if config_dict.get("converter") and isinstance(config_dict["converter"], dict):
            if "vlm_config" in config_dict["converter"]:
                config_dict["converter"]["vlm_config"] = vision_llm.model_dump()

    # Override security groups if provided
    if overrides.security_groups is not None:
        config_dict["security_groups"] = overrides.security_groups

    # Recreate config from dict
    modified_config = CrawlerConfig(**config_dict)

    return modified_config


async def process_document(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    user: dict[str, Any] = Depends(verify_token),
) -> ProcessedDocument:
    """Process a document to extract metadata without uploading.

    curl -X POST http://localhost:8000/v1/documents/process \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F "collection_name=my_collection" (optional)

    Args:
        file: Uploaded file to process
        collection_name: Optional collection name to load config from
        user: Authenticated user information from token

    Returns:
        ProcessedDocument with extracted metadata

    Raises:
        HTTPException: Various error codes based on failure type
    """
    import uuid
    from pathlib import Path

    from src.crawler import CrawlerConfig
    from src.crawler.converter import create_converter
    from src.crawler.converter.types import DocumentInput
    from src.crawler.document import Document
    from src.crawler.llm.embeddings import EmbedderConfig
    from src.crawler.llm.llm import LLMConfig
    from src.crawler.vector_db.database_client import DatabaseClientConfig
    from src.endpoints.pipeline_registry import get_registry

    temp_file_path: Path | None = None

    try:
        # Load config from collection or use defaults
        if collection_name:
            milvus_token: str = user.get("milvus_token", "")
            if not milvus_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Milvus token is required for database access",
                )
            config = load_config_from_collection(collection_name, milvus_token)
        else:
            # Use default config for processing
            registry = get_registry()
            if registry.has_pipeline("irads"):
                config = registry.get_config("irads")
            else:
                # Fallback to minimal config
                config = CrawlerConfig(
                    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
                    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
                    vision_llm=LLMConfig.ollama(model_name="llava:latest"),
                    database=DatabaseClientConfig(
                        provider="milvus",
                        collection="temp",
                        host="localhost",
                        port=19530,
                    ),
                    metadata_schema={},
                )

        # Save uploaded file to temporary location
        file_ext = Path(file.filename or "document").suffix
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"process_{uuid.uuid4()}{file_ext}"

        # Write file content
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = len(content)

        # Create document
        document = Document.create(source=str(temp_file_path))

        # Convert to markdown
        converter = create_converter(config.converter)
        doc_input = DocumentInput.from_document(document)
        converted = converter.convert(doc_input)
        document.markdown = converted.markdown

        # Extract metadata
        from src.crawler.extractor.extractor import MetadataExtractor
        from src.crawler.llm.llm import get_llm

        llm = get_llm(config.llm)
        extractor = MetadataExtractor(config=config.extractor, llm=llm)
        extraction_result = extractor.run(document)
        metadata = extraction_result.metadata or {}

        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        return ProcessedDocument(
            metadata=metadata,
            file_name=file.filename or "document",
            file_size=file_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up temporary file on error
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        ) from e


def load_config_from_collection(
    collection_name: str,
    milvus_token: str,
) -> CrawlerConfig:
    """Load CrawlerConfig from a collection's description.

    Args:
        collection_name: Name of the collection
        milvus_token: Milvus authentication token

    Returns:
        CrawlerConfig instance loaded from collection description

    Raises:
        HTTPException: If collection not found or config invalid
    """

    from src.crawler.main import CrawlerConfig
    from src.milvus_client import get_milvus_client

    client = get_milvus_client(milvus_token)

    # Check if collection exists
    collections = client.list_collections()
    if collection_name not in collections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        )

    # Get collection description
    collection_info = client.describe_collection(collection_name)
    safe_dict = loads(dumps(collection_info, default=str))

    # Extract description JSON
    description_str = safe_dict.get("description", "")
    if not description_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{collection_name}' does not have a description with pipeline config",
        )

    try:
        description_json = loads(description_str) if isinstance(description_str, str) else description_str
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse collection description: {str(e)}",
        ) from e

    # Load config from description
    try:
        config = CrawlerConfig.from_collection_description(
            description_json=description_json,
            collection_name=collection_name,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid pipeline config in collection description: {str(e)}",
        ) from e

    return config


async def upload_document(
    pipeline_name: str | None = None,
    collection_name: str | None = None,
    file: UploadFile = File(...),
    config_overrides: str | None = Form(None),
    user: dict[str, Any] = Depends(verify_token),
) -> UploadResponse:
    """Upload and process a document through a predefined pipeline or collection.

    curl -X POST http://localhost:8000/v1/documents/upload/{pipeline_name} \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"embedding_model": "nomic-embed-text", "security_groups": ["group1"]}'

    OR

    curl -X POST http://localhost:8000/v1/documents/upload?collection_name=my_collection \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"security_groups": ["group1"]}'

    Args:
        pipeline_name: Name of the pipeline to use (required if collection_name not provided)
        collection_name: Name of the collection to upload to (loads config from collection description)
        file: Uploaded file to process
        config_overrides: Optional JSON string with configuration overrides
        user: Authenticated user information from token

    Returns:
        UploadResponse with processing results

    Raises:
        HTTPException: Various error codes based on failure type
    """
    import time

    start_time = time.time()
    temp_file_path: Path | None = None

    try:
        # Validate that either pipeline_name or collection_name is provided
        if not pipeline_name and not collection_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either pipeline_name or collection_name",
            )

        # Get Milvus token from user info (needed for both collection and pipeline paths)
        milvus_token: str = user.get("milvus_token", "")
        if not milvus_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Milvus token is required for database access",
            )

        # Load config from collection or pipeline
        if collection_name:
            # Load config from collection description
            base_config = load_config_from_collection(collection_name, milvus_token)
            pipeline_name = collection_name  # Use collection name for response
        else:
            # Validate pipeline exists
            if pipeline_name is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="pipeline_name is required when collection_name is not provided",
                )
            registry = get_registry()
            if not registry.has_pipeline(pipeline_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Pipeline '{pipeline_name}' not found. Available pipelines: {registry.list_pipelines()}",
                )

            # Get base pipeline config
            base_config = registry.get_config(pipeline_name)

        # Parse config overrides
        overrides = ConfigOverrides()
        if config_overrides:
            try:
                overrides_dict = json.loads(config_overrides)
                overrides = ConfigOverrides(**overrides_dict)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid config_overrides JSON: {str(e)}",
                ) from e

        # Apply config overrides
        config = _override_config(base_config, overrides)

        # Save uploaded file to temporary location
        file_ext = Path(file.filename or "document").suffix
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"upload_{uuid.uuid4()}{file_ext}"

        # Write file content
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Override database config to use JWT token directly
        # Create a custom database config that overrides the token property
        original_db_config = config.database

        class TokenDatabaseConfig(DatabaseClientConfig):
            """Database config that uses JWT token for authentication."""

            jwt_token: str = ""

            @property
            def token(self) -> str:
                """Return the JWT token for authentication."""
                return self.jwt_token

        # Create a new database config with JWT token
        token_db_config = TokenDatabaseConfig(
            provider=original_db_config.provider,
            collection=original_db_config.collection,
            host=original_db_config.host,
            port=original_db_config.port,
            username=original_db_config.username,
            password=original_db_config.password,
            partition=original_db_config.partition,
            recreate=original_db_config.recreate,
            collection_description=original_db_config.collection_description,
            jwt_token=milvus_token,
        )
        config.database = token_db_config

        # Initialize crawler
        try:
            crawler = Crawler(config=config)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to vector database: {str(e)}",
            ) from e

        # Process file through crawler
        try:
            crawler.crawl(str(temp_file_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}",
            ) from e

        # Get document ID from the processed file
        # The crawler creates documents with UUIDs, but we don't have direct access
        # For now, use the filename as document identifier
        document_id = temp_file_path.stem

        processing_time = time.time() - start_time

        return UploadResponse(
            message="Document processed and uploaded successfully",
            pipeline_name=pipeline_name,
            document_id=document_id,
            chunks_created=0,  # Crawler doesn't return stats, would need to modify
            processing_time_sec=processing_time,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing document: {str(e)}",
        ) from e
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors


async def upload_document_to_collection(
    collection_name: str,
    file: UploadFile = File(...),
    metadata: str = Form(...),
    user: dict[str, Any] = Depends(verify_token),
) -> UploadResponse:
    """Upload a document to a collection with metadata.

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'metadata={"title":"Example","author":"John Doe"}'

    Args:
        collection_name: Name of the collection to upload to
        file: Uploaded file to process
        metadata: JSON string with document metadata
        user: Authenticated user information from token

    Returns:
        UploadResponse with processing results

    Raises:
        HTTPException: Various error codes based on failure type
    """
    # Parse metadata JSON
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metadata JSON: {str(e)}",
        ) from e

    # Get Milvus token from user info
    milvus_token: str = user.get("milvus_token", "")
    if not milvus_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Milvus token is required for database access",
        )

    # Load config from collection
    # TODO: Is this used?
    # base_config = load_config_from_collection(collection_name, milvus_token)

    # Merge metadata into config_overrides as document metadata
    # The metadata will be used when processing the document
    config_overrides_dict = {"metadata": metadata_dict}

    # Convert to JSON string for upload_document
    config_overrides_json = json.dumps(config_overrides_dict)

    # Call existing upload logic
    return await upload_document(
        pipeline_name=None,
        collection_name=collection_name,
        file=file,
        config_overrides=config_overrides_json,
        user=user,
    )


# Initialize pipeline registry with predefined pipelines
def _initialize_pipelines() -> None:
    """Register all predefined pipelines in the registry."""
    from src.crawler.arxiv_math import create_arxiv_math_config
    from src.crawler.irads import create_irad_config

    registry = get_registry()

    # Register irads pipeline
    registry.register("irads", create_irad_config)

    # Register arxiv_math pipeline
    registry.register("arxiv_math", create_arxiv_math_config)


# Initialize pipelines on module import
_initialize_pipelines()
