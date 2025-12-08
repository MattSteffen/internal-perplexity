"""Document pipeline upload endpoint handler."""

import json
import uuid
from json import dumps, loads
from pathlib import Path
from typing import Any

from crawler import Crawler, CrawlerConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import CollectionDescription, DatabaseClientConfig
from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.auth_utils import verify_token
from src.milvus_client import get_milvus_client
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
    collection_name: str
    document_id: str | None = None
    chunks_created: int = 0
    processing_time_sec: float = 0.0


# TODO: Don't use -F collection name, use a post request with the collection name in the body.
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

    from crawler.document import Document

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
            db = get_milvus_client(milvus_token)
            if db.has_collection(collection_name):
                description = db.describe_collection(collection_name).get('description', '')
                if description:
                    config = CrawlerConfig.from_collection_description(description, milvus_token)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Collection '{collection_name}' does not have a description",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection '{collection_name}' not found",
                )
        else:
            # Use default config for processing
            registry = get_registry()
            config = registry.get_config("default")

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
        document = Document.create(source=str(temp_file_path), security_group=config.security_groups)

        crawler = Crawler(config=config)
        crawler.converter.convert(document)
        metadata = crawler.extractor.run(document)

        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        return ProcessedDocument(
            metadata=metadata.metadata,
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

# TODO: This should use the same process as in process_document, but instead of just getting the metadata, it should upload the document to the collection by calling crawler.crawl_document
# TODO: Should remove all the pipeline_name logic and just use the collection_name
# TODO: This should be the url /v1/upload, have the collection name be in the body instead of the query parameter
async def upload_document(
    collection_name: str | None = None,
    file: UploadFile = File(...),
    config_overrides: str | None = Form(None),
    user: dict[str, Any] = Depends(verify_token),
) -> UploadResponse:
    """Upload and process a document through a predefined pipeline or collection.

    curl -X POST http://localhost:8000/v1/documents/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"embedding_model": "nomic-embed-text", "security_groups": ["group1"]}'

    OR

    curl -X POST http://localhost:8000/v1/documents/upload?collection_name=my_collection \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"security_groups": ["group1"]}'

    Args:
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
        # Validate that collection_name is provided
        if not collection_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either collection_name",
            )

        # Get Milvus token from user info (needed for both collection and pipeline paths)
        milvus_token: str = user.get("milvus_token", "")
        if not milvus_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Milvus token is required for database access",
            )

        # Load config from collection or pipeline
        print("Loading config from collection...")
        if collection_name:
            db = get_milvus_client(milvus_token)
            if db.has_collection(collection_name):
                description = db.describe_collection(collection_name).get('description', '')
                if description:
                    collection_desc = CollectionDescription.from_json(description)
                    config = collection_desc.collection_config
                    print("Collection config ->", config.database)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Collection '{collection_name}' does not have a description",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection '{collection_name}' not found",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide collection_name",
            )

        # overrides = CrawlerConfig.from_dict(config_overrides) # TODO: Make sure this works
        # config = config.merge_with(overrides)

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
        username, password = milvus_token.split(":")
        config.database.username = username
        config.database.password = password
        config.database.recreate = False

        # Initialize crawler
        try:
            print("Initializing crawler...")
            crawler = Crawler(config=config)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to vector database: {str(e)}",
            ) from e

        # Process file through crawler
        try:
            print("Crawling document...")
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
            collection_name=collection_name,
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

