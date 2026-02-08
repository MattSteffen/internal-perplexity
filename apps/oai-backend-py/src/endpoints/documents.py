"""Document pipeline upload endpoint handler."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from crawler import Crawler, CrawlerConfig
from crawler.document import Document
from crawler.vector_db import CollectionDescription
from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from pymilvus import MilvusClient  # type: ignore

from src.auth_utils import verify_token
from src.endpoints.pipeline_registry import get_registry

logger = logging.getLogger(__name__)


class ConfigOverrides(BaseModel):
    """Configuration overrides for pipeline processing."""

    embedding_model: str | None = None
    llm_model: str | None = None
    vision_model: str | None = None
    security_groups: list[str] | None = None


class ProcessedDocument(BaseModel):
    """Response model for document processing endpoint."""

    metadata: dict[str, Any]
    markdown_content: str
    file_name: str
    file_size: int | None = None


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    message: str
    collection_name: str
    document_id: str | None = None
    chunks_created: int = 0
    processing_time_sec: float = 0.0


def _load_config_from_collection(collection_name: str, milvus_client: MilvusClient) -> tuple[CrawlerConfig, CollectionDescription]:
    """Load CrawlerConfig and CollectionDescription from a collection. Raises HTTPException on failure."""
    if not milvus_client.has_collection(collection_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        )
    description = milvus_client.describe_collection(collection_name).get("description", "")
    if not description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{collection_name}' does not have a description",
        )
    collection_desc = CollectionDescription.from_json(description)
    if not collection_desc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse collection description for '{collection_name}'",
        )
    return collection_desc.collection_config, collection_desc


def _parse_security_groups(security_groups: str | None, default: list[str]) -> list[str]:
    """Parse security_groups JSON string or return default. Raises HTTPException on invalid JSON."""
    if not security_groups:
        return default
    try:
        parsed = json.loads(security_groups)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid security_groups JSON format",
        )


async def process_document(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    user: dict[str, Any] = Depends(verify_token),
    milvus_client: MilvusClient | None = None,
) -> ProcessedDocument:
    """Process a document to extract metadata and markdown without uploading.

    Returns both the extracted metadata and converted markdown content.

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/process \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf"

    Args:
        file: Uploaded file to process
        collection_name: Optional collection name to load config from
        user: Authenticated user information from token

    Returns:
        ProcessedDocument with extracted metadata and markdown content

    Raises:
        HTTPException: Various error codes based on failure type
    """
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
            if milvus_client is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Milvus client dependency is missing",
                )
            config, _ = _load_config_from_collection(collection_name, milvus_client)
        else:
            registry = get_registry()
            config = CrawlerConfig.from_dict(registry.get("standard"))

        # Save uploaded file to temporary location
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / file.filename

        # Write file content
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = len(content)

        # Create document
        document = Document.create(source=str(temp_file_path), security_group=config.security_groups)
        print("Created document", document.source)

        crawler = Crawler(config=config)
        document.markdown = crawler.converter.convert(document)
        print("Converted document", document.markdown[:100])
        metadata_result = crawler.extractor.run(document)
        print("Metadata result", metadata_result.metadata)

        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        return ProcessedDocument(
            metadata=metadata_result.metadata,
            markdown_content=document.markdown or "",
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


async def upload_document(
    collection_name: str,
    user: dict[str, Any],
    file: UploadFile | None = None,
    markdown_content: str | None = None,
    metadata_override: str | None = None,
    security_groups: str | None = None,
    milvus_client: MilvusClient | None = None,
) -> UploadResponse:
    """Upload and process a document to a collection.

    Both modes use the crawler's crawl_document pipeline. The crawler skips
    convert and extract when markdown/metadata are already set on the document.

    Modes:
    1. File: upload a raw file; crawler converts, extracts, chunks, embeds, upserts.
    2. Pre-processed: provide markdown_content (and optional metadata_override);
       crawler skips convert/extract and runs chunk, embed, upsert.

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf"

    Pre-processed:

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F 'markdown_content=# My Document...' \
      -F 'metadata_override={"title": "My Doc", "author": ["John"]}' \
      -F 'security_groups=["public", "team_a"]'

    Args:
        collection_name: Name of the collection to upload to
        file: Optional uploaded file to process (mutually exclusive with markdown_content)
        markdown_content: Optional pre-converted markdown (skips conversion)
        metadata_override: Optional JSON string with pre-extracted metadata (skips extraction)
        security_groups: Optional JSON string with security groups for the document
        user: Authenticated user information from token

    Returns:
        UploadResponse with processing results

    Raises:
        HTTPException: Various error codes based on failure type
    """
    start_time = time.time()
    temp_file_path: Path | None = None

    try:
        # Validate that collection_name is provided
        if not collection_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="collection_name is required",
            )

        # Validate input: must have either file or markdown_content
        if file is None and markdown_content is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either 'file' or 'markdown_content'",
            )

        # Get Milvus token from user info
        milvus_token: str = user.get("milvus_token", "")
        if not milvus_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Milvus token is required for database access",
            )

        # Parse username from token for RBAC check
        # TODO: There should be a function to parse the token into a username and password.
        if ":" not in milvus_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
            )
        username, password = milvus_token.split(":", 1)

        logger.info(f"Loading config from collection '{collection_name}'...")
        if milvus_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Milvus client dependency is missing",
            )
        config, collection_desc = _load_config_from_collection(collection_name, milvus_client)
        db = milvus_client

        # RBAC check: verify user has write access to collection
        # Check if user's roles include any of the collection's security groups
        try:
            user_info = db.describe_user(username)
            user_roles = set(user_info.get("roles", []))
            collection_security_groups = set(collection_desc.collection_security_groups or ["public"])

            # Check for overlap between user roles and collection security groups
            if not user_roles.intersection(collection_security_groups):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User '{username}' does not have write access to collection '{collection_name}'. " f"Required roles: {list(collection_security_groups)}, user roles: {list(user_roles)}",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to check user permissions for '{username}': {e}")
            # Continue if RBAC check fails (graceful degradation)

        doc_security_groups = _parse_security_groups(security_groups, config.security_groups or ["public"])

        # Override database config to use user's token
        config.database = config.database.copy_with_overrides(
            username=username,
            password=password,
            recreate=False,
        )

        # Initialize crawler
        try:
            logger.info("Initializing crawler...")
            crawler = Crawler(config=config)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to vector database: {str(e)}",
            ) from e

        # Build document and run crawler pipeline (both modes use crawl_document)
        if markdown_content is not None:
            # Pre-processed mode: markdown and metadata set; crawler skips convert/extract
            logger.info("Using pre-processed markdown content...")
            metadata = {}
            if metadata_override:
                try:
                    metadata = json.loads(metadata_override)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid metadata_override JSON format",
                    )
            document = Document.create(
                source=f"api_upload_{uuid.uuid4()}",
                security_group=doc_security_groups,
            )
            document.markdown = markdown_content
            document.metadata = metadata
        else:
            # File upload mode: save to temp path, optional metadata override
            logger.info("Processing uploaded file...")
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file_path = temp_dir / file.filename
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            document = Document.create(
                source=str(temp_file_path),
                security_group=doc_security_groups,
            )

        try:
            logger.info("Crawling document...")
            success, num_chunks = crawler.crawl_document(document)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}",
            ) from e

        processing_time = time.time() - start_time
        message = "Document already exists" if not success else "Document processed and uploaded successfully"
        return UploadResponse(
            message=message,
            collection_name=collection_name,
            document_id=document.document_id,
            chunks_created=num_chunks if success else 0,
            processing_time_sec=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing document: {str(e)}")
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
                pass
