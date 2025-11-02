"""Collections endpoint handler."""

from json import dumps, loads
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from src.milvus_client import get_milvus_client


class CollectionMetadata(BaseModel):
    """Collection metadata model.

    Wraps Milvus collection information in a type-safe way.
    """

    # Allow any additional fields from Milvus
    model_config = {"extra": "allow"}

    def __init__(self, **data: Any) -> None:
        """Initialize collection metadata from Milvus data."""
        super().__init__(**data)


class CollectionsResponse(BaseModel):
    """Response model for collections listing endpoint."""

    collections: list[str]
    collection_metadata: dict[str, CollectionMetadata]


async def list_collections() -> CollectionsResponse:
    """Handle collections listing requests.

    Returns all collections from Milvus with their metadata.

    curl -X GET http://localhost:8000/v1/collections
    """
    try:
        client = get_milvus_client()

        # List all collections
        collections = client.list_collections()

        # Build response with collection names and metadata
        collection_metadata: dict[str, CollectionMetadata] = {}

        for collection_name in collections:
            try:
                # Get collection description/statistics
                # MilvusClient.describe_collection returns collection info dict
                collection_info = client.describe_collection(collection_name)
                # Convert protobuf objects to serializable Python types
                safe_dict = loads(dumps(collection_info, default=str))

                # Ensure safe_dict is a dict and includes the name
                if isinstance(safe_dict, dict):
                    # Add collection name if not present
                    if "name" not in safe_dict:
                        safe_dict["name"] = collection_name
                    collection_metadata[collection_name] = CollectionMetadata(**safe_dict)
                else:
                    # If it's not a dict, wrap it
                    collection_metadata[collection_name] = CollectionMetadata(
                        name=collection_name,
                        description=safe_dict,
                    )

            except Exception as e:
                # If we can't get metadata for a collection, include minimal info
                collection_metadata[collection_name] = CollectionMetadata(
                    name=collection_name,
                    error=f"Failed to retrieve metadata: {str(e)}",
                )

        return CollectionsResponse(
            collections=collections,
            collection_metadata=collection_metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}",
        ) from e
