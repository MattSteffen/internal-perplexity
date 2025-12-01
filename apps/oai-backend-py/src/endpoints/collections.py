"""Collections endpoint handler."""

import logging
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from src.milvus_client import get_milvus_client

logger = logging.getLogger(__name__)

# Import CollectionDescription for parsing collection descriptions
try:
    from crawler.vector_db import CollectionDescription
except ImportError:
    raise ImportError("crawler package not available")


class CollectionInfo(BaseModel):
    """Collection info model."""

    description: str
    metadata_schema: dict[str, Any]
    pipeline_name: str
    num_documents: int
    required_roles: list[str]


class CollectionsResponse(BaseModel):
    """Response model for collections listing endpoint."""

    collection_names: list[str]
    collections: dict[str, CollectionInfo]


class RolesResponse(BaseModel):
    """Response model for roles listing endpoint."""

    roles: list[str]


class User(BaseModel):
    """User model with id, name, and roles."""

    id: str
    name: str
    roles: list[str]


class UsersResponse(BaseModel):
    """Response model for users listing endpoint."""

    users: list[User]


class CreateCollectionRequest(BaseModel):
    """Request model for creating a new collection."""

    collection_name: str = Field(..., min_length=1, description="Name of the collection to create")
    pipeline_name: str | None = Field(None, description="Name of predefined pipeline to use")
    custom_config: dict[str, Any] | None = Field(None, description="Full CrawlerConfig dict for custom pipeline")
    config_overrides: dict[str, Any] | None = Field(None, description="Configuration overrides for predefined pipeline")
    description: str | None = Field(None, description="Human-readable description of the collection")
    roles: list[str] = Field(..., description="List of roles to grant read access to the collection", default_factory=lambda: ["admin"])
    metadata_schema: dict[str, Any] | None = Field(None, description="Optional JSON schema override for metadata")


class CreateCollectionResponse(BaseModel):
    """Response model for collection creation endpoint."""

    collection_name: str
    message: str
    roles: list[str]


async def list_collections(token: str | None = None) -> CollectionsResponse:
    """Handle collections listing requests.

    Returns all collections from Milvus with their metadata.

    curl -X GET http://localhost:8000/v1/collections
    """
    try:
        client = get_milvus_client(token)

        # List all collections
        collections = client.list_collections()

        # Build response with collection names and metadata
        collection_descriptions: dict[str, CollectionInfo] = {}

        for collection_name in collections:
            try:
                collection_info: CollectionInfo = CollectionInfo(
                    description="",
                    metadata_schema={},
                    pipeline_name="",
                    num_documents=0,
                    required_roles=[],
                )
                # Get collection description/statistics
                # MilvusClient.describe_collection returns collection info dict
                collection_info_dict = client.describe_collection(collection_name)
                collection_info.num_documents = client.get_collection_stats(collection_name)["row_count"]  # TODO: This is not num_documents, but is num_chunks, fix later

                # Extract and parse collection description to get library_context and metadata_schema
                try:
                    collection_desc = CollectionDescription.from_json(collection_info_dict["description"])
                    collection_info.pipeline_name = collection_desc.pipeline_name
                    collection_info.metadata_schema = collection_desc.metadata_schema
                    collection_info.description = collection_desc.description
                    # Get the roles that are required to access the collection
                    # Each role is granted a privilege group, if that group is in the collection_security_groups, then the role can access the collection
                    all_roles = client.list_roles()
                    for r in all_roles:
                        role_desc = client.describe_role(r)
                        for privilege in role_desc["privileges"]:
                            if privilege in collection_desc.collection_security_groups:
                                collection_info.required_roles.append(r)
                except Exception as e:
                    raise ValueError(f"Failed to parse collection description for {collection_name}: {str(e)}") from e

                collection_descriptions[collection_name] = collection_info

            except Exception as e:
                # If we can't get metadata for a collection, include minimal info
                logger.warning(f"Failed to retrieve metadata for collection '{collection_name}': {str(e)}")

        return CollectionsResponse(
            collection_names=list(collection_descriptions.keys()),
            collections=collection_descriptions,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}",
        ) from e


async def list_roles(token: str | None = None) -> RolesResponse:
    """Handle roles listing requests.

    Returns all roles from Milvus with their privileges.

    curl -X GET http://localhost:8000/v1/roles \
      -H "Authorization: Bearer $TOKEN"
    """
    try:
        client = get_milvus_client(token)

        # List all roles
        role_names: list[str] = client.list_roles()
        return RolesResponse(roles=role_names)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list roles: {str(e)}",
        ) from e


async def list_users(token: str | None = None) -> UsersResponse:
    """Handle users listing requests.

    Returns all users from Milvus with their roles.

    curl -X GET http://localhost:8000/v1/users \
      -H "Authorization: Bearer $TOKEN"
    """
    try:
        client = get_milvus_client(token)

        # List all users
        usernames = client.list_users()

        # Get details for each user
        users: list[User] = []
        for username in usernames:
            try:
                user_info = client.describe_user(username)
                # user_info is {'user_name': user-name, 'roles': set(...)}
                roles_set = user_info.get("roles", set())
                # Convert set to list of strings
                if isinstance(roles_set, set):
                    roles_list = [str(r) for r in roles_set]
                elif isinstance(roles_set, list):
                    roles_list = [str(r) for r in roles_set]
                else:
                    roles_list = []
                users.append(User(id=username, name=username, roles=roles_list))
            except Exception as e:
                # If we can't get details for a user, include minimal info
                logger.warning(f"Failed to describe user '{username}': {str(e)}")
                users.append(User(id=username, name=username, roles=[]))

        return UsersResponse(users=users)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list users: {str(e)}",
        ) from e


async def create_collection(
    request: CreateCollectionRequest,
    token: str | None = None,
) -> CreateCollectionResponse:
    """Create a new collection with pipeline configuration and permissions.

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "pipeline_name": "irads",
        "config_overrides": {"embedding_model": "nomic-embed-text"},
        "default_permissions": "public"
      }'

    Args:
        request: Collection creation request with pipeline and permission settings
        token: Milvus authentication token

    Returns:
        CreateCollectionResponse with collection details

    Raises:
        HTTPException: Various error codes based on failure type
    """
    from crawler import CrawlerConfig
    from crawler.vector_db import DatabaseClientConfig

    from src.endpoints.document_pipelines import ConfigOverrides, _override_config
    from src.endpoints.pipeline_registry import get_registry

    # get the user name and password from the token
    user_name, password = token.split(":")
    if not user_name or not password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    try:
        client = get_milvus_client(token)

        # Validate collection name (basic Milvus naming rules)
        if not request.collection_name or not request.collection_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Collection name cannot be empty",
            )

        # Check if collection already exists
        existing_collections = client.list_collections()
        if request.collection_name in existing_collections:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection '{request.collection_name}' already exists",
            )

        # Validate that either pipeline_name OR custom_config is provided (not both)
        if request.pipeline_name and request.custom_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot specify both pipeline_name and custom_config",
            )

        if not request.pipeline_name and not request.custom_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must specify either pipeline_name or custom_config",
            )

        # Build CrawlerConfig
        # TODO: Distinguish between pipeline and collection config, the source of the config.
        config: CrawlerConfig
        if request.pipeline_name:
            # Validate pipeline exists
            registry = get_registry()
            if not registry.has_pipeline(request.pipeline_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Pipeline '{request.pipeline_name}' not found. Available pipelines: {registry.list_pipelines()}",
                )

            # Get base pipeline config
            base_config = registry.get_config(request.pipeline_name)

            # Apply config overrides if provided
            if request.config_overrides:
                overrides = ConfigOverrides(**request.config_overrides)
                config = _override_config(base_config, overrides)
            else:
                config = base_config

        else:
            # Custom config provided
            if request.custom_config is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="custom_config is required when pipeline_name is not provided",
                )
            try:
                config = CrawlerConfig.from_dict(request.custom_config)
                if request.default_permissions:
                    config.security_groups = [request.default_permissions]
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid custom_config: {str(e)}",
                ) from e

        # Override collection name in database config
        original_db_config = config.database
        db_config = DatabaseClientConfig(
            provider=original_db_config.provider,
            collection=request.collection_name,
            host=original_db_config.host,
            port=original_db_config.port,
            username=user_name,
            password=password,
            partition=original_db_config.partition,
            recreate=False,  # Don't recreate on API creation
            collection_description=request.description or original_db_config.collection_description,
        )
        config.database = db_config

        # Override metadata schema if provided
        if request.metadata_schema:
            config.metadata_schema = request.metadata_schema

        # Get embedding dimension
        from crawler.llm.embeddings import get_embedder

        embedder = get_embedder(config.embeddings)
        embedding_dimension = embedder.get_dimension()

        from crawler.vector_db.database_utils import get_db

        get_db(config.database, embedding_dimension, config)

        # Grant privileges to the collection
        # TODO: The privilege granted should be a specific one imported from crawler.
        for role in request.roles:
            # TODO: Should use admin client to grant privileges.
            client.grant_privilege_v2(role, "ClusterReadOnly", config.database.collection_name)

        return CreateCollectionResponse(
            collection_name=request.collection_name,
            message="Collection created successfully",
            roles=request.roles,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        ) from e
