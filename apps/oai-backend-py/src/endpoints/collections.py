"""Collections endpoint handler."""

import logging
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from src.auth_utils import extract_username_from_token
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
    num_documents: int
    num_chunks: int
    num_partitions: int
    required_roles: list[str]
    access_level: str = "public"  # One of: "public", "private", "admin"


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
    template_name: str | None = Field(None, description="Name of predefined template to use (e.g., 'standard', 'academic')")
    custom_config: dict[str, Any] | None = Field(None, description="Full CrawlerConfig dict for custom pipeline")
    config_overrides: dict[str, Any] | None = Field(None, description="Configuration overrides for predefined pipeline")
    description: str | None = Field(None, description="Human-readable description of the collection")
    roles: list[str] = Field(default_factory=lambda: ["public"], description="List of roles to grant read access to the collection")
    metadata_schema: dict[str, Any] | None = Field(None, description="Optional JSON schema override for metadata")
    access_level: str = Field(default="public", description="Access level: public, private, or admin")


class PipelineInfo(BaseModel):
    """Information about a pipeline template."""

    name: str
    description: str
    metadata_schema: dict[str, Any]
    chunk_size: int
    embedding_model: str
    llm_model: str


class PipelinesResponse(BaseModel):
    """Response model for pipelines listing endpoint."""

    pipelines: list[PipelineInfo]


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
                    num_documents=0,
                    num_chunks=0,
                    num_partitions=0,
                    required_roles=[],
                    access_level="public",
                )
                # Get collection description/statistics
                # MilvusClient.describe_collection returns collection info dict
                collection_info_dict = client.describe_collection(collection_name)
                # Get num_chunks from row_count
                collection_info.num_chunks = client.get_collection_stats(collection_name).get("row_count", 0)
                # Get num_partitions by listing partitions
                try:
                    partitions = client.list_partitions(collection_name=collection_name)
                    print("Partitions ->", partitions)
                    collection_info.num_partitions = len(partitions) if partitions else 0
                except Exception as e:
                    logger.warning(f"Failed to get partitions for collection '{collection_name}': {str(e)}")
                    collection_info.num_partitions = 0

                # Extract and parse collection description to get library_context and metadata_schema
                collection_desc = None
                try:
                    collection_desc = CollectionDescription.from_json(collection_info_dict["description"])
                    collection_info.metadata_schema = collection_desc.metadata_schema
                    collection_info.description = collection_desc.description
                    # Get access_level from database config
                    # Valid values: "public", "private", "admin"
                    access_level = collection_desc.collection_config.database.access_level
                    if access_level in ["public", "private", "admin"]:
                        collection_info.access_level = access_level
                    else:
                        logger.warning(f"Invalid access_level '{access_level}' for {collection_name}, defaulting to 'public'")
                        collection_info.access_level = "public"
                except Exception as e:
                    # If parsing fails, set default values
                    logger.warning(f"Failed to parse collection description for {collection_name}: {str(e)}")
                    collection_info.access_level = "public"

                # Get the roles that are required to access the collection
                # Each role is granted a privilege group, if that group is in the collection_security_groups, then the role can access the collection
                if collection_desc:
                    try:
                        all_roles = client.list_roles()
                        for r in all_roles:
                            role_desc = client.describe_role(r)
                            for privilege in role_desc.get("privileges", []):
                                if isinstance(privilege, str) and privilege in collection_desc.collection_security_groups:
                                    collection_info.required_roles.append(r)
                    except Exception as e:
                        logger.warning(f"Failed to list roles for {collection_name}: {str(e)}")

                # Get num_documents using search function
                # Use search with empty text, no filters, and limit 10000 to get all documents
                try:
                    from src.endpoints.search import SearchRequest
                    from src.endpoints.search import search as search_function

                    # Extract username from token for user dict
                    username = ""
                    if token:
                        try:
                            username = extract_username_from_token(token)
                        except ValueError:
                            logger.warning(f"Failed to extract username from token for collection '{collection_name}'")
                            username = ""

                    user_dict = {
                        "username": username,
                        "milvus_token": token or "",
                    }

                    # Create search request with empty text, no filters, limit 10000
                    search_request = SearchRequest(
                        collection=collection_name,
                        text="",
                        filters=[],
                        limit=10000,
                    )

                    # Call search function to get all documents
                    search_response = await search_function(search_request, user_dict)
                    collection_info.num_documents = search_response.total

                except Exception as e:
                    # If search fails, log warning but don't fail the entire collection listing
                    logger.warning(f"Failed to get document count for collection '{collection_name}': {str(e)}")
                    # num_documents remains 0

                collection_descriptions[collection_name] = collection_info

            except Exception as e:
                # If we can't get metadata for a collection, include minimal info
                logger.warning(f"Failed to retrieve metadata for collection '{collection_name}': {str(e)}")

        return CollectionsResponse(
            collection_names=list[str](collection_descriptions.keys()),
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


async def list_pipelines() -> PipelinesResponse:
    """List all available pipeline templates.

    curl -X GET http://localhost:8000/v1/pipelines

    Returns:
        PipelinesResponse with list of available pipeline templates
    """
    from src.endpoints.pipeline_registry import get_registry

    registry = get_registry()
    pipeline_info_list = registry.get_pipeline_info()

    pipelines = [
        PipelineInfo(
            name=info["name"],
            description=info["description"],
            metadata_schema=info["metadata_schema"],
            chunk_size=info["chunk_size"],
            embedding_model=info["embedding_model"],
            llm_model=info["llm_model"],
        )
        for info in pipeline_info_list
    ]

    return PipelinesResponse(pipelines=pipelines)


async def create_collection(
    request: CreateCollectionRequest,
    token: str | None = None,
) -> CreateCollectionResponse:
    """Create a new collection with access level.

    Can use either a template_name or custom_config to define the collection pipeline.

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "template_name": "standard",
        "access_level": "public"
      }'

    Or with custom config:

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "custom_config": {...full CrawlerConfig dict...},
        "access_level": "public"
      }'

    Args:
        request: Collection creation request with access level settings
        token: Milvus authentication token

    Returns:
        CreateCollectionResponse with collection details

    Raises:
        HTTPException: Various error codes based on failure type
    """
    from crawler import CrawlerConfig
    from crawler.vector_db import DatabaseClientConfig

    from src.endpoints.pipeline_registry import get_registry

    # Get the user name and password from the token
    if not token or ":" not in token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
        )
    user_name, password = token.split(":", 1)
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

        # Validate access_level
        if request.access_level not in ["public", "private", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid access_level: {request.access_level}. Must be one of: public, private, admin",
            )

        # Build CrawlerConfig from template or custom config
        config: CrawlerConfig
        if request.custom_config is not None:
            # Use custom config
            try:
                config = CrawlerConfig.from_dict(request.custom_config)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid custom_config: {str(e)}",
                ) from e
        elif request.template_name is not None:
            # Use template
            registry = get_registry()
            if not registry.has_pipeline(request.template_name):
                available = registry.list_pipelines()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Template '{request.template_name}' not found. Available: {available}",
                )
            config = registry.get_config(request.template_name)
        else:
            # Default to 'standard' template
            registry = get_registry()
            config = registry.get_config("standard")

        # Override collection name in database config
        config.name = request.collection_name
        original_db_config = config.database
        db_config = DatabaseClientConfig(
            provider=original_db_config.provider,
            collection=request.collection_name,
            host=original_db_config.host,
            port=original_db_config.port,
            username=user_name,
            password=password,
            partition=original_db_config.partition,
            access_level=request.access_level,
            recreate=False,  # Don't recreate on API creation
            collection_description=request.description or original_db_config.collection_description,
        )
        config.database = db_config

        # Override metadata schema if provided
        if request.metadata_schema:
            config.metadata_schema = request.metadata_schema

        # Override security groups based on roles
        if request.roles:
            config.security_groups = request.roles

        # Get embedding dimension
        from crawler.llm.embeddings import get_embedder

        embedder = get_embedder(config.embeddings)
        embedding_dimension = embedder.get_dimension()

        from crawler.vector_db.database_utils import get_db

        get_db(config.database, embedding_dimension, config)

        # Grant privileges to the collection for specified roles
        # Uses CollectionReadOnly privilege to allow reading from the collection
        granted_roles: list[str] = []
        failed_roles: list[str] = []

        for role in request.roles:
            try:
                # Check if role exists first
                existing_roles = client.list_roles()
                if role not in existing_roles:
                    logger.warning(f"Role '{role}' does not exist, skipping privilege grant")
                    failed_roles.append(role)
                    continue

                # Grant CollectionReadOnly privilege to the role
                client.grant_privilege_v2(
                    role_name=role,
                    privilege="CollectionReadOnly",
                    collection_name=request.collection_name,
                    db_name="default",
                )
                granted_roles.append(role)
                logger.info(f"Granted CollectionReadOnly to role '{role}' on '{request.collection_name}'")
            except Exception as e:
                # Log warning but continue - user may not have admin permissions
                logger.warning(f"Failed to grant privileges to role '{role}': {str(e)}")
                failed_roles.append(role)

        # Build response message
        message = "Collection created successfully"
        if failed_roles:
            message += f". Note: Failed to grant privileges to roles: {failed_roles}"

        return CreateCollectionResponse(
            collection_name=request.collection_name,
            message=message,
            roles=granted_roles if granted_roles else request.roles,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        ) from e
