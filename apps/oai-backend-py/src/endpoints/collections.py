"""Collections endpoint handler."""

import logging
from json import dumps, loads
from typing import Any, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from src.milvus_client import get_milvus_client

logger = logging.getLogger(__name__)

# Import CollectionDescription for parsing collection descriptions
try:
    from crawler.vector_db import CollectionDescription
    from crawler.vector_db.milvus_utils import extract_collection_description
except ImportError:
    # Fallback if crawler package is not available
    CollectionDescription = None
    extract_collection_description = None


# TODO: This needs to have a sepcific format for the frontend to display it correctly.
# TODO: This also needs to have the parsing of the description field to be json formatted for the frontend to display it correctly.
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


class Role(BaseModel):
    """Role model with name and privileges."""

    role: str
    privileges: list[str]


class RolesResponse(BaseModel):
    """Response model for roles listing endpoint."""

    roles: list[Role]


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
    default_permissions: Literal["admin_only", "public"] = Field(
        default="admin_only",
        description="Default permission level: admin_only or public (all authenticated users)",
    )
    metadata_schema: dict[str, Any] | None = Field(None, description="Optional JSON schema override for metadata")


class CreateCollectionResponse(BaseModel):
    """Response model for collection creation endpoint."""

    collection_name: str
    message: str
    pipeline_name: str | None = None
    permissions: dict[str, Any]


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
        collection_metadata: dict[str, CollectionMetadata] = {}

        for collection_name in collections:
            try:
                # Get collection description/statistics
                # MilvusClient.describe_collection returns collection info dict
                collection_info = client.describe_collection(collection_name)
                # Convert protobuf objects to serializable Python types
                safe_dict = loads(dumps(collection_info, default=str))

                # Extract and parse collection description to get library_context and metadata_schema
                library_context: str | None = None
                collection_desc: CollectionDescription | None = None
                if CollectionDescription and extract_collection_description:
                    try:
                        description_str = extract_collection_description(collection_info)
                        if description_str:
                            collection_desc = CollectionDescription.from_json(description_str)
                            if collection_desc:
                                library_context = collection_desc.library_context
                    except Exception as e:
                        logger.debug(f"Failed to parse collection description for {collection_name}: {str(e)}")
                        # Fallback: try to extract description from safe_dict
                        if isinstance(safe_dict, dict):
                            schema = safe_dict.get("schema", {})
                            if isinstance(schema, dict):
                                description_str = schema.get("description")
                            else:
                                description_str = safe_dict.get("description")
                            if description_str and isinstance(description_str, str):
                                try:
                                    collection_desc = CollectionDescription.from_json(description_str)
                                    if collection_desc:
                                        library_context = collection_desc.library_context
                                except Exception:
                                    # If parsing fails, use the raw description as fallback
                                    library_context = description_str
                        else:
                            library_context = str(safe_dict) if safe_dict else None

                # Ensure safe_dict is a dict and includes the name
                if isinstance(safe_dict, dict):
                    # Add collection name if not present
                    if "name" not in safe_dict:
                        safe_dict["name"] = collection_name
                    # Add parsed CollectionDescription fields to metadata for frontend access
                    if collection_desc:
                        # Store the full CollectionDescription JSON in the description field
                        # This allows the frontend to parse it and extract metadata_schema, pipeline_config, etc.
                        safe_dict["description"] = collection_desc.to_json()
                        # Also add individual fields for easier access (frontend can use these directly)
                        safe_dict["library_context"] = collection_desc.library_context
                        safe_dict["metadata_schema"] = collection_desc.metadata_schema
                        if collection_desc.collection_config_json:
                            # Extract pipeline_config and permissions from collection_config_json if present
                            config_json = collection_desc.collection_config_json
                            if "pipeline_config" in config_json:
                                safe_dict["pipeline_config"] = config_json["pipeline_config"]
                            if "permissions" in config_json:
                                safe_dict["permissions"] = config_json["permissions"]
                    elif library_context is not None:
                        # Fallback: if we have library_context but no parsed CollectionDescription,
                        # use library_context as description
                        safe_dict["description"] = library_context
                    collection_metadata[collection_name] = CollectionMetadata(**safe_dict)
                else:
                    # If it's not a dict, wrap it
                    collection_metadata[collection_name] = CollectionMetadata(
                        name=collection_name,
                        description=library_context or str(safe_dict) if safe_dict else None,
                    )

            except Exception as e:
                # If we can't get metadata for a collection, include minimal info
                logger.warning(f"Failed to retrieve metadata for collection '{collection_name}': {str(e)}")
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


async def list_roles(token: str | None = None) -> RolesResponse:
    """Handle roles listing requests.

    Returns all roles from Milvus with their privileges.

    curl -X GET http://localhost:8000/v1/roles \
      -H "Authorization: Bearer $TOKEN"
    """
    try:
        client = get_milvus_client(token)

        # List all roles
        role_names = client.list_roles()

        # Get details for each role
        roles: list[Role] = []
        for role_name in role_names:
            try:
                role_info = client.describe_role(role_name)
                # role_info is {'role': role-name, 'privileges': []}
                privileges = role_info.get("privileges", [])
                # Ensure privileges is a list of strings
                if isinstance(privileges, list):
                    privileges_list = [str(p) for p in privileges]
                else:
                    privileges_list = []
                roles.append(Role(role=role_name, privileges=privileges_list))
            except Exception as e:
                # If we can't get details for a role, include minimal info
                logger.warning(f"Failed to describe role '{role_name}': {str(e)}")
                roles.append(Role(role=role_name, privileges=[]))

        return RolesResponse(roles=roles)

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
    from crawler.vector_db.milvus_utils import create_schema
    from src.endpoints.document_pipelines import ConfigOverrides, _override_config
    from src.endpoints.pipeline_registry import get_registry

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
        config: CrawlerConfig
        pipeline_name: str | None = None

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
            pipeline_name = request.pipeline_name

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
            username=original_db_config.username,
            password=original_db_config.password,
            partition=original_db_config.partition,
            recreate=False,  # Don't recreate on API creation
            collection_description=request.description or original_db_config.collection_description,
        )
        config.database = db_config

        # Override metadata schema if provided
        metadata_schema = request.metadata_schema or config.metadata_schema

        # Build pipeline config for storage in collection_config_json
        pipeline_config: dict[str, Any] = {}
        if pipeline_name:
            pipeline_config["pipeline_name"] = pipeline_name
            if request.config_overrides:
                pipeline_config["overrides"] = request.config_overrides
        else:
            pipeline_config["full_config"] = config.model_dump()

        # Build permissions info
        permissions: dict[str, Any] = {
            "default": request.default_permissions,
        }

        # Get embedding dimension
        from crawler.llm.embeddings import get_embedder

        embedder = get_embedder(config.embeddings)
        embedding_dimension = embedder.get_dimension()

        # Build library context from description
        library_context = request.description or config.database.collection_description or f"Collection: {request.collection_name}"

        # Build collection_config_json with pipeline_config and permissions for storage
        collection_config_json: dict[str, Any] = {
            **config.model_dump(),
            "pipeline_config": pipeline_config,
            "permissions": permissions,
        }

        # Create collection schema with description
        collection_schema = create_schema(
            embedding_dimension,
            user_metadata_json_schema=metadata_schema,
            library_context=library_context,
            collection_config_json=collection_config_json,
        )

        # Create the collection
        from crawler.vector_db.milvus_utils import create_index

        index_params = create_index(client)
        client.create_collection(
            collection_name=request.collection_name,
            dimension=embedding_dimension,
            schema=collection_schema,
            index_params=index_params,
            vector_field_name="text_embedding",
            auto_id=True,
        )

        # Create partition if specified
        if config.database.partition:
            client.create_partition(request.collection_name, config.database.partition)

        # Set up collection-level RBAC if needed
        # For now, we only handle default_permissions (admin_only vs public)
        # Document-level permissions will be handled when uploading documents
        if request.default_permissions == "public":
            # Note: In a full implementation, you might want to grant CollectionRead
            # and CollectionQuery privileges to a "public" role here
            # For now, we'll just store the permission setting in the description
            logger.info(f"Collection '{request.collection_name}' set to public access")

        return CreateCollectionResponse(
            collection_name=request.collection_name,
            message="Collection created successfully",
            pipeline_name=pipeline_name,
            permissions=permissions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        ) from e
