"""Collections endpoint handler."""

import logging
from typing import Any, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, model_validator
from pymilvus import MilvusClient  # type: ignore

from src.milvus_client import MilvusClientContext

logger = logging.getLogger(__name__)

# Import CollectionDescription for parsing collection descriptions
try:
    from crawler.vector_db import CollectionDescription
except ImportError:
    raise ImportError("crawler package not available")

ACCESS_LEVELS: tuple[str, ...] = ("public", "private", "group_only", "admin")


class CollectionInfo(BaseModel):
    """Collection info model."""

    description: str
    metadata_schema: dict[str, Any]
    num_documents: int
    num_chunks: int
    num_partitions: int
    required_roles: list[str]
    access_level: str = "public"  # One of: "public", "private", "group_only", "admin"


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
    """Request model for creating a new collection.

    Collection name is taken from crawler_config.database.collection.
    """

    access_level: Literal["public", "private", "group_only", "admin"] = Field(
        default="public",
        description="Access level for the collection",
    )
    access_groups: list[str] = Field(
        default_factory=list,
        description="List of role/group names to grant read access; required when access_level is group_only",
    )
    crawler_config: dict[str, Any] = Field(
        ...,
        description="Full CrawlerConfig JSON (must include database.collection)",
    )

    @model_validator(mode="after")
    def validate_access_groups_when_group_only(self) -> CreateCollectionRequest:
        if self.access_level == "group_only" and not self.access_groups:
            raise ValueError("access_groups must be non-empty when access_level is group_only")
        return self


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


async def list_collections(context: MilvusClientContext) -> CollectionsResponse:
    """Handle collections listing requests.

    Returns all collections from Milvus with their metadata.

    curl -X GET http://localhost:8000/v1/collections
    """
    try:
        client = context.client
        token = context.token

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
                try:
                    collection_info_dict = client.describe_collection(collection_name)
                except Exception as e:
                    logger.warning(f"Failed to describe collection '{collection_name}': {str(e)}")
                    continue
                # Get num_chunks from row_count
                collection_info.num_chunks = client.get_collection_stats(collection_name).get("row_count", 0)
                # Get num_partitions by listing partitions
                try:
                    partitions = client.list_partitions(collection_name=collection_name)
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
                    access_level = collection_desc.collection_config.database.access_level
                    if access_level in ACCESS_LEVELS:
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

                    # Create search request with empty text, no filters, limit 10000
                    search_request = SearchRequest(
                        collection=collection_name,
                        text="",
                        filters=[],
                        limit=10000,
                    )

                    # Call search function to get all documents
                    search_response = await search_function(search_request, context)
                    # Crawler's db.disconnect() invalidates process-wide gRPC state; refresh app client.
                    context.pool.invalidate(context.cache_key)
                    client = context.pool.get(context.cache_key, token)
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


async def list_roles(client: MilvusClient) -> RolesResponse:
    """Handle roles listing requests.

    Returns all roles from Milvus with their privileges.

    curl -X GET http://localhost:8000/v1/roles \
      -H "Authorization: Bearer $TOKEN"
    """
    try:
        # List all roles
        role_names: list[str] = client.list_roles()
        return RolesResponse(roles=role_names)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list roles: {str(e)}",
        ) from e


async def list_users(client: MilvusClient) -> UsersResponse:
    """Handle users listing requests.

    Returns all users from Milvus with their roles.

    curl -X GET http://localhost:8000/v1/users \
      -H "Authorization: Bearer $TOKEN"
    """
    try:
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


async def get_pipeline_config(name: str) -> dict[str, Any]:
    """Return full crawler-config JSON for the given pipeline.

    curl -X GET http://localhost:8000/v1/pipelines/standard

    Returns:
        Full crawler config dict for the pipeline. 404 if not found.
    """
    from src.endpoints.pipeline_registry import get_registry

    registry = get_registry()
    try:
        return registry.get(name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{name}' not found. Available: {list(registry.list_pipelines())}",
        ) from None


def _milvus_host_port_from_uri(uri: str) -> tuple[str, int]:
    """Parse Milvus URI into host and port for DatabaseClientConfig."""
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port if parsed.port is not None else 19530
    return host, port


def _compute_security_groups(
    access_level: str,
    access_groups: list[str],
    creator_role: str,
    base_groups: list[str] | None = None,
) -> list[str]:
    """Compute security_groups for the stored crawler config so app-level RBAC passes.

    Ensures creator_role is always included. Deduplicates while preserving order.
    """
    base = list(base_groups or [])
    if access_level == "public":
        return list(dict.fromkeys([*base, creator_role]))
    if access_level == "private":
        return [creator_role]
    if access_level == "group_only":
        return list(dict.fromkeys([*access_groups, creator_role]))
    if access_level == "admin":
        return list(dict.fromkeys(["admin", creator_role]))
    return list(dict.fromkeys([*base, creator_role]))


def _grant_collection_privilege(
    client: Any,
    role_name: str,
    privilege: str,
    collection_name: str,
    db_name: str = "default",
) -> None:
    """Grant a Milvus RBAC privilege on a collection to a role. Raises on failure."""
    client.grant_privilege_v2(
        role_name=role_name,
        privilege=privilege,
        collection_name=collection_name,
        db_name=db_name,
    )


async def create_collection(
    request: CreateCollectionRequest,
    context: MilvusClientContext,
) -> CreateCollectionResponse:
    """Create a new collection from crawler_config; set security_groups and grant Milvus RBAC.

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "access_level": "public",
        "access_groups": [],
        "crawler_config": {"name": "my_collection", "database": {"collection": "my_collection", ...}, ...}
      }'

    Collection name is taken from crawler_config.database.collection.

    Permission semantics:
    - security_groups on the stored config always include the creator (username from token);
      for public: base groups + creator; private: creator only; group_only: access_groups + creator; admin: admin + creator.
    - Milvus RBAC: the creator role (username) and, when access_level is group_only, each access_groups role
      are granted CollectionReadWrite on the new collection. All listed roles must exist in Milvus (400 if missing);
      on grant failure the collection is dropped (best-effort) and 500 is returned.
    """
    from crawler import CrawlerConfig
    from crawler.vector_db import DatabaseClientConfig

    from src.milvus_client import _milvus_settings

    token = context.token
    # Get the user name and password from the token
    if ":" not in token:
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
        # Parse CrawlerConfig from request; validate database.collection
        db_section = request.crawler_config.get("database") or {}
        collection_name = (db_section.get("collection") or "").strip()
        if not collection_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="crawler_config.database.collection must be present and non-empty",
            )

        try:
            config = CrawlerConfig.from_dict(request.crawler_config)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid crawler_config: {str(e)}",
            ) from e

        client = context.client
        existing_collections = client.list_collections()
        if collection_name in existing_collections:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection '{collection_name}' already exists",
            )

        # Override config: collection name, access_level, security_groups, db connection from token and backend URI
        config.name = collection_name
        original_db_config = config.database
        host, port = _milvus_host_port_from_uri(_milvus_settings.uri)
        db_config = DatabaseClientConfig(
            provider=original_db_config.provider,
            collection=collection_name,
            host=host,
            port=port,
            username=user_name,
            password=password,
            partition=original_db_config.partition,
            access_level=request.access_level,
            recreate=False,
            collection_description=original_db_config.collection_description,
        )
        config.database = db_config
        base_groups = getattr(config, "security_groups", None) or []
        config.security_groups = _compute_security_groups(
            request.access_level,
            request.access_groups,
            creator_role=user_name,
            base_groups=base_groups if isinstance(base_groups, list) else list(base_groups),
        )

        # Create collection via crawler path so description is set as the crawler does
        from crawler.llm.embeddings import get_embedder
        from crawler.vector_db.database_utils import get_db

        embedder = get_embedder(config.embeddings)
        embedding_dimension = embedder.get_dimension()
        db = get_db(config.database, embedding_dimension, config)
        db.connect(create_if_missing=True)
        # Crawler may have invalidated process-wide gRPC state; use fresh client for RBAC
        context.pool.invalidate(context.cache_key)
        client = context.pool.get(context.cache_key, token)

        # Grant CollectionReadWrite to creator and (when group_only) to access_groups; fail fast on missing roles
        creator_role = user_name
        roles_to_grant: list[str] = [creator_role]
        if request.access_level == "group_only":
            roles_to_grant = list(dict.fromkeys([creator_role, *request.access_groups]))

        existing_roles = client.list_roles()
        missing_roles = [r for r in roles_to_grant if r not in existing_roles]
        if missing_roles:
            try:
                client.drop_collection(collection_name)
            except Exception as cleanup_err:
                logger.warning("Best-effort cleanup after missing roles failed: %s", cleanup_err)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Roles do not exist in Milvus; cannot grant privileges: {missing_roles}. " "Ensure the creator and access_groups roles exist (e.g. via Milvus admin).",
            ) from None

        try:
            for role in roles_to_grant:
                _grant_collection_privilege(client, role, "CollectionReadWrite", collection_name, db_name="default")
                logger.info("Granted CollectionReadWrite to role '%s' on '%s'", role, collection_name)
        except Exception as e:
            try:
                client.drop_collection(collection_name)
            except Exception as cleanup_err:
                logger.warning("Best-effort cleanup after grant failure failed: %s", cleanup_err)
            logger.exception("Failed to grant privileges: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Collection was created but granting privileges failed: {str(e)}",
            ) from e

        return CreateCollectionResponse(
            collection_name=collection_name,
            message="Collection created successfully",
            roles=roles_to_grant,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        ) from e
