"""Milvus client singleton for database operations."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymilvus import MilvusClient  # type: ignore


class MilvusSettings(BaseSettings):
    """Milvus connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="MILVUS_",
        case_sensitive=False,
    )

    uri: str = Field(
        default="http://localhost:19530",
        description="Milvus connection URI",
    )


_milvus_settings = MilvusSettings()

# Cache Milvus client instances per token
_milvus_clients: dict[str | None, MilvusClient] = {}


def get_milvus_client(token: str | None = None) -> MilvusClient:
    """Get or create a Milvus client instance for the given token.

    Creates a new client instance for each unique token to support
    multiple users with different authentication tokens.

    Args:
        token: The token to use for authentication (format: username:password).

    Returns:
        MilvusClient instance authenticated with the provided token.
    """
    global _milvus_clients

    # Use token as cache key (None for no token)
    cache_key = token

    # Create new client if not cached for this token
    if cache_key not in _milvus_clients:
        _milvus_clients[cache_key] = MilvusClient(uri=_milvus_settings.uri, token=token)

    return _milvus_clients[cache_key]
