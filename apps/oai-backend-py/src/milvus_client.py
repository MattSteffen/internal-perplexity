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

# Singleton Milvus client instance
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """Get or create the singleton Milvus client instance.

    Returns:
        MilvusClient instance.
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=_milvus_settings.uri)
    return _milvus_client
