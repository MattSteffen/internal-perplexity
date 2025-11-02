"""Configuration settings for the application."""

import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="OAI_",
        case_sensitive=False,
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for Ollama OpenAI-compatible API",
    )
    api_key: str = Field(
        default="ollama",
        description="API key for Ollama (Ollama doesn't require real auth but expects this header)",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    # Keycloak OAuth2 settings
    keycloak_url: str = Field(
        default="",
        description="Keycloak realm URL (e.g., https://keycloak.yourdomain.com/realms/myrealm)",
    )
    client_id: str = Field(default="", description="OAuth2 client ID")
    client_secret: str = Field(default="", description="OAuth2 client secret")
    redirect_uri: str = Field(
        default="http://localhost:3000/api/auth/callback",
        description="OAuth2 redirect URI",
    )
    frontend_redirect_url: str = Field(
        default="http://localhost:3000/dashboard",
        description="Frontend redirect URL after authentication",
    )


settings = Settings()


# -------------------------------
# --- Radchat Config Models ---
# -------------------------------


class OllamaConfig(BaseModel):
    """Ollama client configuration."""

    base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embedding_model: str = Field(default=os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:v2"))
    llm_model: str = Field(default=os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b"))
    request_timeout: int = Field(default=int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")))
    context_length: int = Field(default=int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32000")))


class MilvusConfig(BaseModel):
    """Milvus connection configuration."""

    host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    port: str = Field(default=os.getenv("MILVUS_PORT", "19530"))
    username: str = Field(default=os.getenv("MILVUS_USERNAME", "matt"))
    password: str = Field(default=os.getenv("MILVUS_PASSWORD", "steffen"))
    collection_name: str = Field(default=os.getenv("IRAD_COLLECTION_NAME", "arxiv3"))


class SearchConfig(BaseModel):
    """Milvus search configuration."""

    nprobe: int = Field(default=int(os.getenv("MILVUS_NPROBE", "10")))
    search_limit: int = Field(default=int(os.getenv("MILVUS_SEARCH_LIMIT", "5")))
    hybrid_limit: int = Field(default=int(os.getenv("MILVUS_HYBRID_SEARCH_LIMIT", "10")))
    rrf_k: int = Field(default=int(os.getenv("MILVUS_RRF_K", "100")))
    drop_ratio: float = Field(default=float(os.getenv("MILVUS_DROP_RATIO", "0.2")))
    output_fields: list[str] = Field(
        default=[
            "metadata",
            "default_text",
            "default_document_id",
            "default_chunk_index",
            "default_source",
        ]
    )


class AgentConfig(BaseModel):
    """Agent configuration."""

    max_tool_calls: int = Field(default=int(os.getenv("AGENT_MAX_TOOL_CALLS", "5")))
    default_role: str = Field(default=os.getenv("AGENT_DEFAULT_ROLE", "system"))
    logging_level: str = Field(default=os.getenv("AGENT_LOGGING_LEVEL", "INFO"))


class UserValves(BaseModel):
    """User-provided configuration overrides."""

    COLLECTION_NAME: str = Field(default_factory=lambda: radchat_config.milvus.collection_name)
    MILVUS_USERNAME: str = Field(default_factory=lambda: radchat_config.milvus.username)
    MILVUS_PASSWORD: str = Field(default_factory=lambda: radchat_config.milvus.password)


class RadchatConfig(BaseModel):
    """Unified configuration for Radchat agent."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def update_from_valves(self, valves: UserValves) -> None:
        """Override runtime configuration from user-provided valves.

        Args:
            valves: UserValves instance with override values.
        """
        if hasattr(valves, "MILVUS_USERNAME") and valves.MILVUS_USERNAME:
            self.milvus.username = valves.MILVUS_USERNAME
        if hasattr(valves, "MILVUS_PASSWORD") and valves.MILVUS_PASSWORD:
            self.milvus.password = valves.MILVUS_PASSWORD
        if hasattr(valves, "COLLECTION_NAME") and valves.COLLECTION_NAME:
            self.milvus.collection_name = valves.COLLECTION_NAME


# Instantiate a global, immutable base config
radchat_config: RadchatConfig = RadchatConfig()
