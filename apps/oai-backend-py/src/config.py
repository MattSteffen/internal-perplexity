"""Configuration settings for the application."""

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


# -------------------------------
# --- Radchat Config Models ---
# -------------------------------


class OllamaConfig(BaseSettings):
    """Ollama client configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    embedding_model: str = Field(default="all-minilm:v2", validation_alias="OLLAMA_EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-oss:20b", validation_alias="OLLAMA_LLM_MODEL")
    request_timeout: int = Field(default=300, validation_alias="OLLAMA_REQUEST_TIMEOUT")
    context_length: int = Field(default=32000, validation_alias="OLLAMA_CONTEXT_LENGTH")


class MilvusConfig(BaseSettings):
    """Milvus connection configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    uri: str | None = Field(default=None, validation_alias="MILVUS_URI")
    host: str = Field(default="localhost", validation_alias="MILVUS_HOST")
    port: str = Field(default="19530", validation_alias="MILVUS_PORT")
    username: str = Field(default="matt", validation_alias="MILVUS_USERNAME")
    password: str = Field(default="steffen", validation_alias="MILVUS_PASSWORD")
    collection_name: str = Field(default="arxiv3", validation_alias="IRAD_COLLECTION_NAME")

    @property
    def resolved_uri(self) -> str:
        if self.uri:
            return self.uri
        return f"http://{self.host}:{self.port}"


class SearchConfig(BaseSettings):
    """Milvus search configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    nprobe: int = Field(default=10, validation_alias="MILVUS_NPROBE")
    search_limit: int = Field(default=5, validation_alias="MILVUS_SEARCH_LIMIT")
    hybrid_limit: int = Field(default=10, validation_alias="MILVUS_HYBRID_SEARCH_LIMIT")
    rrf_k: int = Field(default=100, validation_alias="MILVUS_RRF_K")
    drop_ratio: float = Field(default=0.2, validation_alias="MILVUS_DROP_RATIO")
    output_fields: list[str] = Field(
        default=[
            "metadata",
            "text",
            "document_id",
            "chunk_index",
            "source",
        ]
    )


class AgentConfig(BaseSettings):
    """Agent configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    max_tool_calls: int = Field(default=5, validation_alias="AGENT_MAX_TOOL_CALLS")
    default_role: str = Field(default="system", validation_alias="AGENT_DEFAULT_ROLE")
    logging_level: str = Field(default="INFO", validation_alias="AGENT_LOGGING_LEVEL")


class MilvusPoolConfig(BaseSettings):
    """Milvus client pool configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    ttl_seconds: int = Field(default=15 * 60, validation_alias="MILVUS_POOL_TTL_SECONDS")
    max_size: int = Field(default=250, validation_alias="MILVUS_POOL_MAX_SIZE")


class CorsConfig(BaseSettings):
    """CORS configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OAI_",
        case_sensitive=False,
        env_parse_delimiter=",",
    )

    allow_origins: list[str] = Field(default_factory=lambda: ["*"], validation_alias="CORS_ALLOW_ORIGINS")
    allow_credentials: bool = Field(default=True, validation_alias="CORS_ALLOW_CREDENTIALS")
    allow_methods: list[str] = Field(default_factory=lambda: ["*"], validation_alias="CORS_ALLOW_METHODS")
    allow_headers: list[str] = Field(default_factory=lambda: ["*"], validation_alias="CORS_ALLOW_HEADERS")


class ToolingConfig(BaseSettings):
    """Configuration for internal tooling behavior."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    milvus_search_max_workers: int = Field(default=4, validation_alias="MILVUS_SEARCH_MAX_WORKERS")


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


class AppConfig(BaseModel):
    """Centralized application configuration."""

    settings: Settings = Field(default_factory=Settings)
    radchat: RadchatConfig = Field(default_factory=RadchatConfig)
    milvus_pool: MilvusPoolConfig = Field(default_factory=MilvusPoolConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    tooling: ToolingConfig = Field(default_factory=ToolingConfig)


# Instantiate a global, immutable base config
app_config: AppConfig = AppConfig()
radchat_config: RadchatConfig = app_config.radchat
settings: Settings = app_config.settings
