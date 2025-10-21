import time
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import ollama
from typing import List, Dict, Optional
from tqdm import tqdm

from pymilvus.client import abstract


class EmbedderConfig(BaseModel):
    """
    Configuration for embedding model providers.

    This model provides type-safe configuration for connecting to embedding
    services with automatic validation of parameters.

    Attributes:
        model: Name of the embedding model to use
        base_url: Base URL for the embedding service API
        api_key: API key for authentication (if required)
        provider: Provider name (e.g., 'ollama', 'openai')
        dimension: Optional pre-configured embedding dimension (auto-detected if None)
    """

    model: str = Field(
        ..., min_length=1, description="Name of the embedding model to use"
    )
    base_url: str = Field(
        ..., min_length=1, description="Base URL for the embedding service API"
    )
    api_key: str = Field(
        default="",
        description="API key for authentication (if required by the provider)",
    )
    provider: str = Field(
        default="ollama", description="Provider name (e.g., 'ollama', 'openai')"
    )
    dimension: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional pre-configured embedding dimension (auto-detected if None)",
    )

    model_config = {
        "validate_assignment": True,
    }

    @classmethod
    def ollama(
        cls,
        model: str,
        base_url: str = "http://localhost:11434",
        dimension: Optional[int] = None,
    ) -> "EmbedderConfig":
        """Create Ollama embedder configuration."""
        return cls(
            model=model, base_url=base_url, provider="ollama", dimension=dimension
        )

    @classmethod
    def openai(
        cls,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        dimension: Optional[int] = None,
    ) -> "EmbedderConfig":
        """Create OpenAI embedder configuration."""
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="openai",
            dimension=dimension,
        )


class Embedder(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def embed(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        Args:
            query: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries.

        Args:
            queries: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the dimension of the embedding model"""
        pass


def get_embedder(config: EmbedderConfig) -> Embedder:
    if config.provider == "ollama":
        return OllamaEmbedder(config)
    raise ValueError(f"unsupported provider: {config.provider}")


class OllamaEmbedder(Embedder):
    """Ollama implementation of the Embedder interface."""

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.embedder = ollama.Client(host=config.base_url)
        self._dimension = None

    def embed(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        Args:
            query: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embedder.embeddings(model=self.config.model, prompt=query)
        return embedding.embedding

    def embed_batch(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries.

        Args:
            queries: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedder.embed(model=self.config.model, input=queries)
        return embeddings.embeddings

    def get_dimension(self) -> int:
        """Returns the dimension of the embedding model."""
        # Use configured dimension if available
        if self.config.dimension is not None:
            if self._dimension is None or self._dimension != self.config.dimension:
                self._dimension = self.config.dimension
            return self._dimension

        # Otherwise probe once and cache
        if self._dimension is None:
            # Calculate dimension on first request
            test_embedding = self.embed(query="test")
            self._dimension = len(test_embedding)
        return self._dimension
