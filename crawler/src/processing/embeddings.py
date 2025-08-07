from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict

from pymilvus.client import abstract


@dataclass
class EmbedderConfig:
    """Configuration for the embedder."""

    model: str
    base_url: str
    api_key: str = ""
    provider: str = "ollama"

    @classmethod
    def from_dict(cls, config: Dict[str, any]):
        return cls(
            model=config.get("model"),
            base_url=config.get("base_url"),
            api_key=config.get("api_key", ""),
            provider=config.get("provider", "ollama"),
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
        self.embedder = OllamaEmbeddings(model=config.model, base_url=config.base_url)
        self._dimension = None

    def embed(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        Args:
            query: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embedder.embed_query(query)

    def get_dimension(self) -> int:
        """Returns the dimension of the embedding model."""
        if self._dimension is None:
            # Calculate dimension on first request
            self._dimension = len(self.embedder.embed_query("test"))
        return self._dimension


def test():
    """Test function for the OllamaEmbedder."""
    config = EmbedderConfig(model="all-minilm:v2", base_url="http://localhost:11434")

    embedder = OllamaEmbedder()
    embedder.init(config)

    # Test single embedding
    result = embedder.embed("test")
    print(f"Embedding dimension: {len(result)}")
    print(f"First 5 values: {result[:5]}")


if __name__ == "__main__":
    test()
