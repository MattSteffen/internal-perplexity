import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict
from tqdm import tqdm

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
        model = config.get("model")
        if not model:
            raise ValueError("Embedder model cannot be empty")

        base_url = config.get("base_url")
        if not base_url:
            raise ValueError("Embedder base_url cannot be empty")

        return cls(
            model=model,
            base_url=base_url,
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
    """Ollama implementation of the Embedder interface with comprehensive logging."""

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.embedder = OllamaEmbeddings(model=config.model, base_url=config.base_url)
        self._dimension = None

        # Setup logging
        self.logger = logging.getLogger('OllamaEmbedder')
        self.logger.propagate = False  # Prevent duplicate messages
        self.logger.info(f"Initialized OllamaEmbedder with model: {config.model}")
        self.logger.debug(f"Base URL: {config.base_url}")

    def embed(self, query: str) -> List[float]:
        """Generate embedding for a single query string with logging.

        Args:
            query: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        embed_start_time = time.time()

        self.logger.debug("🧮 Starting embedding generation...")
        self.logger.debug(f"Input text length: {len(query)} characters")

        try:
            embedding = self.embedder.embed_query(query)

            embed_time = time.time() - embed_start_time
            embedding_dimension = len(embedding)

            self.logger.debug("✅ Embedding generated successfully")
            self.logger.debug("📊 Embedding Statistics:")
            self.logger.debug(f"   • Processing time: {embed_time:.3f}s")
            self.logger.debug(f"   • Embedding dimension: {embedding_dimension}")
            self.logger.debug(f"   • Processing rate: {len(query)/embed_time:.0f} chars/sec")

            return embedding

        except Exception as e:
            embed_time = time.time() - embed_start_time
            self.logger.error(f"❌ Embedding generation failed after {embed_time:.3f}s: {e}")
            raise

    def embed_batch(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries with progress tracking.

        Args:
            queries: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        batch_start_time = time.time()

        self.logger.info(f"🧮 Starting batch embedding generation for {len(queries)} queries...")

        embeddings = []
        stats = {
            'total_queries': len(queries),
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'total_chars': sum(len(query) for query in queries)
        }

        # Process with progress bar
        with tqdm(total=len(queries), desc="Embedding queries", unit="query") as pbar:
            for i, query in enumerate(queries):
                query_start_time = time.time()

                try:
                    embedding = self.embedder.embed_query(query)
                    embeddings.append(embedding)
                    stats['successful_embeddings'] += 1

                    query_time = time.time() - query_start_time
                    pbar.set_postfix_str(f"Query {i+1}/{len(queries)} ({query_time:.3f}s)")
                    pbar.update(1)

                except Exception as e:
                    stats['failed_embeddings'] += 1
                    self.logger.error(f"❌ Failed to embed query {i+1}: {e}")
                    pbar.update(1)
                    continue

        batch_time = time.time() - batch_start_time

        # Log batch statistics
        self.logger.info("=== Batch embedding completed ===")
        self.logger.info("📊 Batch Embedding Statistics:")
        self.logger.info(f"   • Total queries: {stats['total_queries']}")
        self.logger.info(f"   • Successful embeddings: {stats['successful_embeddings']}")
        self.logger.info(f"   • Failed embeddings: {stats['failed_embeddings']}")
        self.logger.info(f"   • Total characters processed: {stats['total_chars']}")
        self.logger.info(f"   • Total processing time: {batch_time:.2f}s")
        self.logger.info(f"   • Average time per query: {batch_time/stats['total_queries']:.3f}s")
        self.logger.info(f"   • Queries per second: {stats['successful_embeddings']/batch_time:.1f}")

        if stats['failed_embeddings'] > 0:
            self.logger.warning(f"⚠️  {stats['failed_embeddings']} queries failed to embed")

        return embeddings

    def get_dimension(self) -> int:
        """Returns the dimension of the embedding model with logging."""
        if self._dimension is None:
            self.logger.info("🔍 Determining embedding dimension...")
            try:
                # Calculate dimension on first request
                test_embedding = self.embedder.embed_query("test")
                self._dimension = len(test_embedding)
                self.logger.info(f"✅ Embedding dimension determined: {self._dimension}")
            except Exception as e:
                self.logger.error(f"❌ Failed to determine embedding dimension: {e}")
                raise
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
