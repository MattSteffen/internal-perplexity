from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Optional, List, Dict, Any
import yaml

# Try using langchain's milvus so you can upload full documents with metadata.

class LocalEmbedder:
    def __init__(self, embeddings_config: Dict[str, Any] = {}):
        """Initialize embeddings model based on provider.
        
        Args:
          config: Configuration dictionary
            - embeddings
              - provider: Ollama or OpenAI
              - model: Ollama model name or OpenAI model name
              - url: Ollama URL or OpenAI URL
              - api_key: OpenAI API key
        """
        self.provider: str = embeddings_config.get("provider")
        self.model_name = embeddings_config.get("model")
        self.url = embeddings_config.get("url")
        self.api_key = embeddings_config.get("api_key", "")
        
        if self.provider == "ollama":
            self.embedder = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.url
            )
        elif self.provider == "openai":
            if not self.api_key:
                raise ValueError("API key required for OpenAI embeddings")
            self.embedder = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.url
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        self.dimension = len(self.embedder.embed_query("test"))

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""
        return self.embedder.embed_query(query)
        
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.embed_documents(texts)
    
def test():
    config = {"embeddings": {"provider": "ollama", "model": "all-minilm:v2", "url": "http://localhost:11434"}}
    embedder = LocalEmbedder(config)
    print(embedder.embed_query("test"))
    print(embedder.embed_queries(["test", "test2"]))