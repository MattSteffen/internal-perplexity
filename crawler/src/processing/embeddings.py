from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Optional, List
import yaml

class LocalEmbedder:
    def __init__(self, 
                 source: str = "ollama",
                 model_name: str = "all-minilm:v2",
                 url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """Initialize embeddings model based on source.
        
        Args:
            source: Either "ollama" or "openai"
            model_name: Name of model to use
            url: Base URL for API calls
            api_key: API key (required for OpenAI)
        """
        config = load_config("../schema.yaml")

        self.source: str = source
        self.dimension: int = 0
        self.model_name = config.get("embedding_model", "all-minilm:v2")
        self.url = config.get("embedding_url", "http://localhost:11434")
        self.api_key = config.get("embedding_api_key", None)
        
        if source == "ollama":
            self.embedder = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.url
            )
        elif source == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI embeddings")
            self.embedder = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.url
            )
        else:
            raise ValueError(f"Unsupported embedding source: {source}")
        
        self.dimension = len(self.embedder.embed_query("test"))

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""
        return self.embedder.embed_query(query)
        
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.embed_documents(texts)



def load_config(config_file: str) -> dict:
    """Loads the collection schema configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

# test / demo
if __name__ == "__main__":
    embedder = LocalEmbedder(source="ollama")
    text = "This is a sample text to be embedded."
    embedding = embedder.embed_query(text)
    print(embedding)