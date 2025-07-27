from pymilvus import MilvusClient
import ollama
from typing import List, Optional

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "all-minilm:v2"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "test_collection"

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return None

def connect_milvus() -> Optional[MilvusClient]:
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token=MILVUS_TOKEN)
        if not client.has_collection(collection_name=COLLECTION_NAME):
            print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
            return None
        client.load_collection(collection_name=COLLECTION_NAME)
        return client
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return None