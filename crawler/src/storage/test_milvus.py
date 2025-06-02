import requests
from milvus import MilvusStorage
from config import MILVUS_CONFIG, MILVUS_CONFIG_2, SCHEMA_1, SCHEMA_2

def get_ollama_embedding(text: str) -> list:
    """Get embedding from Ollama."""
    response = requests.post('http://localhost:11434/api/embeddings',
                           json={'model': 'all-minilm:v2', 'prompt': text})
    return response.json()['embedding']

def create_test_data(schema_type: int = 1) -> list:
    """Create test data with different metadata schemas."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
        "Machine learning is transforming industries.",
        "Data science combines statistics and programming.",
        "Artificial intelligence is the future of technology."
    ]
    
    data = []
    for i, text in enumerate(texts):
        item = {
            "text": text,
            "text_embedding": get_ollama_embedding(text),
            "chunk_index": i,
            "source": f"test_doc_{i}.txt",
            "minio": f"http://minio.example.com/docs/test_doc_{i}.txt"
        }
        
        if schema_type == 1:
            item.update({
                "title": f"Document {i}",
                "author": f"Author {i}",
                "tags": ["test", f"tag{i}", "example"]
            })
        else:
            item.update({
                "category": f"Category {i}",
                "importance": i * 10,
                "meta_data": {
                    "created_at": "2024-03-20",
                    "version": "1.0"
                }
            })
        
        data.append(item)
    
    return data

def test_milvus_storage():
    """Test Milvus storage with different configurations."""
    # Test with first configuration and schema
    storage1 = MilvusStorage(MILVUS_CONFIG, recreate=True)
    storage1.create_collection(embedding_size=384, schema=SCHEMA_1)
    data1 = create_test_data(schema_type=1)
    storage1.insert_data(data1)
    print("Successfully inserted data into first collection")

    # Test with second configuration and schema
    storage2 = MilvusStorage(MILVUS_CONFIG_2, recreate=True)
    storage2.create_collection(embedding_size=384, schema=SCHEMA_2)
    data2 = create_test_data(schema_type=2)
    storage2.insert_data(data2)
    print("Successfully inserted data into second collection")

if __name__ == "__main__":
    test_milvus_storage() 