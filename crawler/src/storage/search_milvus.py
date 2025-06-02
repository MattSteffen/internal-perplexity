import requests
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from config import MILVUS_CONFIG, MILVUS_CONFIG_2
from typing import Optional

def perform_search(
    client: MilvusClient,
    queries: list[str],
    collection_name: str,
    partition_name: Optional[str] = None,
    filters: list[str] = [],
) -> list[str]:
    print("STARTING SEARCH")
    # There are 3 types of searches to perform, semantic on text, full-text on text, and full-text on metadata.
    # Each must be performed for each query, not all at once.
    search_requests = []
    for query in queries:
        search_parameters_semantic_on_text = {
            "data": [get_ollama_embedding(query)],
            "anns_field": "text_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            },
            "expr": " and ".join(filters),
            "limit": 10,
        }
        search_requests.append(
            AnnSearchRequest(**search_parameters_semantic_on_text)
        )

        search_parameters_full_text_on_text = {
            "data": [query],
            "anns_field": "text_sparse_embedding",
            "param": {
                "drop_ratio_search": 0.2,
            },
            "expr": " and ".join(filters),
            "limit": 10,
        }
        search_requests.append(
            AnnSearchRequest(**search_parameters_full_text_on_text)
        )

        search_parameters_full_text_on_metadata = {
            "data": [query],
            "anns_field": "metadata_sparse_embedding",
            "param": {
                "drop_ratio_search": 0.2,
            },
            "expr": " and ".join(filters),
            "limit": 10,
        }
        search_requests.append(
            AnnSearchRequest(**search_parameters_full_text_on_metadata)
        )

    ranker = RRFRanker(100)
    # Perform the search
    result = client.hybrid_search(
        collection_name=collection_name,
        partition_names=[partition_name] if partition_name else None,
        reqs=search_requests,
        ranker=ranker,
        output_fields=["text", "source", "chunk_index"],
        limit=100,
    )

    return result


def get_ollama_embedding(text: str) -> list:
    """Get embedding from Ollama."""
    response = requests.post('http://localhost:11434/api/embeddings',
                           json={'model': 'all-minilm:v2', 'prompt': text})
    return response.json()['embedding']

def search_collection(client: MilvusClient, collection_name: str, query_text: str, top_k: int = 3):
    """Search a collection using vector similarity."""
    query_embedding = get_ollama_embedding(query_text)
    
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        anns_field="text_embedding",
        search_params={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["text", "source", "chunk_index"]
    )
    
    return results

def main():
    # Initialize clients
    client1 = MilvusClient(uri=f"http://{MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Tell me about programming languages",
        "What is the future of technology?"
    ]
    
    print("\nSearching Collection 1:")
    print("-" * 50)
    for query in queries:
        print(f"\nQuery: {query}")
        results = perform_search(client1, [query], MILVUS_CONFIG["collection"], filters=["chunk_index == 1"])
        # results = search_collection(client1, MILVUS_CONFIG["collection"], query)
        for hits in results:
            for hit in hits:
                print(f"Score: {hit["distance"]:.4f}")
                print(f"Text: {hit['entity']['text']}")
                print(f"Source: {hit['entity']['source']}")
                print(f"Chunk Index: {hit['entity']['chunk_index']}")
                print("-" * 30)
    

if __name__ == "__main__":
    main() 