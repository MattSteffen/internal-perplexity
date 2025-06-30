
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Extracted from milvus_search.py
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
import ollama

# --- Constants and Schemas ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "all-minilm:v2"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "test_collection"
OUTPUT_FIELDS = ["source", "chunk_index", "metadata", "title", "author", "date", "keywords", "unique_words"]

SearchInputSchema = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Performs a semantic search using the given queries and optional filters.",
        "parameters": {
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of queries for semantic search",
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter expressions to apply to the search",
                    "default": [],
                },
            },
        },
    },
}

QueryInputSchema = {
    "type": "function",
    "function": {
        "name": "query",
        "description": "Runs a filtered query without semantic search. Only filters are used.",
        "parameters": {
            "type": "object",
            "required": ["filters"],
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter expressions to apply to the query",
                    "default": [],
                }
            },
        },
    },
}

# --- Pydantic Models for API ---
class SearchInput(BaseModel):
    queries: List[str]
    filters: Optional[List[str]] = []

class QueryInput(BaseModel):
    filters: List[str]

# --- FastAPI App ---
app = FastAPI(title="Tools Server")

# --- Milvus Connection and Helper Functions (from milvus_search.py) ---
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

# --- API Routes ---
@app.get("/openapi.json")
async def get_openapi_spec():
    return {
        "search": SearchInputSchema,
        "query": QueryInputSchema,
    }

@app.post("/search")
async def search_endpoint(input_data: SearchInput):
    milvus_client = connect_milvus()
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Could not connect to Milvus database.")
    
    # This is a simplified version of the search logic from milvus_search.py
    search_requests = []
    for query in input_data.queries:
        search_params = {
            "data": [get_embedding(query)],
            "anns_field": "text_embedding",
            "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
            "expr": " and ".join(input_data.filters),
            "limit": 10,
        }
        search_requests.append(AnnSearchRequest(**search_params))

    ranker = RRFRanker(k=100)
    results = milvus_client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=search_requests,
        ranker=ranker,
        output_fields=OUTPUT_FIELDS,
        limit=10
    )
    return results

@app.post("/query")
async def query_endpoint(input_data: QueryInput):
    milvus_client = connect_milvus()
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Could not connect to Milvus database.")

    query_results = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter=" and ".join(input_data.filters),
        output_fields=OUTPUT_FIELDS,
        limit=100,
    )
    return query_results

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
