from .schemas import SearchInput, QueryInput
from .utils import connect_milvus, get_embedding
from pymilvus import AnnSearchRequest, RRFRanker

OUTPUT_FIELDS = ["source", "chunk_index", "metadata", "title", "author", "date", "keywords", "unique_words"]

def search(input_data: SearchInput):
    milvus_client = connect_milvus()
    if not milvus_client:
        return {"error": "Could not connect to Milvus database."}

    search_requests = []
    for query in input_data.queries:
        search_params = {
            "data": [get_embedding(query)],
            "anns_field": "text_embedding",
            "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
            "expr": " and ".join(input_data.filters) if input_data.filters else None,
            "limit": 10,
        }
        search_requests.append(AnnSearchRequest(**search_params))

    ranker = RRFRanker(k=100)
    results = milvus_client.hybrid_search(
        collection_name="test_collection",
        reqs=search_requests,
        ranker=ranker,
        output_fields=OUTPUT_FIELDS,
        limit=10
    )
    return results

def query(input_data: QueryInput):
    milvus_client = connect_milvus()
    if not milvus_client:
        return {"error": "Could not connect to Milvus database."}

    query_results = milvus_client.query(
        collection_name="test_collection",
        filter=" and ".join(input_data.filters),
        output_fields=OUTPUT_FIELDS,
        limit=100,
    )
    return query_results
