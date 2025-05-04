from pymilvus import (
    MilvusClient, DataType
)

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=5)

from pymilvus import MilvusClient

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="AUTOINDEX",
    metric_type="IP"
)

index_params.add_index(
    field_name="sparse",
    index_name="sparse_index",
    index_type="AUTOINDEX",  # Index type for sparse vectors
    metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
    # params={"drop_ratio_build": 0.2},  # The ratio of small vector values to be dropped during indexing
)
from pymilvus import MilvusClient

client.create_collection(
    collection_name="my_collection",
    schema=schema,
    index_params=index_params
)


from pymilvus import MilvusClient

data=[
    {"id": 0, "text": "Artificial intelligence was founded as an academic discipline in 1956.", "sparse":{9637: 0.30856525997853057, 4399: 0.19771651149001523, }, "dense": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]},
    {"id": 1, "text": "Alan Turing was the first person to conduct substantial research in AI.", "sparse":{6959: 0.31025067641541815, 1729: 0.8265339135915016, }, "dense": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.9029438446296592, 0.43742130801983836]},
    {"id": 2, "text": "Born in Maida Vale, London, Turing was raised in southern England.", "sparse":{1220: 0.15303302147479103, 7335: 0.9436728846033107, }, "dense": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.29836057404, -0.398398650]}]

res = client.insert(
    collection_name="my_collection",
    data=data
)

from pymilvus import AnnSearchRequest

query_dense_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

search_param_1 = {
    "data": [query_dense_vector],
    "anns_field": "dense",
    "param": {
        "metric_type": "IP",
        "params": {"nprobe": 10.0}
    },
    "limit": 2
}
request_1 = AnnSearchRequest(**search_param_1)

query_sparse_vector = {3573: 0.34701499565746674}, {5263: 0.2639375518635271}
search_param_2 = {
    "data": [query_sparse_vector],
    "anns_field": "sparse",
    "param": {
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0.2}
    },
    "limit": 2
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]
# reqs = [request_1]
from pymilvus import RRFRanker

ranker = RRFRanker(100)


res = client.hybrid_search(
    collection_name="my_collection",
    reqs=reqs,
    ranker=ranker,
    output_fields=["text"],
    limit=2
)
for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit.get('entity'))



# from pymilvus import MilvusClient, DataType, Function, FunctionType

# client = MilvusClient(
#     uri="http://localhost:19530",
#     token="root:Milvus"
# )

# schema = MilvusClient.create_schema()

# schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
# schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True)
# schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# bm25_function = Function(
#     name="text_bm25_emb", # Function name
#     input_field_names=["text"], # Name of the VARCHAR field containing raw text data
#     output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
#     function_type=FunctionType.BM25, # Set to `BM25`
# )

# schema.add_function(bm25_function)


# index_params = MilvusClient.prepare_index_params()

# index_params.add_index(
#     field_name="sparse",

#     index_type="SPARSE_INVERTED_INDEX",
#     metric_type="BM25",
#     params={
#         "inverted_index_algo": "DAAT_MAXSCORE",
#         "bm25_k1": 1.2,
#         "bm25_b": 0.75
#     }

# )

# client.create_collection(
#     collection_name='my_collection', 
#     schema=schema, 
#     index_params=index_params
# )


# client.insert('my_collection', [
#     {'text': 'information retrieval is a field of study.'},
#     {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
#     {'text': 'data mining and information retrieval overlap in research.'},
# ])


# search_params = {
#     'params': {'drop_ratio_search': 0.2},
# }

# res = client.search(
#     collection_name='my_collection', 
#     data=['whats the focus of information retrieval?'],
#     anns_field='sparse',
#     limit=3,
#     search_params=search_params
# )


# print(res)
# client.drop_collection('my_collection')