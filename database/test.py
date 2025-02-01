from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import random

print("Connecting to Milvus...")
# Connect to Milvus Docker container
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
print("Successfully connected to Milvus")

# Collection configuration
collection_name = "demo_documents"
dimension = 128  # Vector dimension

print(f"\nCreating collection schema '{collection_name}'...")
# Create collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
]

schema = CollectionSchema(fields, description="Document search demo")
collection = Collection(collection_name, schema)
print("Collection schema created successfully")

print("\nGenerating sample data...")
# Generate sample data
documents = [
    "Artificial intelligence fundamentals",
    "Machine learning techniques", 
    "Deep neural networks architecture",
    "Natural language processing applications",
    "Computer vision systems"
]

# Insert data with random vectors
data = [
    [i for i in range(len(documents))],  # IDs
    documents,  # Text content
    [[random.random() for _ in range(dimension)] for _ in documents]  # Vectors
]

print(f"Inserting {len(documents)} documents into collection...")
insert_result = collection.insert(data)
collection.flush()  # Ensure data is persisted
print("Data inserted successfully")

print("\nCreating index for vector search...")
# Create index for efficient search
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2", 
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="vector",
    index_params=index_params
)
print("Index created successfully")

# Load the collection into memory before searching
print("\nLoading collection into memory...")
collection.load()
print("Collection loaded successfully")

print("\nPerforming vector search...")
# Perform a vector search
search_vectors = [[random.random() for _ in range(dimension)]]

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

results = collection.search(
    data=search_vectors,
    anns_field="vector",
    param=search_params,
    limit=3,
    output_fields=["text"]
)

print("\nSearch Results:")
# Display results
for hits in results:
    print("Matched documents:")
    for hit in hits:
        print(f"- Text: {hit.entity.get('text')}")
        print(f"  Distance: {hit.distance:.4f}\n")
