from pymilvus import MilvusClient, DataType
import numpy as np

# ============================================================================
# SECTION 1: SETUP MILVUS CLUSTER AND PERMISSIONS
# ============================================================================

# Initialize admin client
admin_client = MilvusClient(
    uri="http://10.43.73.179:19530",
    token="root:Milvus",
)

# Clean up existing collections if they exist
HAPPY_COLLECTION = "happy_dsp_docs"
SAD_COLLECTION = "sad_dsp_docs"

for collection in [HAPPY_COLLECTION, SAD_COLLECTION]:
    if admin_client.has_collection(collection):
        admin_client.drop_collection(collection)

# Create users
users = {
    "happy_reader": "HappyPass123",
    "sad_reader": "SadPass123",
    "dsp_admin": "AdminPass123",
}

for username, password in users.items():
    try:
        admin_client.create_user(user_name=username, password=password)
    except Exception as e:
        print(f"User {username} may already exist: {e}")

# Create roles
roles = ["happy_read_only", "sad_read_only", "dsp_full_access"]

for role in roles:
    try:
        admin_client.create_role(role_name=role)
    except Exception as e:
        print(f"Role {role} may already exist: {e}")

# Assign roles to users
admin_client.grant_role(user_name="happy_reader", role_name="happy_read_only")
admin_client.grant_role(user_name="sad_reader", role_name="sad_read_only")
admin_client.grant_role(user_name="dsp_admin", role_name="dsp_full_access")

print("✓ Users and roles created")

# ============================================================================
# SECTION 2: CREATE COLLECTIONS AND INSERT SAMPLE DATA
# ============================================================================

# Create schema for collections
DIMENSION = 128

schema = admin_client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("content", DataType.VARCHAR, max_length=2048)

index_params = admin_client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})

# Create happy collection
admin_client.create_collection(
    collection_name=HAPPY_COLLECTION,
    schema=schema,
    index_params=index_params,
)

# Create sad collection
admin_client.create_collection(
    collection_name=SAD_COLLECTION,
    schema=schema,
    index_params=index_params,
)

print("✓ Collections created")

# Grant privileges after collections exist using built-in privilege groups
admin_client.grant_privilege_v2(
    role_name="happy_read_only",
    privilege="CollectionReadOnly",
    collection_name=HAPPY_COLLECTION,
    db_name="default",
)

admin_client.grant_privilege_v2(
    role_name="sad_read_only",
    privilege="CollectionReadOnly",
    collection_name=SAD_COLLECTION,
    db_name="default",
)

for collection in [HAPPY_COLLECTION, SAD_COLLECTION]:
    admin_client.grant_privilege_v2(
        role_name="dsp_full_access",
        privilege="CollectionAdmin",
        collection_name=collection,
        db_name="default",
    )

print("✓ Privileges granted")

# Insert happy DSP documents
happy_docs = [
    {
        "title": "Breakthrough in Fast Fourier Transform Optimization",
        "content": "Researchers achieved remarkable speedups in FFT computations, enabling real-time processing of high-resolution audio. The new algorithm reduces complexity while maintaining perfect accuracy.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
    {
        "title": "Revolutionary Filter Design Simplifies Audio Processing",
        "content": "A novel IIR filter design methodology makes audio enhancement accessible to everyone. The elegant solution provides crystal-clear output with minimal computational overhead.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
    {
        "title": "Joyful Advances in Wavelet Transform Applications",
        "content": "Wavelet analysis brings exciting new possibilities to image compression. The beautiful mathematical properties enable lossless compression with incredible efficiency and visual clarity.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
]

admin_client.insert(collection_name=HAPPY_COLLECTION, data=happy_docs)
print(f"✓ Inserted {len(happy_docs)} happy documents")

# Insert sad DSP documents
sad_docs = [
    {
        "title": "Disappointing Limitations in Sampling Theory",
        "content": "New research reveals fundamental constraints in signal reconstruction. Aliasing artifacts prove unavoidable in many practical scenarios, limiting the effectiveness of digital signal processing.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
    {
        "title": "Tragic Failures in Noise Reduction Algorithms",
        "content": "Despite decades of research, noise removal remains frustratingly difficult. The inherent trade-off between noise suppression and signal distortion continues to plague audio engineers.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
    {
        "title": "Melancholic Reality of Quantization Error",
        "content": "Quantization noise inevitably degrades digital signals. The unavoidable loss of information in analog-to-digital conversion represents a sorrowful compromise in signal fidelity.",
        "embedding": np.random.rand(DIMENSION).tolist(),
    },
]

admin_client.insert(collection_name=SAD_COLLECTION, data=sad_docs)
print(f"✓ Inserted {len(sad_docs)} sad documents")

# ============================================================================
# SECTION 3: TEST ACCESS CONTROL WITH DIFFERENT USERS
# ============================================================================

# Create query vector
query_vector = np.random.rand(DIMENSION).tolist()

print("\n" + "="*70)
print("TESTING ACCESS CONTROL")
print("="*70)

# Test 1: Happy reader (should only access happy collection)
print("\n1. Happy Reader searching happy collection:")
happy_client = MilvusClient(
    uri="http://localhost:19530",
    token="happy_reader:HappyPass123",
)

try:
    results = happy_client.search(
        collection_name=HAPPY_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✓ Success! Found {len(results[0])} results")
    for hit in results[0]:
        print(f"   - {hit['entity']['title']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n   Happy Reader attempting sad collection (should fail):")
try:
    results = happy_client.search(
        collection_name=SAD_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✗ Unexpected success - access control failed!")
except Exception as e:
    print(f"   ✓ Correctly denied access")

# Test 2: Sad reader (should only access sad collection)
print("\n2. Sad Reader searching sad collection:")
sad_client = MilvusClient(
    uri="http://localhost:19530",
    token="sad_reader:SadPass123",
)

try:
    results = sad_client.search(
        collection_name=SAD_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✓ Success! Found {len(results[0])} results")
    for hit in results[0]:
        print(f"   - {hit['entity']['title']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n   Sad Reader attempting happy collection (should fail):")
try:
    results = sad_client.search(
        collection_name=HAPPY_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✗ Unexpected success - access control failed!")
except Exception as e:
    print(f"   ✓ Correctly denied access")

# Test 3: Admin (should access both collections)
print("\n3. Admin searching both collections:")
admin_dsp_client = MilvusClient(
    uri="http://localhost:19530",
    token="dsp_admin:AdminPass123",
)

print("\n   Admin searching happy collection:")
try:
    results = admin_dsp_client.search(
        collection_name=HAPPY_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✓ Success! Found {len(results[0])} results")
    for hit in results[0]:
        print(f"   - {hit['entity']['title']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n   Admin searching sad collection:")
try:
    results = admin_dsp_client.search(
        collection_name=SAD_COLLECTION,
        data=[query_vector],
        limit=2,
        output_fields=["title", "content"],
    )
    print(f"   ✓ Success! Found {len(results[0])} results")
    for hit in results[0]:
        print(f"   - {hit['entity']['title']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*70)
print("ACCESS CONTROL TEST COMPLETE")
print("="*70)


# from pymilvus import MilvusClient

# client = MilvusClient(
#     uri="http://localhost:19530",  # replace with your own Milvus server address
#     token="root:Milvus",
# )

# # Create Collections
# # TODO: Create the collections along with the configuration of each collection
# collection_list: list[str] = [
#     "test_arxiv2",
#     "test_arxiv3",
#     "test_arxiv4",
# ]

# for collection in collection_list:
#     client.create_collection(collection_name=collection)


# # Create users
# class User:
#     def __init__(self, username, password):
#         self.username = username
#         self.password = password
#         self.role = "admin"


# # TODO: Generate the list of all users maybe once
# user_list: list[User] = [
#     # User("root", "Milvus"),
#     User("admin", "Admin123"),
#     User("user", "User123"),
# ]

# for user in user_list:
#     client.create_user(user_name=user.username, password=user.password)


# # Create roles
# # TODO: Try to mimic the roles rincon uses
# class Role:
#     def __init__(self, name, description):
#         self.name = name
#         self.description = description


# role_list: list[Role] = [
#     Role("admin", "Admin role"),
#     Role("user", "User role"),
# ]

# for role in role_list:
#     client.create_role(role_name=role.name, description=role.description)

# # Grant Priveledges
# permissions = {
#     "admin": [("CollectionAdmin", "test_arxiv2")],
#     "user": [("CollectionReadOnly", "test_arxiv2")],
# }

# for role, permissions in permissions.items():
#     for permission in permissions:
#         client.grant_privilege_v2(
#             role_name=role,
#             privilege=permission[0],
#             collection_name=permission[1],
#             db_name="default",
#         )


# # Assign roles to users
# for user in user_list:
#     client.grant_role(
#         user_name=user.username,
#         role_name=user.role,
#     )



