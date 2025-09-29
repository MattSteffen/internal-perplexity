from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",  # replace with your own Milvus server address
    token="root:Milvus",
)

# Create Collections
# TODO: Create the collections along with the configuration of each collection
collection_list: list[str] = [
    "test_arxiv2",
    "test_arxiv3",
    "test_arxiv4",
]

for collection in collection_list:
    client.create_collection(collection_name=collection)


# Create users
class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.role = "admin"


# TODO: Generate the list of all users maybe once
user_list: list[User] = [
    User("root", "Milvus"),
    User("admin", "Admin123"),
    User("user", "User123"),
]

for user in user_list:
    client.create_user(user_name=user.username, password=user.password)


# Create roles
# TODO: Try to mimic the roles rincon uses
class Role:
    def __init__(self, name, description):
        self.name = name
        self.description = description


role_list: list[Role] = [
    Role("admin", "Admin role"),
    Role("user", "User role"),
]

for role in role_list:
    client.create_role(role_name=role.name, description=role.description)

# Grant Priveledges
permissions = {
    "admin": [("CollectionAdmin", "test_arxiv2")],
    "user": [("CollectionReadOnly", "test_arxiv2")],
}

for role, permissions in permissions.items():
    for permission in permissions:
        client.grant_privilege_v2(
            role_name=role,
            privilege=permission[0],
            collection_name=permission[1],
            db_name="default",
        )


# Assign roles to users
for user in user_list:
    client.grant_role(
        user_name=user.username,
        role_name=user.role,
    )
