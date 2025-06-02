MILVUS_CONFIG = {
    "host": "localhost",
    "port": "19530",
    "username": "",
    "password": "",
    "collection": "test_collection",
    "partition": "test_partition"
}

MILVUS_CONFIG_2 = {
    "host": "localhost",
    "port": "19530",
    "username": "",
    "password": "",
    "collection": "test_collection_2",
    "partition": "test_partition_2"
}

# Example JSON schemas for metadata
SCHEMA_1 = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Document title"
        },
        "author": {
            "type": "string",
            "description": "Document author"
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Document tags"
        }
    }
}

SCHEMA_2 = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "Document category"
        },
        "importance": {
            "type": "integer",
            "description": "Importance score"
        },
        "meta_data": {
            "type": "object",
            "description": "Additional metadata"
        }
    }
} 