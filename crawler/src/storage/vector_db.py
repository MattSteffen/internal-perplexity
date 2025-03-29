import logging
from typing import List, Dict, Any, Optional
import numpy as np
import re 
import yaml 
from uuid import uuid4

MAX_DOC_LENGTH = 10240

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("Pymilvus not installed. Using mock implementation for development.")

class VectorStorage:
    def __init__(self, config: dict = {}):
        # Now use nested config sections for various parameters.
        self.config = config
        milvus_config = config.get("milvus", {})
        embeddings_config = config.get("embeddings", {})

        self.host = milvus_config.get("host", "localhost")
        self.port = milvus_config.get("port", 19530)
        # Use top-level "collection" key for the collection name.
        self.collection_name = config.get("collection", "documents")
        self.dim = embeddings_config.get("dimension", 384)
        # If you want read_only to be controlled separately, you can either leave it top-level
        # or add another key (for example under "vector_db"). Here we assume top-level.
        self.read_only = config.get("read_only", False)
        self.collection = None

        # Milvus
        self.client = None

    def __enter__(self):
        milvus_config = self.config.get("milvus", {})
        # Configure with security credentials from the "milvus" section.
        user = milvus_config.get("user")
        password = milvus_config.get("password")
        secure = milvus_config.get("secure", False)
        if user and password:
            connections.connect(host=self.host, port=self.port, user=user, password=password, secure=secure)
        else:
            connections.connect(host=self.host, port=self.port)
        
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            # self.collection = Collection(self.collection_name) # TODO: This is a bug, don't delete every time.
            utility.drop_collection(self.collection_name)
            self._create_collection()
            self.collection.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.collection:
            self.collection.release()
        # Disconnect using the collection name.
        connections.disconnect(self.collection_name)

    def close(self):
        self.__exit__(None, None, None)

    def _create_collection(self):
        # Get the schema config from the "metadata" section.
        schema_config = self.config.get("metadata", {}).get("schema")
        if not schema_config:
            schema_config = load_schema_config(self.schema_config_path)
        # Also pass milvus-specific config to the collection creation helper.
        milvus_config = self.config.get("milvus", {})
        self.collection = create_collection(schema_config, self.collection_name, self.dim, milvus_config)

    def insert_data(self, texts: list, embeddings: list, metadatas: list):
        # TODO: Confirm that the schema is generated correctly.
        # TODO: Confirm that the metadata matches the schema
        """
        Inserts data into Milvus collection, ensuring no duplicate records
        (based on a unique combination of "source" and "chunk_index") are inserted.
        """
        if self.read_only:
            print("Storage is in read-only mode. Insert operation skipped.")
            return

        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length")
        
        # new_entries = []
        # for i in range(len(texts)):
        #     meta = metadatas[i]
        #     source = meta.get('source', '')
        #     chunk_index = meta.get('chunk_index')
        #     if chunk_index is None:
        #         raise ValueError("Each metadata dict must include a 'chunk_index' field")
        #     new_entries.append((source, chunk_index))
        
        # seen = set()
        # indices_to_check = []
        # for i, key in enumerate(new_entries):
        #     print(f"Checking key: {key}")
        #     if key not in seen:
        #         seen.add(key)
        #         indices_to_check.append(i)
        #     else:
        #         print(f"Skipping duplicate within batch for key {key}")
        
        # print("lens: ", len(seen), len(indices_to_check), len(texts))

        # if len(indices_to_check) == 0:
        #     print("No new entries to insert (all are duplicates in batch).")
        #     return
        
        # filter_clauses = []
        # for key in seen:
        #     s, ci = key
        #     filter_clauses.append(f'(source == "{s}" and chunk_index == {ci})')
        # filter_expr = " or ".join(filter_clauses)
        
        # existing_records = self.collection.query(expr=filter_expr, output_fields=["source", "chunk_index"])
        # existing_keys = {(rec["source"], rec["chunk_index"]) for rec in existing_records}
        
        # final_indices = []
        # for i in indices_to_check:
        #     key = new_entries[i]
        #     if key in existing_keys:
        #         print(f"Skipping duplicate existing entry for key {key}")
        #     else:
        #         final_indices.append(i)
        
        # if not final_indices:
        #     print("No new entries to insert after duplicate check.")
        #     return

        # texts_to_insert = [texts[i] for i in final_indices]
        # embeddings_to_insert = [embeddings[i] for i in final_indices]
        final_indexes = list(range(len(texts)))

        # Get metadata fields from the "metadata" schema.
        metadata_schema = self.config.get("metadata", {}).get("schema", {})
        metadata_fields = list(metadata_schema.get("properties", {}).keys())
        
        data: list[dict] = []
        for i in final_indexes:
            field_values = {field: metadatas[i].get(field, "") for field in metadata_fields} # only use the fields defined in the schema
            field_values["embedding"] = embeddings[i]
            field_values["text"] = texts[i][:MAX_DOC_LENGTH]
            data.append(field_values)
        

        # Optionally, if a partition is defined at the top level, retrieve it.
        self.collection.insert(data, partition_name=self.config.get('partition', None))
        self.collection.flush()
        print(f"Inserted {len(texts)} new chunks.")

    def search(self, query_embedding: list, limit: int = 5, filters: list[str] = []) -> list:
        # TODO: if config.read_only:
        """
        Search for similar chunks using a query embedding.
        Optionally filter by a list of authors or other valid filters.

        Args:
            query_embedding (list): The query vector. If empty or None, the search will
                                    only filter based on the provided filters.
            limit (int): Maximum number of results to return.
            filters (list[str], optional): A list of filter clauses as strings.
                                           Each filter should follow the pattern:
                                           "field == value" or "field in [value1, value2, ...]".

        Returns:
            list: List of dictionaries containing search results with metadata.
        """
        # Define a set of allowed fields for filtering.
        allowed_fields = {"source", "chunk_index", "text", "title", "author", "author_role", "url"}

        def is_valid_filter(filter_str: str) -> bool:
            """
            A simple validator for filter expressions.
            The expected format is optionally enclosed in parentheses, then:
              field (==|in|<|>) value
            """
            pattern = r'^\s*\(?\s*([a-zA-Z_]+)\s*(==|in|<|>)\s*(.+)\s*\)?\s*$'
            match = re.match(pattern, filter_str)
            if not match:
                return False
            field = match.group(1)
            return field in allowed_fields

        # Validate the filters provided and keep only those that are valid.
        valid_filter_list = []
        for f in filters:
            if is_valid_filter(f):
                valid_filter_list.append(f)
            else:
                print(f"Warning: Invalid filter skipped: {f}")

        # Combine valid filters using 'and'
        filter_expr = " and ".join(valid_filter_list) if valid_filter_list else ""

        if query_embedding:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["text", "source", "title", "author", "author_role", "url", "chunk_index"]
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get("text"),
                        "source": hit.entity.get("source"),
                        "title": hit.entity.get("title"),
                        "author": hit.entity.get("author"),
                        "author_role": hit.entity.get("author_role"),
                        "url": hit.entity.get("url"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "distance": hit.distance
                    })

            return formatted_results
        else:
            # No embedding provided: query using the filter expression (or all docs if no filter).
            results = self.collection.query(
                expr=filter_expr,
                output_fields=["text", "source", "title", "author", "author_role", "url", "chunk_index"]
            )
            return results[:limit]





def load_schema_config(config_file: str) -> dict:
    """Loads the collection schema configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config



def build_collection_schema(schema_config: dict, default_dim: int = 384) -> CollectionSchema:
    """
    Builds a CollectionSchema based on the provided schema configuration.
    The passed schema_config is now the actual JSON schema (not nested under "schema").
    """
    # Now access properties directly from the schema_config.
    fields_config = schema_config.get("properties", {})
    # Always add the auto-generated id field.
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=default_dim),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=MAX_DOC_LENGTH),
    ]
    
    for field_name, field_def in fields_config.items():
        field_type = field_def["type"]
        if field_type == "string":
            if field_name != "text": # Not allowed to duplicate text
                max_length = field_def.get("maxLength", 1024)
                fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length))
        elif field_type == "integer":
            fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
        elif field_type == "float_vector":
            if field_name != "embedding": # Not allowed to duplicate embedding
                dim = field_def.get("dim", default_dim)
                fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim))
        elif field_type == "array":
            max_length = field_def.get("maxLength", 1024)
            fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length))
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
    
    description = schema_config.get("description", "Document chunks with embeddings")

    return CollectionSchema(fields, description=description)

def create_collection(schema_config: dict, collection_name: str, default_dim: int, milvus_config: dict) -> Collection:
    """
    Creates and returns a new Milvus collection based on the schema configuration.
    The milvus_config is used to obtain index parameters.
    """
    schema = build_collection_schema(schema_config, default_dim)
    collection = Collection(collection_name, schema)
    
    # Get index details from the milvus config.
    index_field = milvus_config.get("index_field", "embedding")
    index_params = milvus_config.get("index_params", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    collection.create_index(index_field, index_params)
    
    return collection



# sample config:
# {'vector_db': {'enabled': True}, 'milvus': {'host': 'localhost', 'port': 19530, 'user': 'minioadmin', 'password': 'minioadmin', 'secure': False, 'index_field': 'embedding', 'index_params': {'index_type': 'IVF_FLAT', 'metric_type': 'L2', 'params': {'nlist': 128}}}, 'llm': {'model': 'llama-3.3-70b-versatile', 'provider': 'groq', 'base_url': 'https://api.groq.com'}, 'vision_llm': {'model': 'gemma3', 'provider': 'ollama', 'base_url': 'http://localhost:11434'}, 'embeddings': {'model': 'all-minilm:v2', 'provider': 'ollama', 'base_url': 'http://localhost:11434', 'dimension': 384}, 'logging': {'level': 'INFO', 'file': 'crawler.log', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}, 'processing': {'extractors': [{'type': 'json', 'enabled': True, 'metadata_mapping': {'title': 'paper_title', 'authors': 'author', 'year': 'conference_year'}}], 'chunk_size': 800}, 'name': 'default_collection', 'description': 'Conference documents and papers with embeddings', 'metadata': {'extra_embeddings': [], 'schema': {'$schema': 'http://json-schema.org/draft-07/schema#', 'title': 'Document', 'type': 'object', 'properties': {'text': {'type': 'string', 'maxLength': 1024, 'description': 'Text content of the document chunk.'}, 'embedding': {'type': 'float_vector', 'dim': 384, 'description': 'Embedding vector of the document chunk.'}, 'source': {'type': 'string', 'maxLength': 1024, 'description': 'Source identifier of the document chunk.'}, 'title': {'type': 'string', 'maxLength': 255, 'description': 'Title of the document.'}, 'author': {'type': 'array', 'maxItems': 255, 'items': {'type': 'string', 'description': 'An author of the document.'}, 'description': 'List of authors of the document.'}, 'author_role': {'type': 'string', 'maxLength': 255, 'description': 'Role of the author in the document (e.g., writer, editor).'}, 'url': {'type': 'string', 'maxLength': 1024, 'description': 'URL associated with the document.'}, 'chunk_index': {'type': 'integer', 'description': 'Index of the document chunk.'}}}}, 'path': '../../data/conference', 'collection': 'conference_docs'}