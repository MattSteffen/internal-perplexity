import logging
from typing import List, Dict, Any, Optional
import numpy as np
import re 
import yaml 

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
    # TODO: use config path instead of all the arguments
    # TODO: have a config option that says read only or read/write
    def __init__(self, host: str = 'localhost', port: str = '19530', 
                 collection_name: str = 'documents', dim: int = 384,
                 schema_config_path: str = "milvus_schema.yaml"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.schema_config_path = schema_config_path
        self.collection = None

    def __enter__(self):
        # TODO: Configure with security credentials
        connections.connect(host=self.host, port=self.port)
        
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.collection:
            self.collection.release()
        connections.disconnect("default") # TODO: collection name which is from config file
    def close(self):
        self.__exit__(None, None, None)
    # def close(self):
    #     """Disconnect from Milvus."""
    #     connections.disconnect("default")

    def _create_collection(self):
        # Load schema configuration from YAML
        # TODO: use self.config instead.
        config = load_schema_config(self.schema_config_path)
        self.collection = create_collection(config, self.collection_name, self.dim)

    def insert_data(self, texts: list, embeddings: list, metadatas: list):
        # TODO: if config.read_only: don't do anything
        """
        Inserts data into Milvus collection, ensuring no duplicate records
        (based on a unique combination of "source" and "chunk_index") are inserted.
        
        Args:
            texts (list): List of text chunks.
            embeddings (list): List of embeddings for each chunk.
            metadatas (list): List of metadata dictionaries. Each metadata **must**
                              include the keys "source" and "chunk_index".
        """
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length")
        
        # Build a list of unique keys from new records: (source, chunk_index)
        new_entries = []
        for i in range(len(texts)):
            meta = metadatas[i]
            source = meta.get('source', '')
            chunk_index = meta.get('chunk_index')
            if chunk_index is None:
                raise ValueError("Each metadata dict must include a 'chunk_index' field")
            new_entries.append((source, chunk_index))
        
        # Remove duplicates within the new batch
        seen = set()
        indices_to_check = []
        for i, key in enumerate(new_entries):
            print(f"Checking key: {key}")
            if key not in seen:
                seen.add(key)
                indices_to_check.append(i)
            else:
                print(f"Skipping duplicate within batch for key {key}")
        
        print("lens: ", len(seen), len(indices_to_check), len(texts))

        # If all entries are duplicates, return early
        if len(indices_to_check) == 0:
            print("No new entries to insert (all are duplicates in batch).")
            return
        
        # Build a filter expression to query for existing records.
        filter_clauses = []
        for key in seen:
            s, ci = key
            filter_clauses.append(f'(source == "{s}" and chunk_index == {ci})')
        filter_expr = " or ".join(filter_clauses)
        
        # Query the collection for existing entries that match any of these keys.
        existing_records = self.collection.query(expr=filter_expr, output_fields=["source", "chunk_index"])
        existing_keys = {(rec["source"], rec["chunk_index"]) for rec in existing_records}
        
        # Determine final indices to insert (exclude any already existing records)
        final_indices = []
        for i in indices_to_check:
            key = new_entries[i]
            if key in existing_keys:
                print(f"Skipping duplicate existing entry for key {key}")
            else:
                final_indices.append(i)
        
        if not final_indices:
            print("No new entries to insert after duplicate check.")
            return

        # Prepare data for insertion (using metadata values from the final indices)
        # TODO: This is hardcoded. It should be created from config options.
        texts_to_insert = [texts[i] for i in final_indices]
        embeddings_to_insert = [embeddings[i] for i in final_indices]
        sources_to_insert = [metadatas[i].get('source', '') for i in final_indices]
        titles_to_insert = [metadatas[i].get('title', '') for i in final_indices]
        authors_to_insert = [metadatas[i].get('author', '') for i in final_indices]
        author_roles_to_insert = [metadatas[i].get('author_role', '') for i in final_indices]
        urls_to_insert = [metadatas[i].get('url', '') for i in final_indices]
        chunk_indices_to_insert = [metadatas[i].get('chunk_index') for i in final_indices]

        data = [
            texts_to_insert,
            embeddings_to_insert,
            sources_to_insert,
            titles_to_insert,
            authors_to_insert,
            author_roles_to_insert,
            urls_to_insert,
            chunk_indices_to_insert
        ]

        self.collection.insert(data)
        self.collection.flush()
        print(f"Inserted {len(texts_to_insert)} new chunks.")

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



def build_collection_schema(config: dict, default_dim: int) -> CollectionSchema:
    """
    Builds a CollectionSchema based on the provided configuration.
    """
    fields_config = config.get("fields", [])
    # Always add the auto-generated id field.
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)
    ]
    
    for field in fields_config:
        name = field["name"]
        field_type = field["type"]
        
        if field_type == "string":
            max_length = field.get("max_length", 1024)
            fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length))
        elif field_type == "integer":
            fields.append(FieldSchema(name=name, dtype=DataType.INT64))
        elif field_type == "float_vector":
            dim = field.get("dim", default_dim)
            fields.append(FieldSchema(name=name, dtype=DataType.FLOAT_VECTOR, dim=dim))
        elif field_type == "array":
            max_length = field.get("max_length", 1024)
            fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length))
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
    
    description = config.get("description", "Document chunks with embeddings")
    return CollectionSchema(fields, description=description)

def create_collection(config: dict, collection_name: str, default_dim: int) -> Collection:
    """
    Creates and returns a new Milvus collection based on the configuration.
    """
    schema = build_collection_schema(config, default_dim)
    collection = Collection(collection_name, schema)
    
    # Create index for vector search.
    index_field = config.get("index_field", "embedding")
    index_params = config.get("index_params", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    collection.create_index(index_field, index_params)
    
    return collection
