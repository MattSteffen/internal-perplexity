from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import re


class VectorStorage:
    def __init__(self, host: str = 'localhost', port: str = '19530', 
                 collection_name: str = 'documents', dim: int = 384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None

        connections.connect(host=host, port=port)
        
        if not utility.has_collection(collection_name):
            self._create_collection()
        else:
            self.collection = Collection(collection_name)
            self.collection.load()

    def _create_collection(self):
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name='author', dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name='author_role', dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name='chunk_index', dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="Document chunks with embeddings")
        self.collection = Collection(self.collection_name, schema)
        
        # Create index for vector search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)

    def insert_data(self, texts: list, embeddings: list, metadatas: list):
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
        
        # Build a list of unique keys from new records
        # Each key is a tuple: (source, chunk_index)
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
            # key is a tuple: (source, chunk_index)
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
        # Example filter: (source == "abc" and chunk_index == 1) or (source == "def" and chunk_index == 2)
        filter_clauses = []
        for key in seen:
            s, ci = key
            # Use double quotes for string values in the filter expression
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
              field (==|in) value
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

        # If a query embedding is provided, perform a vector search.
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
            # Apply limit manually if needed.
            return results[:limit]

    def close(self):
        """Disconnect from Milvus."""
        connections.disconnect("default")
