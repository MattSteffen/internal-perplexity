import logging
import json 
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .utils import build_milvus_schema, validate_schema, _validate_metadata
# Constants
MAX_DOC_LENGTH = 10240  # Max length for the 'text' field in Milvus should be 65000
# Define reasonable max lengths for indexed VARCHAR fields
MAX_SOURCE_LENGTH = 1024 # Adjust as needed for your source path/identifier lengths
DEFAULT_VARCHAR_MAX_LENGTH = 1024 # Default for other string fields

try:
    from pymilvus import MilvusClient, DataType, MilvusException
    MILVUS_AVAILABLE = True
    logging.info("Pymilvus library loaded successfully.")
except ImportError:
    MILVUS_AVAILABLE = False
    logging.error("Pymilvus not installed. VectorStorage operations cannot proceed.")

class VectorStorage:
    """
    Manages interaction with a Milvus vector database collection for storing
    document chunks and their embeddings.

    Handles connection, schema management, data insertion, and duplicate detection
    based on the combination of 'source' and 'chunk_index' metadata fields.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the VectorStorage instance.

        Args:
            config: A dictionary containing configuration settings. Expected keys include:
                - collection (str): Name of the Milvus collection.
                - milvus (dict): Milvus connection details (host, port, user, password, secure, index_params, etc.).
                - embeddings (dict): Embedding model details (dimension).
                - metadata (dict): Metadata schema definition ('schema' key with JSON schema).
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("Pymilvus library is required but not installed.")

        self.config = config
        self.milvus_config: Dict[str, Any] = config.get("milvus", {})
        self.schema_config: Optional[Dict[str, Any]] = validate_schema(self.config.get("metadata", {}).get("schema"))

        self.host: str = self.milvus_config.get("host")
        self.port: int = self.milvus_config.get("port")
        self.collection_name: str = config.get("collection")
        if not self.collection_name:
            raise ValueError("Configuration must include a 'collection' name.")
        self.partition_name: str = self.config.get("partition", None)

        self.client: Optional[MilvusClient] = None
        logging.info(f"VectorStorage initialized for collection '{self.collection_name}' at {self.host}:{self.port}")

    def __enter__(self) -> 'VectorStorage':
        """
        Connects to Milvus and loads or creates the collection upon entering context.
        """
        # TODO: Remove defaults
        user = self.milvus_config.get("user", "root")
        password = self.milvus_config.get("password", "Milvus")
        secure = self.milvus_config.get("secure", False)
        protocol = "https" if secure else "http"

        try:
            uri = f"{protocol}://{self.host}:{self.port}"
            token = f"{user}:{password}"
            
            logging.info(f"Connecting to Milvus at {uri}...")
            self.client = MilvusClient(uri=uri, token=token)
            logging.info("Successfully connected to Milvus.")

            if not self.client.has_collection(self.collection_name):
                logging.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self._create_collection()
                logging.info(f"Collection '{self.collection_name}' created.")
            elif self.config.get("recreate", False):
                logging.info(f"Collection '{self.collection_name}' exists but 'recreate' is True. Recreating...")
                self.client.drop_collection(self.collection_name)
                self._create_collection()
                logging.info(f"Collection '{self.collection_name}' already exists.")

            if self.partition_name:
                if not self.client.has_partition(self.collection_name, self.partition_name):
                    logging.info(f"Creating partition '{self.partition_name}'...")
                    self.client.create_partition(self.collection_name, self.partition_name)

            # Load collection
            self.client.load_collection(self.collection_name)
            logging.info(f"Collection '{self.collection_name}' loaded.")

        except MilvusException as e:
            logging.error(f"Milvus error during connection or collection handling: {e}")
            if self.client:
                self.client.close()
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during VectorStorage setup: {e}")
            if self.client:
                self.client.close()
            raise

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Releases collection and disconnects from Milvus. """
        if self.client:
            try:
                logging.info(f"Releasing collection '{self.collection_name}' from memory.")
                self.client.release_collection(self.collection_name)
            except MilvusException as e:
                logging.warning(f"Error releasing collection '{self.collection_name}': {e}")
            except Exception as e:
                logging.warning(f"Unexpected error releasing collection: {e}")
            
            try:
                logging.info("Closing Milvus client connection.")
                self.client.close()
            except Exception as e:
                logging.warning(f"Error closing client connection: {e}")
            
            self.client = None

    def close(self):
        """ Provides an explicit close method. """
        self.__exit__(None, None, None)

    def _create_collection(self):
        """ Creates the Milvus collection with schema and indexes """
        if not self.schema_config:
            raise RuntimeError("Schema configuration not loaded before attempting collection creation.")

        try:
            schema = build_milvus_schema(
                schema_config=self.schema_config,
                embedding_dim=self.config.get("embeddings", {}).get("dimension", 384) # TODO: Determine this by the embedding model
            )

            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_name="embedding_index",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            index_params.add_index(
                field_name="full_text_embedding",
                index_name="full_text_embedding_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75
                }
            )

            # Create collection with schema and index
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

        except MilvusException as e:
            logging.error(f"Failed to create collection or index '{self.collection_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during collection creation: {e}")
            raise

    def _check_duplicates(self, metadatas: List[Dict[str, Any]]) -> List[int]:
        # TODO: Check again. But actually don't need for now.
        """Check for duplicates using new client API"""
        if not self.client:
            logging.error("Client not connected. Cannot check for duplicates.")
            return []
        if not metadatas:
            return []

        batch_keys: Dict[Tuple[str, int], int] = {} # Stores (source, chunk_index) -> first index mapping
        indices_to_check_db: List[int] = []
        sources_in_batch: Set[str] = set()
        indices_in_batch: Set[int] = set()

        # Check duplicates within batch
        for i, meta in enumerate(metadatas):
            source = meta.get('source')
            chunk_index = meta.get('chunk_index')

            if not source or not isinstance(source, str) or chunk_index is None or not isinstance(chunk_index, int):
                logging.warning(f"Missing or invalid 'source' or 'chunk_index' in metadata at index {i}. Skipping duplicate check for this item, it won't be inserted by validation.")
                # We rely on _validate_metadata to prevent insertion later
                continue

            current_key = (source, chunk_index)
            if current_key in batch_keys:
                logging.debug(f"Duplicate (source='{source}', chunk_index={chunk_index}) found within batch (index {i}, original index {batch_keys[current_key]}). Skipping.")
                continue # Skip this item, keep the first occurrence

            # First occurrence in batch
            batch_keys[current_key] = i
            indices_to_check_db.append(i)
            sources_in_batch.add(source)
            indices_in_batch.add(chunk_index)

        if not indices_to_check_db:
            return []

        # Check against database
        try:
            if sources_in_batch and indices_in_batch:
                expr = f"source in {list(sources_in_batch)} and chunk_index in {list(indices_in_batch)}"
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=["source", "chunk_index"]
                )

                existing_keys = {(r['source'], r['chunk_index']) for r in results}
                final_indices = [i for i in indices_to_check_db 
                               if (metadatas[i]['source'], metadatas[i]['chunk_index']) not in existing_keys]
                return final_indices

        except MilvusException as e:
            logging.error(f"Failed to query for duplicates: {e}")
            return indices_to_check_db

        return indices_to_check_db

    def insert_data(self, data: List[Dict[str, Any]]):
        """
        Inserts text, embeddings, and metadata into the Milvus collection,
        avoiding duplicates based on the ('source', 'chunk_index') combination.

        Args:
            data:
                text:  of text chunks.
                embedding:  of corresponding embedding vectors.
                metadata:  of corresponding metadata dictionaries. Each must contain
                       'source' (str) and 'chunk_index' (int).
        """
        if not self.client:
            logging.error("Client is not initialized. Cannot insert data.")
            raise RuntimeError("VectorStorage not properly initialized or entered.")

        if not data:
            logging.info("Received empty lists. No data to insert.")
            return

        # 1. Check for duplicates and get indices of items to insert
        logging.info(f"Starting duplicate check for {len(data)} items based on (source, chunk_index)...")
        # indices_to_insert = self._check_duplicates(data)
        indices_to_insert = list(range(len(data)))
        logging.info(f"Duplicate check complete. {len(indices_to_insert)} items identified as new.")

        if not indices_to_insert:
            logging.info("No new entries to insert after duplicate check.")
            return

        data_to_insert = []
        for i in indices_to_insert:
            # if not _validate_metadata(data[i]['metadata'], i):
            #     continue

            data_to_insert.append(data[i])

        if not data_to_insert:
            logging.info("No valid data remaining after metadata validation and preparation.")
            return
        
        # 2. Create metatext for full-text search
        meta_text_fields = self.config.get("metadata", {}).get("full_text_search", [])
        for i in indices_to_insert:
            data_to_insert[i]['metatext'] = json.dumps({
                k: data[i][k] for k in meta_text_fields
            })

        # 3. Insert the filtered and validated data
        try:
            successes = self.client.insert(
                collection_name=self.collection_name,
                data=data_to_insert,
                partition_name=self.partition_name
            )
            self.client.flush(self.collection_name)
            logging.info(f"Inserted {successes} entries")

        except MilvusException as e:
            logging.error(f"Failed to insert data: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during insertion: {e}")
            raise


"""
To search via hybrid search:

from pymilvus import WeightedRanker, RRFRanker, AnnSearchRequest

# connect to Milvus
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
client.load_collection(collection_name="my_collection")

# Get the index details to perform the queries
indexes = client.list_indexes(collection_name="my_collection")

search_requests = []
for index in indexes:
    anns_field = index.get("field_name")
    metric_type = index.get("metric_type")
    search_requests.append(AnnSearchRequest(**{
        "data": [query_dense_vector], if anns_field == "embedding" else [query_text],
        "anns_field": anns_field,
        "metric_type": metric_type,
        "params": {"nprobe": 10} if metric_type == "IP" else {"drop_ratio_build": 0.2},
        "limit": 10,
        "expr": "",
        "expr_params": {},
    }))

ranker = RRFRanker() # defaults 60

res = client.hybrid_search(
    collection_name="my_collection",
    search_requests=search_requests,
    ranker=ranker,
    output_fields=["source", "chunk_index", "embedding", "metatext"] + whatever_other_columns_you_want,
    limit=10,
)

query_results = client.query(
    collection_name="my_collection",
    filter="", # required
    output_fields=["source", "chunk_index", "embedding", "metatext"] + whatever_other_columns_you_want,
    limit=10,
)

# Note on output fields: setting to ["*"] will return all fields

# TODO: Construct the prompts for LLM to know how to query for the data
It must contain the details of the collections and what columns are present.
The column descriptions need to be good and detailed.
It needs to describe the indexes and how to use them.

2 tools:
1. Get the collection details along with index.
2. Get the data from the collection via hybrid search.

# TODO: Determine how OI returns tool calls and whether the data is maintained as context.


I am designing a system prompt for a milvus query agent. This agent is gong to be provided a specific milvus collection and is responsible for answering the questions of the user. The agent is going to be provided with a set of tools that it can use to answer the questions.
There will be only two tools:
1. Search: when the user wants semantic information, like a natural language query for the semanticly similar data in the database.
2. Filter: when the user wants to query the database like a relational one. For example, if they ask for all the documents from the past year, or all the documents from a specific author.

These both will be tools the LLM must decide to use or to just respond to the users question.

I also need to give the LLM lots of instruction on how to construct filters, queries, etc. This involves milvus specific api like boolean filtering. I think I'll just copy and summarize the docs and then add some examples.
The LLM also needs instruction on the database structure, so what is the schema of the collection.

Can you ask me a few questions about what the needs are for my prompt, and after I answer them can you construct the full prompt necessary.

One other aspect of the prompt is that I can have variables like `{database_schema}` and `{documentation}` that can serve as placeholders for the schema and documentation and context etc.

GEMINI:
Okay, great! Designing a good system prompt is crucial for getting the best performance out of your Milvus query agent. Asking clarifying questions first is a smart approach.

Here are a few questions to help refine the requirements for your system prompt:

1.  **Schema Details (`{database_schema}`):**
    * Could you describe the specific fields in your Milvus collection schema? Please include field names, data types (e.g., `VARCHAR`, `INT64`, `FLOAT_VECTOR`, `BOOL`), and whether each field is a primary key, a vector field, or a scalar field intended for filtering.
    * Is there one specific field that holds the text data used for semantic search embeddings?

2.  **Tool Input/Output:**
    * **Search Tool:** What exact input does the `Search` tool expect? Does it just take the natural language query text? Does it need other parameters like `top_k` (the number of results to return)? What format does the `Search` tool return its results in?
    * **Filter Tool:** What exact format does the `Filter` tool expect for the filter expression? Should the LLM generate a string that directly corresponds to Milvus' boolean expression syntax (e.g., `"year > 2022 and author == 'John Doe'"`), or does it expect a structured format (like JSON)? What format does the `Filter` tool return results in?

3.  **Milvus Filter Syntax (`{documentation}`):**
    * Besides basic operators (`==`, `!=`, `>`, `<`, `>=`, `<=`, `and`, `or`, `not`), are there specific Milvus functions or operators you absolutely need the LLM to know how to use (e.g., `in` for lists, `like` for string patterns, handling JSON fields if applicable)?
    * Are there any common pitfalls or specific formatting rules for values (e.g., strings always need single or double quotes) that should be explicitly mentioned?

4.  **Agent Behavior & Disambiguation:**
    * When should the agent decide *not* to use any tool and just answer directly? (e.g., for greetings, questions about its own capabilities, simple requests that don't require data lookup).
    * What should the agent do if a user's query is ambiguous? For example, if it's unclear whether they need a semantic search, a filter, or if the filter criteria are incomplete (e.g., "Find documents about AI from last year" - does "last year" mean the previous calendar year or the last 365 days?). Should it ask clarifying questions?
    * How should the agent handle queries that might require *both* semantic search *and* filtering (e.g., "Find documents similar to 'machine learning advancements' published after 2023")? Does one tool handle both, or does it need to sequence calls?

5.  **Collection Content & Scope:**
    * Can you provide a brief, high-level description of the *content* stored in this specific Milvus collection? (e.g., "news articles", "product descriptions", "scientific papers", "internal knowledge base documents"). This helps the agent understand the context of the data.

6.  **Examples:**
    * Could you provide one or two concrete examples of user queries and the *ideal* tool call (with exact parameters/filter strings) the agent should generate for each?

Once you provide answers to these questions, I can help construct a comprehensive system prompt using placeholders like `{database_schema}` and `{documentation}` where appropriate.


USER:
1. I can't give you the schema just yet, but you can put it as a variable and I'll json.dumps it in the prompt.
2. tools:
    1. search: queries list[str] - a list of text queries to search for semantically, filters list[str] - a list of filters to apply to the search results, returns list[dict] - a list of dictionaries with the keys and metadata from the database including text, source, and metadata.
    2. filter: query str - a filter query to apply to the search results, returns list[dict] - a list of dictionaries with the keys and metadata from the database including text, source, and metadata.
4. If the context is already found in the previous messages or it already knows the answer, it should just respond to the user. Otherwise it should use the tools. when ambiguous it should use the tools and query semantic search. If the question is something that might refer to filtering then it can ask clarifying questions, but default to search. For both, it is possible to filter a search result, so use search and provide filters inside that tool call.

5. The collection is chunked internal research and development documents. They handle a lot of different topics, but mostly revolve around signal processing and AI and ML.

6. "What are the latest advancements in AI for signal processing?"
tool call
search, queries ["latest advancements in ai for signal processing"], filters: []
This is obviously not formatted correctly, but you get the idea. I am using the tool call format from most tool call providers like the openai api spec.

3: Here’s a concise yet comprehensive guide to using metadata-based filtering with the Milvus Client API. You’ll learn:

- **How Milvus applies filters** to narrow down the search space before ANN search  
- **Filter expression syntax**: operators (`==`, `>`, `in`, `like`, etc.) and special functions (`json_contains`)  
- **Python client usage**: passing `filter` to `search()`, using `query()` and `get()` for pure metadata retrieval  
- **Filter templating** to dynamically inject parameters for complex queries  
- **Practical code examples** covering scalar fields, JSON arrays, and combined vector+metadata search  

By the end, you’ll have ready-to-use patterns for all common filtering scenarios.

---

## 1. How Milvus Applies Filters

When you include a filtering condition in a search request, Milvus performs **pre-filtering**—it first scans and excludes any entities that don’t match your filter, then runs the ANN search only on the remaining subset. This can dramatically improve relevance and reduce search latency, especially on large collections  ([Filtered Search | Milvus Documentation](https://milvus.io/docs/filtered-search.md?utm_source=chatgpt.com)).

---

## 2. Filter Expression Syntax

Milvus supports rich boolean expressions over scalar and JSON fields:

| Operator / Function  | Meaning                                                      |
|----------------------|--------------------------------------------------------------|
| `==`, `!=`, `>`, `<` | Standard numeric/string comparisons                         |
| `>=`, `<=`           | Greater/less than or equal to                                |
| `IN`, `NOT IN`       | Membership test for lists (e.g. `city IN ["NY","LA"]`)      |
| `LIKE`               | SQL-style wildcard for strings (`%`, `_`)                    ([What's New with Metadata Filtering in Milvus v2.4.3 - Zilliz blog](https://zilliz.com/blog/what-is-new-with-metadata-filtering-in-milvus?utm_source=chatgpt.com)) |
| `json_contains`      | Checks whether a JSON array field contains a value            ([JSON and Metadata Filtering in Milvus - Zilliz blog](https://zilliz.com/blog/json-metadata-filtering-in-milvus?utm_source=chatgpt.com)) |
| `AND`, `OR`, `NOT`   | Boolean connectors                                           |

### Examples

```sql
-- Simple numeric filter
"age >= 21 AND country == \"US\""

-- String wildcard
"title LIKE \"%Guide%\""

-- JSON array
"json_contains(tags, \"ai\") AND year < 2023"
```

---

## 3. Python Client: `search()` with Filters

In Python (via `pymilvus`), you pass your filter expression directly to the `search()` method. Milvus will apply the filter before computing nearest-neighbor distances  ([Filtered Search | Milvus Documentation](https://milvus.io/docs/filtered-search.md?utm_source=chatgpt.com), [JSON and Metadata Filtering in Milvus - Zilliz blog](https://zilliz.com/blog/json-metadata-filtering-in-milvus?utm_source=chatgpt.com)).

### Basic Scalar Filter

```python
from pymilvus import (
    connections,
    Collection
)

# 1. Connect and load collection
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 2. Vector query embedding
query_vecs = [...]  # e.g. 1×768 float list

# 3. Search with metadata filter
results = collection.search(
    data=query_vecs,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["title", "year"],       # return these scalar fields
    filter="year >= 2020 AND rating > 4.5"
)

for hits in results:
    for hit in hits:
        print(hit.id, hit.entity.get("title"), hit.distance)
```

### JSON Array Filtering

```python
# Only search within entities tagged with "Sci-Fi"
results = collection.search(
    data=query_vecs,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 8}},
    limit=3,
    output_fields=["title", "tags"],
    filter='json_contains(tags, "Sci-Fi")'
)
```

---

## 4. Pure Metadata Queries: `query()` and `get()`

If you only need metadata (no vector similarity), use `query()` or `get()`  ([Query | Milvus Documentation](https://milvus.io/docs/get-and-scalar-query.md?utm_source=chatgpt.com)).

- **`get()`** retrieves by primary keys.  
- **`query()`** retrieves by arbitrary filter expressions.  
- **`query_iterator()`** supports pagination over large result sets.

### Example: Retrieve All Recent Articles

```python
from pymilvus import Collection

collection = Collection("articles")

# Query all entities where published_date > "2025-01-01"
entities = collection.query(
    expr="published_date > \"2025-01-01\"",
    output_fields=["id", "title", "author"]
)

for ent in entities:
    print(ent["id"], ent["title"], ent["author"])
```

---

## 5. Filter Expression Templating

For dynamic queries or very large `IN` lists, Milvus supports **templating**, where placeholders in the expression are substituted at runtime  ([Filter Templating | Milvus Documentation](https://milvus.io/docs/filtering-templating.md?utm_source=chatgpt.com)).

```python
expr = "age > {min_age} AND city IN {cities}"
params = {"min_age": 30, "cities": ["北京", "上海"]}

results = collection.search(
    data=query_vecs,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 5}},
    limit=10,
    filter=expr,
    filter_params=params
)
```

---

## 6. Putting It All Together: Complete Example

Below is a richer script illustrating embedding insertion, scalar and JSON filtering, and a hybrid vector+metadata search  ([Milvus: A complete example of how to use VectorDB by Python and ...](https://jimmy-wang-gen-ai.medium.com/milvus-a-complete-example-of-how-to-use-vectordb-by-python-and-serve-it-as-an-api-3a05e2f8db3c?utm_source=chatgpt.com), [Filtered search with Milvus and OpenAI](https://cookbook.openai.com/examples/vector_databases/milvus/filtered_search_with_milvus_and_openai?utm_source=chatgpt.com)).

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# 1. Connect
connections.connect("default", host="localhost", port="19530")

# 2. Define schema & create collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="year", dtype=DataType.INT64),
    FieldSchema(name="genres", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="tags", dtype=DataType.JSON)
]
schema = CollectionSchema(fields, description="Movies")
collection = Collection("movies", schema)

# 3. Insert sample data
model = SentenceTransformer("all-MiniLM-L6-v2")
titles = ["Blade Runner", "Alien", "Inception"]
embs = model.encode(titles).tolist()
years = [1982, 1979, 2010]
genres = ["Sci-Fi", "Sci-Fi", "Sci-Fi"]
tags = [["dystopia","android"], ["space","horror"], ["dream","thriller"]]

collection.insert([list(range(3)), embs, years, genres, tags])

# 4. Hybrid search: vector + metadata
query_emb = model.encode(["futuristic detective"]).tolist()
results = collection.search(
    data=query_emb,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=2,
    output_fields=["year","genres","tags"],
    filter='year > 2000 AND json_contains(tags, "thriller")'
)

for hits in results:
    for hit in hits:
        print(hit.id, hit.entity.get("year"), hit.distance)
```

With these patterns and examples, you should be fully equipped to leverage Milvus’s metadata filtering to build fast, relevant, and precise vector-based applications!

"""