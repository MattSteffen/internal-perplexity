# """
# title: myToolName
# author: myName
# funding_url: [any link here will be shown behind a `Heart` button for users to show their support to you]
# version: 1.0.0
# # the version is displayed in the UI to help users keep track of updates.
# license: GPLv3
# description: [recommended]
# requirements: package1>=2.7.0,package2,package3
# """
from pydantic import BaseModel, Field
import os
import requests
import time  # Added for time.sleep
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any  # Added for type hinting
import pymilvus
import json

# --- Standalone Milvus and Embedding Functions ---


def connect_milvus(
    host: str, port: str, collection_name: str
) -> Optional[pymilvus.Collection]:
    """
    Connects to Milvus and returns the specified collection object.

    Args:
        host: Milvus server host address.
        port: Milvus server port.
        collection_name: Name of the collection to connect to.

    Returns:
        A loaded Milvus Collection object or None if connection/loading fails.
    """
    print("Starting connect")
    alias = "rag_connection"  # Unique alias for this connection
    try:

        if not pymilvus.connections.has_connection(alias):
            print(f"Connecting to Milvus at {host}:{port}...")
            pymilvus.connections.connect(alias=alias, host=host, port=port)
        else:
            print(f"Already connected to Milvus using alias '{alias}'")
        print("Has connection")
        if not pymilvus.utility.has_collection(collection_name, using=alias):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return None
        print("Has collection")

        print(f"Accessing Milvus collection: {collection_name}")
        collection = pymilvus.Collection(collection_name, using=alias)
        collection.load()  # Ensure the collection data is loaded into memory
        print(f"Collection '{collection_name}' loaded.")
        return collection
    except Exception as e:
        print(
            f"Error connecting to or loading Milvus collection '{collection_name}': {e}"
        )
        # Clean up connection if it exists but failed later
        if pymilvus.connections.has_connection(alias):
            pymilvus.connections.disconnect(alias)
        return None


def get_embedding(ollama_url: str, text: str, model: str) -> Optional[List[float]]:
    """
    Gets embedding from Ollama for the given text.

    Args:
        ollama_url: Base URL of the Ollama API.
        text: The text to embed.
        model: The embedding model name to use in Ollama.

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,  # Add a timeout
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        return result.get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama ({ollama_url}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during embedding: {e}")
        return None


def search_milvus(
    queries: List[str],  # Changed from single query to list
    collection: pymilvus.Collection,
    ollama_url: str,
    embedding_model: str,
    limit: int = 5,
    filters: Optional[List[str]] = None,  # Added filters parameter
) -> List[Dict[str, Any]]:
    """
    Searches a Milvus collection using one or more query strings and optional filters.

    Args:
        queries: The list of user's query strings (original + expanded).
        milvus_collection: Milvus collection.
        ollama_url: Base URL for the Ollama API (for embedding).
        embedding_model: Name of the Ollama embedding model.
        limit: Maximum number of search results to return *in total*.
        filters: A list of Milvus filter expression strings (e.g., ['author == "John Doe"', 'chunk_index > 0']).

    Returns:
        A list of unique dictionaries, each representing a search result document, ranked by score.
    """
    if collection is None:
        print("Failed to connect to Milvus collection for search.")
        return []

    # Filter out empty queries
    valid_queries = [q for q in queries if q and q.strip()]
    if not valid_queries:
        print("No valid queries provided for Milvus search.")
        return []

    # Get embeddings for all valid queries
    query_embeddings = []
    for query in valid_queries:
        embedding = get_embedding(ollama_url, query, embedding_model)
        if embedding:
            query_embeddings.append(embedding)
        else:
            print(f"Failed to get embedding for query: '{query}', skipping.")

    if not query_embeddings:
        print("Failed to get embeddings for any provided query.")
        return []

    # Construct the filter expression (if any)
    filter_expr = None
    if filters:
        # Ensure filters are valid strings before joining
        valid_filters = [f for f in filters if isinstance(f, str) and f.strip()]
        if valid_filters:
            filter_expr = " and ".join(valid_filters)
            print(f"Using Milvus filter expression: {filter_expr}")
        else:
            print("Provided filters list was empty or contained invalid entries.")

    try:
        # print(
        #     f"Searching collection '{milvus_collection_name}' with {len(query_embeddings)} query vectors."
        # )
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16},
        }  # Adjust nprobe as needed

        collection_details = collection.describe()
        fields = [f.get("name") for f in collection_details.get("fields", [])]

        # Perform search
        results = collection.search(
            data=query_embeddings,  # List of query vectors
            anns_field="embedding",
            param=search_params,
            limit=limit,  # Retrieve potentially more initially to allow for deduplication
            expr=filter_expr,  # Apply the combined filter expression
            output_fields=fields,
        )

        # Process and deduplicate results
        all_hits = []
        # Results is a list of lists (Hit objects), one list per query embedding
        for hit_list in results:
            for hit in hit_list:
                entity = hit.entity
                # Skip results with missing essential fields if necessary
                if entity.get("text") is None:
                    continue
                hit_details = {
                    "id": entity.get("id"),
                    "content": entity.get("text"),
                    "score": hit.score,
                }
                for f in fields:
                    if entity.get(f) is not None:
                        hit_details[f] = entity.get(f)
                all_hits.append(hit_details)

        # Deduplicate based on 'id' (or potentially content hash if IDs are not unique chunks)
        unique_documents_dict = {}
        for doc in all_hits:
            doc_id = doc["id"]
            # Keep the document with the best score (lowest L2) if duplicates found
            if (
                doc_id not in unique_documents_dict
                or doc["score"] < unique_documents_dict[doc_id]["score"]
            ):
                unique_documents_dict[doc_id] = doc

        # Sort the unique documents by score (ascending for L2)
        sorted_documents = sorted(
            unique_documents_dict.values(), key=lambda x: x["score"]
        )

        # Return the top 'limit' unique documents
        final_documents = sorted_documents[:limit]

        print(
            f"Found {len(final_documents)} unique relevant documents in Milvus after processing."
        )
        # Optional: Disconnect if the connection is specific and not reused immediately
        # pymilvus.connections.disconnect(f"rag_connection_{milvus_collection_name}")
        return final_documents

    except pymilvus.exceptions.MilvusException as e:
        print(f"Error searching Milvus: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error during Milvus search: {e}")
        return []


class Tools:
    class Valves(BaseModel):
        # Updated valve for Lambda API Key
        MILVUS_HOST: str = Field(default="host.docker.internal")
        MILVUS_PORT: str = Field(default="19530")
        COLLECTION_NAME: str = Field(default="arxiv")
        OLLAMA_URL: str = Field(default="http://host.docker.internal:11434")
        OLLAMA_EMBEDDING_MODEL: str = Field(default="all-minilm:v2")
        QUERY_EXPANSION: bool = Field(default=False)

    def __init__(self):
        self.citation = False
        self.valves = self.Valves(**{"QUERY_EXPANSION": False})

    # Add your custom tools using pure Python code here, make sure to add type hints
    # Use Sphinx-style docstrings to document your tools, they will be used for generating tools specifications

    async def search_milvus(self, queries: list[str], filters: list[str], __event_emitter__=None) -> str:
        """
        Searches a Milvus vector database using semantic queries.
        This tool retrieves relevant document chunks based on the provided queries.

        :param queries: A list of search query strings. Should include the original user query plus any generated expansions for better coverage.
        :param filters: A list of filters to apply to the search.
        :return: A string summarizing the search results (e.g., number of documents found) or an error message.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Connecting to milvus",
                        "done": False,
                    },
                }
            )

        collection = connect_milvus(
            self.valves.MILVUS_HOST,
            self.valves.MILVUS_HOST,
            self.valves.COLLECTION_NAME,
        )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Connected: {collection != None}",
                        "done": False,
                    },
                }
            )

        res = search_milvus(
            queries,
            collection,
            self.valves.OLLAMA_URL,
            self.valves.OLLAMA_EMBEDDING_MODEL,
            limit=5,
            filters=None,
        )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Got {len(res)} results",
                        "done": True,
                    },
                }
            )

            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "document": ["all the text of the document used"],
                        "metadata": [
                            {
                                "source": "THE CITATION",
                                "date_accessed": "yesterday",
                                "title": "Hoopla",
                            }
                        ],
                        "source": {
                            "name": "Yahoo",
                            "url": "http://localhost:5000",
                            "type": "webpage",
                        },
                    },
                }
            )

        return f"Milvus says: {res}"

    async def expanded_search_milvus(
        self, queries: list[str], filters: list[str], __event_emitter__=None
    ) -> str:
        """
        Searches a Milvus vector database using semantic queries and optional metadata filters.
        This tool retrieves relevant document chunks based on the provided queries and filters.

        :param queries: A list of search query strings. Should include the original user query plus any generated expansions for better coverage.
        :param filters: A list of valid milvus filters that will be combined with ' and ' to offer more fine-grained results.
        :return: A string of all the document data including metadata or an error message.
        """
        # Call ollama to expand the queries provided more context about the user and the collecctions
        new_queries = queries
        new_filters = filters
        return await self.search_milvus(new_queries, new_filters, __event_emitter__)
