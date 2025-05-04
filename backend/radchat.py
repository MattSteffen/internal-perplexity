"""
title: Radchat Function
author: Rincon [mjst, kca]
author_url:
funding_url:
version: 0.1
requirements: pymilvus
"""

import os
import re
import requests
import json, yaml
from typing import Optional, Dict, Any, List, Union, Generator, Iterator, Callable
from pydantic import BaseModel, Field
import collections
from pymilvus import (
    MilvusClient,
)

# OLLAMA_BASE_URL = "http://ollama.a1.autobahn.rinconres.com"
# OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
# COLLECTION_NAME = "dev"
# OUTPUT_FIELDS = ["source", "chunk_index", "text", "metatext"]

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "all-minilm:v2"
OLLAMA_LLM_MODEL = "qwen3:latest"
COLLECTION_NAME = "arxiv"
OUTPUT_FIELDS = ["source", "chunk_index", "metatext", "title", "author", "date", "unique_words"]
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
DOC_LIMIT = 10
MAX_TOOL_CALLS = 3

SearchInputSchema = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Performs a semantic search using the given queries and optional filters.",
        "parameters": {
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of queries for semantic search",
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter expressions to apply to the search",
                    "default": [],
                },
                "text_search": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of text snippets to search for, when a user submits a query with a confusing word, it is likely domain specific, use the text search to find the word in the document",
                    "default": [],
                },
            },
        },
    },
}

QueryInputSchema = {
    "type": "function",
    "function": {
        "name": "query",
        "description": "Runs a filtered query without semantic search. Only filters are used.",
        "parameters": {
            "type": "object",
            "required": ["filters"],
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter expressions to apply to the query",
                    "default": [],
                }
            },
        },
    },
}

class EventEmitter:
    def __init__(self, emitter: Optional[Callable] = None):
        self.emitter = emitter

    async def status(self, description: str, done: bool = False) -> None:
        if self.emitter:
            await self.emitter({
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            })

    async def citation(self, document: List[str], metadata: List[Dict[str, str]], source: Dict[str, str]) -> None:
        if self.emitter:
            await self.emitter({
                "type": "citation",
                "data": {
                    "document": document,
                    "metadata": metadata,
                    "source": source,
                },
            })

    async def error(self, description: str, done: bool = True) -> None:
        if self.emitter:
            await self.emitter({
                "type": "error",
                "data": {
                    "description": description,
                    "done": done,
                },
            })

def connect_milvus(token: str = "") -> Optional[MilvusClient]:
    """
    Connects to Milvus using MilvusClient and returns the client if the
    specified collection exists and is loaded, otherwise returns None.
    """
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    try:
        # 1. Instantiate a MilvusClient
        client = MilvusClient(uri=uri, token=token)  # TODO: Get token
        # 2. Verify the collection exists
        if not client.has_collection(collection_name=COLLECTION_NAME):
            print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
            return None  #
        # 3. Load the collection into memory
        client.load_collection(collection_name=COLLECTION_NAME)  #
        print(f"Collection '{COLLECTION_NAME}' loaded.")
        return client
    except Exception as e:
        print(
            f"Error connecting to or loading Milvus collection '{COLLECTION_NAME}': {e}"
        )
        return None


# def perform_search(query_input: SearchInput, client: MilvusClient) -> List:
def perform_search(
    client: MilvusClient,
    queries: list[str],
    text_search: str = "",
    filters: list[str] = [],
) -> List:
    print("STARTING SEARCH")
    # Get the index details to perform the queries
    # print(client.describe_collection(collection_name=COLLECTION_NAME))
    indexes = client.list_indexes(collection_name=COLLECTION_NAME)
    # print(indexes)

    search_results = []
    for index in indexes:
        print("index", index)
        index_details = client.describe_index(
            collection_name=COLLECTION_NAME, index_name=index
        )
        if index_details.get("field_name") not in ["embedding", "sparse", "full_text_embedding"]:
            print("Skipping index", index_details.get("field_name"), index_details)
            continue
        anns_field = index_details.get("field_name")
        metric_type = index_details.get("metric_type")
        match index_details.get("field_name"):
            case "sparse", "full_text_embedding":
                search_results.extend(
                    client.search(
                        collection_name=COLLECTION_NAME,
                        data=[text_search],
                        anns_field=anns_field,
                        filter=" and ".join(filters),
                        limit=DOC_LIMIT,
                        output_fields=OUTPUT_FIELDS,
                        search_params={"drop_ratio_build": 0.2},
                    )
                )
            case "embedding":
                search_results.extend(
                    client.search(
                        collection_name=COLLECTION_NAME,
                        data=[
                            get_embedding(t)[0]
                            for t in queries
                        ],
                        anns_field=anns_field,
                        filter=" and ".join(filters),
                        limit=DOC_LIMIT,
                        output_fields=OUTPUT_FIELDS,
                        search_params={"nprobe": 10},
                    )
                )

    # Combine results by doc id, format them correctly, sort by highest score.
    print("Completing search")
    return process_docs(search_results)


# def perform_query(query_input: QueryInput, client: MilvusClient) -> List:
def perform_query(filters: list[str], client: MilvusClient) -> List:
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter=filters,  # required
        output_fields=OUTPUT_FIELDS,
        limit=10,
    )

    docs = [
        {k: hit.get("entity", {}).get(k) for k in OUTPUT_FIELDS}
        for hit in query_results[0]
    ]
    return docs


def process_docs(search_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Processes a list of Milvus search results to eliminate duplicates by ID
    (keeping the one with the lowest distance), group results by source,
    sort chunks within each source by chunk_index, and finally sort the
    groups by the minimum distance found within that group.

    Args:
        search_results: A list of dictionaries, where each dictionary
                        represents a search hit and has the structure:
                        {"id": int, "distance": float, "entity": dict}
                        The "entity" dictionary contains fields like
                        "chunk_index", "source", "content", etc.

    Returns:
        A list of processed dictionaries, sorted first by the best distance
        of their source group, and then by chunk_index within each group.
        Duplicates based on 'id' are removed, keeping the entry with the
        lowest 'distance'.
    """
    print("Processing docs", end=" -- ")
    if not search_results:
        return []
    else:
        search_results = search_results[0]

    # 1. Eliminate duplicates by ID, keeping the one with the lowest distance
    unique_docs_by_id: Dict[int, Dict[str, Any]] = {}
    for doc in search_results:
        doc_id = doc.get('id')
        # Ensure doc_id exists and is valid before processing
        if doc_id is None:
            # Optionally log a warning or handle missing IDs
            continue

        if doc_id not in unique_docs_by_id or doc['distance'] < unique_docs_by_id[doc_id]['distance']:
            unique_docs_by_id[doc_id] = doc

    # Convert back to a list of unique documents
    unique_docs_list = list(unique_docs_by_id.values())

    # 2. Group unique documents by source
    grouped_by_source = collections.defaultdict(list)
    # Keep track of the minimum distance found for each source group
    source_min_distance: Dict[str, float] = {}

    for doc in unique_docs_list:
        # Ensure 'entity' and 'source' keys exist
        entity = doc.get('entity', {})
        source = entity.get('source')
        if source is None:
            # Optionally handle documents missing a source
            # For example, assign a default source or skip
            source = "unknown_source" # Example handling

        grouped_by_source[source].append(doc)

        # Update the minimum distance for this source
        current_min = source_min_distance.get(source, float('inf'))
        source_min_distance[source] = min(current_min, doc['distance'])

    # 3. Prepare groups for sorting: sort chunks within each group by chunk_index
    #    and store the group's minimum distance.
    processed_groups = []
    for source, docs in grouped_by_source.items():
        # Sort documents (chunks) within the group by chunk_index
        # Handle potential missing 'chunk_index' by defaulting to a large number or 0
        sorted_chunks = sorted(
            docs,
            key=lambda d: d.get('entity', {}).get('chunk_index', float('inf'))
        )
        # Get the pre-calculated minimum distance for this source group
        min_dist = source_min_distance[source]
        processed_groups.append({'min_distance': min_dist, 'source': source, 'chunks': sorted_chunks})

    # 4. Sort the groups based on their minimum distance (ascending)
    sorted_groups = sorted(processed_groups, key=lambda g: g['min_distance'])

    # 5. Flatten the sorted groups back into a single list
    final_result = []
    for group in sorted_groups:
        final_result.extend(group['chunks'])

    print("Done")
    return [document_to_markdown(document.get("entity")) for document in final_result]

def document_to_markdown(document: Dict[str, Any]) -> str:
    """
    Converts a document to a Markdown string.
    Using the following keys: title, source, url, date, unique_words, tags, author/authors, keywords, summary, and finally content/text
    """

    title = document.get("title", "")
    source = document.get("source", "") or document.get("url", "")
    date = document.get("date", "")
    authors = document.get("author", "") or document.get("authors", "")
    keywords = document.get("keywords", "") or document.get("tags", "") or document.get("unique_words", "")
    summary = document.get("summary", "")
    content = document.get("text", "")
    chunk_index = document.get("chunk_index", "")

    # Construct the Markdown string
    markdown_string = ""
    if title: markdown_string += f"# {title}\n\n"
    if source: markdown_string += f"Source: {source}\n"
    if date: markdown_string += f"Date: {date}\n"
    if authors: markdown_string += f"Authors: {authors}\n"
    if keywords: markdown_string += f"Keywords: {keywords}\n"
    if summary: markdown_string += f"Summary: {summary}\n"
    if title and chunk_index: markdown_string += f"For inline citations, use the following format:`<{title}.{chunk_index}>`\n\n"
    if content: markdown_string += f"Content:\n{content}\n"

    return markdown_string



def get_embedding(text: str) -> Optional[List[float]]:
    """
    Gets embedding from Ollama for the given text.

    Args:
        OLLAMA_BASE_URL: Base URL of the Ollama API.
        text: The text to embed.
        model: The embedding model name to use in Ollama.

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
            timeout=30,  # Add a timeout
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        return result.get("embeddings")
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama ({OLLAMA_BASE_URL}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during embedding: {e}")
        return None

def call_ollama_api(payload: dict, stream: bool = False, timeout: int = 90) -> dict:
    """
    Makes a request to the Ollama API and handles errors.
    
    Args:
        url: The Ollama API endpoint URL.
        payload: The JSON payload to send.
        stream: Whether to stream the response.
        timeout: Request timeout in seconds.
        
    Returns:
        The JSON response data or raises an exception on error.
    """
    print("Connecting to Ollama API", end=" -- ")
    payload["options"] = {"num_ctx": 32768}
    try:
        response = requests.post(OLLAMA_BASE_URL+"/api/chat", json=payload, stream=stream, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        # print("result", result)
        
        # Parse output to remove content in <think></think> tags
        if msg := result.get("message", {}):
            print("Found message")
            # TODO: Improve parsing for other models, this checks qwen3 reasoning for tools.
            if content := msg.get("content"):
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                
                # Check for tool calls in <tool_call></tool_call> tags
                tool_call_matches = re.findall(r'<tool_call>.*?</tool_call>', content, re.DOTALL)
                if tool_call_matches:
                    msg["tool_calls"] = msg.get("tool_calls", [])
                    for tool_call_json in tool_call_matches:
                        try:
                            tool_call_json = tool_call_json.replace("<tool_call>", "").replace("</tool_call>", "").strip()
                            tool_call_data = json.loads(tool_call_json)
                            msg["tool_calls"].append(tool_call_data)
                        except json.JSONDecodeError:
                            print(f"Invalid JSON in tool call: {tool_call_json}")
                    
                # Remove the tool call tags from content
                msg["content"] = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
                print("Final message: ", msg)
                result["message"] = msg
        print("Call Completed")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API ({OLLAMA_BASE_URL}): {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        raise Exception(f"Failed to communicate with Ollama: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        raise Exception("Received invalid response from Ollama")
    except Exception as e:
        print(f"Unexpected error during Ollama API call: {e}")
        raise Exception(f"Unexpected error: {str(e)}")



class Pipe:
    class Valves(BaseModel):
        MILVUS_TOKEN: str = Field("root:Milvus")

    def __init__(self):
        self.type = "pipe"
        self.valves = self.Valves(
            MILVUS_TOKEN=os.getenv("MILVUS_TOKEN", "root:Milvus")
        )

    def pipes(self) -> List[dict]:
        return [{"id": "radchat", "name": "RadChat"}]

    async def pipe(
        self, body: dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        emitter = EventEmitter(__event_emitter__)
        await emitter.status("Beginning Search")

        messages = body["messages"]
        print("Starting pipe, messages:", messages)

        # 1. LIST and DESCRIBE collections from Milvus
        await emitter.status("Connecting to Milvus")
        milvus_client = connect_milvus(self.valves.MILVUS_TOKEN)
        if milvus_client is None:
            await emitter.error("Failed to connect to Milvus database")
            return "Error: Unable to connect to the knowledge database. Please try again later or contact support."
            
        collection_schema_description = milvus_client.describe_collection(
            COLLECTION_NAME
        ).get("fields", {})

        # 3. CREATE system prompt for structure
        await emitter.status("Structuring Queries")
        system_prompt = SystemPrompt.replace(
            "<<database_schema>>", str(collection_schema_description)
        )
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        available_tools = {"search": perform_search, "query": perform_query}

        for tool_call_index in range(MAX_TOOL_CALLS):
            print("Tool call index:", tool_call_index)
            payload = {
                "model": OLLAMA_LLM_MODEL,
                "messages": all_messages,
                "tools": [SearchInputSchema, QueryInputSchema],
                "stream": False,
            }

            try:
                response_data = call_ollama_api(payload)
                
                tool_calls = response_data.get("message", {}).get("tool_calls", [])
                if not tool_calls:
                    break

                await emitter.status("Calling tools")
                    
                for tool in tool_calls:
                    print("Tool call:", tool)
                    if func := available_tools.get(tool.get('function', {}).get('name', '')):
                        output = func(client=milvus_client, **tool.get('function', {}).get('arguments', {}))
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": "\n\n".join(output),
                                "name": tool.get('function', {}).get('name', ''),
                            }
                        )
                    print("done with tool call")
                print("done with all tool calls")
                
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                print(error_message)
                await emitter.error(error_message)
                return f"An error occurred while processing your request: {str(e)}"

        await emitter.status("Finalizing results...", done=True)
        final_content = response_data.get("message", {}).get("content", "")

        return final_content
        # if __event_emitter__:
        #     for r in docs:
        #         await __event_emitter__(
        #             {
        #                 "type": "citation",
        #                 "data": {
        #                     "document": [r.get("text")],
        #                     "metadata": [
        #                         {
        #                             "source": r.get("author", "unknown source"),
        #                             "date_accessed": "yesterday",
        #                             "title": r.get("title", "unknown title"),
        #                         }
        #                     ],
        #                     "source": {"name": r.get("title", "unknown name")},
        #                 },
        #             }
        #         )

    #     payload = {"stream": body.get("stream", False)}
    #     if body.get("stream", False):
    #         return self.stream_response(
    #             f"{self.valves.OLLAMA_BASE_URL}/api/chat",
    #             {
    #                 **payload,
    #                 "model": "gemma3:27b",
    #                 "messages": chat_messages,
    #             },
    #         )
    #     return self.non_stream_response(
    #         f"{self.valves.OLLAMA_BASE_URL}/api/chat",
    #         {
    #             **payload,
    #             "model": "gemma3:27b",
    #             "messages": chat_messages,
    #         },
    #     )

    # def stream_response(self, url, payload):
    #     try:
    #         with requests.post(
    #             url, json=payload, stream=True, timeout=(3.05, 60)
    #         ) as response:
    #             if response.status_code != 200:
    #                 raise Exception(
    #                     f"HTTP Error {response.status_code}: {response.text}"
    #                 )

    #             for line in response.iter_lines():
    #                 if line:
    #                     line = line.decode("utf-8")
    #                     try:
    #                         data = json.loads(line)
    #                         if "message" in data and "content" in data["message"]:
    #                             content = data["message"]["content"]
    #                             if content:
    #                                 yield content
    #                         elif "done" in data and data["done"]:
    #                             break
    #                     except json.JSONDecodeError:
    #                         print(f"Failed to parse JSON: {line}")
    #     except Exception as e:
    #         print(f"Error in stream_response: {e}")
    #         yield f"Error: {e}"

    # def non_stream_response(self, url, payload):
    #     try:
    #         response = requests.post(url, json=payload, timeout=(3.05, 60))
    #         if response.status_code != 200:
    #             raise Exception(f"HTTP Error {response.status_code}: {response.text}")

    #         res = response.json()
    #         return res.get("message", {}).get("content", "")
    #     except Exception as e:
    #         print(f"Error in non_stream_response: {e}")
    #         return f"Error: {e}"


SystemPrompt = """
You are a specialized AI assistant responsible for answering questions about a specific Milvus collection. Your primary goal is to help users retrieve relevant information from this collection using the tools provided.

**1. Milvus Collection Context:**

- **Content:** The collection contains chunked documents from internal research and development efforts. The main topics cover signal processing, Artificial Intelligence (AI), and Machine Learning (ML).
- **Schema:** You MUST use the following schema to understand the available fields, their data types, and which fields can be used for filtering. Pay close attention to field names and types when constructing filter expressions.
  ```json
  <<database_schema>>
  ```

**2. Available Tools:**

You have the following tools to interact with the Milvus collection:

- **`search`**: Use this tool for semantic search, optionally combined with metadata filtering.

  - **Parameters:**
    - `queries` (list[str]): A list of natural language strings to search for based on semantic similarity. Usually, you'll provide one query reflecting the user's intent.
    - `filters` (list[str]): A list of filter expressions (strings) to apply _before_ the semantic search. Use this to narrow down results based on metadata criteria. Each string in the list must follow the Milvus boolean expression syntax. If no filtering is needed, provide an empty list `[]`.
  - **Returns:** `list[dict]` - A list of matching documents, including their metadata (e.g., text snippets, source, scalar fields).

- **`filter`**: Use this tool _only_ when the user wants to retrieve documents based _solely_ on metadata criteria, without any semantic search component.
  - **Parameters:**
    - `query` (str): A single filter expression string that specifies the metadata conditions for retrieving documents. This string must follow the Milvus boolean expression syntax.
  - **Returns:** `list[dict]` - A list of documents matching the filter criteria, including their metadata.

**3. How to Answer User Queries:**

- **Check History/Knowledge First:** Before using any tool, check if the user's question can be answered based on the current conversation history. If so, answer directly without using a tool. Examples: greetings, simple clarifications, questions about your capabilities.
- **Semantic Queries:** If the user asks a question seeking information based on meaning, concepts, or topics (e.g., "What documents discuss X?", "Find information similar to Y"), use the `search` tool. Provide the core semantic query in the `queries` parameter.
- **Metadata-Only Queries:** If the user asks a question that only involves filtering by specific metadata fields (e.g., "List all documents by author Jane Doe", "Show me items where year > 2023", "Get documents with ID 12345"), use the `filter` tool. Provide the complete filter expression in the `query` parameter.
- **Hybrid Queries (Semantic + Filter):** If the user's query combines semantic meaning with metadata filters (e.g., "Find documents _about_ neural networks _published after_ 2022", "Search for 'signal processing techniques' _where the author is_ John Smith"), you MUST use the `search` tool. Put the semantic part in the `queries` parameter and the filtering conditions in the `filters` parameter (as a list containing one or more filter strings).
- **Ambiguity:**
  - If a query is ambiguous but seems to have a semantic component, default to using the `search` tool. Include potential filters in the `filters` parameter if they can be inferred, otherwise use an empty list `[]`.
  - If a query seems purely filter-based but the criteria are unclear (e.g., "Find recent documents"), you may ask one clarifying question. However, if the user doesn't clarify or the intent remains unclear, default to using the `search` tool with the ambiguous term as the query (e.g., `queries=["recent documents"]`, `filters=[]`).

**4. Constructing Filter Expressions:**

- **Syntax Reference:** You MUST strictly adhere to the Milvus boolean expression syntax when creating filter strings for the `filters` parameter of the `search` tool or the `query` parameter of the `filter` tool. The detailed syntax rules, available operators (`==`, `>`, `!=`, `IN`, `LIKE`, `AND`, `OR`, `NOT`), handling of strings (use double quotes `"`), numbers, and special functions like `json_contains` are provided in the documentation below.
- **Use the Schema:** Always refer to the `database_schema` provided above to ensure you are using correct field names and comparing values of the appropriate type.
- **Documentation:**

  ```markdown
  Milvus supports rich boolean expressions over scalar and JSON fields:
  | Operator / Function | Meaning |
  |----------------------|--------------------------------------------------------------|
  | `==`, `!=`, `>`, `<` | Standard numeric/string comparisons |
  | `>=`, `<=` | Greater/less than or equal to |
  | `IN`, `NOT IN` | Membership test for lists (e.g. `city IN ["NY","LA"]`) |
  | `LIKE` | SQL-style wildcard for strings (`%`, `_`) |
  | `json_contains` | Checks whether a JSON array field contains a value |
  | `AND`, `OR`, `NOT` | Boolean connectors |
  Examples
  -- Simple numeric filter
  "age >= 21 AND country == \"US\""
  -- String wildcard
  "title LIKE \"%Guide%\""
  -- JSON array
  "json_contains(tags, \"ai\") AND year < 2023"
  ```

**5. Responding to the User:**

- After receiving results from a tool (which will be a `list[dict]`), do not just output the raw data.
- Synthesize the information from the returned dictionaries into a clear, concise, and helpful natural language response.
- Highlight the key findings or relevant snippets from the retrieved documents based on the user's query. Mention relevant metadata if appropriate (like source, author, date).
- If a tool returns no results, inform the user clearly that no matching documents were found based on their criteria.

**6. Examples:**

- **User Query:** "What are the latest advancements in AI for signal processing?"
  - **Tool Call:**
    ```json
    {
      "tool_calls": [{
        "name": "search",
        "arguments": "{
          \"queries\": [\"latest advancements in AI for signal processing\"],
          \"filters\": [],
          \"text_search\": []
        }"
      }]
    }
    ```
- **User Query:** "Find documents that discuss "neural networks in legal research" or "AI for case law analysis", but only those tagged with domain:legaltech and from the year 2023."
  - **Tool Call:** (Assuming 'doc_type' and 'publish_year' are valid fields in the schema)
    ```json
    {
      "tool_calls": [{
        "function": {
            "name": "search",
            "arguments": {
            "queries": [
                "neural networks in legal research",
                "AI for case law analysis"
            ],
            "filters": [
                ["domain == \"legaltech\" AND year == 2023"]
            ],
            "text_search": []
            }
        }
      }]
    }
    ```
- **User Query:** "Show me all documents written by 'Dr. Evelyn Reed'."
  - **Tool Call:** (Assuming 'author' is a valid field in the schema)
    ```json
    {
      "tool_calls": [{
        "function": {
            "name": "query",
            "arguments": {
              "filters": ["author == \"Dr. Evelyn Reed\""]
            }
        }
      }]
    }
    ```
- **User Query:** "Find documents containing the exact phrase 'quantum computing applications' in healthcare research."
  - **Tool Call:**
    ```json
    {
      "tool_calls": [{
        "function": {
            "name": "search",
            "arguments": {
              "queries": ["healthcare research"],
              "filters": [],
              "text_search": ["quantum computing applications"]
            }
        }
      }]
    }
    ```
    Follow these instructions carefully to effectively query the Milvus database and provide accurate, relevant answers to the user.
"""