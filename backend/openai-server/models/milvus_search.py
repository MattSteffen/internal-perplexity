
"""
title: Radchat Function
author: Rincon [mjst, kca]
author_url:
funding_url:
version: 0.1
requirements: pymilvus
"""

import os
import uuid
import time
import logging
import json
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict

import ollama
from pydantic import BaseModel, Field, ConfigDict
from pymilvus import (
    MilvusClient,
    AnnSearchRequest,
    RRFRanker,
)


# --- Configuration ---
# Using environment variables for configuration improves security and flexibility.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:v2")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:1.7b")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_arxiv2")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# --- Constants ---
OUTPUT_FIELDS = ["source", "chunk_index", "metadata", "title", "author", "date", "keywords", "unique_words", "text"]
MAX_TOOL_CALLS = 5
REQUEST_TIMEOUT = 300
# Milvus search parameters
NPROBE = 10
SEARCH_LIMIT = 10
HYBRID_SEARCH_LIMIT = 100
RRF_K = 100
DROP_RATIO = 0.2

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Models for Type Safety ---
class MilvusDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    chunk_index: int
    metadata: Optional[str] = ""
    title: Optional[str] = ""
    author: Optional[Union[List[str], str]] = Field(default_factory=list)
    date: Optional[int] = 0
    keywords: Optional[List[str]] = Field(default_factory=list)
    unique_words: Optional[List[str]] = Field(default_factory=list)
    text: Optional[str] = ""
    distance: Optional[float] = None

# --- Tool Schemas ---
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

# --- Core Functions ---
def connect_milvus(token: str = "") -> Optional[MilvusClient]:
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    try:
        client = MilvusClient(uri=uri, token=token)
        if not client.has_collection(collection_name=COLLECTION_NAME):
            logging.error(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
            return None
        client.load_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' loaded.")
        return client
    except Exception as e:
        logging.error(f"Error connecting to or loading Milvus collection '{COLLECTION_NAME}': {e}")
        return None

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        logging.error(f"Error getting embedding from Ollama ({OLLAMA_BASE_URL}): {e}")
        return None

def perform_search(client: MilvusClient, queries: list[str], filters: list[str] = []) -> list[MilvusDocument]:
    search_requests = []
    filter_expr = " and ".join(filters) if filters else ""

    search_configs = [
        {"field": "text_sparse_embedding", "param": {"drop_ratio_search": DROP_RATIO}, "data_transform": lambda q: [q]},
        {"field": "metadata_sparse_embedding", "param": {"drop_ratio_search": DROP_RATIO}, "data_transform": lambda q: [q]},
    ]

    for query in queries:
        embedding = get_embedding(query)
        if embedding:
            search_requests.append(AnnSearchRequest(
                data=[embedding],
                anns_field="text_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": NPROBE}},
                expr=filter_expr,
                limit=SEARCH_LIMIT
            ))
        
        for config in search_configs:
            search_requests.append(AnnSearchRequest(
                data=config["data_transform"](query),
                anns_field=config["field"],
                param=config["param"],
                expr=filter_expr,
                limit=SEARCH_LIMIT
            ))

    if not search_requests:
        return []

    result = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=search_requests,
        ranker=RRFRanker(k=RRF_K),
        output_fields=OUTPUT_FIELDS,
        limit=HYBRID_SEARCH_LIMIT,
    )

    return [MilvusDocument(**doc['entity'], distance=doc['distance']) for doc in result[0]]

def perform_query(filters: list[str], client: MilvusClient) -> List[MilvusDocument]:
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter=" and ".join(filters),
        output_fields=OUTPUT_FIELDS,
        limit=100,
    )
    return [MilvusDocument(**doc) for doc in query_results]

def document_to_markdown(document: MilvusDocument) -> str:
    parts = []
    if document.title:
        parts.append(f"### {document.title}")
    if document.author:
        authors = document.author
        if isinstance(authors, list):
            parts.append(f"**Authors:** {', '.join(authors)}")
        else:
            parts.append(f"**Authors:** {authors}")
    if document.date:
        parts.append(f"**Date:** {document.date}")
    if document.source:
        parts.append(f"**Source:** `{document.source}` (Chunk: {document.chunk_index})")

    # Dynamically render other fields, excluding those already handled or empty.
    fields_to_ignore = {'title', 'author', 'date', 'source', 'chunk_index', 'text'}
    
    other_data = []
    for key, value in document.model_dump().items():
        if key not in fields_to_ignore and value:
            key_title = key.replace('_', ' ').title()
            if isinstance(value, list):
                value_str = '`, `'.join(map(str, value))
                other_data.append(f"**{key_title}:** `{value_str}`")
            else:
                other_data.append(f"**{key_title}:** {value}")
    
    if other_data:
        parts.extend(other_data)

    if document.text:
        parts.append("\n---\n" + document.text)
        
    return "\n".join(parts)

def consolidate_documents(documents: List[MilvusDocument]) -> List[MilvusDocument]:
    """
    Consolidates documents with the same source, combining their text and metadata,
    and sorts the results by distance.
    """
    if not documents:
        return []

    source_groups = defaultdict(list)
    for doc in documents:
        source_groups[doc.source].append(doc)

    consolidated_docs = []
    for source, docs in source_groups.items():
        sorted_chunks = sorted(docs, key=lambda d: d.chunk_index)
        base_doc = sorted_chunks[0]
        
        combined_text = "\n\n---\n\n".join([d.text for d in sorted_chunks if d.text])
        
        # Combine list-based fields
        combined_keywords = sorted(list(set(kw for d in sorted_chunks for kw in d.keywords)))
        combined_unique_words = sorted(list(set(word for d in sorted_chunks for word in d.unique_words)))
        
        min_distance = min((d.distance for d in sorted_chunks if d.distance is not None), default=None)

        # Create a new consolidated document
        consolidated_data = base_doc.model_dump()
        consolidated_data.update({
            "text": combined_text,
            "keywords": combined_keywords,
            "unique_words": combined_unique_words,
            "distance": min_distance,
            "chunk_index": 0, # Represents the consolidated document
        })
        
        consolidated_docs.append(MilvusDocument(**consolidated_data))

    # Sort the final list of consolidated documents by distance
    return sorted(consolidated_docs, key=lambda d: d.distance if d.distance is not None else float('inf'))

def build_citations(documents: List[MilvusDocument]) -> list:
    consolidated_docs = consolidate_documents(documents)
    
    citations = []
    for doc in consolidated_docs:
        citations.append({
            "source": {"name": doc.source, "url": ""},
            "document": [doc.text],
            "metadata": doc.model_dump(exclude={'text', 'distance'}),
            "distance": doc.distance
        })
    return citations

def to_openai_chunk(ollama_chunk: ollama.ChatResponse) -> dict:
    """Converts an Ollama streaming chunk to the OpenAI format."""
    message = ollama_chunk.message
    finish_reason = ollama_chunk.done_reason
    
    
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.chunk",
        "created": ollama_chunk.created_at,
        "model": ollama_chunk.model,
        "choices": [{
            "index": 0,
            "delta": {"content": message.content},
            "finish_reason": finish_reason
        }]
    }

async def pipe(body: dict):
    """Orchestrates a streaming agentic loop."""
    messages = body.get("messages", [])
    milvus_client = connect_milvus(MILVUS_TOKEN)
    if not milvus_client:
        yield {"error": "Unable to connect to the knowledge database."}
        return

    initial_search_results = perform_search(client=milvus_client, queries=[messages[-1].get("content")])
    preliminary_context = "\n\n".join([document_to_markdown(d) for d in initial_search_results])
    
    try:
        schema_info = str(milvus_client.describe_collection(COLLECTION_NAME).get("fields", {}))
    except Exception as e:
        logging.error(f"Failed to describe collection: {e}")
        schema_info = "{}"

    system_prompt = SystemPrompt.replace(
        "<<database_schema>>", schema_info
    ).replace(
        "<<preliminary_context>>", preliminary_context
    )

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system_prompt
        all_messages = messages
    else:
        all_messages = [{"role": "system", "content": system_prompt}] + messages

    available_tools = {"search": perform_search, "query": perform_query}
    all_sources = initial_search_results
    ollama_client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
    final_content = ""

    for i in range(MAX_TOOL_CALLS):
        logging.info(f"Agent loop iteration {i+1}/{MAX_TOOL_CALLS}")
        stream = await ollama_client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=all_messages,
            tools=[SearchInputSchema, QueryInputSchema],
            stream=True
        )
        
        tool_calls: list[ollama.Message.ToolCall] = []
        async for chunk in stream:
            yield to_openai_chunk(chunk)
            if new_tool_calls := chunk.message.tool_calls:
                logging.info(f"Tool calls received: {new_tool_calls}")
                tool_calls.extend(new_tool_calls)
            if content_chunk := chunk.message.content:
                final_content += content_chunk

        if not tool_calls:
            logging.info("No tool calls, breaking loop.")
            break

        all_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for tool in tool_calls:
            function_name = tool.function.name
            function_args = tool.function.arguments
            
            if func := available_tools.get(function_name):
                try:
                    tool_output = func(client=milvus_client, **function_args)
                    all_sources.extend(tool_output)
                    all_messages.append({
                        "role": "tool",
                        "content": "\n\n".join([document_to_markdown(d) for d in tool_output]),
                        "name": function_name,
                    })
                except Exception as e:
                    logging.error(f"Error executing tool {function_name}: {e}")
                    all_messages.append({
                        "role": "tool",
                        "content": f"Error executing tool {function_name}: {e}",
                        "name": function_name,
                    })

    yield {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.final",
        "created": int(time.time()),
        "model": OLLAMA_LLM_MODEL,
        "sources": build_citations(all_sources),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_content},
            "finish_reason": "stop"
        }]
    }

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
  - **Returns:** `list[str]` - A list of matching documents, including their metadata (e.g., text snippets, source, scalar fields).

- **`query`**: Use this tool _only_ when the user wants to retrieve documents based _solely_ on metadata criteria, without any semantic search component.
  - **Parameters:**
    - `filters` (str): A single filter expression string that specifies the metadata conditions for retrieving documents. This string must follow the Milvus boolean expression syntax.
  - **Returns:** `list[str]` - A list of documents matching the filter criteria, including their metadata.

**3. How to Answer User Queries:**

- **Check History/Knowledge First:** Before using any tool, check if the user's question can be answered based on the current conversation history. If so, answer directly without using a tool. Examples: greetings, simple clarifications, questions about your capabilities.
- **Semantic Queries:** If the user asks a question seeking information based on meaning, concepts, or topics (e.g., "What documents discuss X?", "Find information similar to Y"), use the `search` tool. Provide the core semantic query in the `queries` parameter.
- **Metadata-Only Queries:** If the user asks a question that only involves filtering by specific metadata fields (e.g., "List all documents by author Jane Doe", "Show me items where year > 2023", "Get documents with ID 12345"), use the `filter` tool. Provide the complete filter expression in the `query` parameter.
- **Hybrid Queries (Semantic + Filter):** If the user's query combines semantic meaning with metadata filters (e.g., "Find documents _about_ neural networks _published after_ 2022", "Search for 'signal processing techniques' _where the author is_ John Smith"), you MUST use the `search` tool. Put the semantic part in the `queries` parameter and the filtering conditions in the `filters` parameter (as a list containing one or more filter strings).
- **Ambiguity:**
  - If a query is ambiguous but seems to have a semantic component, default to using the `search` tool. Include potential filters in the `filters` parameter if they can be inferred, otherwise use an empty list `[]`.
  - If a query seems purely filter-based but the criteria are unclear (e.g., "Find recent documents"), you may ask one clarifying question. However, if the user doesn't clarify or the intent remains unclear, default to using the `search` tool with the ambiguous term as the query (e.g., `queries=["recent documents"]`, `filters=[]`).

**4. Constructing Filter Expressions:**

- **Syntax Reference:** You MUST strictly adhere to the Milvus boolean expression syntax when creating filter strings for the `filters` parameter of the `search` tool or the `query` parameter of the `filter` tool. The detailed syntax rules, available operators (`==`, `>`, `!=`, `<`, `>=`, `<=`, `IN`, `NOT IN`, `LIKE`, `json_contains`), handling of strings (use double quotes `"`), numbers, and special functions like `json_contains` are provided in the documentation below.
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

- After receiving results from a tool (which will be a `list[str]`), do not just output the raw data.
- Synthesize the information from the returned strings into a clear, concise, and helpful natural language response.
- Highlight the key findings or relevant snippets from the retrieved documents based on the user's query. Mention relevant metadata if appropriate (like source, author, date).
- If a tool returns no results, inform the user clearly that no matching documents were found based on their criteria.
- If your response includes details from the documents, put their title at the bottom of the response indicating their use.
- If the documents do not provide enough information to answer the user's question, inform them that you couldn't find relevant documents in the collection.

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
        }"
      }]
    }
    ```
- **User Query:** "Find documents that discuss \"neural networks in legal research\" or \"AI for case law analysis\", but only those tagged with domain:legaltech and from the year 2023."
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
            }
        }
      }]
    }
    ```
    Follow these instructions carefully to effectively query the Milvus database and provide accurate, relevant answers to the user.

    Some Preliminary Context for the user's query taken from a simple semantic search of the user's query. It may not be a response to the query especially if they need to filter the data for a certain attribute. Use this context to inform your search and response.
    <<preliminary_context>>

    Be clear in your responses. If the preliminary context can provide an answer use it and don't call search. Otherwise use it to inform your search.
    If the tool calls do not provide enough information to answer the user's question, inform them that you couldn't find relevant documents in the collection.
"""
