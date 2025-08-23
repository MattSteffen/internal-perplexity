"""
title: Radchat Function
author: Rincon [mjst, kca]
version: 0.1
requirements: pymilvus, ollama
"""

from pydantic import BaseModel, Field
from typing import Optional


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
OLLAMA_BASE_URL = "http://ollama.a1.autobahn.rinconres.com"  # os.getenv(
#     "OLLAMA_BASE_URL", "http://ollama.a1.autobahn.rinconres.com"
# )
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:latest")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "all_irads")
MILVUS_HOST = os.getenv("MILVUS_HOST", "10.43.210.111")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# --- Constants ---
OUTPUT_FIELDS = [
    "source",
    "chunk_index",
    "metadata",
    "title",
    "author",
    "date",
    "keywords",
    "unique_words",
    "text",
]
MAX_TOOL_CALLS = 5
REQUEST_TIMEOUT = 300
# Milvus search parameters
NPROBE = 10
SEARCH_LIMIT = 5
HYBRID_SEARCH_LIMIT = 10
RRF_K = 100
DROP_RATIO = 0.2

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            "required": [],
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
        logging.error(
            f"Error connecting to or loading Milvus collection '{COLLECTION_NAME}': {e}"
        )
        return None


def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        logging.error(f"Error getting embedding from Ollama ({OLLAMA_BASE_URL}): {e}")
        return None


def perform_search(
    client: MilvusClient, queries: list[str] = [], filters: list[str] = []
) -> list[MilvusDocument]:
    search_requests = []
    filter_expr = " and ".join(filters) if filters else ""

    search_configs = [
        {
            "field": "text_sparse_embedding",
            "param": {"drop_ratio_search": DROP_RATIO},
            "data_transform": lambda q: [q],
        },
        {
            "field": "metadata_sparse_embedding",
            "param": {"drop_ratio_search": DROP_RATIO},
            "data_transform": lambda q: [q],
        },
    ]

    for query in queries:
        embedding = get_embedding(query)
        if embedding:
            search_requests.append(
                AnnSearchRequest(
                    data=[embedding],
                    anns_field="text_embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": NPROBE}},
                    expr=filter_expr,
                    limit=SEARCH_LIMIT,
                )
            )

        for config in search_configs:
            search_requests.append(
                AnnSearchRequest(
                    data=config["data_transform"](query),
                    anns_field=config["field"],
                    param=config["param"],
                    expr=filter_expr,
                    limit=SEARCH_LIMIT,
                )
            )

    if not search_requests:
        if len(filters) > 0:
            return perform_query(filters, client)
        return []

    result = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=search_requests,
        ranker=RRFRanker(k=RRF_K),
        output_fields=OUTPUT_FIELDS,
        limit=HYBRID_SEARCH_LIMIT,
    )

    return [
        MilvusDocument(**doc["entity"], distance=doc["distance"]) for doc in result[0]
    ]


def perform_query(filters: list[str], client: MilvusClient) -> List[MilvusDocument]:
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter=" and ".join(filters),
        output_fields=OUTPUT_FIELDS,
        limit=100,
    )
    return [MilvusDocument(**doc) for doc in query_results]


def get_metadata(client: MilvusClient) -> list[str]:
    """gets all the entries in the database (firts 1000) and their title + authors + date, returns those."""
    all_docs = perform_query([], client)
    data = []
    for doc in all_docs:
        data.append(f"Title: {doc.title}\nAuthors: {doc.author}\nDate: {doc.date}")
    return data


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
    fields_to_ignore = {"title", "author", "date", "source", "chunk_index", "text"}

    other_data = []
    for key, value in document.model_dump().items():
        if key not in fields_to_ignore and value:
            key_title = key.replace("_", " ").title()
            if isinstance(value, list):
                value_str = "`, `".join(map(str, value))
                other_data.append(f"**{key_title}:** `{value_str}`")
            else:
                other_data.append(f"**{key_title}:** {value}")

    if other_data:
        parts.extend(other_data)

    if document.text:
        parts.append("\n---\n" + document.text)

    return "\n".join(parts)


def prettify_doc(document: MilvusDocument) -> str:
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
    fields_to_ignore = {"title", "author", "date", "source", "chunk_index", "text"}

    other_data = []
    for key, value in document.model_dump().items():
        if key not in fields_to_ignore and value:
            key_title = key.replace("_", " ").title()
            if isinstance(value, list):
                value_str = "`, `".join(map(str, value))
                other_data.append(f"**{key_title}:** `{value_str}`")
            else:
                other_data.append(f"**{key_title}:** {value}")

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
        combined_keywords = sorted(
            list(set(kw for d in sorted_chunks for kw in d.keywords))
        )
        combined_unique_words = sorted(
            list(set(word for d in sorted_chunks for word in d.unique_words))
        )

        min_distance = min(
            (d.distance for d in sorted_chunks if d.distance is not None), default=None
        )

        # Create a new consolidated document
        consolidated_data = base_doc.model_dump()
        consolidated_data.update(
            {
                "text": combined_text,
                "keywords": combined_keywords,
                "unique_words": combined_unique_words,
                "distance": min_distance,
                "chunk_index": 0,  # Represents the consolidated document
            }
        )

        consolidated_docs.append(MilvusDocument(**consolidated_data))

    # Sort the final list of consolidated documents by distance
    return sorted(
        consolidated_docs,
        key=lambda d: d.distance if d.distance is not None else float("inf"),
    )


def build_citations(documents: List[MilvusDocument]) -> list:
    consolidated_docs = consolidate_documents(documents)

    citations = []
    for doc in consolidated_docs:
        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [prettify_doc(doc)],
                "metadata": doc.model_dump(exclude={"text", "distance"}),
                "distance": doc.distance,
            }
        )
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
        "choices": [
            {
                "index": 0,
                "delta": {"content": message.content},
                "finish_reason": finish_reason,
            }
        ],
    }


class Pipe:
    class Valves(BaseModel):
        COLLECTION_NAME: str = Field(default=COLLECTION_NAME)

    def __init__(self):
        self.valves = self.Valves()
        self.citations = False

    async def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict = None,
    ):
        """Orchestrates a streaming agentic loop."""

        messages = body.get("messages", [])
        milvus_client = connect_milvus(MILVUS_TOKEN)
        if not milvus_client:
            yield {"error": "Unable to connect to the knowledge database."}
            return

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Fetching data...",
                    "done": False,
                    "hidden": False,
                },
            }
        )
        initial_search_results = perform_search(
            client=milvus_client, queries=[messages[-1].get("content")]
        )
        preliminary_context = "\n\n".join(
            [document_to_markdown(d) for d in initial_search_results]
        )

        try:
            schema_info = str(
                milvus_client.describe_collection(COLLECTION_NAME).get("fields", {})
            )
        except Exception as e:
            logging.error(f"Failed to describe collection: {e}")
            schema_info = "{}"

        meta = get_metadata(milvus_client)
        system_prompt = (
            SystemPrompt.replace("<<database_schema>>", schema_info)
            .replace("<<preliminary_context>>", preliminary_context)
            .replace("<<database_metadata>>", "\n\n".join(meta))
        )

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
            all_messages = messages
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        available_tools = {"search": perform_search}
        all_sources = initial_search_results
        ollama_client = ollama.AsyncClient(host=OLLAMA_BASE_URL, timeout=300)
        final_content = ""

        for i in range(MAX_TOOL_CALLS):
            logging.info(f"Agent loop iteration {i+1}/{MAX_TOOL_CALLS}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Waiting on Ollama",
                        "done": False,
                        "hidden": False,
                    },
                }
            )
            stream = await ollama_client.chat(
                model=OLLAMA_LLM_MODEL,
                messages=all_messages,
                tools=[SearchInputSchema],
                options={"num_ctx": 32000},
                stream=True,
            )

            tool_calls: list[ollama.Message.ToolCall] = []
            async for chunk in stream:
                print("Got chunk", chunk.message)
                if new_tool_calls := chunk.message.tool_calls:
                    logging.info(f"Tool calls received: {new_tool_calls}")
                    tool_calls.extend(new_tool_calls)
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Tool called: {new_tool_calls[0].function.name}",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                else:
                    yield to_openai_chunk(chunk)
                if content_chunk := chunk.message.content:
                    final_content += content_chunk

            if not tool_calls:
                logging.info("No tool calls, breaking loop.")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Done researching.",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )
                break

            all_messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

            for tool in tool_calls:
                function_name = tool.function.name
                function_args = tool.function.arguments

                if func := available_tools.get(function_name):
                    try:
                        tool_output = func(client=milvus_client, **function_args)
                        all_sources.extend(tool_output)
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": "\n\n".join(
                                    [document_to_markdown(d) for d in tool_output]
                                ),
                                "name": function_name,
                            }
                        )
                    except Exception as e:
                        logging.error(f"Error executing tool {function_name}: {e}")
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": f"Error executing tool {function_name}: {e}",
                                "name": function_name,
                            }
                        )

        for cit in build_citations(all_sources):
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": cit,
                }
            )
        yield {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.final",
            "created": int(time.time()),
            "model": OLLAMA_LLM_MODEL,
            # "sources": build_citations(all_sources),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": "stop",
                }
            ],
        }


SystemPrompt = """
You are a specialized document retrieval assistant for an internal research and development (IRAD) document collection. Your SOLE PURPOSE is to help users find and extract information from this specific document collection. You cannot and will not provide information from outside this collection.

## Core Principles:
- **Document-Only Responses**: All answers must be grounded in the retrieved documents
- **Explicit Source Attribution**: Always cite which documents inform your response
- **Acknowledge Limitations**: If information isn't in the collection, clearly state this
- **No External Knowledge**: Never supplement with information not found in the documents

## Document Collection Context:

**Content Focus**: Internal R&D documents covering signal processing, Artificial Intelligence (AI), and Machine Learning (ML)

**Database Schema**:
```json
<<database_schema>>
```

**Available Documents**:
```
<<database_metadata>>
```

## Available Tool:

**`search`** - Retrieves relevant documents from the collection
- `queries` (list[str]): Semantic search terms for finding conceptually similar content
- `filters` (list[str]): Milvus filter expressions for metadata-based filtering
- Returns: List of matching document chunks with metadata

## Query Processing Strategy:

**1. Information Available Check**: First check if the user's question can be answered from:
   - Previous search results in this conversation
   - The document metadata provided above
   - If yes, answer directly without searching

**2. Query Classification**:
   - **Semantic Search**: User seeks topical/conceptual information → Use `queries` parameter
   - **Metadata Filtering**: User wants documents by specific criteria → Use `filters` parameter  
   - **Hybrid Search**: Combines both semantic and metadata needs → Use both parameters

**3. Search Decision Matrix**:
   - Documents about a topic/concept → `queries: ["topic"]`, `filters: []`
   - Documents by author/date/title → `queries: []`, `filters: ["criteria"]`
   - Topic + specific criteria → `queries: ["topic"]`, `filters: ["criteria"]`

## Response Guidelines:
In the case that the user is looking for details about the IRADs in question:
**When Documents Are Found**:
- Synthesize information from retrieved documents into a coherent answer
- Include specific details, quotes, or key findings relevant to the query
- If documents partially address the query, acknowledge what's covered and what's missing

In the case that the user is looking for generic details from the database:
**When documents are found**:
- Be brief with your responses (`how many irads were written by <some-author>` -> `10, their titles are: <list the irads>`)
- Do not provide too many details unless requested

In All Cases:
**When No Documents Are Found**:
- State clearly: "I could not find any documents in the collection that address [specific query]"
- Do not speculate or provide information from outside the collection
- Optionally suggest related search terms if appropriate

**Important Constraints**:
- Never provide information not contained in the retrieved documents
- Never say "I don't know" - instead specify that the information isn't in the document collection
- Always be explicit about the source and scope of your knowledge
- If asked about topics outside the collection scope (signal processing, AI, ML R&D), clearly state the collection's limitations


## 4. Constructing Filter Expressions:

- **Syntax Reference:** You MUST strictly adhere to the Milvus boolean expression syntax when creating filter strings for the `filters` parameter of the `search` tool or the `query` parameter of the `filter` tool. The detailed syntax rules, available operators (`==`, `>`, `!=`, `<`, `>=`, `<=`, `IN`, `NOT IN`, `LIKE`, `ARRAY_CONTAINS`, `ARRAY_CONTAINS_ALL`, `ARRAY_CONTAINS_ANY`, `ARRAY_LENGTH`), handling of strings (use double quotes `"`), numbers, and special functions are provided in the documentation below.

- **Use the Schema:** Always refer to the `database_schema` provided above to ensure you are using correct field names and comparing values of the appropriate type. Pay special attention to ARRAY fields (`author`, `keywords`, `unique_words`) which require specific ARRAY operators.

- **Critical Field Types:**
  - `author`: ARRAY of VARCHAR - use `ARRAY_CONTAINS`, `ARRAY_CONTAINS_ANY`, `ARRAY_CONTAINS_ALL`
  - `keywords`: ARRAY of VARCHAR - use `ARRAY_CONTAINS`, `ARRAY_CONTAINS_ANY`, `ARRAY_CONTAINS_ALL`
  - `unique_words`: ARRAY of VARCHAR - use `ARRAY_CONTAINS`, `ARRAY_CONTAINS_ANY`, `ARRAY_CONTAINS_ALL`
  - `date`: INT64 representing year (YYYY format) - use numeric comparisons
  - `title`: VARCHAR - use string comparisons and `LIKE` for pattern matching
  - `text`: VARCHAR - use string comparisons and `LIKE` for pattern matching

- **Documentation:**

  ```markdown
  Milvus supports rich boolean expressions over scalar and ARRAY fields:
 
  | Operator / Function | Meaning | Use Case |
  |---------------------|---------|----------|
  | `==`, `!=`, `>`, `<` | Standard numeric/string comparisons | Numbers, strings, years |
  | `>=`, `<=` | Greater/less than or equal to | Numbers, years |
  | `IN`, `NOT IN` | Membership test for lists | String/numeric values |
  | `LIKE` | SQL-style wildcard for strings (`%`, `_`) | Text pattern matching |
  | `ARRAY_CONTAINS` | Checks if array contains specific value | Single author, keyword, etc. |
  | `ARRAY_CONTAINS_ALL` | Checks if array contains all specified values | Multiple required items |
  | `ARRAY_CONTAINS_ANY` | Checks if array contains any of specified values | Any of several options |
  | `ARRAY_LENGTH` | Filters by array length | Arrays with certain sizes |
  | `AND`, `OR`, `NOT` | Boolean connectors | Combining conditions |

  ## Examples Based on Your Schema:

  ### Author Filtering (ARRAY field):
  -- Find documents by a specific author
  "ARRAY_CONTAINS(author, \"John Smith\")"
 
  -- Find documents with any of several authors
  "ARRAY_CONTAINS_ANY(author, [\"John Smith\", \"Jane Doe\", \"Bob Wilson\"])"
 
  -- Find documents authored by both specific authors
  "ARRAY_CONTAINS_ALL(author, [\"John Smith\", \"Jane Doe\"])"
 
  -- Find documents with multiple authors
  "ARRAY_LENGTH(author) > 1"

  ### Date Filtering (INT64 field - year format):
  -- Documents from a specific year
  "date == 2023"
 
  -- Documents from recent years
  "date >= 2020"
 
  -- Documents from a range of years
  "date >= 2015 AND date <= 2023"
 
  -- Documents from specific years
  "date IN [2020, 2021, 2022, 2023]"

  ### Keywords Filtering (ARRAY field):
  -- Documents with specific keyword
  "ARRAY_CONTAINS(keywords, \"artificial intelligence\")"
 
  -- Documents with any AI-related keywords
  "ARRAY_CONTAINS_ANY(keywords, [\"AI\", \"machine learning\", \"neural networks\"])"

  ### Title and Text Filtering (VARCHAR fields):
  -- Title contains specific word
  "title LIKE \"%Guide%\""
 
  -- Text contains pattern
  "text LIKE \"%machine learning%\""

  ### Combined Examples:
  -- Recent documents by specific author with AI keywords
  "date >= 2020 AND ARRAY_CONTAINS(author, \"John Smith\") AND ARRAY_CONTAINS_ANY(keywords, [\"AI\", \"artificial intelligence\"])"
 
  -- Documents with multiple authors from last 5 years
  "ARRAY_LENGTH(author) > 1 AND date >= 2019"
 
  -- Documents with specific title pattern and recent date
  "title LIKE \"%Analysis%\" AND date >= 2022"
 
  -- Documents by multiple specific authors OR containing specific keywords
  "ARRAY_CONTAINS_ANY(author, [\"Smith\", \"Johnson\"]) OR ARRAY_CONTAINS(keywords, \"research\")"
  ```

**Important Notes:**
- NEVER use `json_contains` on ARRAY fields - use `ARRAY_CONTAINS` instead
- The `date` field stores years as integers (e.g., 2023), not date strings
- ARRAY fields (`author`, `keywords`, `unique_words`) are nullable, so check for NULL if needed
- Always use double quotes for string values in filters
- When filtering ARRAY fields, the comparison value should match the element type (VARCHAR for these arrays)

## 5. Responding to the User:

**Processing Search Results**:
- Never output raw search results directly to the user
- Synthesize information from the returned document chunks into clear, natural language responses
- Extract and highlight key findings, relevant snippets, and important details that address the user's specific query
- Include relevant metadata (author, title, date) when it adds value to the response

**When Documents Are Found**:
- Provide a comprehensive answer based on the retrieved content
- Quote specific passages when they directly answer the question
- Explain how the documents relate to the user's query
- If documents partially address the query, clearly state what is covered and what remains unanswered
- End with: "**Sources**: [List document titles used]"

**When No Documents Are Found**:
- State explicitly: "I could not find any documents in the collection that address [specific query topic]"
- Do not speculate or provide information from outside the collection
- Do not simply say "I don't know" - be specific about what's missing from the collection
- If appropriate, suggest alternative search terms or related topics that might be in the collection

**Quality Standards**:
- Responses must be grounded entirely in the retrieved documents
- Maintain professional, helpful tone while staying within document boundaries
- If retrieved documents don't provide sufficient information to fully answer the question, acknowledge this limitation explicitly
- Never supplement document content with external knowledge, even if it seems helpful


## 6. Examples:

- **User Query:** "What are the latest advancements in AI for signal processing?"
  - **Tool Call:**
    ```json
    {
      "queries": ["latest advancements in AI for signal processing"],
      "filters": []
    }
    ```

- **User Query:** "Find documents about neural networks in legal research from 2023 with AI keywords."
  - **Tool Call:**
    ```json
    {
      "queries": ["neural networks in legal research"],
      "filters": ["date == 2023 AND ARRAY_CONTAINS_ANY(keywords, [\"AI\", \"artificial intelligence\", \"neural networks\"])"]
    }
    ```

- **User Query:** "Show me all documents written by Dr. Evelyn Reed."
  - **Tool Call:**
    ```json
    {
      "queries": [],
      "filters": ["ARRAY_CONTAINS(author, \"Dr. Evelyn Reed\")"]
    }
    ```

- **User Query:** "Find documents containing quantum computing applications in healthcare from the last 3 years."
  - **Tool Call:**
    ```json
    {
      "queries": ["quantum computing applications healthcare"],
      "filters": ["date >= 2022"]
    }
    ```

- **User Query:** "Get documents by either John Smith or Jane Doe that mention machine learning."
  - **Tool Call:**
    ```json
    {
      "queries": ["machine learning"],
      "filters": ["ARRAY_CONTAINS_ANY(author, [\"John Smith\", \"Jane Doe\"])"]
    }
    ```

- **User Query:** "Show research papers with multiple authors from 2020-2023."
  - **Tool Call:**
    ```json
    {
      "queries": ["research papers"],
      "filters": ["ARRAY_LENGTH(author) > 1 AND date >= 2020 AND date <= 2023"]
    }
    ```

- **User Query:** "Find documents with titles containing 'Analysis' published after 2021."
  - **Tool Call:**
    ```json
    {
      "queries": [],
      "filters": ["title LIKE \"%Analysis%\" AND date > 2021"]
    }
    ```

## Preliminary Context Handling:
Context: <<preliminary_context>>

**Usage Instructions**:
- If preliminary context fully answers the user's query, use it directly and cite the sources
- If preliminary context is incomplete or requires filtering, use it to inform your search strategy
- If preliminary context is irrelevant, proceed with fresh search based on user query

Remember: Your value lies in accurate retrieval and synthesis of information from this specific document collection. Stay within these bounds to provide reliable, grounded responses.
"""
