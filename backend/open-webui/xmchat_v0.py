"""
title: XMChat Function
author: Rincon [mjst, kca]
version: 0.1
requirements: pymilvus, ollama
"""

"""
What is XMChat?
- XMChat is a chatbot that can answer questions about X-Midas.
- Capabilities:
    - answer coding questions, like "how do I do X?" and "what does this command do?" and "what command does X?"
    - answer questions about the X-Midas codebase, like "what does this file do?" and "what is this function?" and "what is this class?"
    - Provide links back to the actual documentation or codebase for user's reference

- Tools:
    - search milvus: collections include: docs, mattermost chats, help, explain files, code examples
    - generate more search queries, and summarize the results (like an agentic search engine)

Big changes to make:
- XMChat will have more tools
    - search by collections (collecctions will include: docs, mattermost chats, help, explain files, code examples)
    - Search by multiple collections at once
    - generate more search queries, and summarize the results (like an agentic search engine)


Data is extracted from the following sources and schemas and stored in milvus:
learnxm.json
```json
[
{
        "subject": "What is MIDAS?",
        "description": "This section defines MIDAS as a Multi-user Interactive Development and Analysis System, explaining that X-Midas is MIDAS with an X11 GUI and plot primitives. It is described as a government-sponsored signal processing software framework that is open-source but not freely distributed, with a focus on signal generation, acquisition, analysis, algorithm development, application prototyping, and deliverable system development, emphasizing code re-usability and a combined hardware and software infrastructure. The primary goal is to minimize the time and cost of developing and delivering new signal processing techniques.",
        "tags": [
            "MIDAS",
            "X-Midas",
            "signal processing",
            "software framework",
            "open-source",
            "GUI",
            "plot primitives"
        ],
        "url": "https://rrc.rinconres.com/~xmmgr/latest/learnxm/background.html"
    },
]
```

xm_docs.json
```json
[
    {
        "subject": "X-Midas",
        "description": "X-Midas is an open-source signal processing software environment for interactive analysis, data acquisition, and digital signal processing algorithm and application development. It is designed for multi-user interaction, scalability, and extensibility, supporting various programming languages and platforms.",
        "tags": [
            "signal processing",
            "software environment",
            "interactive analysis",
            "data acquisition",
            "algorithm development",
            "open-source",
            "scalability",
            "extensibility"
        ],
        "xm_path": "README.md"
    },
]
```

processed_xm_qa.json (matter most conversations)
```json
[
    {
        "id": "qpnk13mguircxrqspjuaknh4sc",
        "question": "Is there a primitive in the midas baseline to interpolate 5000 files?",
        "answer": "Looks like `interpolate` handles 3000 and 5000 too.",
        "users": [
            "djza"
        ],
        "time": "2025-04-19 01:05:22.549 +0000 UTC",
        "context": "2025-04-21 18:50:36.578 +0000 UTC: zoc: \"@kca @ged \nExporting this channel into a CSV as per request 44843\"\n\n2025-04-21 16:09:29.798 +0000 UTC: hwp: \"Thank you!\"\n\n2025-04-19 01:05:22.549 +0000 UTC: djza: \"~~Is there a primitive in the midas baseline to interpolate 5000 files?~~  Looks like `interpolate` handles 3000 and 5000 too.  Just doesn't support cubic hermite for my use case.\"\n\n2025-04-17 23:39:46.342 +0000 UTC: rda: \"There's also the `/BWDISC=<Hz>` switch. \n It  drops connection if sample rate changes more than <Hz>.  Might as well stop if someone changes the sample rate on you.\"\n\n2025-04-17 23:38:31.102 +0000 UTC: rda: \"Use the `MAXEMPTY=` switch.  It's the maximum number of \"empty\" warnings before exiting - it's a fixed time interval per warning.  The default is 999999 (~ never).  Set it to something like `/MAXEMPTY=5` and see if it does what you need.\"\n\n2025-04-17 23:14:02.395 +0000 UTC: jph: \"If all of the primitives had been data-driven, then each of them would have shut itself down when the data ran out, and then the pipe section would have ended naturally.  But in this case, the primitive count would never drop to zero because not every primitive was data-driven.\"\n\n2025-04-17 23:11:30.496 +0000 UTC: jph: \"In this case, it was needed to automatically close out a pipe section when all the data ran out, because I had pipe monitors or other event-based primitives that were holding things open.\"\n\n2025-04-17 23:10:01.54 +0000 UTC: jph: \"This is an example of an ad-hoc primitive that I made that has the ability to wait until all pipes are empty and then send an EXIT message and set Mc->break_:\nhttps://gitlab.rinconres.com/rincon/kb/-/snippets/94\"\n\n2025-04-17 23:03:55.669 +0000 UTC: jph: \"I have limited direct experience with the PSM option tree, but one ad-hoc option would be to create a custom primitive whose job is just to watch a pipe, and if the pipe sits empty for a certain amount of time, it sets `Mc->break_` or sends an `EXIT` message.\"\n\n2025-04-17 22:49:05.275 +0000 UTC: hwp: \"let me try typing that again. I have a macro that captures some timecode data from serveral tuners using sourcenic and compares them. \n\nIs there a way to have the pipe or sourcenic automatically quit if its not receiving any packets? I see the /TIMEOUT switch in pipe but that's for the whole routine regardless of if Im getting packets or not, correct?\"\n\n2025-04-17 19:14:58.794 +0000 UTC: jph: \"My understanding is that Dr. Parker decided to declare a minor victory and hand off the baton to the next generation at that point.  As far as I know he is no longer pursuing additional public releases of X-Midas source code\"\n\n2025-04-17 19:13:36.209 +0000 UTC: jph: \"Mike Parker's efforts for a public X-Midas led to a [one-time release](https://gitlab/jph/pastebin/-/wikis/XM2020-Message-from-Mike-Parker) of Midas source code at version 5.4.0.\"\n\n2025-04-17 19:13:31.372 +0000 UTC: jph: \"NextMidas has not been publicly available online for a few years now.\""
    },
]
```
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
OLLAMA_BASE_URL = "http://ollama.a1.autobahn.rinconres.com"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "mistral-small3.2:latest"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "xmidas"
MILVUS_HOST = "10.43.210.111"  # TODO: Replace with http://svc.mjst.milvus.milvus
MILVUS_PORT = "19530"

# --- Constants ---
# TODO: Don't use output fields, just return everything.
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


def document_to_markdown(document: MilvusDocument) -> str:
    parts = []
    if document.source:
        parts.append(f"**Source:** `{document.source}` (Chunk: {document.chunk_index})")

    # Dynamically render other fields, excluding those already handled or empty.
    fields_to_ignore = {"source", "chunk_index", "text"}

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


def build_citations(documents: List[MilvusDocument]) -> list:
    citations = []
    for doc in documents:
        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [document_to_markdown(doc)],
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

        system_prompt = SystemPrompt.replace(
            "<<database_schema>>", schema_info
        ).replace("<<preliminary_context>>", preliminary_context)

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
You are a specialized document retrieval assistant for a collection of all documentation, help/explain files for X-Midas. Your SOLE PURPOSE is to help users find and extract information from this specific document collection. You cannot and will not provide information from outside this collection.

## Core Principles:
- **Document-Only Responses**: All answers must be grounded in the retrieved documents
- **Explicit Source Attribution**: Always cite which documents inform your response
- **Acknowledge Limitations**: If information isn't in the collection, clearly state this
- **No External Knowledge**: Never supplement with information not found in the documents

## Document Collection Context:

**Content Focus**: X-Midas Documentation describing everything from how to write, use, and learn X-Midas.

**Database Schema**:
```json
<<database_schema>>
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
Respond to the user briefly. Provide code examples only when asked.

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
