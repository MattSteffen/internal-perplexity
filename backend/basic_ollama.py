hardcoded_api_key = "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"

import os
import requests
import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pymilvus # Ensure pymilvus is installed: pip install pymilvus

# System prompt for the LLM to perform query expansion and filtering
SYSTEM_PROMPT_QUERY_EXPANSION = """You are an expert assistant specializing in analyzing user queries and chat history to optimize document retrieval from a Milvus vector database.
Your goal is to generate relevant search queries and metadata filters to help find the most pertinent information using the available tool.

Instructions:
1.  Carefully review the latest user query and the preceding conversation history provided.
2.  Identify key entities, concepts, and any specific constraints mentioned (e.g., author names, document titles, topics, roles).
3.  Generate 1 to 3 *additional* semantic search queries (`extra_queries`) that capture different facets, rephrasings, or related concepts of the user's request. These queries will be used alongside the original query for semantic search in the vector database.
4.  Identify any explicit or implicit requests to filter documents based on their metadata. Construct filter expressions (`filters`) for the Milvus database based on the available fields.
5.  You MUST use the `milvus_search_helper` tool to output the generated `extra_queries` and `filters`. Do not provide any conversational text, only the tool call.

Available metadata fields for filtering:
* `title` (string): Document title. Use `==` for exact match (e.g., `title == "Introduction to RAG"`).
* `author` (string): Document author(s). Use `==` for exact match (e.g., `author == "Matt Steffen"`). Remember string values need double quotes within the filter string.
* `author_role` (string): Author's role. Use `==` (e.g., `author_role == "editor"`).
* `url` (string): Associated URL. Use `==`.
* `chunk_index` (integer): Index of the text chunk. Use operators like `==`, `>`, `<`, `>=`, `<=` (e.g., `chunk_index > 5`).
* `description` (string): Document description. Direct filtering on long text fields is often inefficient; prefer semantic search queries unless a very specific phrase is targeted with `==`.

Example:
User Query: "Find me documents about Large Language Models written by Geoffrey Hinton."
Chat History: [...]
Tool Call Expected:
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "milvus_search_helper",
      "arguments": '{ "extra_queries": ["Hinton LLM research", "Neural network advancements Hinton"], "filters": ["author == \\"Geoffrey Hinton\\""] }'
    }
  }]
}

Example 2 (No specific filters):
User Query: "Tell me about vector databases."
Chat History: [...]
Tool Call Expected:
{
  "tool_calls": [{
    "id": "call_def456",
    "type": "function",
    "function": {
      "name": "milvus_search_helper",
      "arguments": '{ "extra_queries": ["vector database explanation", "how do vector DBs work", "milvus vs pinecone"], "filters": [] }'
    }
  }]
}


If no specific filters are identifiable, provide an empty list `[]` for `filters`. If no good extra queries can be generated, provide an empty list `[]` for `extra_queries`.
You MUST call the `milvus_search_helper` tool. Do not provide a conversational answer outside the tool call.
"""

# Tool definition for Groq API
MILVUS_HELPER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "milvus_search_helper",
        "description": "Generates relevant search queries and metadata filters for retrieving documents from a Milvus vector database based on user query and conversation history.",
        "parameters": {
            "type": "object",
            "properties": {
                "extra_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 1-3 additional search queries relevant to the user's request to improve document retrieval via semantic search."
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of filter strings for Milvus metadata query expression (e.g., 'author == \"Matt Steffen\"', 'chunk_index > 5'). Available fields: title, author, author_role, url, chunk_index, description."
                }
            },
            "required": ["extra_queries", "filters"]
        }
    }
}


# --- Standalone Milvus and Embedding Functions ---

def connect_milvus(host: str, port: str, collection_name: str) -> Optional[pymilvus.Collection]:
    """
    Connects to Milvus and returns the specified collection object.

    Args:
        host: Milvus server host address.
        port: Milvus server port.
        collection_name: Name of the collection to connect to.

    Returns:
        A loaded Milvus Collection object or None if connection/loading fails.
    """
    alias = "rag_connection" # Unique alias for this connection
    try:
        if not pymilvus.connections.has_connection(alias):
             print(f"Connecting to Milvus at {host}:{port}...")
             pymilvus.connections.connect(alias=alias, host=host, port=port)
        else:
             print(f"Already connected to Milvus using alias '{alias}'")

        if not pymilvus.utility.has_collection(collection_name, using=alias):
             print(f"Error: Collection '{collection_name}' does not exist.")
             return None

        print(f"Accessing Milvus collection: {collection_name}")
        collection = pymilvus.Collection(collection_name, using=alias)
        collection.load() # Ensure the collection data is loaded into memory
        print(f"Collection '{collection_name}' loaded.")
        return collection
    except Exception as e:
        print(f"Error connecting to or loading Milvus collection '{collection_name}': {e}")
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
            timeout=30 # Add a timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        return result.get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama ({ollama_url}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during embedding: {e}")
        return None
    
def search_milvus(
    queries: List[str], # Changed from single query to list
    milvus_host: str,
    milvus_port: str,
    milvus_collection_name: str,
    ollama_url: str,
    embedding_model: str,
    limit: int = 5,
    filters: Optional[List[str]] = None # Added filters parameter
) -> List[Dict[str, Any]]:
    """
    Searches a Milvus collection using one or more query strings and optional filters.

    Args:
        queries: The list of user's query strings (original + expanded).
        milvus_host: Milvus server host.
        milvus_port: Milvus server port.
        milvus_collection_name: Name of the Milvus collection.
        ollama_url: Base URL for the Ollama API (for embedding).
        embedding_model: Name of the Ollama embedding model.
        limit: Maximum number of search results to return *in total*.
        filters: A list of Milvus filter expression strings (e.g., ['author == "John Doe"', 'chunk_index > 0']).

    Returns:
        A list of unique dictionaries, each representing a search result document, ranked by score.
    """
    collection = connect_milvus(milvus_host, milvus_port, milvus_collection_name)
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
        print(f"Searching collection '{milvus_collection_name}' with {len(query_embeddings)} query vectors.")
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}} # Adjust nprobe as needed

        collection_details = collection.describe()
        fields = [f.get('name') for f in collection_details.get('fields', [])]

        # Perform search
        results = collection.search(
            data=query_embeddings, # List of query vectors
            anns_field="embedding",
            param=search_params,
            limit=limit, # Retrieve potentially more initially to allow for deduplication
            expr=filter_expr, # Apply the combined filter expression
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
                    "score": hit.score
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
            if doc_id not in unique_documents_dict or doc["score"] < unique_documents_dict[doc_id]["score"]:
                unique_documents_dict[doc_id] = doc

        # Sort the unique documents by score (ascending for L2)
        sorted_documents = sorted(unique_documents_dict.values(), key=lambda x: x["score"])

        # Return the top 'limit' unique documents
        final_documents = sorted_documents[:limit]

        print(f"Found {len(final_documents)} unique relevant documents in Milvus after processing.")
        # Optional: Disconnect if the connection is specific and not reused immediately
        # pymilvus.connections.disconnect(f"rag_connection_{milvus_collection_name}")
        return final_documents

    except pymilvus.exceptions.MilvusException as e:
        print(f"Error searching Milvus: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error during Milvus search: {e}")
        return []

# --- Ollama Interaction Function ---

def call_ollama_chat(ollama_url: str, payload: Dict[str, Any]) -> str:
    """
    Makes a non-streaming chat request to Ollama.

    Args:
        ollama_url: Base URL of the Ollama API.
        payload: The JSON payload for the /api/chat endpoint.

    Returns:
        The content of the response message, or an error string.
    """
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=(10, 180) # connection timeout 10s, read timeout 180s
        )
        response.raise_for_status() # Check for HTTP errors

        res = response.json()
        content = res.get("message", {}).get("content", "")
        print("Received response from Ollama.")
        return content

    except requests.exceptions.Timeout:
         print(f"Error: Ollama request timed out ({ollama_url}/api/chat)")
         return "Error: The request to the language model timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama chat API: {e}")
        return f"Error: Could not communicate with the language model ({e})"
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON response from Ollama: {response.text}")
         return "Error: Received an invalid response from the language model."
    except Exception as e:
        print(f"Unexpected error during Ollama chat call: {e}")
        return f"Error: An unexpected error occurred ({e})."
# Define a type hint for the Groq response
GroqChatResponse = Union[
    Dict[str, str], # For text responses: {"type": "text", "content": "..."}
    Dict[str, Any]  # For tool calls: {"type": "tool_call", "name": "...", "arguments": {...}}
]


def call_groq_chat(
    payload: Dict[str, Any],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None # Can be "auto", "none", or {"type": "function", "function": {"name": "my_function"}}
) -> GroqChatResponse:
    """
    Makes a non-streaming chat request to the Groq API, supporting optional tool calling.

    Args:
        payload: The JSON payload for the Groq /openai/v1/chat/completions endpoint.
                 Must include 'model' and 'messages'.
        tools: An optional list of tool definitions (in OpenAI format).
        tool_choice: Optional control over tool usage ('auto', 'none', or specific tool).

    Returns:
        A dictionary indicating the response type:
        - {"type": "text", "content": "..."} if a text message is returned.
        - {"type": "tool_call", "name": "...", "arguments": {...}} if a tool call is requested.
        - {"type": "error", "content": "..."} if an error occurred.
    """
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hardcoded_api_key}",
        "Content-Type": "application/json",
    }

    # Ensure the request is non-streaming
    payload['stream'] = False

    # Add tools and tool_choice to payload if provided
    if tools:
        payload['tools'] = tools
    if tool_choice:
        payload['tool_choice'] = tool_choice

    # Validate required payload keys
    if "model" not in payload:
         print("Error: Groq payload must include a 'model' key.")
         return {"type": "error", "content": "Error: Payload must include a 'model' key."}
    if "messages" not in payload or not isinstance(payload["messages"], list):
         print("Error: Groq payload must include a 'messages' key containing a list.")
         return {"type": "error", "content": "Error: Payload must include a 'messages' key containing a list."}


    try:
        # print(f"Sending request to Groq: {json.dumps(payload, indent=2)}") # Debug: Print payload
        response = requests.post(
            groq_api_url,
            headers=headers,
            json=payload,
            timeout=(10, 180) # connection timeout 10s, read timeout 180s
        )
        response.raise_for_status()

        res = response.json()
        # print(f"Received response from Groq: {json.dumps(res, indent=2)}") # Debug: Print response

        choices = res.get("choices", [])
        if not choices:
             print("Error: Groq response contained no 'choices'.")
             return {"type": "error", "content": "Error: Received no response choices from the language model."}

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason")

        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls and finish_reason == "tool_calls":
            # Assuming only one tool call per response for simplicity, as requested
            if len(tool_calls) > 1:
                 print("Warning: Multiple tool calls received, processing only the first one.")

            first_tool_call = tool_calls[0]
            tool_name = first_tool_call.get("function", {}).get("name")
            tool_arguments_str = first_tool_call.get("function", {}).get("arguments")

            if not tool_name or not tool_arguments_str:
                 print(f"Error: Malformed tool call received from Groq: {first_tool_call}")
                 return {"type": "error", "content": "Error: Received incomplete tool call information."}

            try:
                tool_arguments = json.loads(tool_arguments_str)
                print(f"Groq requested tool call: '{tool_name}' with arguments: {tool_arguments}")
                return {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_arguments
                }
            except json.JSONDecodeError:
                print(f"Error: Could not decode tool call arguments JSON: {tool_arguments_str}")
                return {"type": "error", "content": "Error: Failed to parse tool call arguments."}

        # If no tool call, get the text content
        content = message.get("content", "")
        if content is None and not tool_calls :
             print("Warning: Groq response message content is null, but no tool call was made.")
             content = "" # Treat null content without tool calls as empty string

        # print("Received text response from Groq.")
        return {"type": "text", "content": content}

    except requests.exceptions.Timeout:
         print(f"Error: Groq request timed out ({groq_api_url})")
         return {"type": "error", "content": "Error: The request to the language model timed out."}
    except requests.exceptions.RequestException as e:
        error_details = ""
        status_code = None
        if e.response is not None:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                error_details = f" Status Code: {status_code}. Response: {error_data}"
            except json.JSONDecodeError:
                error_details = f" Status Code: {status_code}. Response: {e.response.text}"
        print(f"Error calling Groq chat API: {e}{error_details}")
        auth_hint = " (Hint: Check if your Groq API key is correct and active)" if status_code in [401, 403] else ""
        return {"type": "error", "content": f"Error: Could not communicate with the language model{auth_hint}. ({e})"}
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON response from Groq: {response.text}")
         return {"type": "error", "content": "Error: Received an invalid response from the language model."}
    except Exception as e:
        print(f"Unexpected error during Groq chat call: {e}")
        return {"type": "error", "content": f"Error: An unexpected error occurred ({e})."}

class Pipe:
    """
    A RAG pipeline using Groq for query expansion/filtering and Ollama/Milvus for retrieval.
    - Takes user query and chat history.
    - Calls Groq with a tool to generate extra queries and filters.
    - Performs Milvus search using original query, extra queries, and filters.
    - Calls Groq again with retrieved context to generate the final answer.
    """
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(default="http://host.docker.internal:11434")
        # Defaulting Groq model here for generation, can be overridden
        GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile") # Or another capable Groq model
        GROQ_MODEL_QUERY_EXPANSION: str = Field(default="llama-3.3-70b-versatile") # Faster model for tool use
        OLLAMA_EMBEDDING_MODEL: str = Field(default="all-minilm:v2") # Match your embedding setup
        MILVUS_HOST: str = Field(default="host.docker.internal")
        MILVUS_PORT: str = Field(default="19530")
        MILVUS_COLLECTION: str = Field(default="sample") # Use a descriptive name
        MILVUS_SEARCH_LIMIT: int = Field(default=5)
        MAX_CONTEXT_TOKENS: int = Field(default=4096) # Rough token limit for final prompt
        HISTORY_LENGTH: int = Field(default=5) # How many past messages to include in history

    def __init__(self):
        self.valves = self.Valves(
            OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            GROQ_MODEL=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            GROQ_MODEL_QUERY_EXPANSION=os.getenv("GROQ_MODEL_QUERY_EXPANSION", "llama-3.3-70b-versatile"),
            OLLAMA_EMBEDDING_MODEL=os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:v2"),
            MILVUS_HOST=os.getenv("MILVUS_HOST", "host.docker.internal"),
            MILVUS_PORT=os.getenv("MILVUS_PORT", "19530"),
            MILVUS_COLLECTION=os.getenv("MILVUS_COLLECTION", "sample"),
            MILVUS_SEARCH_LIMIT=int(os.getenv("MILVUS_SEARCH_LIMIT", "5")),
            MAX_CONTEXT_TOKENS=int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
             HISTORY_LENGTH=int(os.getenv("HISTORY_LENGTH", "5")),
        )
        self.type = "pipe"
        # Basic check for API key (won't validate it, just checks if placeholder)
        if hardcoded_api_key == "YOUR_GROQ_API_KEY_HERE":
             print("\n*** WARNING: Groq API key is not set. Tool calling and generation will fail. ***\n")
        print("RAG Pipe Initialized with config:", self.valves.model_dump()) # Use model_dump() for pydantic v2

    def pipes(self) -> List[dict]:
        """Defines the available pipeline(s) this class offers."""
        return [{"id": "groq-rag-tool-pipeline", "name": f"{self.valves.GROQ_MODEL}-rag-tools"}]

    def _get_text_content(self, content: Union[str, List[Dict]]) -> str:
        """Extracts text from Ollama message content (str or list)."""
        if isinstance(content, list):
            # Handle multimodal content if necessary, extract text parts
            text_content = " ".join(
                item["text"] for item in content if item.get("type") == "text"
            )
            # Add handling for images if needed, e.g., "[Image Input]"
            # image_parts = [item for item in content if item.get("type") == "image_url"]
            # if image_parts: text_content += f" [Image Input{'s' if len(image_parts) > 1 else ''}]"

        else:
            text_content = str(content) # Ensure it's a string
        return text_content.strip()

    def pipe(self, body: dict) -> str:
        """Main RAG pipeline execution method with tool-based query enhancement."""
        messages_full_history: List[Dict[str, Any]] = body.get("messages", [])
        if not messages_full_history:
            return "Error: No messages provided in the request body."

        # --- 1. Extract History and Last User Query ---
        last_message = messages_full_history[-1]
        if last_message.get("role") != "user":
            # This case might happen with function/tool calls in some UIs, adjust logic if needed
            print("Warning: Last message is not from user. Processing might be affected.")
            # For now, let's try to find the *last* user message further back
            user_query = ""
            for msg in reversed(messages_full_history):
                if msg.get("role") == "user":
                    user_query = self._get_text_content(msg.get("content", ""))
                    break
            if not user_query:
                 return "Error: Could not find a user message in the history."
        else:
            user_query = self._get_text_content(last_message.get("content", ""))


        if not user_query:
            return "Error: Last user message has no text content."

        # Prepare history for query expansion (limit length)
        history_for_expansion = messages_full_history[-(self.valves.HISTORY_LENGTH + 1):-1] # Get messages before the last one

        print(f"\n--- RAG Pipeline Start ---")
        print(f"Original User Query: '{user_query}'")
        print(f"History for Expansion ({len(history_for_expansion)} messages): {history_for_expansion}")


        # --- 2. Call Groq for Query Expansion and Filtering (Tool Call) ---
        print(f"Requesting query expansion/filtering from Groq model: {self.valves.GROQ_MODEL_QUERY_EXPANSION}")
        expansion_messages = [{"role": "system", "content": SYSTEM_PROMPT_QUERY_EXPANSION}]
        expansion_messages.extend(history_for_expansion) # Add historical context
        expansion_messages.append({"role": "user", "content": user_query}) # Add the latest query

        expansion_payload = {
            "model": self.valves.GROQ_MODEL_QUERY_EXPANSION,
            "messages": expansion_messages,
            # Temperature 0 for deterministic tool use? Optional.
            # "temperature": 0.0
        }

        # Call Groq, forcing the tool use
        tool_response = call_groq_chat(
            expansion_payload,
            tools=[MILVUS_HELPER_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "milvus_search_helper"}}
        )

        search_queries = [user_query] # Always include the original query
        milvus_filters = []

        if tool_response["type"] == "tool_call" and tool_response["name"] == "milvus_search_helper":
            generated_args = tool_response.get("arguments", {})
            extra_queries = generated_args.get("extra_queries", [])
            milvus_filters = generated_args.get("filters", []) # Expects a list of strings

            if extra_queries:
                print(f"Groq generated {len(extra_queries)} extra queries: {extra_queries}")
                search_queries.extend(extra_queries) # Add them to the list
            if milvus_filters:
                print(f"Groq generated {len(milvus_filters)} filters: {milvus_filters}")
            else:
                 print("Groq did not generate any specific filters.")

        elif tool_response["type"] == "error":
            print(f"Warning: Groq call for query expansion failed: {tool_response['content']}")
            print("Proceeding with original query and no filters.")
        else:
            # This shouldn't happen if tool_choice forces the tool, but handle defensively
            print(f"Warning: Groq did not return the expected tool call. Response: {tool_response}")
            print("Proceeding with original query and no filters.")


        # --- 3. Perform RAG Search with Enhanced Parameters ---
        print(f"Performing Milvus search with {len(search_queries)} queries and {len(milvus_filters)} filters.")
        rag_results = search_milvus(
            queries=search_queries,
            milvus_host=self.valves.MILVUS_HOST,
            milvus_port=self.valves.MILVUS_PORT,
            milvus_collection_name=self.valves.MILVUS_COLLECTION,
            ollama_url=self.valves.OLLAMA_BASE_URL,
            embedding_model=self.valves.OLLAMA_EMBEDDING_MODEL,
            limit=self.valves.MILVUS_SEARCH_LIMIT,
            filters=milvus_filters, # Pass the extracted filters
        )

        # --- 4. Construct Context for Final Answer Generation ---
        context = ""
        if rag_results:
            context += "Use the following relevant information retrieved from the knowledge base to answer the user's query:\n\n"
            # Basic token counting (approximation)
            current_tokens = 0
            limit_tokens = self.valves.MAX_CONTEXT_TOKENS * 0.6 # Reserve space for prompt/query/answer

            for idx, result in enumerate(rag_results):
                result_text = (
                    f"Source: {result.get('source', 'N/A')}\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    # f"Author: {result.get('author', 'N/A')}\n" # Optional: Include author if useful
                    # f"Chunk Index: {result.get('chunk_index', -1)}\n" # Optional: Include index if useful
                    f"Content: {result.get('content', '')}\n"
                    # f"(Relevance Score: {result.get('score', 0.0):.4f})\n" # Optional: Score might confuse LLM
                    f"---\n"
                )
                result_tokens = len(result_text.split()) # Simple space-based token count

                if current_tokens + result_tokens < limit_tokens:
                    context += result_text
                    current_tokens += result_tokens
                else:
                    print(f"Context limit reached ({current_tokens} tokens), stopping context construction.")
                    break # Stop adding context if near limit

            context += "\n---\nEnd of Retrieved Context.\n\n"
            print(f"Constructed context with ~{current_tokens} tokens from {idx+1 if idx is not None else 0} Milvus results.")
        else:
            print("No relevant context found in Milvus for the query and filters.")
            context = "No specific information was found in the knowledge base regarding the user's query. Answer based on your general knowledge, stating that the knowledge base didn't contain relevant details.\n\n"


        # --- 5. Prepare Messages for Final Answer Generation ---
        # Use history, context, and the original user query
        final_messages = []
        # System prompt for the final answer generation stage
        final_system_prompt = f"You are a helpful assistant. {context.strip()} Answer the following user query based *primarily* on the provided context. If the context is insufficient or doesn't contain the answer, say so clearly and answer based on your general knowledge."
        final_messages.append({"role": "system", "content": final_system_prompt})

        # Add relevant history (excluding the last user query which comes next)
        history_for_answer = messages_full_history[-(self.valves.HISTORY_LENGTH + 1):-1]
        final_messages.extend(history_for_answer)

        # Add the last user query
        final_messages.append({"role": "user", "content": user_query})

        # --- 6. Call Groq for Final Answer Generation ---
        print(f"Requesting final answer from Groq model: {self.valves.GROQ_MODEL}")
        final_payload = {
            "model": self.valves.GROQ_MODEL,
            "messages": final_messages,
            "stream": False,
             # Adjust temperature for creativity vs factualness
             # "temperature": 0.5
        }

        # Call Groq *without* tools for the final text response
        final_response_data = call_groq_chat(final_payload) # No tools needed here

        # --- 7. Return Final Response ---
        if final_response_data["type"] == "text":
            final_response = final_response_data["content"]
            print(f"Groq generated final response: {final_response[:150]}...") # Log snippet
        elif final_response_data["type"] == "error":
            final_response = final_response_data["content"] # Return the error message
            print(f"Error during final Groq call: {final_response}")
        else:
            # Should not happen as we didn't request tools
            final_response = "Error: Received unexpected response type during final answer generation."
            print(final_response)

        print("--- RAG Pipeline End ---")
        return final_response