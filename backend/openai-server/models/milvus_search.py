"""
title: Radchat Function
author: Rincon [mjst, kca]
author_url:
funding_url:
version: 0.1
requirements: pymilvus
"""

# TODO: Figure out if authors are stored as list or str.

from http.client import REQUEST_TIMEOUT
import os
import uuid
import time
import re
import ollama
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pymilvus import (
    MilvusClient,
    MilvusException,
    AnnSearchRequest,
    RRFRanker,
)

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "all-minilm:v2"
OLLAMA_LLM_MODEL = "qwen3:1.7b"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "test_collection"
OUTPUT_FIELDS = ["source", "chunk_index", "metadata", "title", "author", "date", "keywords", "unique_words", "text"]
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MAX_TOOL_CALLS = 5
REQUEST_TIMEOUT = 300

# --- Pydantic Models for Type Safety ---
class MilvusDocument(BaseModel):
    source: str
    chunk_index: int
    metadata: Optional[str] = ""
    title: Optional[str] = ""
    author: Optional[List[str]| str] = []
    date: Optional[int] = 0
    keywords: Optional[List[str]] = []
    unique_words: Optional[List[str]] = []
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
            print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
            return None
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded.")
        return client
    except Exception as e:
        print(f"Error connecting to or loading Milvus collection '{COLLECTION_NAME}': {e}")
        return None

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Error getting embedding from Ollama ({OLLAMA_BASE_URL}): {e}")
        return None

def perform_search(client: MilvusClient, queries: list[str], filters: list[str] = []) -> list[MilvusDocument]:
    search_requests = []
    for query in queries:
        search_requests.append(AnnSearchRequest(
            data=[get_embedding(query)],
            anns_field="text_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=" and ".join(filters),
            limit=10
        ))
        search_requests.append(AnnSearchRequest(
            data=[query],
            anns_field="text_sparse_embedding",
            param={"drop_ratio_search": 0.2},
            expr=" and ".join(filters),
            limit=10
        ))
        search_requests.append(AnnSearchRequest(
            data=[query],
            anns_field="metadata_sparse_embedding",
            param={"drop_ratio_search": 0.2},
            expr=" and ".join(filters),
            limit=10
        ))

    result = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=search_requests,
        ranker=RRFRanker(k=100),
        output_fields=OUTPUT_FIELDS,
        limit=100,
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
        parts.append(f"**Authors:** {', '.join(document.author)}")
    if document.date:
        parts.append(f"**Date:** {document.date}")
    if document.source:
        parts.append(f"**Source:** `{document.source}` (Chunk: {document.chunk_index})")
    if document.keywords:
        parts.append(f"**Keywords:** `{'`, `'.join(document.keywords)}`")
    if document.text:
        parts.append("\n---\n" + document.text)
    return "\n".join(parts)

def build_citations(documents: List[MilvusDocument]) -> list:
    doc_map = {doc.source: doc for doc in sorted(documents, key=lambda d: d.chunk_index)}
    return [
        {
            "source": {"name": doc.source, "url": ""},
            "document": [d.text for d in documents if d.source == source],
            "metadata": doc.model_dump(exclude={'text'}),
            "distance": doc.distance
        }
        for source, doc in doc_map.items()
    ]

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
            "delta": message.content,
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
    
    system_prompt = SystemPrompt.replace(
        "<<database_schema>>", str(milvus_client.describe_collection(COLLECTION_NAME).get("fields", {}))
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

    for _ in range(MAX_TOOL_CALLS):
        stream = await ollama_client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=all_messages,
            tools=[SearchInputSchema, QueryInputSchema],
            stream=True
        )
        
        tool_calls = []
        async for chunk in stream:
            yield to_openai_chunk(chunk)
            if new_tool_calls := chunk.get("message", {}).get("tool_calls"):
                print(f"Tool calls: {new_tool_calls}")
                tool_calls.extend(new_tool_calls)
            if content_chunk := chunk.get("message", {}).get("content"):
                final_content += content_chunk

        if not tool_calls:
            break

        all_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for tool in tool_calls:
            function_name = tool.get('function', {}).get('name')
            function_args = tool.get('function', {}).get('arguments', {})
            
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
You are a specialized AI assistant...
"""