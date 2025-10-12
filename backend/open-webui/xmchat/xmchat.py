"""
XMChat Agentic Assistant (single-file reference)
- Always-RAG, schema-aware, multi-collection hybrid search with rank fusion
- Tools: plan_queries, search, filter_query (optional)
- Strictly grounded synthesis with visible citations and status events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
from pydantic import BaseModel, ConfigDict, Field
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker


# ---------------- Configuration ----------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://ollama.a1.autobahn.rinconres.com"
)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral-small3.2:latest")

# If using a single Milvus collection, keep this; else set per-collection names.
MILVUS_HOST = os.getenv("MILVUS_HOST", "10.43.210.111")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "xmidas")

# Vector search params
NPROBE = 10
SEARCH_LIMIT = 8
HYBRID_SEARCH_LIMIT = 10
RRF_K = 100
DROP_RATIO = 0.2

# Retrieval budget
FIRST_PASS_K = 24
FINAL_PASS_K = 8
MAX_TOOL_CALLS = 4
REQUEST_TIMEOUT = 300

# Output fields to pull back
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


# ---------------- Logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------- Data Models ----------------


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
    collection: Optional[str] = None  # if you add it to schema


class PlanResult(BaseModel):
    intent: str
    collections: List[str]
    queries: List[str]
    filters: List[str] = Field(default_factory=list)
    need_chats: bool = False
    need_examples: bool = False
    confidence: float = 0.6


# ---------------- Milvus Connection ----------------


def connect_milvus(token: str = "") -> Optional[MilvusClient]:
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    try:
        client = MilvusClient(uri=uri, token=token)
        if not client.has_collection(collection_name=DEFAULT_COLLECTION):
            logging.error(f"Error: Collection '{DEFAULT_COLLECTION}' does not exist.")
            return None
        client.load_collection(collection_name=DEFAULT_COLLECTION)
        logging.info(f"Collection '{DEFAULT_COLLECTION}' loaded.")
        return client
    except Exception as e:
        logging.error(f"Milvus connection error: {e}")
        return None


# ---------------- Embeddings ----------------


def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        resp = client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return resp.get("embedding")
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None


# ---------------- Search Utilities ----------------


def _make_ann_reqs_for_query(
    query: str,
    dense_embedding: Optional[List[float]],
    filters_expr: str,
    limit: int,
) -> List[AnnSearchRequest]:
    reqs: List[AnnSearchRequest] = []
    if dense_embedding:
        reqs.append(
            AnnSearchRequest(
                data=[dense_embedding],
                anns_field="text_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": NPROBE}},
                expr=filters_expr,
                limit=limit,
            )
        )
    # Sparse fields (if present)
    for field in ("text_sparse_embedding", "metadata_sparse_embedding"):
        try:
            reqs.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field=field,
                    param={"drop_ratio_search": DROP_RATIO},
                    expr=filters_expr,
                    limit=limit,
                )
            )
        except Exception as _:
            # Field may not exist; ignore.
            pass
    return reqs


def _filters_to_expr(filters: List[str]) -> str:
    if not filters:
        return ""
    # Expect already valid Milvus boolean syntax
    return " AND ".join(f"({f})" for f in filters if f.strip())


def _rrf_fuse(
    results_by_collection: Dict[str, List[MilvusDocument]],
    k: int = RRF_K,
    top_k: int = FINAL_PASS_K,
) -> List[MilvusDocument]:
    # Reciprocal rank fusion across collections
    scores: Dict[Tuple[str, int], float] = defaultdict(float)
    docs_by_key: Dict[Tuple[str, int], MilvusDocument] = {}

    for coll, docs in results_by_collection.items():
        for rank, d in enumerate(docs, start=1):
            key = (d.source, d.chunk_index)
            docs_by_key[key] = d
            scores[key] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: List[MilvusDocument] = []
    for (src, idx), _ in ranked[:top_k]:
        fused.append(docs_by_key[(src, idx)])
    return fused


def _dedupe_by_source_chunk(docs: List[MilvusDocument]) -> List[MilvusDocument]:
    seen = set()
    out = []
    for d in docs:
        key = (d.source, d.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _extract_snippet(text: str, query_terms: List[str]) -> str:
    if not text:
        return ""
    # Simple heuristic snippet extraction
    lowered = text.lower()
    best_pos = min(
        (lowered.find(t.lower()) for t in query_terms if t),
        default=0,
    )
    window = 300
    start = max(0, best_pos - window // 2)
    end = min(len(text), start + window)
    snippet = text[start:end]
    return snippet.strip()


def document_to_markdown(
    document: MilvusDocument, query_terms: Optional[List[str]] = None
) -> str:
    parts = []
    parts.append(f"Source: `{document.source}` (Chunk: {document.chunk_index})")
    if document.title:
        parts.append(f"Title: {document.title}")
    if isinstance(document.author, list) and document.author:
        parts.append(f"Author: {', '.join(document.author)}")
    if document.date:
        parts.append(f"Date: {document.date}")
    if document.keywords:
        parts.append(f"Keywords: {', '.join(document.keywords[:10])}")
    if document.text:
        snippet = (
            _extract_snippet(document.text, query_terms or [])
            if query_terms
            else document.text[:300]
        )
        parts.append("---")
        parts.append(snippet)
    return "\n".join(parts)


def build_citations(
    documents: List[MilvusDocument], query_terms: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    citations = []
    for doc in documents:
        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [document_to_markdown(doc, query_terms)],
                "metadata": doc.model_dump(exclude={"text", "distance"}),
                "distance": doc.distance,
            }
        )
    return citations


# ---------------- Tools ----------------


class ToolSchemas:
    plan_queries = {
        "type": "function",
        "function": {
            "name": "plan_queries",
            "description": (
                "Plan retrieval: choose collections, expand queries, and propose "
                "optional Milvus filters. Always include at least 2 query variants."
            ),
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User question or task.",
                    }
                },
            },
        },
    }

    search = {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Hybrid semantic search over Milvus. Accepts queries, optional "
                "filters and collections. Returns top results."
            ),
            "parameters": {
                "type": "object",
                "required": ["queries"],
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Target collections, e.g., docs, help, explain, "
                            "code_examples, chats"
                        ),
                        "default": [],
                    },
                    "top_k": {
                        "type": "integer",
                        "default": HYBRID_SEARCH_LIMIT,
                    },
                },
            },
        },
    }

    filter_query = {
        "type": "function",
        "function": {
            "name": "filter_query",
            "description": (
                "Run a Milvus scalar/array filter query (no vector). Use for "
                "title LIKE, year ranges, author contains, etc."
            ),
            "parameters": {
                "type": "object",
                "required": ["filters"],
                "properties": {
                    "filters": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "limit": {"type": "integer", "default": 50},
                },
            },
        },
    }


def _detect_collections_available(
    milvus: MilvusClient,
) -> List[str]:
    # If using single collection, reflect via metadata later.
    try:
        return [DEFAULT_COLLECTION]
    except Exception:
        return [DEFAULT_COLLECTION]


def _collection_filter_expr_from_names(
    names: List[str],
) -> Optional[str]:
    # If you add a 'collection' VARCHAR field in Milvus, use:
    # return f'collection IN ({", ".join(f\'"{n}"\' for n in names)})'
    # With current schema lacking 'collection', return None and rely on 'source'.
    return None


def perform_hybrid_search(
    client: MilvusClient,
    queries: List[str],
    filters: List[str],
    collections: List[str],
    top_k: int,
    first_pass_k: int,
) -> List[MilvusDocument]:
    if not queries:
        return []

    # If you have multiple Milvus collections, iterate here. For now, single.
    target_collections = (
        collections if collections else _detect_collections_available(client)
    )

    results_by_collection: Dict[str, List[MilvusDocument]] = {}

    filter_expr = _filters_to_expr(filters)
    query_terms = [q for q in queries if q]

    for coll in target_collections:
        search_requests: List[AnnSearchRequest] = []
        for q in queries:
            emb = get_embedding(q)
            search_requests.extend(
                _make_ann_reqs_for_query(
                    q, emb, filters_expr=filter_expr, limit=max(top_k, first_pass_k)
                )
            )

        if not search_requests:
            continue

        try:
            result = client.hybrid_search(
                collection_name=DEFAULT_COLLECTION,
                reqs=search_requests,
                ranker=RRFRanker(k=RRF_K),
                output_fields=OUTPUT_FIELDS,
                limit=max(top_k, first_pass_k),
            )
        except Exception as e:
            logging.error(f"Milvus hybrid_search error: {e}")
            continue

        docs: List[MilvusDocument] = [
            MilvusDocument(**hit["entity"], distance=hit.get("distance"))
            for hit in result[0]
        ]
        # Dedupe and take a manageable slice
        docs = _dedupe_by_source_chunk(docs)[: max(top_k, first_pass_k)]
        results_by_collection[coll] = docs

    fused = _rrf_fuse(results_by_collection, k=RRF_K, top_k=top_k)
    return fused


def perform_filter_query(
    client: MilvusClient,
    filters: List[str],
    collections: List[str],
    limit: int,
) -> List[MilvusDocument]:
    target_collections = (
        collections if collections else _detect_collections_available(client)
    )
    expr = _filters_to_expr(filters)
    results: List[MilvusDocument] = []

    for coll in target_collections:
        try:
            rows = client.query(
                collection_name=DEFAULT_COLLECTION,
                filter=expr,
                output_fields=OUTPUT_FIELDS,
                limit=limit,
            )
            results.extend([MilvusDocument(**r) for r in rows])
        except Exception as e:
            logging.error(f"Milvus query error: {e}")
            continue

    return _dedupe_by_source_chunk(results)[:limit]


# ---------------- Streaming Conversion ----------------


def to_openai_chunk(ollama_chunk: ollama.ChatResponse) -> dict:
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


# ---------------- Agent ----------------

SYSTEM_PROMPT = """
You are XMChat, a specialized, document-grounded coding assistant for X‑Midas.
You must only answer using content retrieved from the Milvus collection(s).
If the retrieved context does not contain the answer, say so and suggest
alternative searches. Always cite sources used.

Core rules:
- Document-only responses; no external knowledge
- Cite sources; include file/title and chunk index
- Be concise; code examples only if asked or clearly necessary
- Prefer official docs/help/explain over chats for authoritative answers
- If partially covered, state what remains unknown

Milvus schema reminder (fields may include):
- source (VARCHAR), chunk_index (INT), title (VARCHAR), author (ARRAY<VARCHAR>),
  date (INT64 year), keywords (ARRAY<VARCHAR>), unique_words (ARRAY<VARCHAR>),
  text (VARCHAR), text_embedding (VECTOR), text_sparse_embedding (SPARSE),
  metadata_sparse_embedding (SPARSE)

Filter syntax reminder:
- Use Milvus boolean expressions with ==, !=, >, >=, <, <=, IN, NOT IN, LIKE,
  ARRAY_CONTAINS, ARRAY_CONTAINS_ANY, ARRAY_CONTAINS_ALL, AND, OR, NOT.
- Strings in double quotes. Years are INT64.

When planning:
- Choose collections among: docs, help, explain, code_examples, chats
- Generate 3–6 semantically varied queries targeting primitives, switches,
  file/function names, and synonyms.
- Propose filters if the query mentions title, date, author, or keywords.

Your output must be grounded in the context provided by tools.
"""


class Pipe:  # XMChatAgent
    def __init__(self) -> None:
        self.milvus = connect_milvus(MILVUS_TOKEN)
        self.citations: List[MilvusDocument] = []
        self.ollama = ollama.AsyncClient(host=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT)

    async def emit_status(
        self, emitter, description: str, done: bool = False, hidden: bool = False
    ):
        if not emitter:
            return
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": hidden,
                },
            }
        )

    async def plan_queries(self, messages: List[Dict[str, Any]]) -> PlanResult:
        user_q = messages[-1]["content"]
        sys = {
            "role": "system",
            "content": (
                "You are a retrieval planner. Return JSON with fields: "
                "intent, collections, queries, filters, need_chats, "
                "need_examples, confidence. Collections must be chosen from: "
                '["docs","help","explain","code_examples","chats"]. '
                "Produce at least 3 queries."
            ),
        }
        planner_prompt = {
            "role": "user",
            "content": f"Question: {user_q}\nReturn JSON only.",
        }

        resp = await self.ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[sys, planner_prompt],
            options={"num_ctx": 16000},
        )
        content = resp["message"]["content"].strip()
        try:
            data = json.loads(content)
            return PlanResult(**data)
        except Exception:
            # Safe fallback
            return PlanResult(
                intent="qa",
                collections=["docs", "help", "explain"],
                queries=[user_q, f"What is {user_q} in X-Midas?"],
                filters=[],
                need_chats=False,
                need_examples=True,
                confidence=0.5,
            )

    def run_search_tool(
        self,
        queries: List[str],
        filters: List[str],
        collections: List[str],
        top_k: int,
    ) -> List[MilvusDocument]:
        if not self.milvus:
            return []
        return perform_hybrid_search(
            client=self.milvus,
            queries=queries,
            filters=filters,
            collections=collections,
            top_k=top_k,
            first_pass_k=FIRST_PASS_K,
        )

    def run_filter_query_tool(
        self, filters: List[str], collections: List[str], limit: int
    ) -> List[MilvusDocument]:
        if not self.milvus:
            return []
        return perform_filter_query(
            client=self.milvus,
            filters=filters,
            collections=collections,
            limit=limit,
        )

    async def synthesize_answer(
        self,
        all_messages: List[Dict[str, Any]],
        stream: bool,
        emitter,
    ):
        await self.emit_status(emitter, "Synthesizing answer…")
        chat_stream = await self.ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=all_messages,
            tools=[ToolSchemas.search, ToolSchemas.filter_query],
            options={"num_ctx": 32000},
            stream=True,
        )
        final_content = ""
        async for chunk in chat_stream:
            if stream:
                # Forward token chunks to caller
                yield to_openai_chunk(chunk)
            if content_chunk := chunk.message.content:
                final_content += content_chunk
        if not stream:
            yield {
                "id": str(uuid.uuid4()),
                "object": "chat.completion.final",
                "created": int(time.time()),
                "model": OLLAMA_LLM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

    async def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict | None = None,
    ):
        # Body: { messages: [...], stream: bool }
        stream = bool(body.get("stream", True))
        messages = body.get("messages", [])
        if not messages:
            yield {"error": "No messages provided."}
            return

        if not self.milvus:
            yield {"error": "Unable to connect to the knowledge database."}
            return

        # Build system prompt + user messages
        all_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + messages

        # 1) Plan
        await self.emit_status(__event_emitter__, "Planning queries…")
        plan = await self.plan_queries(messages)
        await self.emit_status(
            __event_emitter__,
            f"Searching: {', '.join(plan.collections)}…",
        )

        # 2) First-pass search
        results = self.run_search_tool(
            queries=plan.queries,
            filters=plan.filters,
            collections=plan.collections,
            top_k=FINAL_PASS_K,
        )

        query_terms = plan.queries[:4]
        preliminary_context = "\n\n".join(
            [document_to_markdown(d, query_terms) for d in results]
        )
        self.citations = results[:]

        # 3) If weak grounding, refine once
        if len(results) < max(3, FINAL_PASS_K // 2):
            await self.emit_status(__event_emitter__, "Refining and broadening search…")
            refined_queries = plan.queries + [
                q + " switches",
                q + " example",
                q + " usage",
            ]
            # Optionally include chats/examples if not already
            refined_colls = list(
                dict.fromkeys(
                    plan.collections
                    + (["code_examples"] if plan.need_examples else [])
                    + (["chats"] if plan.need_chats else [])
                )
            )
            results2 = self.run_search_tool(
                queries=refined_queries,
                filters=plan.filters,
                collections=refined_colls,
                top_k=FINAL_PASS_K,
            )
            self.citations.extend(results2)
            preliminary_context = "\n\n".join(
                [document_to_markdown(d, query_terms) for d in self.citations]
            )

        # 4) Synthesis with grounded context
        context_block = (
            "Retrieved Context:\n\n" + preliminary_context[:12000]
            if preliminary_context
            else "Retrieved Context: (none)"
        )
        all_messages.append({"role": "system", "content": context_block})

        # 5) Stream synthesized answer
        async for out in self.synthesize_answer(
            all_messages=all_messages, stream=stream, emitter=__event_emitter__
        ):
            yield out

        # 6) Emit citations
        for cit in build_citations(self.citations, query_terms):
            if __event_emitter__:
                await __event_emitter__({"type": "citation", "data": cit})

        # 7) Done
        await self.emit_status(__event_emitter__, "Done.", done=True, hidden=True)


# ---------------- Minimal CLI/Test Harness ----------------


async def _dummy_emitter(event: Dict[str, Any]):
    # Replace with your UI event bus
    kind = event.get("type")
    if kind == "status":
        logging.info(f"STATUS: {event['data']['description']}")
    elif kind == "citation":
        logging.info(
            f"CITATION: {event['data']['source']['name']} "
            f"(dist={event['data'].get('distance')})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="+")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    async def main():
        agent = XMChatAgent()
        body = {
            "messages": [{"role": "user", "content": " ".join(args.question)}],
            "stream": not args.no_stream,
        }
        async for piece in agent.pipe(body, __event_emitter__=_dummy_emitter):
            if piece.get("object") == "chat.completion.chunk":
                print(piece["choices"][0]["delta"]["content"], end="", flush=True)
            elif piece.get("object") == "chat.completion.final":
                print(piece["choices"][0]["message"]["content"], flush=True)

    asyncio.run(main())
