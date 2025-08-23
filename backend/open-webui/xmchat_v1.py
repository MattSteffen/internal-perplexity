"""
X-Midas Coding Assistant - Agentic RAG System
- Proprietary language support with no external LLM knowledge
- Advanced code-aware RAG with multi-stage retrieval
- Specialized tools for code understanding, debugging, and generation
- Multi-turn conversation with context preservation
- Hybrid search with code-specific ranking and filtering
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

# Using a single Milvus collection with partitions for different document types
MILVUS_HOST = os.getenv("MILVUS_HOST", "10.43.210.111")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "xmidas")

# Available partitions within the xmidas collection (empty = search all partitions)
DEFAULT_PARTITIONS = [
    "docs",
    "help",
    "explain",
    "code_examples",
    "chats",
    "api_reference",
    "tutorials",
]

# Vector search params - optimized for code
NPROBE = 10
SEARCH_LIMIT = 8
HYBRID_SEARCH_LIMIT = 12  # Increased for code diversity
RRF_K = 100
DROP_RATIO = 0.2

# Code-specific retrieval settings
CODE_FIRST_PASS_K = 32  # More results for code queries
CODE_FINAL_K = 12  # More context for code understanding
API_RELATION_K = 8  # Related APIs/functions
EXAMPLE_K = 6  # Code examples
DEBUG_K = 8  # Debug-related content

# Retrieval budget
FIRST_PASS_K = 24
FINAL_PASS_K = 8
MAX_TOOL_CALLS = 6  # Increased for more sophisticated tool chains
MAX_AGENT_ITERATIONS = 3  # Multi-turn agent reasoning
REQUEST_TIMEOUT = 300

# Code understanding settings
MAX_CODE_CHUNK_SIZE = 2000  # Larger chunks for code context
MIN_CODE_SIMILARITY = 0.7  # Threshold for code similarity matching
CONTEXT_WINDOW_SIZE = 16000  # Larger context for code reasoning

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
    partition: Optional[str] = None  # if you add it to schema


class PlanResult(BaseModel):
    intent: str
    partitions: List[str]
    queries: List[str]
    filters: List[str] = Field(default_factory=list)
    need_chats: bool = False
    need_examples: bool = False
    need_api_relations: bool = False
    need_debug_info: bool = False
    need_similar_code: bool = False
    confidence: float = 0.6
    query_type: str = (
        "general"  # general, api_lookup, code_generation, debugging, explanation
    )


class CodePattern(BaseModel):
    pattern_type: str  # function_call, class_definition, import_statement, etc.
    pattern_code: str
    context: str
    similarity_score: float
    source_file: str


class APIRelationship(BaseModel):
    function_name: str
    related_functions: List[str]
    dependencies: List[str]
    usage_examples: List[str]
    documentation: str


class DebugContext(BaseModel):
    error_pattern: str
    likely_causes: List[str]
    solutions: List[str]
    prevention_tips: List[str]
    related_issues: List[str]


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
    # Enhanced tool schemas for X-Midas coding assistant

    plan_queries = {
        "type": "function",
        "function": {
            "name": "plan_queries",
            "description": (
                "Analyze user question and plan comprehensive retrieval strategy. "
                "Identify query type, generate multiple search variations, and determine "
                "which tools and collections to use for the best results."
            ),
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User question or coding task.",
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
                "Hybrid semantic search over Milvus with code-optimized ranking. "
                "Best for finding documentation, examples, and general information."
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
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target partitions within xmidas collection: docs, help, explain, code_examples, chats, api_reference, tutorials",
                        "default": [],
                    },
                    "top_k": {
                        "type": "integer",
                        "default": CODE_FINAL_K,
                    },
                },
            },
        },
    }

    find_similar_code = {
        "type": "function",
        "function": {
            "name": "find_similar_code",
            "description": (
                "Find similar code patterns, functions, or implementations. "
                "Use for code generation, refactoring, or understanding usage patterns."
            ),
            "parameters": {
                "type": "object",
                "required": ["code_pattern"],
                "properties": {
                    "code_pattern": {
                        "type": "string",
                        "description": "Code snippet or pattern to find similar examples for.",
                    },
                    "pattern_type": {
                        "type": "string",
                        "enum": [
                            "function",
                            "class",
                            "method",
                            "statement",
                            "expression",
                            "general",
                        ],
                        "default": "general",
                    },
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["code_examples"],
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": MIN_CODE_SIMILARITY,
                    },
                },
            },
        },
    }

    find_api_relations = {
        "type": "function",
        "function": {
            "name": "find_api_relations",
            "description": (
                "Find related functions, classes, and dependencies for a given API. "
                "Shows how functions work together and their relationships."
            ),
            "parameters": {
                "type": "object",
                "required": ["function_name"],
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of function, class, or API to find relations for.",
                    },
                    "include_dependencies": {
                        "type": "boolean",
                        "default": True,
                    },
                    "include_examples": {
                        "type": "boolean",
                        "default": True,
                    },
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["docs", "api_reference", "code_examples"],
                    },
                },
            },
        },
    }

    debug_helper = {
        "type": "function",
        "function": {
            "name": "debug_helper",
            "description": (
                "Help debug errors by finding similar issues, common solutions, "
                "and prevention tips from the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "required": ["error_description"],
                "properties": {
                    "error_description": {
                        "type": "string",
                        "description": "Error message, unexpected behavior, or problem description.",
                    },
                    "include_similar_issues": {
                        "type": "boolean",
                        "default": True,
                    },
                    "include_prevention": {
                        "type": "boolean",
                        "default": True,
                    },
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["help", "chats", "troubleshooting"],
                    },
                },
            },
        },
    }

    explain_concept = {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": (
                "Explain X-Midas language concepts, syntax, or features with "
                "examples and detailed documentation."
            ),
            "parameters": {
                "type": "object",
                "required": ["concept"],
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept, keyword, or language feature to explain.",
                    },
                    "include_examples": {
                        "type": "boolean",
                        "default": True,
                    },
                    "difficulty_level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "default": "intermediate",
                    },
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["docs", "help", "explain", "tutorials"],
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
                "Run precise scalar/array filter queries. Use for finding specific "
                "files, authors, date ranges, or keyword matches."
            ),
            "parameters": {
                "type": "object",
                "required": ["filters"],
                "properties": {
                    "filters": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "partitions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                    },
                },
            },
        },
    }


def perform_hybrid_search(
    client: MilvusClient,
    queries: List[str],
    filters: List[str],
    partitions: List[str],
    top_k: int,
    first_pass_k: int,
) -> List[MilvusDocument]:
    if not queries:
        return []

    # Use partitions within the single xmidas collection
    # If no partitions specified, search all partitions (empty partition_name)
    target_partitions = partitions if partitions else []
    partition_name = target_partitions[0] if len(target_partitions) == 1 else ""

    # For multiple partitions, we'll need to make separate searches and combine results
    if len(target_partitions) > 1:
        # Search each partition separately and combine results
        all_results = []
        for partition in target_partitions:
            try:
                result = client.hybrid_search(
                    collection_name=DEFAULT_COLLECTION,
                    partition_names=[
                        partition
                    ],  # Milvus expects partition_names as list
                    reqs=_build_search_requests(queries, filters, top_k, first_pass_k),
                    ranker=RRFRanker(k=RRF_K),
                    output_fields=OUTPUT_FIELDS,
                    limit=max(top_k, first_pass_k),
                )

                docs: List[MilvusDocument] = [
                    MilvusDocument(**hit["entity"], distance=hit.get("distance"))
                    for hit in result[0]
                ]
                all_results.extend(docs)
            except Exception as e:
                logging.error(
                    f"Milvus hybrid_search error for partition {partition}: {e}"
                )
                continue

        # Dedupe and take final slice
        all_results = _dedupe_by_source_chunk(all_results)
        return _rrf_fuse({"combined": all_results}, k=RRF_K, top_k=top_k)
    else:
        # Single partition or all partitions (empty partition_name)
        try:
            result = client.hybrid_search(
                collection_name=DEFAULT_COLLECTION,
                partition_names=(
                    [partition_name] if partition_name else []
                ),  # Empty list = all partitions
                reqs=_build_search_requests(queries, filters, top_k, first_pass_k),
                ranker=RRFRanker(k=RRF_K),
                output_fields=OUTPUT_FIELDS,
                limit=max(top_k, first_pass_k),
            )

            docs: List[MilvusDocument] = [
                MilvusDocument(**hit["entity"], distance=hit.get("distance"))
                for hit in result[0]
            ]
            return _dedupe_by_source_chunk(docs)[:top_k]
        except Exception as e:
            logging.error(f"Milvus hybrid_search error: {e}")
            return []


def _build_search_requests(
    queries: List[str], filters: List[str], top_k: int, first_pass_k: int
) -> List[AnnSearchRequest]:
    """Build search requests for hybrid search."""
    search_requests: List[AnnSearchRequest] = []
    filter_expr = _filters_to_expr(filters)

    for q in queries:
        emb = get_embedding(q)
        search_requests.extend(
            _make_ann_reqs_for_query(
                q, emb, filters_expr=filter_expr, limit=max(top_k, first_pass_k)
            )
        )

    return search_requests


def perform_filter_query(
    client: MilvusClient,
    filters: List[str],
    partitions: List[str],
    limit: int,
) -> List[MilvusDocument]:
    # Use partitions within the single xmidas collection
    target_partitions = partitions if partitions else []
    expr = _filters_to_expr(filters)
    results: List[MilvusDocument] = []

    if len(target_partitions) > 1:
        # Query each partition separately
        for partition in target_partitions:
            try:
                rows = client.query(
                    collection_name=DEFAULT_COLLECTION,
                    partition_names=[partition],
                    filter=expr,
                    output_fields=OUTPUT_FIELDS,
                    limit=limit,
                )
                results.extend([MilvusDocument(**r) for r in rows])
            except Exception as e:
                logging.error(f"Milvus query error for partition {partition}: {e}")
                continue
    else:
        # Single partition or all partitions (empty partition_names)
        partition_names = [target_partitions[0]] if target_partitions else []
        try:
            rows = client.query(
                collection_name=DEFAULT_COLLECTION,
                partition_names=partition_names,  # Empty list = all partitions
                filter=expr,
                output_fields=OUTPUT_FIELDS,
                limit=limit,
            )
            results.extend([MilvusDocument(**r) for r in rows])
        except Exception as e:
            logging.error(f"Milvus query error: {e}")

    return _dedupe_by_source_chunk(results)[:limit]


# ---------------- New Tool Implementations ----------------


def find_similar_code_tool(
    client: MilvusClient,
    code_pattern: str,
    pattern_type: str = "general",
    partitions: List[str] = None,
    min_similarity: float = MIN_CODE_SIMILARITY,
) -> List[CodePattern]:
    """Find similar code patterns using semantic search."""
    if not client:
        return []

    # Create specialized queries for code similarity
    queries = [code_pattern]

    if pattern_type == "function":
        queries.extend(
            [
                f"function {code_pattern}",
                f"def {code_pattern}",
                f"function definition {code_pattern}",
            ]
        )
    elif pattern_type == "class":
        queries.extend(
            [
                f"class {code_pattern}",
                f"class definition {code_pattern}",
                f"object {code_pattern}",
            ]
        )

    results = perform_hybrid_search(
        client=client,
        queries=queries,
        filters=[],
        partitions=partitions or ["code_examples"],
        top_k=EXAMPLE_K,
        first_pass_k=CODE_FIRST_PASS_K,
    )

    # Filter by similarity threshold and format as CodePattern
    code_patterns = []
    for doc in results:
        if doc.distance and doc.distance >= min_similarity:
            pattern = CodePattern(
                pattern_type=pattern_type,
                pattern_code=doc.text[:MAX_CODE_CHUNK_SIZE],
                context=f"Source: {doc.source}, Chunk: {doc.chunk_index}",
                similarity_score=1.0 - doc.distance,  # Convert distance to similarity
                source_file=doc.source,
            )
            code_patterns.append(pattern)

    return code_patterns[:EXAMPLE_K]


def find_api_relations_tool(
    client: MilvusClient,
    function_name: str,
    include_dependencies: bool = True,
    include_examples: bool = True,
    partitions: List[str] = None,
) -> APIRelationship:
    """Find API relationships and related functions."""
    if not client:
        return APIRelationship(
            function_name=function_name,
            related_functions=[],
            dependencies=[],
            usage_examples=[],
            documentation="",
        )

    # Search for the function and related content
    queries = [
        f"function {function_name}",
        f"API {function_name}",
        f"{function_name} usage",
        f"{function_name} parameters",
        f"{function_name} return value",
    ]

    if include_dependencies:
        queries.append(f"{function_name} dependencies")
    if include_examples:
        queries.append(f"{function_name} example")

    results = perform_hybrid_search(
        client=client,
        queries=queries,
        filters=[],
        partitions=partitions or ["docs", "api_reference", "code_examples"],
        top_k=API_RELATION_K,
        first_pass_k=CODE_FIRST_PASS_K,
    )

    # Extract relationships from results
    related_functions = []
    dependencies = []
    usage_examples = []
    documentation_parts = []

    for doc in results:
        text = doc.text.lower()
        doc_text = doc.text

        # Look for function mentions
        if "function " in text or "def " in text:
            # Extract function names (simplified regex-free approach)
            words = doc_text.split()
            for i, word in enumerate(words):
                if word in ["function", "def"] and i + 1 < len(words):
                    func_name = words[i + 1].strip("(),")
                    if func_name and func_name != function_name:
                        related_functions.append(func_name)

        # Look for import/dependency statements
        if "import " in text or "from " in text:
            lines = doc_text.split("\n")
            for line in lines:
                if line.strip().startswith(("import ", "from ")):
                    dependencies.append(line.strip())

        # Look for examples
        if "example" in text or "usage" in text:
            usage_examples.append(doc_text[:500])

        # Collect documentation
        documentation_parts.append(doc_text[:1000])

    return APIRelationship(
        function_name=function_name,
        related_functions=list(set(related_functions))[:10],
        dependencies=list(set(dependencies))[:10],
        usage_examples=usage_examples[:5],
        documentation=" ".join(documentation_parts)[:2000],
    )


def debug_helper_tool(
    client: MilvusClient,
    error_description: str,
    include_similar_issues: bool = True,
    include_prevention: bool = True,
    partitions: List[str] = None,
) -> DebugContext:
    """Help debug errors by finding similar issues and solutions."""
    if not client:
        return DebugContext(
            error_pattern=error_description,
            likely_causes=[],
            solutions=[],
            prevention_tips=[],
            related_issues=[],
        )

    queries = [
        f"error {error_description}",
        f"problem {error_description}",
        f"issue {error_description}",
        f"debug {error_description}",
        f"fix {error_description}",
    ]

    if include_similar_issues:
        queries.append(f"similar errors to {error_description}")
    if include_prevention:
        queries.append(f"prevent {error_description}")

    results = perform_hybrid_search(
        client=client,
        queries=queries,
        filters=[],
        partitions=partitions or ["help", "chats", "troubleshooting"],
        top_k=DEBUG_K,
        first_pass_k=FIRST_PASS_K,
    )

    # Extract debug information
    likely_causes = []
    solutions = []
    prevention_tips = []
    related_issues = []

    for doc in results:
        text = doc.text.lower()
        doc_text = doc.text

        if "cause" in text or "reason" in text:
            likely_causes.append(doc_text[:300])
        if "solution" in text or "fix" in text or "resolve" in text:
            solutions.append(doc_text[:300])
        if "prevent" in text or "avoid" in text:
            prevention_tips.append(doc_text[:300])

        # Related issues are the source documents
        related_issues.append(f"{doc.source} (chunk {doc.chunk_index})")

    return DebugContext(
        error_pattern=error_description,
        likely_causes=likely_causes[:5],
        solutions=solutions[:5],
        prevention_tips=prevention_tips[:5],
        related_issues=related_issues[:5],
    )


def explain_concept_tool(
    client: MilvusClient,
    concept: str,
    include_examples: bool = True,
    difficulty_level: str = "intermediate",
    partitions: List[str] = None,
) -> List[MilvusDocument]:
    """Explain X-Midas concepts with examples and documentation."""
    if not client:
        return []

    queries = [
        f"explain {concept}",
        f"what is {concept}",
        f"{concept} definition",
        f"{concept} syntax",
        f"{concept} usage",
    ]

    if include_examples:
        queries.extend(
            [
                f"{concept} example",
                f"how to use {concept}",
                f"{concept} tutorial",
            ]
        )

    # Add difficulty-specific queries
    if difficulty_level == "beginner":
        queries.append(f"{concept} for beginners")
    elif difficulty_level == "advanced":
        queries.append(f"advanced {concept}")

    return perform_hybrid_search(
        client=client,
        queries=queries,
        filters=[],
        partitions=partitions or ["docs", "help", "explain", "tutorials"],
        top_k=CODE_FINAL_K,
        first_pass_k=CODE_FIRST_PASS_K,
    )


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
You are X-Midas Assistant, a specialized coding assistant for the proprietary X-Midas programming language.
You must ONLY use information retrieved from the knowledge base - no external programming knowledge.
Always cite sources with file names and chunk indices.

CORE PRINCIPLES:
- Document-only responses; never use external knowledge about programming
- Always cite sources; include source file and chunk index
- Provide complete, working code examples when requested
- Prioritize official documentation over user discussions
- If information is missing, suggest specific search terms to find it

KNOWLEDGE BASE SCHEMA:
- source (VARCHAR): File or document name
- chunk_index (INT): Position in the document
- title (VARCHAR): Document or section title
- author (ARRAY<VARCHAR>): Content authors
- date (INT64): Year of content
- keywords (ARRAY<VARCHAR>): Topic keywords
- unique_words (ARRAY<VARCHAR>): Important terms
- text (VARCHAR): Actual content
- Plus vector embeddings for semantic search

QUERY TYPES YOU HANDLE:
1. API_LOOKUP: Find function/class definitions, parameters, return types
2. CODE_GENERATION: Generate code examples, templates, boilerplate
3. DEBUGGING: Help solve errors, understand behavior, find common issues
4. EXPLANATION: Explain concepts, syntax, language features
5. GENERAL: Other coding questions and discussions

SEARCH STRATEGY:
- Generate 4-6 varied queries for different aspects of the question
- Use both specific terms and synonyms
- Include error messages, function names, class names exactly as written
- Search across docs, help, explain, code_examples, chats, api_reference, and tutorials partitions
- Use filters for specific file types, years, or authors when relevant

RESPONSE FORMAT:
- Start with direct answer or solution
- Include relevant code examples from knowledge base
- Cite all sources used
- Suggest related searches if answer is incomplete
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
        user_q = messages[-1]["content"].lower()

        # Intelligent query type detection
        query_type = "general"
        if any(
            keyword in user_q
            for keyword in ["function", "method", "api", "class", "object", "def "]
        ):
            query_type = "api_lookup"
        elif any(
            keyword in user_q
            for keyword in [
                "error",
                "problem",
                "issue",
                "bug",
                "fail",
                "crash",
                "debug",
            ]
        ):
            query_type = "debugging"
        elif any(
            keyword in user_q
            for keyword in [
                "example",
                "sample",
                "template",
                "how to",
                "write",
                "create",
            ]
        ):
            query_type = "code_generation"
        elif any(
            keyword in user_q
            for keyword in ["explain", "what is", "how does", "understand", "learn"]
        ):
            query_type = "explanation"

        sys = {
            "role": "system",
            "content": (
                "You are an intelligent query planner for X-Midas coding assistant. "
                "Analyze the user's question and create an optimal retrieval strategy.\n\n"
                "Return JSON with fields:\n"
                "- intent: brief description of what user wants\n"
                "- partitions: list from ['docs','help','explain','code_examples','chats','api_reference','tutorials']\n"
                "- queries: 4-6 varied search queries\n"
                "- filters: Milvus filter expressions if specific criteria needed\n"
                "- need_chats: true if user discussions might help\n"
                "- need_examples: true if code examples are relevant\n"
                "- need_api_relations: true if function/class relationships needed\n"
                "- need_debug_info: true if debugging help is relevant\n"
                "- need_similar_code: true if similar code patterns would help\n"
                "- confidence: 0.0-1.0 based on clarity of request\n"
                f"- query_type: detected type from {query_type}\n\n"
                "Optimize for the proprietary X-Midas language - focus on exact terminology and patterns."
            ),
        }

        planner_prompt = {
            "role": "user",
            "content": f"Question: {messages[-1]['content']}\n\nDetected query type: {query_type}\n\nReturn JSON only.",
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
        except Exception as e:
            logging.error(f"Planning failed: {e}, using fallback")
            # Smart fallback based on detected query type
            fallback_configs = {
                "api_lookup": PlanResult(
                    intent="API lookup",
                    partitions=["docs", "api_reference", "code_examples"],
                    queries=[
                        messages[-1]["content"],
                        f"{messages[-1]['content']} function",
                        f"{messages[-1]['content']} usage",
                    ],
                    filters=[],
                    need_chats=False,
                    need_examples=True,
                    need_api_relations=True,
                    need_debug_info=False,
                    need_similar_code=False,
                    confidence=0.8,
                    query_type="api_lookup",
                ),
                "debugging": PlanResult(
                    intent="Debug help",
                    partitions=["help", "chats", "troubleshooting"],
                    queries=[
                        messages[-1]["content"],
                        f"fix {messages[-1]['content']}",
                        f"solve {messages[-1]['content']}",
                    ],
                    filters=[],
                    need_chats=True,
                    need_examples=True,
                    need_api_relations=False,
                    need_debug_info=True,
                    need_similar_code=False,
                    confidence=0.7,
                    query_type="debugging",
                ),
                "code_generation": PlanResult(
                    intent="Code generation",
                    partitions=["code_examples", "docs", "tutorials"],
                    queries=[
                        messages[-1]["content"],
                        f"{messages[-1]['content']} example",
                        f"how to {messages[-1]['content']}",
                    ],
                    filters=[],
                    need_chats=False,
                    need_examples=True,
                    need_api_relations=False,
                    need_debug_info=False,
                    need_similar_code=True,
                    confidence=0.6,
                    query_type="code_generation",
                ),
                "explanation": PlanResult(
                    intent="Explanation",
                    partitions=["docs", "help", "explain", "tutorials"],
                    queries=[
                        messages[-1]["content"],
                        f"explain {messages[-1]['content']}",
                        f"{messages[-1]['content']} tutorial",
                    ],
                    filters=[],
                    need_chats=False,
                    need_examples=True,
                    need_api_relations=False,
                    need_debug_info=False,
                    need_similar_code=False,
                    confidence=0.8,
                    query_type="explanation",
                ),
            }

            return fallback_configs.get(
                query_type,
                PlanResult(
                    intent="General question",
                    partitions=["docs", "help", "explain"],
                    queries=[
                        messages[-1]["content"],
                        f"What is {messages[-1]['content']} in X-Midas?",
                    ],
                    filters=[],
                    need_chats=False,
                    need_examples=True,
                    need_api_relations=False,
                    need_debug_info=False,
                    need_similar_code=False,
                    confidence=0.5,
                    query_type="general",
                ),
            )

    def run_search_tool(
        self,
        queries: List[str],
        filters: List[str],
        partitions: List[str],
        top_k: int,
    ) -> List[MilvusDocument]:
        if not self.milvus:
            return []
        return perform_hybrid_search(
            client=self.milvus,
            queries=queries,
            filters=filters,
            partitions=partitions,
            top_k=top_k,
            first_pass_k=FIRST_PASS_K,
        )

    def run_filter_query_tool(
        self, filters: List[str], partitions: List[str], limit: int
    ) -> List[MilvusDocument]:
        if not self.milvus:
            return []
        return perform_filter_query(
            client=self.milvus,
            filters=filters,
            partitions=partitions,
            limit=limit,
        )

    def run_find_similar_code_tool(
        self,
        code_pattern: str,
        pattern_type: str = "general",
        collections: List[str] = None,
        min_similarity: float = MIN_CODE_SIMILARITY,
    ) -> List[CodePattern]:
        if not self.milvus:
            return []
        return find_similar_code_tool(
            client=self.milvus,
            code_pattern=code_pattern,
            pattern_type=pattern_type,
            collections=collections,
            min_similarity=min_similarity,
        )

    def run_find_api_relations_tool(
        self,
        function_name: str,
        include_dependencies: bool = True,
        include_examples: bool = True,
        collections: List[str] = None,
    ) -> APIRelationship:
        if not self.milvus:
            return APIRelationship(
                function_name=function_name,
                related_functions=[],
                dependencies=[],
                usage_examples=[],
                documentation="",
            )
        return find_api_relations_tool(
            client=self.milvus,
            function_name=function_name,
            include_dependencies=include_dependencies,
            include_examples=include_examples,
            collections=collections,
        )

    def run_debug_helper_tool(
        self,
        error_description: str,
        include_similar_issues: bool = True,
        include_prevention: bool = True,
        collections: List[str] = None,
    ) -> DebugContext:
        if not self.milvus:
            return DebugContext(
                error_pattern=error_description,
                likely_causes=[],
                solutions=[],
                prevention_tips=[],
                related_issues=[],
            )
        return debug_helper_tool(
            client=self.milvus,
            error_description=error_description,
            include_similar_issues=include_similar_issues,
            include_prevention=include_prevention,
            collections=collections,
        )

    def run_explain_concept_tool(
        self,
        concept: str,
        include_examples: bool = True,
        difficulty_level: str = "intermediate",
        collections: List[str] = None,
    ) -> List[MilvusDocument]:
        if not self.milvus:
            return []
        return explain_concept_tool(
            client=self.milvus,
            concept=concept,
            include_examples=include_examples,
            difficulty_level=difficulty_level,
            collections=collections,
        )

    def _extract_function_names(self, text: str) -> List[str]:
        """Extract function names from user query using simple heuristics."""
        import re

        # Look for common function patterns
        patterns = [
            r"\bfunction\s+(\w+)",
            r"\bdef\s+(\w+)",
            r"\b(\w+)\s*\([^)]*\)",  # function calls
            r"\b(\w+)\s*\([^}]*{",  # function definitions
        ]

        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            functions.extend(matches)

        # Also look for common X-Midas API keywords
        x_midas_keywords = ["process", "transform", "analyze", "convert", "validate"]
        words = text.lower().split()
        for word in words:
            if word in x_midas_keywords or ("_" in word and len(word) > 3):
                functions.append(word)

        return list(set(functions))[:3]  # Limit and deduplicate

    def _extract_code_patterns(self, text: str) -> List[str]:
        """Extract code patterns from user query."""
        import re

        # Look for code-like patterns
        patterns = [
            r"`([^`]+)`",  # Backtick code blocks
            r"```[\s\S]*?```",  # Triple backtick code blocks
            r"\b\w+\s*\([^)]*\)",  # Function calls
            r"\b\w+\.\w+\s*\([^)]*\)",  # Method calls
        ]

        code_patterns = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            code_patterns.extend([m.strip("`") for m in matches])

        # If no patterns found, return the whole query as potential code
        if not code_patterns and len(text) < 200:
            code_patterns = [text]

        return code_patterns[:2]  # Limit to 2 patterns

    async def synthesize_answer(
        self,
        all_messages: List[Dict[str, Any]],
        stream: bool,
        emitter,
        plan: PlanResult,
    ):
        await self.emit_status(emitter, "Synthesizing answer…")

        # Select appropriate tools based on query type
        available_tools = [ToolSchemas.search, ToolSchemas.filter_query]

        if plan.query_type == "api_lookup":
            available_tools.extend([ToolSchemas.find_api_relations])
        elif plan.query_type == "debugging":
            available_tools.extend([ToolSchemas.debug_helper])
        elif plan.query_type == "code_generation":
            available_tools.extend([ToolSchemas.find_similar_code])
        elif plan.query_type == "explanation":
            available_tools.extend([ToolSchemas.explain_concept])

        # Add general-purpose tools
        if plan.need_similar_code:
            available_tools.append(ToolSchemas.find_similar_code)
        if plan.need_debug_info:
            available_tools.append(ToolSchemas.debug_helper)
        if plan.need_api_relations:
            available_tools.append(ToolSchemas.find_api_relations)

        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in available_tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        chat_stream = await self.ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=all_messages,
            tools=unique_tools,
            options={"num_ctx": CONTEXT_WINDOW_SIZE},
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
            f"Searching: {', '.join(plan.partitions)}…",
        )

        # 2) Execute specialized tools based on plan
        all_results = []
        tool_contexts = []

        # Always start with basic search
        await self.emit_status(
            __event_emitter__,
            f"Searching {len(plan.queries)} queries across {len(plan.partitions)} partitions…",
        )
        search_results = self.run_search_tool(
            queries=plan.queries,
            filters=plan.filters,
            partitions=plan.partitions,
            top_k=CODE_FINAL_K,
        )
        all_results.extend(search_results)

        # Execute specialized tools based on query type and needs
        tool_calls = 0

        # API Relations Tool
        if plan.need_api_relations and tool_calls < MAX_TOOL_CALLS:
            await self.emit_status(__event_emitter__, "Finding API relationships…")
            # Extract function names from the query
            function_names = self._extract_function_names(messages[-1]["content"])
            for func_name in function_names[:2]:  # Limit to 2 functions
                api_relations = self.run_find_api_relations_tool(
                    function_name=func_name,
                    include_dependencies=True,
                    include_examples=True,
                    partitions=plan.partitions,
                )
                if api_relations.related_functions or api_relations.documentation:
                    tool_contexts.append(
                        f"API Relations for {func_name}:\n{api_relations.model_dump_json()}"
                    )
                    tool_calls += 1

        # Debug Helper Tool
        if plan.need_debug_info and tool_calls < MAX_TOOL_CALLS:
            await self.emit_status(__event_emitter__, "Analyzing error patterns…")
            debug_context = self.run_debug_helper_tool(
                error_description=messages[-1]["content"],
                include_similar_issues=True,
                include_prevention=True,
                partitions=plan.partitions,
            )
            if debug_context.likely_causes or debug_context.solutions:
                tool_contexts.append(
                    f"Debug Analysis:\n{debug_context.model_dump_json()}"
                )
                tool_calls += 1

        # Similar Code Tool
        if plan.need_similar_code and tool_calls < MAX_TOOL_CALLS:
            await self.emit_status(__event_emitter__, "Finding similar code patterns…")
            code_patterns = self._extract_code_patterns(messages[-1]["content"])
            for pattern in code_patterns[:1]:  # Limit to 1 pattern
                similar_code = self.run_find_similar_code_tool(
                    code_pattern=pattern,
                    pattern_type="general",
                    partitions=["code_examples"],
                    min_similarity=MIN_CODE_SIMILARITY,
                )
                if similar_code:
                    tool_contexts.append(
                        f"Similar Code Patterns:\n{json.dumps([p.model_dump() for p in similar_code])}"
                    )
                    tool_calls += 1

        # Concept Explanation Tool
        if plan.query_type == "explanation" and tool_calls < MAX_TOOL_CALLS:
            await self.emit_status(
                __event_emitter__, "Gathering comprehensive explanation…"
            )
            concept_results = self.run_explain_concept_tool(
                concept=messages[-1]["content"],
                include_examples=True,
                difficulty_level="intermediate",
                partitions=plan.partitions,
            )
            if concept_results:
                all_results.extend(concept_results)
                tool_calls += 1

        # 3) Refine search if needed
        if len(all_results) < max(3, CODE_FINAL_K // 2) and tool_calls < MAX_TOOL_CALLS:
            await self.emit_status(__event_emitter__, "Broadening search scope…")
            refined_queries = (
                plan.queries
                + [q + " example" for q in plan.queries[:2]]
                + [q + " usage" for q in plan.queries[:2]]
            )

            refined_partitions = list(
                dict.fromkeys(plan.partitions + ["code_examples", "help"])
            )
            refined_results = self.run_search_tool(
                queries=refined_queries,
                filters=plan.filters,
                partitions=refined_partitions,
                top_k=CODE_FINAL_K,
            )
            all_results.extend(refined_results)

        # Combine all results and deduplicate
        self.citations = _dedupe_by_source_chunk(all_results)

        # Build comprehensive context
        query_terms = plan.queries[:4]
        search_context = "\n\n".join(
            [document_to_markdown(d, query_terms) for d in self.citations]
        )
        tool_context = "\n\n".join(tool_contexts)

        preliminary_context = f"{search_context}\n\n{tool_context}".strip()

        # 4) Synthesis with grounded context
        context_block = (
            "Retrieved Context:\n\n" + preliminary_context[:12000]
            if preliminary_context
            else "Retrieved Context: (none)"
        )
        all_messages.append({"role": "system", "content": context_block})

        # 5) Stream synthesized answer with enhanced tools
        async for out in self.synthesize_answer(
            all_messages=all_messages,
            stream=stream,
            emitter=__event_emitter__,
            plan=plan,
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
