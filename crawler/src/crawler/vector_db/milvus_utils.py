"""
General schema for all datasets (all system fields use default_ prefix):
- default_document_id: The id of the document (not milvus unique id)
- default_chunk_index: The index of the chunk in the document
- default_text: The text chunk of the document
- default_text_embedding: The embedding of the text chunk
- default_text_sparse_embedding: The full-text-search embedding of the chunk
- default_metadata: The metadata of the document
- default_metadata_sparse_embedding: The full-text-search embedding of the metadata
- default_minio: The source of the document, the url to the original document in minio

Extra schema designed by users:
- title: The title of the document
- author: The author of the document
- date: The date of the document
- keywords: The keywords of the document
- unique_words: The unique words of the document
- etc.

User metadata can now contain any keys without conflict since system fields are prefixed.
"""

import json
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
)

# Constants
MAX_DOC_LENGTH = 65535  # Max length for VARCHAR enforced by Milvus
DEFAULT_VARCHAR_MAX_LENGTH = 20480
DEFAULT_ARRAY_CAPACITY = 100  # Default max number of elements in an array field
DEFAULT_ARRAY_VARCHAR_MAX_LENGTH = 512  # Default max length for string elements within an array
DEFAULT_SECURITY_GROUP = ["public"]  # TODO: Ensure that public is a valid security group and all users get added to it upon creation
# Helper mapping for JSON schema types to Milvus DataType for array elements
JSON_TYPE_TO_MILVUS_ELEMENT_TYPE = {
    "string": DataType.VARCHAR,
    "integer": DataType.INT64,
    "number": DataType.DOUBLE,
    "float": DataType.DOUBLE,
    "boolean": DataType.BOOL,
}


def _create_base_schema(embedding_size) -> CollectionSchema:
    field_schema_primary_id = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        description="Auto-generated unique record ID",
    )
    field_schema_document_id = FieldSchema(
        name="default_document_id",
        dtype=DataType.VARCHAR,  # a uuid
        max_length=64,
        description="The uuid of the document",
    )
    field_schema_minio = FieldSchema(
        name="default_minio",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="The source url of the document in minio",
    )
    field_schema_chunk_index = FieldSchema(
        name="default_chunk_index",
        dtype=DataType.INT64,
        description="The index of the chunk in the document",
    )
    field_schema_text = FieldSchema(
        name="default_text",
        dtype=DataType.VARCHAR,
        enable_analyzer=True,
        max_length=MAX_DOC_LENGTH,
        description="The text of the document",
    )
    field_schema_text_embedding = FieldSchema(
        name="default_text_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=embedding_size,
        description="The embedding of the text",
    )
    field_schema_text_sparse_embedding = FieldSchema(
        name="default_text_sparse_embedding",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        description="The full-text-search embedding of the text",
    )
    field_schema_default_metadata = FieldSchema(
        name="default_metadata",
        dtype=DataType.VARCHAR,
        max_length=MAX_DOC_LENGTH,
        enable_analyzer=True,
        description="The metadata of the document as a JSON string",
    )
    field_schema_metadata_sparse_embedding = FieldSchema(
        name="default_metadata_sparse_embedding",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        description="The full-text-search embedding of the metadata",
    )
    field_schema_metadata = FieldSchema(
        name="metadata",
        dtype=DataType.JSON,
        description="The metadata of the document as outlined by the user.",
    )
    field_schema_benchmark_questions = FieldSchema(
        name="benchmark_questions",
        dtype=DataType.VARCHAR,
        max_length=MAX_DOC_LENGTH,
        description="Benchmark questions generated for the document as a JSON array string",
        default_value="[]",
    )

    # RBAC implementation
    # Note: Milvus does not support default values for ARRAY fields
    # The default security group must be set at the application level when inserting documents
    field_schema_security_group = FieldSchema(
        name="security_group",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=20,
        max_length=100,
        description="The security group of the document for RBAC row-level access control",
    )

    # functions to build the full-text-search indexes
    function_full_text_search = Function(
        name="full_text_search_on_chunks",
        input_field_names=["default_text"],
        output_field_names=["default_text_sparse_embedding"],
        function_type=FunctionType.BM25,
    )
    function_full_text_search_metadata = Function(
        name="full_text_search_on_metadata",
        input_field_names=["default_metadata"],
        output_field_names=["default_metadata_sparse_embedding"],
        function_type=FunctionType.BM25,
    )
    fields = [
        field_schema_primary_id,
        field_schema_document_id,
        field_schema_minio,
        field_schema_chunk_index,
        field_schema_default_metadata,
        field_schema_text,
        field_schema_text_embedding,
        field_schema_text_sparse_embedding,
        field_schema_metadata,
        field_schema_metadata_sparse_embedding,
        field_schema_benchmark_questions,
        field_schema_security_group,
    ]
    functions = [function_full_text_search, function_full_text_search_metadata]
    return fields, functions


def create_schema(
    embedding_size: int,
    user_metadata_json_schema: dict[str, any] = None,
    library_context: str = None,
    collection_config_json: str | None = None,
) -> CollectionSchema:
    fields, functions = _create_base_schema(embedding_size)
    description = create_description(fields, user_metadata_json_schema, library_context, collection_config_json)
    # TODO: This must have properly formatted collection description.
    schema = CollectionSchema(
        fields=fields,
        functions=functions,
        description=description,
        enable_dynamic_field=True,
    )

    return schema


def create_index(client: MilvusClient):
    # Prepare index parameters
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="default_text_embedding",
        index_name="text_embedding_index",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    index_params.add_index(
        field_name="default_text_sparse_embedding",
        index_name="text_sparse_embedding_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )

    index_params.add_index(
        field_name="default_metadata_sparse_embedding",
        index_name="metadata_sparse_embedding_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )

    # Create bitmap index for security column
    index_params.add_index(
        field_name="security_group",
        index_type="BITMAP",
    )

    return index_params


def parse_collection_config(description: str) -> dict[str, Any] | None:
    """
    Parse the collection config JSON from a collection description.

    The description should be pure JSON that can be loaded directly with json.loads.
    Expected keys: CrawlerConfig, library_context, metadata_schema, llm_prompt

    Args:
        description: Collection description string (should be pure JSON)

    Returns:
        Dictionary containing the collection config, or None if not found/invalid
    """
    if not description:
        return None

    try:
        # Try to parse as pure JSON
        data = json.loads(description.strip())

        # Validate that it has the expected structure
        if "CrawlerConfig" in data:
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def create_description(
    fields: list["FieldSchema"],
    user_metadata_json_schema: dict[str, Any],
    library_context: str,
    collection_config_json: str | None = None,
) -> str:
    """
    Build a description for a Milvus collection.

    If collection_config_json is provided, it is used as the description (pure JSON).
    Otherwise, builds an LLM-friendly, informative description.

    Inputs:
      - fields: List[FieldSchema] from pymilvus
      - user_metadata_json_schema: JSON Schema for the user-supplied metadata blob
      - library_context: Human-readable description of what documents live here
      - collection_config_json: Optional JSON string containing collection configuration
        (if provided, this becomes the entire description)

    Output:
      - If collection_config_json is provided: returns it as-is (pure JSON)
      - Otherwise: returns a human-readable description for LLMs

    Notes for LLMs:
      - The user metadata is stored as a JSON object in the `metadata` field.
      - When constructing filters, use Milvus JSON path syntax and JSON operators
        against the `metadata` field (see examples below).
    """
    # If collection config JSON is provided, use it as the description (pure JSON)
    if collection_config_json:
        return collection_config_json

    def _is_json_metadata_field(fs: list["FieldSchema"]) -> bool:
        for f in fs:
            if f.name == "metadata":
                # Be tolerant of dtype string representations
                if "JSON" in str(getattr(f, "dtype", "")).upper():
                    return True
        return True  # Default true per system design; caller states it's JSON.

    def _schema_props(schema: dict[str, Any]) -> dict[str, Any]:
        return schema.get("properties", {}) if isinstance(schema, dict) else {}

    is_json_metadata = _is_json_metadata_field(fields)
    props = _schema_props(user_metadata_json_schema)

    parts: list[str] = []

    # 1) Purpose/context
    parts.append(f"Collection purpose and library context:\n{library_context}")
    parts.append("")

    # 2) System-defined fields
    parts.append("System-defined metadata schema for the collection:")
    for field in fields:
        parts.append(f"- {field.name}: ({field.dtype}) {field.description}")
    parts.append("")

    # 3) User-defined JSON metadata schema
    parts.append("User-defined metadata schema for the collection (JSON):")
    parts.append(json.dumps(user_metadata_json_schema, indent=2))
    parts.append("Note: The user metadata is stored as a JSON object in the 'metadata' field.")
    if not is_json_metadata:
        parts.append("WARNING: The 'metadata' field does not appear to be a Milvus JSON field. " "JSON-path filtering requires a JSON field type.")
    parts.append("")

    # 4) Milvus JSON filtering quick reference (grounded in docs)
    parts.append("Milvus JSON filtering quick reference:")
    parts.append("- Access JSON keys with JSON path syntax: metadata[\"key\"] or metadata['key']")
    parts.append("- Examples:\n  " 'filter = \'metadata["category"] == "electronics"\'\n  ' "filter = 'metadata[\"price\"] > 50'\n  " 'filter = \'json_contains(metadata["tags"], "featured")\'')
    parts.append("- JSON operators:\n" "  - json_contains(identifier, expr)\n" "  - json_contains_all(identifier, expr)\n" "  - json_contains_any(identifier, expr)")
    parts.append("- Use standard boolean/filter operators where appropriate: " "==, !=, >, <, >=, <=, IN, NOT IN, LIKE, AND, OR, NOT")
    parts.append("- Ensure the collection is loaded and vector fields are indexed when using " "search with filters.")
    parts.append("- JSON fields work best with flat structures. Deeply nested objects are " "treated as strings.")
    parts.append("")

    # 5) Tailored examples for the provided metadata schema
    parts.append("Examples tailored to this collection's JSON metadata schema:")
    ex_lines: list[str] = []

    # Helper to add examples by property type
    for name, spec in props.items():
        ptype = spec.get("type")
        # Arrays
        if ptype == "array":
            items_type = (spec.get("items") or {}).get("type")
            if items_type == "string":
                # Single value
                ex_lines.append(f"# {name}: array of strings\n" f'filter = \'json_contains(metadata["{name}"], "example")\'')
                # Any of
                ex_lines.append(f'filter = \'json_contains_any(metadata["{name}"], ' f'["val1", "val2"])\'')
                # All of
                ex_lines.append(f'filter = \'json_contains_all(metadata["{name}"], ' f'["valA", "valB"])\'')
            else:
                # Generic array examples
                ex_lines.append(f"# {name}: array\n" f'filter = \'json_contains(metadata["{name}"], "value")\'')
        # Integers / numbers
        elif ptype in ("integer", "number"):
            # Range and membership examples
            ex_lines.append(f"# {name}: numeric\n" f"filter = 'metadata[\"{name}\"] >= 2020'")
            ex_lines.append(f"filter = 'metadata[\"{name}\"] IN [2020, 2021, 2022]'")
        # Booleans
        elif ptype == "boolean":
            ex_lines.append(f"# {name}: boolean\n" f"filter = 'metadata[\"{name}\"] == true'")
        # Strings
        elif ptype == "string":
            ex_lines.append(f"# {name}: string\n" f'filter = \'metadata["{name}"] == "Exact Title"\'')
            ex_lines.append(f'filter = \'metadata["{name}"] LIKE "%keyword%"\'')
        # Fallback
        else:
            ex_lines.append(f"# {name}: type={ptype}\n" f'# Use JSON path access: metadata["{name}"] with appropriate operators')

    # Provide combined, practical examples for the common fields in your schema
    # Author (array of strings)
    if "author" in props and props["author"].get("type") == "array":
        ex_lines.append("# Filter by author (array of strings)\n" 'filter = \'json_contains(metadata["author"], "John Doe")\'')
        ex_lines.append('filter = \'json_contains_any(metadata["author"], ' '["John Doe", "Jane Smith"])\'')

    # Date (integer year)
    if "date" in props and props["date"].get("type") in ("integer", "number"):
        ex_lines.append("# Filter by publication year >= 2021\n" "filter = 'metadata[\"date\"] >= 2021'")
        ex_lines.append("# Filter by year range\n" 'filter = \'metadata["date"] >= 2018 AND metadata["date"] <= 2024\'')

    # Keywords (array of strings)
    if "keywords" in props and props["keywords"].get("type") == "array":
        ex_lines.append('# Must contain the keyword "machine learning"\n' 'filter = \'json_contains(metadata["keywords"], "machine learning")\'')
        ex_lines.append("# Contains any of these keywords\n" 'filter = \'json_contains_any(metadata["keywords"], ["AI", "ML"])\'')
        ex_lines.append("# Contains all of these keywords\n" 'filter = \'json_contains_all(metadata["keywords"], ' '["neural networks", "optimization"])\'')

    # Unique words (array of strings)
    if "unique_words" in props and props["unique_words"].get("type") == "array":
        ex_lines.append('# Unique words must include "backpropagation"\n' 'filter = \'json_contains(metadata["unique_words"], "backpropagation")\'')

    # Title and other strings
    for s in ("title", "description"):
        if s in props and props[s].get("type") == "string":
            ex_lines.append(f"# {s} fuzzy match\n" f'filter = \'metadata["{s}"] LIKE "%Analysis%"\'')

    # Combined filter example
    ex_lines.append(
        "# Combined filter example (year + author + keywords)\n"
        'filter = \'metadata["date"] >= 2020 AND '
        'json_contains(metadata["author"], "Dr. Reed") AND '
        'json_contains_any(metadata["keywords"], ["AI", "ML"])\''
    )

    # Render examples
    parts.append("Filter expression examples:")
    parts.append("```text")
    parts.extend(ex_lines)
    parts.append("```")

    return "\n".join(parts)
