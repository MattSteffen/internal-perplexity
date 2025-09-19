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

import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from pymilvus import (
    CollectionSchema,
    FieldSchema,
    MilvusClient,
    Function,
    FunctionType,
    DataType,
)

# Constants
MAX_DOC_LENGTH = 65535  # Max length for VARCHAR enforced by Milvus
DEFAULT_VARCHAR_MAX_LENGTH = 20480
DEFAULT_ARRAY_CAPACITY = 100  # Default max number of elements in an array field
DEFAULT_ARRAY_VARCHAR_MAX_LENGTH = (
    512  # Default max length for string elements within an array
)
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
    ]
    functions = [function_full_text_search, function_full_text_search_metadata]
    return fields, functions


def create_schema(
    embedding_size: int, user_metadata_json_schema: Dict[str, any] = None
):
    fields, functions = _create_base_schema(embedding_size)

    schema = CollectionSchema(
        fields=fields,
        functions=functions,
        description="The schema for the collection",
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

    return index_params
