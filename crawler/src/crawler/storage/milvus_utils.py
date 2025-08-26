"""
General schema for all datasets:
- document_id: The id of the document (not milvus unique id)
- chunk_index: The index of the chunk in the document
- text: The text chunk of the document
- text_embedding: The embedding of the text chunk
- text_sparse_embedding: The full-text-search embedding of the chunk
- metadata: The metadata of the document
- metadata_sparse_embedding: The full-text-search embedding of the metadata
- minio: The source of the document, the url to the original document in minio

Extra schema designed by users:
- title: The title of the document
- author: The author of the document
- date: The date of the document
- keywords: The keywords of the document
- unique_words: The unique words of the document
- etc.

The metadata designed by users is what occupies the metadata in general schema.
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
MAX_DOC_LENGTH = 65535 # Max length for VARCHAR enforced by Milvus
DEFAULT_VARCHAR_MAX_LENGTH = 20480
DEFAULT_ARRAY_CAPACITY = 100 # Default max number of elements in an array field
DEFAULT_ARRAY_VARCHAR_MAX_LENGTH = 512 # Default max length for string elements within an array
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
        name='id',
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        description="Auto-generated unique record ID",
    )
    field_schema_document_id = FieldSchema(
        name="document_id",
        dtype=DataType.VARCHAR, # a uuid
        max_length=64,
        description="The uuid of the document",
    )
    field_schema_minio = FieldSchema(
        name="minio",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="The source url of the document in minio",
    )
    field_schema_chunk_index = FieldSchema(
        name="chunk_index",
        dtype=DataType.INT64,
        description="The index of the chunk in the document",
    )
    field_schema_text = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        enable_analyzer=True,
        max_length=MAX_DOC_LENGTH,
        description="The text of the document",
    )
    field_schema_text_embedding = FieldSchema(
        name="text_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=embedding_size,
        description="The embedding of the text",
    )
    field_schema_text_sparse_embedding = FieldSchema(
        name="text_sparse_embedding",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        description="The full-text-search embedding of the text",
    )
    field_schema_metadata = FieldSchema(
        name="metadata",
        dtype=DataType.VARCHAR,
        max_length=MAX_DOC_LENGTH,
        enable_analyzer=True,
        description="The metadata of the document as a JSON string",
    )
    field_schema_metadata_sparse_embedding = FieldSchema(
        name="metadata_sparse_embedding",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        description="The full-text-search embedding of the metadata",
    )
    field_schema_benchmark_questions = FieldSchema(
        name="benchmark_questions",
        dtype=DataType.VARCHAR,
        max_length=MAX_DOC_LENGTH,
        description="Benchmark questions generated for the document as a JSON array string",
        default_value="[]"
    )

    function_full_text_search = Function(
        name="full_text_search_on_chunks",
        input_field_names=["text"],
        output_field_names=["text_sparse_embedding"],
        function_type=FunctionType.BM25
    )
    function_full_text_search_metadata = Function(
        name="full_text_search_on_metadata",
        input_field_names=["metadata"],
        output_field_names=["metadata_sparse_embedding"],
        function_type=FunctionType.BM25
    )
    fields=[
        field_schema_primary_id,
        field_schema_document_id,
        field_schema_minio,
        field_schema_chunk_index,
        field_schema_text,
        field_schema_text_embedding,
        field_schema_text_sparse_embedding,
        field_schema_metadata,
        field_schema_metadata_sparse_embedding,
        field_schema_benchmark_questions,
    ]
    functions=[function_full_text_search, function_full_text_search_metadata]
    return fields, functions


def _build_user_defined_fields(schema_config: Dict[str, Any]) -> List[FieldSchema]:
    """
    Builds Milvus FieldSchema objects for user-defined metadata fields only.
    
    Args:
        schema_config: JSON schema configuration with 'properties' containing field definitions
        
    Returns:
        List of FieldSchema objects for user-defined fields
        
    Raises:
        ValueError: If schema_config is invalid or contains unsupported types
        ImportError: If Pymilvus is not installed
    """
    if not schema_config:
        logging.warning("Empty schema configuration provided.")
        return []
    # TODO: raise error if not valid jsonschema or if properties is not a dict
    non_allowed_fields = ["id", "document_id", "source", "chunk_index", "text", "text_embedding", "text_sparse_embedding", "metadata", "metadata_sparse_embedding"]
    
    fields_config = schema_config.get("properties", {})
    user_fields: List[FieldSchema] = []
    
    for field_name, field_def in fields_config.items():
        if not isinstance(field_def, dict) or "type" not in field_def:
            logging.warning(f"Skipping invalid field definition for '{field_name}': {field_def}")
            continue
        if field_name in non_allowed_fields:
            logging.warning(f"Skipping field '{field_name}' as it's a reserved field name")
            continue
        
        field_type = field_def.get("type")
        field_description = field_def.get("description", f"User metadata field: {field_name}")
        milvus_field: Optional[FieldSchema] = None
        
        try:
            match field_type:
                case "string":
                    max_length = max(
                        field_def.get("maxLength", DEFAULT_VARCHAR_MAX_LENGTH), 
                        DEFAULT_VARCHAR_MAX_LENGTH
                    )
                    milvus_field = FieldSchema(
                        name=field_name, 
                        dtype=DataType.VARCHAR, 
                        max_length=max_length, 
                        description=field_description,
                        default_value=""
                    )
                
                case "integer":
                    milvus_field = FieldSchema(
                        name=field_name, 
                        dtype=DataType.INT64, 
                        description=field_description,
                        default_value=0
                    )
                
                case "number" | "float":
                    milvus_field = FieldSchema(
                        name=field_name, 
                        dtype=DataType.DOUBLE, 
                        description=field_description,
                        default_value=0.0
                    )
                
                case "boolean":
                    milvus_field = FieldSchema(
                        name=field_name, 
                        dtype=DataType.BOOL, 
                        description=field_description,
                        default_value=False
                    )
                
                case "array":
                    milvus_field = _build_array_field(field_name, field_def, field_description)
                
                case "object":
                    milvus_field = FieldSchema(
                        name=field_name,
                        dtype=DataType.JSON,
                        description=field_description
                    )
                    logging.info(f"Mapping object field '{field_name}' to Milvus JSON type.")
                
                case _:
                    raise ValueError(f"Unsupported field type '{field_type}' for field '{field_name}'.")
            
            if milvus_field:
                user_fields.append(milvus_field)
                logging.debug(f"Added user field '{field_name}' (Type: {milvus_field.dtype})")
        
        except Exception as e:
            logging.error(f"Error processing field '{field_name}' with definition {field_def}: {e}")
            raise
    
    logging.info(f"Built {len(user_fields)} user-defined fields: {[f.name for f in user_fields]}")
    return user_fields


def _build_array_field(field_name: str, field_def: Dict[str, Any], field_description: str) -> FieldSchema:
    """
    Helper function to build array field schemas.
    
    Args:
        field_name: Name of the field
        field_def: Field definition from JSON schema
        field_description: Description for the field
        
    Returns:
        FieldSchema for the array field
    """
    items_def = field_def.get("items")
    
    if not isinstance(items_def, dict) or "type" not in items_def:
        logging.warning(f"Array field '{field_name}' missing valid 'items' definition. Using VARCHAR fallback.")
        return _build_array_fallback_field(field_name, field_def, field_description)
    
    element_type_str = items_def.get("type")
    milvus_element_type = JSON_TYPE_TO_MILVUS_ELEMENT_TYPE.get(element_type_str)
    
    if not milvus_element_type:
        logging.warning(f"Unsupported array element type '{element_type_str}' for field '{field_name}'. Using VARCHAR fallback.")
        return _build_array_fallback_field(field_name, field_def, field_description)
    
    # Build native Milvus array field
    max_capacity = min(
        max(1, field_def.get("maxItems", DEFAULT_ARRAY_CAPACITY)), 
        DEFAULT_ARRAY_CAPACITY
    )
    
    field_kwargs = {
        "name": field_name,
        "dtype": DataType.ARRAY,
        "element_type": milvus_element_type,
        "max_capacity": max_capacity,
        "description": field_description,
        "nullable": True
    }
    
    # Add max_length for VARCHAR array elements
    if milvus_element_type == DataType.VARCHAR:
        element_max_length = min(
            items_def.get("maxLength", DEFAULT_ARRAY_VARCHAR_MAX_LENGTH),
            DEFAULT_ARRAY_VARCHAR_MAX_LENGTH
        )
        field_kwargs["max_length"] = element_max_length
    
    logging.info(f"Mapping array field '{field_name}' to native Milvus ARRAY({milvus_element_type}, capacity={max_capacity})")
    return FieldSchema(**field_kwargs)


def _build_array_fallback_field(field_name: str, field_def: Dict[str, Any], field_description: str) -> FieldSchema:
    """
    Fallback for unsupported array types - serialize as VARCHAR.
    """
    max_length = field_def.get("maxLength", MAX_DOC_LENGTH)
    return FieldSchema(
        name=field_name,
        dtype=DataType.VARCHAR,
        max_length=max_length,
        description=f"{field_description} (serialized array)",
        default_value="[]"
    )



def create_schema(embedding_size: int, user_metadata_json_schema: Dict[str, any] =None):
    fields, functions = _create_base_schema(embedding_size)
    
    # Add user metadata fields
    fields.extend(_build_user_defined_fields(user_metadata_json_schema))
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
        field_name="text_embedding",
        index_name="text_embedding_index",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    index_params.add_index(
        field_name="text_sparse_embedding",
        index_name="text_sparse_embedding_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )

    index_params.add_index(
        field_name="metadata_sparse_embedding",
        index_name="metadata_sparse_embedding_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )

    return index_params


# TODO: Function to validate metadata compared to json schema, concat strings to correct max length.