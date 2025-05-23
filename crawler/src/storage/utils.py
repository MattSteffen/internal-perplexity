from importlib import metadata
import logging
import os
import yaml
from typing import List, Dict, Any, Optional, Set, Tuple

"""
Utility functions for the vector database.
The goals of these functions are to:
- Create a schema for the vector database using the provided JSON schema.
- Validate the schema.
- Validate the metadata. This means that the metadata follows the schema.
"""


# Constants
MAX_DOC_LENGTH = 65000
MAX_SOURCE_LENGTH = 2048
DEFAULT_VARCHAR_MAX_LENGTH = 1024
DEFAULT_ARRAY_CAPACITY = 100 # Default max number of elements in an array field
DEFAULT_ARRAY_VARCHAR_MAX_LENGTH = 256 # Default max length for string elements within an array

# TODO: Build full text search and sematic search based on config.
# TODO: How to determine indexes and search strategry.

try:
    from pymilvus import (
        FieldSchema,
        CollectionSchema,
        DataType,
        Function,
        FunctionType,
    )
    MILVUS_AVAILABLE = True
    logging.info("Pymilvus library loaded successfully.")
except ImportError:
    MILVUS_AVAILABLE = False
    logging.error("Pymilvus not installed. VectorStorage operations cannot proceed.")

# Helper mapping for JSON schema types to Milvus DataType for array elements
JSON_TYPE_TO_MILVUS_ELEMENT_TYPE = {
    "string": DataType.VARCHAR,
    "integer": DataType.INT64,
    "number": DataType.DOUBLE,
    "float": DataType.DOUBLE,
    "boolean": DataType.BOOL,
}
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_schema(schema_config) -> Dict[str, Any]:
    # TODO: Make sure the schema is valid JSON-Schema
    """Loads schema from config, performs basic validation."""
    if schema_config:
        logging.info("Using schema provided directly in configuration.")
        # Basic validation
        if not isinstance(schema_config, dict) or "properties" not in schema_config:
                raise ValueError("Invalid schema configuration: must be a dictionary with a 'properties' key.")
        return schema_config
    else:
        logging.error("Schema configuration ('metadata.schema') is missing.")
        raise ValueError("Schema configuration must be provided in the config object.")

def _validate_metadata(x,y):
    # TODO: Implement metadata validation
    # Example: is the varchar  field length within the max length?
    return True

def validate_metadata(
    metadata_item: Dict[str, Any],
    schema_config: Dict[str, Any],
    item_index: Optional[int] = None
) -> bool:
    """
    Validates a metadata item against a given schema configuration.
    Checks performed:
    - For string fields: ensures the length does not exceed the effective 'maxLength'.
      The effective 'maxLength' is the minimum of the schema-defined 'maxLength'
      (or DEFAULT_VARCHAR_MAX_LENGTH if not defined) and DEFAULT_VARCHAR_MAX_LENGTH.
    - For array fields containing strings: ensures each string element's length
      does not exceed its effective 'maxLength'. The effective 'maxLength' for array
      elements is calculated similarly, using DEFAULT_ARRAY_VARCHAR_MAX_LENGTH.

    Args:
        metadata_item: The dictionary of metadata for a single item.
        schema_config: The schema configuration, expected to have a "properties" key
                       which defines fields and their types/constraints.
        item_index: Optional index of the item in a larger batch, for logging purposes.

    Returns:
        True if the metadata_item is valid according to the checks, False otherwise.
    """
    if not MILVUS_AVAILABLE: # Should not happen if VectorStorage guards this
        logging.error("Pymilvus not available, cannot perform full metadata validation logic tied to Milvus types.")
        return False # Or raise, depending on strictness

    if not isinstance(schema_config, dict) or "properties" not in schema_config:
        log_prefix = f"Item index {item_index}: " if item_index is not None else ""
        logging.error(
            f"{log_prefix}Invalid schema_config provided for validation: Missing 'properties' key."
        )
        return False

    fields_config = schema_config.get("properties", {})
    log_prefix = f"Item index {item_index}" if item_index is not None else "Current item"

    for field_name, field_def in fields_config.items():
        if not isinstance(field_def, dict) or "type" not in field_def:
            logging.warning(
                f"{log_prefix} - Validation: Field definition for '{field_name}' in schema "
                f"is invalid or missing 'type'. Skipping validation for this field."
            )
            continue

        if field_name not in metadata_item:
            # If field is required by schema, this would be an error.
            # For now, only validating present fields.
            continue

        value = metadata_item[field_name]
        field_type = field_def.get("type")

        # 1. Validate simple string fields
        if field_type == "string":
            if not isinstance(value, str):
                logging.warning(
                    f"{log_prefix} - Validation: Field '{field_name}' is defined as string "
                    f"but found type {type(value).__name__}. Skipping length check."
                )
                continue

            # Effective maxLength for Milvus field (as used in build_milvus_schema)
            # If schema defines maxLength, it's min(schema_maxLength, DEFAULT_VARCHAR_MAX_LENGTH)
            # If schema doesn't define maxLength, it's DEFAULT_VARCHAR_MAX_LENGTH
            effective_max_length = min(
                field_def.get("maxLength", DEFAULT_VARCHAR_MAX_LENGTH),
                DEFAULT_VARCHAR_MAX_LENGTH
            )

            if len(value) > effective_max_length:
                logging.warning(
                    f"{log_prefix} - Validation Failed: Field '{field_name}' (length {len(value)}) "
                    f"exceeds effective maxLength ({effective_max_length}). Value starts with: '{value[:50]}...'"
                )
                # set the new value to the truncated version
                metadata_item[field_name] = value[:effective_max_length]

        # 2. Validate string elements within array fields
        # elif field_type == "array":
        #     if not isinstance(value, list):
        #         logging.warning(
        #             f"{log_prefix} - Validation: Field '{field_name}' is defined as array "
        #             f"but found type {type(value).__name__}. Skipping element checks."
        #         )
        #         continue

        #     items_def = field_def.get("items")
        #     if isinstance(items_def, dict) and items_def.get("type") == "string":
        #         # Effective maxLength for string elements in an array (as used in build_milvus_schema)
        #         effective_element_max_length = min(
        #             items_def.get("maxLength", DEFAULT_ARRAY_VARCHAR_MAX_LENGTH),
        #             DEFAULT_ARRAY_VARCHAR_MAX_LENGTH
        #         )

        #         for i, item_element in enumerate(value):
        #             if isinstance(item_element, str):
        #                 if len(item_element) > effective_element_max_length:
        #                     logging.warning(
        #                         f"{log_prefix} - Validation Failed: Field '{field_name}', "
        #                         f"element at index {i} (length {len(item_element)}) "
        #                         f"exceeds item maxLength ({effective_element_max_length}). Element starts with: '{item_element[:50]}...'"
        #                     )
        #                     # set the new value to the truncated version
        #                     metadata_item[field_name][i] = item_element[:effective_element_max_length]
        #             # else: element is not a string, skip length check for this element.
        #             # Could add a warning if type doesn't match items_def.type.

    return metadata_item


# --- Updated build_milvus_schema ---
def build_milvus_schema(schema_config: Dict[str, Any], embedding_dim: int) -> 'CollectionSchema':
    """
    Builds a Milvus CollectionSchema based on the provided JSON schema-like configuration.
    Supports native Milvus ARRAY types based on 'items' definition.
    Uses match/case for type handling.

    Args:
        schema_config: The schema configuration dictionary (e.g., loaded from YAML).
        embedding_dim: The dimension for the 'embedding' vector field.

    Returns:
        A Pymilvus CollectionSchema object.

    Raises:
        ValueError: If schema_config is invalid, unsupported types/structures are found.
        ImportError: If Pymilvus is not installed.
    """
    if not MILVUS_AVAILABLE:
        raise ImportError("Pymilvus library is not installed. Cannot build Milvus schema.")

    if not isinstance(schema_config, dict) or "properties" not in schema_config:
         raise ValueError("Invalid schema_config provided. Must be a dict with 'properties'.")

    fields_config = schema_config.get("properties", {})
    schema_description = schema_config.get("description", "Collection storing document chunks and embeddings")

    # --- Standard Required Fields ---
    fields: List[FieldSchema] = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description="Auto-generated unique record ID"),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="Dense vector embedding"),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=MAX_DOC_LENGTH, description="Original text chunk"),
        FieldSchema(name='metatext', dtype=DataType.VARCHAR, max_length=20000, description="Metadata associated with the document chunk indexed via sparse vectors for full-text search", enable_analyzer=True),
        FieldSchema(name='full_text_embedding', dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=MAX_SOURCE_LENGTH, description="Source of the document", default_value="unknown"),
        FieldSchema(name='chunk_index', dtype=DataType.INT64, description="Index of the chunk within the source", default_value=-1)
    ]
    processed_field_names: Set[str] = {'id', 'embedding', 'text', 'metatext', 'full_text_embedding', 'source', 'chunk_index'}


    # --- Add Fields from Schema Config ---
    for field_name, field_def in fields_config.items():
        if field_name in processed_field_names:
            continue

        if not isinstance(field_def, dict) or "type" not in field_def:
             logging.warning(f"Skipping invalid field definition for '{field_name}': {field_def}")
             continue

        field_type = field_def.get("type")
        field_description = field_def.get("description", f"Metadata field: {field_name}")
        milvus_field: Optional[FieldSchema] = None

        try:
            # --- Use match/case for type-based handling ---
            match field_type:
                case "string":
                    max_length = min(field_def.get("maxLength", DEFAULT_VARCHAR_MAX_LENGTH), DEFAULT_VARCHAR_MAX_LENGTH)
                    milvus_field = FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=field_description, default_value="Unknown")

                case "integer":
                    milvus_field = FieldSchema(name=field_name, dtype=DataType.INT64, description=field_description, default_value=0)

                case "number" | "float": # Combine number and float
                    milvus_field = FieldSchema(name=field_name, dtype=DataType.DOUBLE, description=field_description, default_value=0.0)

                case "boolean":
                    milvus_field = FieldSchema(name=field_name, dtype=DataType.BOOL, description=field_description, default_value=False)

                case "array":
                    items_def = field_def.get("items")
                    if isinstance(items_def, dict) and "type" in items_def:
                        element_type_str = items_def.get("type")
                        milvus_element_type = JSON_TYPE_TO_MILVUS_ELEMENT_TYPE.get(element_type_str)

                        if milvus_element_type:
                            # Use 'maxItems' from schema if present, otherwise default
                            max_capacity = min(max(0,int(field_def.get("maxItems", DEFAULT_ARRAY_CAPACITY))), DEFAULT_ARRAY_CAPACITY)
                            element_max_length = min(items_def.get("maxLength", DEFAULT_VARCHAR_MAX_LENGTH), DEFAULT_VARCHAR_MAX_LENGTH) if milvus_element_type == DataType.VARCHAR else None

                            milvus_field = FieldSchema(
                                name=field_name,
                                dtype=DataType.ARRAY,
                                element_type=DataType.VARCHAR,
                                max_capacity=max_capacity,
                                max_length=element_max_length, # Max length for VARCHAR elements
                                description=field_description,
                                nullable=True
                            )
                            logging.info(f"Mapping array field '{field_name}' to Milvus ARRAY({milvus_element_type}, capacity={max_capacity}).")
                        else:
                            logging.warning(f"Unsupported element type '{element_type_str}' for native Milvus array in field '{field_name}'. Falling back to VARCHAR serialization.")
                            # Fallback to VARCHAR serialization
                            max_length = field_def.get("maxLength", 4096)
                            milvus_field = FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=f"{field_description} (serialized)", default_value="[]")
                    else:
                        logging.warning(f"Array field '{field_name}' is missing valid 'items' definition. Falling back to VARCHAR serialization.")
                        # Fallback to VARCHAR serialization
                        max_length = field_def.get("maxLength", 4096)
                        milvus_field = FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=f"{field_description} (serialized)", default_value="[]")

                case "object":
                    # TODO: Implement object field handling by casting as JSON - https://milvus.io/docs/use-json-fields.md
                    raise NotImplementedError("Object fields are not yet supported.")

                case _: # Default case for unsupported types
                    raise ValueError(f"Unsupported field type '{field_type}' specified for field '{field_name}'.")

            # --- Add the constructed field ---
            if milvus_field:
                fields.append(milvus_field)
                processed_field_names.add(field_name)
                logging.debug(f"Added field '{field_name}' (Type: {milvus_field.dtype}) from schema config.")
            # else: handled by `continue` or exception within match case

        except Exception as e:
             # Catch potential errors during field processing (e.g., invalid maxLength)
             logging.error(f"Error processing schema field '{field_name}' with definition {field_def}: {e}")
             raise # Re-raise to halt schema creation on error

    logging.info(f"Built collection schema with {len(fields)} fields: {[f.name for f in fields]}")
    # Ensure enable_dynamic_field=False unless explicitly needed
    schema = CollectionSchema(fields=fields, description=schema_description, enable_dynamic_field=False)


    # Set sparse text search parameters
    bm_function = Function(
        name="text_bm25_embedding",
        input_field_names=["metatext"],
        output_field_names=["full_text_embedding"],
        function_type=FunctionType.BM25
    )
    schema.add_function(bm_function)

    return schema
