# tools/ Package Overview

This package defines tool schemas and execution logic for function/tool calling.

## Files

### `__init__.py`
Exports the `tool_registry` singleton.

### `base.py`
Tool protocol definition with `get_definition()` and async `execute()` contract.

### `milvus_search.py`
Milvus search tool implementation with schema definition, execution, and helpers.

### `registry.py`
Tool registry that stores available tools, returns their definitions, and executes tools by name.

### `weather.py`
Mock weather tool used for demonstration and testing.
