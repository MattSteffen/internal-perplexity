"""
JSON Document Reader

This module provides the JSONReader class for reading and extracting content from
JSON files.
"""
# TODO: Simplify this, just cast it all to string.
import os
import json
from typing import Any, Optional

from .base import DocumentReader
from ..document_content import DocumentContent


class JSONReader(DocumentReader):
    """Reader for JSON files."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read JSON file and extract content.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                json_data = json.load(f)
                
                # Format JSON prettily and add as text
                formatted_json = json.dumps(json_data, indent=2)
                content.add_text(formatted_json)
                
                # Try to extract any nested tables or structured data
                self._process_json_object(json_data, content)
                
                # Basic metadata
                content.set_metadata(self.get_file_metadata(file_path))
                
            return content
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content
    
    def _process_json_object(self, data, content, table_index=0, path=""):
        """Process JSON objects to extract table-like structures.
        
        Args:
            data: JSON data to process
            content: DocumentContent to add extracted data to
            table_index: Current table index
            path: Current path in the JSON structure
            
        Returns:
            Updated table index
        """
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # This looks like a table: list of dictionaries
            # First, get all possible keys from all objects
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())
            
            # Create a table representation
            table_rows = [",".join(all_keys)]  # Header row
            
            for item in data:
                row = []
                for key in all_keys:
                    value = item.get(key, "")
                    # Handle nested structures by converting to string
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row.append(str(value))
                table_rows.append(",".join(row))
            
            table_text = "\n".join(table_rows)
            path_info = f" at {path}" if path else ""
            content.add_table(f"[JSON Array as Table{path_info}:\n{table_text}\n]", None, table_index)
            return table_index + 1
        
        # Process nested structures
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    table_index = self._process_json_object(value, content, table_index, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                if isinstance(item, (dict, list)):
                    table_index = self._process_json_object(item, content, table_index, new_path)
        
        return table_index 