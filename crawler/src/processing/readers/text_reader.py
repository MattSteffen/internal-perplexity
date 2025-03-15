"""
Text Document Reader

This module provides the TextReader class for reading and extracting content from
plain text files.
"""

import os
from typing import Any, Optional

from .base import DocumentReader
from processing.document_content import DocumentContent


class TextReader(DocumentReader):
    """Reader for plain text files."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read text file and extract content.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                content.add_text(text)
                
                # Basic metadata
                content.set_metadata(self.get_file_metadata(file_path)) # TODO: use text not file_path
                
            return content
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content 