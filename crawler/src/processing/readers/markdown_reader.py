"""
Markdown Document Reader

This module provides the MarkdownReader class for reading and extracting content from
Markdown files.
"""

import os
import markdown
from bs4 import BeautifulSoup
from typing import Any, Optional

from .base import DocumentReader
from ..document_content import DocumentContent


class MarkdownReader(DocumentReader):
    """Reader for Markdown files."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read markdown file and extract content.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                md_content = f.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                content.add_text(text)
                
                # Extract images if any
                img_index = 0
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    alt = img.get('alt', 'No description')
                    
                    # For local images referenced in the markdown
                    if src and not src.startswith(('http://', 'https://')):
                        img_path = os.path.join(os.path.dirname(file_path), src)
                        if os.path.exists(img_path):
                            try:
                                with open(img_path, 'rb') as img_file:
                                    img_data = img_file.read()
                                
                                description = self.process_image(img_data) if self.vision_llm else f"[Image: {alt}]"
                                content.add_image(img_data, description, None, img_index)
                                img_index += 1
                            except Exception as img_err:
                                print(f"Error processing image {src}: {img_err}")
                    else:
                        content.add_text(f"[External image: {src}, Description: {alt}]")
                
                # Basic metadata
                content.set_metadata(self.get_file_metadata(file_path))
                
            return content
        except Exception as e:
            print(f"Error reading markdown file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content 