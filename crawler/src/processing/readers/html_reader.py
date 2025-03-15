"""
HTML Document Reader

This module provides the HTMLReader class for reading and extracting content from
HTML files.
"""

import os
from bs4 import BeautifulSoup
from typing import Any, Optional

from .base import DocumentReader
from ..document_content import DocumentContent


class HTMLReader(DocumentReader):
    """Reader for HTML files."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read HTML file and extract content.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()
                content.add_text(text)
                
                # Extract images if any
                img_index = 0
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    alt = img.get('alt', 'No description')
                    
                    # For local images referenced in the HTML
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
                
                # Extract tables
                table_index = 0
                for table in soup.find_all('table'):
                    table_text = table.get_text()
                    content.add_table(f"[Table content:\n{table_text}\n]", None, table_index)
                    table_index += 1
                
                # Extract metadata from head tags
                meta_data = {
                    'source': file_path,
                    'format': 'html',
                    'size_bytes': os.path.getsize(file_path),
                    'title': soup.title.string if soup.title else ""
                }
                
                # Extract other meta tags
                for meta in soup.find_all('meta'):
                    name = meta.get('name', meta.get('property', ''))
                    if name and meta.get('content'):
                        meta_data[name] = meta.get('content')
                
                content.set_metadata(meta_data)
                
            return content
        except Exception as e:
            print(f"Error reading HTML file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content 