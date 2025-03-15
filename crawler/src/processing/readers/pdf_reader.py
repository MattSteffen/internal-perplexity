"""
PDF Document Reader

This module provides the PDFReader class for reading and extracting content from
PDF files using PyMuPDF.
"""

import os
import fitz  # PyMuPDF
from typing import Any, Optional

from .base import DocumentReader
from ..document_content import DocumentContent


class PDFReader(DocumentReader):
    """Reader for PDF files using PyMuPDF."""
    
    def read(self, file_path: str) -> DocumentContent:
        """Read PDF file and extract content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        content = DocumentContent()
        
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            # Extract document metadata
            pdf_metadata = {
                "source": file_path,
                "format": "pdf",
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creationDate": doc.metadata.get("creationDate", ""),
                "modDate": doc.metadata.get("modDate", ""),
                "pageCount": len(doc),
                "size_bytes": os.path.getsize(file_path)
            }
            
            content.set_metadata(pdf_metadata)
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Extract text
                page_text = page.get_text("text")
                content.add_text(page_text, page_num)
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]  # get the XREF of the image
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    
                    # Process the image
                    img_description = self.process_image(image_data)
                    content.add_image(image_data, img_description, page_num, img_index)
                
                # Try to detect tables (simplified approach)
                blocks = page.get_text("dict")["blocks"]
                table_index = 0
                for b in blocks:
                    if "lines" in b and len(b["lines"]) > 2:
                        # Check if this might be a table based on structure
                        rect = fitz.Rect(b["bbox"])
                        lines_count = len(b["lines"])
                        spans_per_line = [len(line["spans"]) for line in b["lines"]]
                        
                        # Simple heuristic: if multiple lines with multiple spans each
                        if lines_count > 2 and max(spans_per_line) > 2:
                            # Extract table text from the specified area
                            table_text = page.get_text("text", clip=rect)
                            content.add_table(f"[Table content:\n{table_text}\n]", page_num, table_index)
                            table_index += 1
            
            # Close the document
            doc.close()
            
            return content
        
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            content.add_text(f"Error reading file: {str(e)}")
            return content 