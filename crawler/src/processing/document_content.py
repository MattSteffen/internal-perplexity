"""
Document Content Container

This module provides the DocumentContent class, which serves as a container for
extracted document content including text, images, tables, and metadata.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ImageContent:
    """Container for image data and metadata."""
    data: bytes
    description: str
    page: Optional[int] = None
    index: int = 0


@dataclass
class TableContent:
    """Container for table data and metadata."""
    content: str
    page: Optional[int] = None
    index: int = 0


class DocumentContent:
    """Container for extracted document content with text, images, tables, etc."""
    
    def __init__(self):
        """Initialize an empty DocumentContent object."""
        self.text_blocks: List[str] = []
        self.images: List[ImageContent] = []
        self.tables: List[TableContent] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_text(self, text: str, page_num: Optional[int] = None) -> None:
        """Add a text block with optional page number.
        
        Args:
            text: The text content to add
            page_num: Optional page number (0-based index)
        """
        if page_num is not None:
            self.text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
        else:
            self.text_blocks.append(text)
    
    def add_image(self, 
                 image_data: bytes, 
                 description: str, 
                 page_num: Optional[int] = None, 
                 img_index: int = 0) -> None:
        """Add an image with its description and location info.
        
        Args:
            image_data: Binary image data
            description: Text description of the image
            page_num: Optional page number (0-based index)
            img_index: Index of the image on the page
        """
        self.images.append(ImageContent(
            data=image_data,
            description=description,
            page=page_num,
            index=img_index
        ))
        
        if page_num is not None:
            self.text_blocks.append(f"--- Image on page {page_num + 1}, #{img_index + 1} ---\n{description}")
        else:
            self.text_blocks.append(f"--- Image #{img_index + 1} ---\n{description}")
    
    def add_table(self, 
                 table_text: str, 
                 page_num: Optional[int] = None, 
                 table_index: int = 0) -> None:
        """Add a table with its content and location info.
        
        Args:
            table_text: Text representation of the table
            page_num: Optional page number (0-based index)
            table_index: Index of the table on the page
        """
        self.tables.append(TableContent(
            content=table_text,
            page=page_num,
            index=table_index
        ))
        
        if page_num is not None:
            self.text_blocks.append(f"--- Table on page {page_num + 1}, #{table_index + 1} ---\n{table_text}")
        else:
            self.text_blocks.append(f"--- Table #{table_index + 1} ---\n{table_text}")
    
    def get_text(self) -> str:
        """Get all content as a single text string.
        
        Returns:
            Concatenated text content with appropriate separators
        """
        return "\n\n".join(self.text_blocks)
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set document metadata.
        
        Args:
            metadata: Dictionary of metadata key-value pairs
        """
        self.metadata = metadata
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update document metadata with new values.
        
        Args:
            metadata: Dictionary of metadata key-value pairs to update
        """
        self.metadata.update(metadata)
    
    def __str__(self) -> str:
        """String representation of the document content.
        
        Returns:
            A summary of the document content
        """
        return (
            f"DocumentContent with {len(self.text_blocks)} text blocks, "
            f"{len(self.images)} images, {len(self.tables)} tables, "
            f"and {len(self.metadata)} metadata fields"
        ) 