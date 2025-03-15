"""
Base Document Reader

This module provides the base DocumentReader class that all specific document readers
should inherit from. It defines the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union
import io
from PIL import Image
import pytesseract

# Import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_content import DocumentContent


class DocumentReader(ABC):
    """Base class for document type readers.
    
    This abstract class defines the interface that all document readers must implement.
    It provides common functionality like image processing that can be used by all readers.
    """
    
    def __init__(self, llm: Any, vision_llm: Optional[Any] = None):
        """Initialize the document reader.
        
        Args:
            llm: Language model for text processing
            vision_llm: Optional vision model for image processing
        """
        self.llm = llm
        self.vision_llm = vision_llm
    
    @abstractmethod
    def read(self, file_path: str) -> DocumentContent:
        """Read document and extract content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentContent object containing the extracted content
        """
        pass
    
    def process_image(self, image_data: bytes) -> str:
        """Process image data and return description.
        
        This method uses the vision LLM if available, otherwise falls back to OCR.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Text description of the image
        """
        if not image_data:
            return "[Image: No data available]"
        
        try:
            if not self.vision_llm:
                # Use OCR as fallback if no vision LLM
                img = Image.open(io.BytesIO(image_data))
                text = pytesseract.image_to_string(img)
                return f"[Image content: {text or 'No text detected'}]"
            
            # Use vision LLM to describe the image
            description = self.vision_llm.invoke(image_data, "Describe this image in detail")
            return f"[Image description: {description}]"
        except Exception as e:
            print(f"Error processing image: {e}")
            return "[Image: Unable to process]"
    
    def get_file_metadata(self, file_path: str) -> dict:
        """Get basic file metadata.
        # TODO: use the structured llm for this? or do that outside?
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with basic file metadata
        """
        _, ext = os.path.splitext(file_path)
        return {
            'source': file_path,
            'format': ext[1:],  # Remove the leading dot
            'size_bytes': os.path.getsize(file_path)
        } 