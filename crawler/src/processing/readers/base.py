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
    
    def process_image(self, image_data: bytes, context: str = "") -> str:
        """Process image data and return description.
        
        This method first analyzes surrounding text context with LLM to guide the vision LLM
        in describing the image's relevance to the text. Optimized for technical content,
        diagrams, and information-rich visualizations.
        
        Args:
            image_data: Binary image data
            context: Text surrounding the image
            
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
            
            # First analyze context with text LLM
            if context:
                context_prompt = f"""Analyze this text and extract:
1. The technical concepts being discussed
2. Any specific metrics, measurements, or quantities mentioned
3. The type of information that a supporting diagram or figure would likely illustrate
4. Your answer should be 50 words or less.

Text: {context}"""
                context_analysis = self.llm.invoke(context_prompt)
                
                # Use context analysis to guide vision LLM
                vision_prompt = f"""Analyze this image as a technical figure or diagram. Focus on:
1. The type of visualization (e.g., flowchart, architecture diagram, graph, technical drawing)
2. Key components, entities, or data points shown
3. Relationships, flows, or patterns depicted
4. Any text, labels, or legends present
5. Quantitative information or measurements displayed

Given this context from the surrounding text: {context_analysis}

Describe how the image conveys technical information and how it relates to the context. If any aspects seem unclear or potentially incorrect, note that."""
                
                description = self.vision_llm.invoke(image_data, vision_prompt)
            else:
                # Fallback to general technical description if no context
                description = self.vision_llm.invoke(image_data, """Analyze this image as a technical figure or diagram. Describe:
1. The type of visualization (e.g., flowchart, architecture diagram, graph, technical drawing)
2. Key components, entities, or data points shown
3. Relationships, flows, or patterns depicted
4. Any text, labels, or legends present
5. Quantitative information or measurements displayed
6. The main technical concept or information this image is trying to convey
7. Your answer should be 50 words or less.""")
                
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