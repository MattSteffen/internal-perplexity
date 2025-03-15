"""
Document Readers Package

This package provides various document reader classes for different file formats.
"""

from .base import DocumentReader
from .text_reader import TextReader
from .pdf_reader import PDFReader
from .markdown_reader import MarkdownReader
from .html_reader import HTMLReader
from .json_reader import JSONReader
from .csv_reader import CSVReader
from .docx_reader import DocxReader
from .pptx_reader import PptxReader

__all__ = [
    'DocumentReader',
    'TextReader',
    'PDFReader',
    'MarkdownReader',
    'HTMLReader',
    'JSONReader',
    'CSVReader',
    'DocxReader',
    'PptxReader',
] 