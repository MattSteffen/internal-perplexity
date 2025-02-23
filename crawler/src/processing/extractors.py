import json
import os
from abc import ABC, abstractmethod

# TODO: Make this call the hanlder for each file in the directory individually not necessarily folder-wide


class ExtractorHandler(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    @abstractmethod
    def extract(self):
        """Yields text and metadata from documents."""
        pass

    def files(self, extension: str):
        """Yields paths to files with given extension in the directory."""
        for file_name in os.listdir(self.dir_path):
            if file_name.endswith(extension):
                yield os.path.join(self.dir_path, file_name)

class JSONHandler(ExtractorHandler):
    def extract(self):
        """Yields text and metadata from JSON documents."""
        for i, file_path in enumerate(self.files('.json')):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            text = data.get('content', [])
            metadata = {k: v for k, v in data.items() if k != 'content'}
            metadata['source'] = file_path # TODO: use full path
            metadata['format'] = 'json'
            metadata['chunk_index'] = i
            yield text, metadata

class TXTHandler(ExtractorHandler):
    def extract(self):
        """Yields text and metadata from TXT documents."""
        for file_path in self.files('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
            
            metadata = {
                'source': file_path,
                'format': 'txt'
            }
            yield text, metadata

def get_handler(dir_path: str) -> ExtractorHandler:
    """Factory function to get appropriate handler based on files in directory."""
    files = os.listdir(dir_path)
    if any(f.endswith('.json') for f in files):
        return JSONHandler(dir_path)
    elif any(f.endswith('.txt') for f in files):
        return TXTHandler(dir_path)
    else:
        raise ValueError(f"No supported files found in {dir_path}")

# test / demo
if __name__ == "__main__":
    handler = get_handler("../../../data/conference/")
    for text, metadata in handler.extract():
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")




# Maybe something more along the lines of this:

# import docx
# import pptx
# import json
# import markdown
# import pandas as pd
# from bs4 import BeautifulSoup
# import re
# from pathlib import Path

# class DocumentProcessor:
#     def __init__(self):
#         self.supported_extensions = {
#             '.docx': self.process_docx,
#             '.pptx': self.process_pptx,
#             '.md': self.process_markdown,
#             '.json': self.process_json,
#             '.csv': self.process_csv,
#             '.txt': self.process_text
#         }
    
#     def process_file(self, file_path):
#         """
#         Main method to process any supported file type
#         Returns cleaned text ready for LLM input
#         """
#         file_path = Path(file_path)
#         if file_path.suffix not in self.supported_extensions:
#             raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
#         return self.supported_extensions[file_path.suffix](file_path)
    
#     def process_docx(self, file_path):
#         """Process Word documents"""
#         doc = docx.Document(file_path)
        
#         # Extract text from paragraphs
#         full_text = []
#         for para in doc.paragraphs:
#             if para.text.strip():  # Skip empty paragraphs
#                 full_text.append(para.text.strip())
        
#         # Extract text from tables
#         for table in doc.tables:
#             for row in table.rows:
#                 row_text = []
#                 for cell in row.cells:
#                     if cell.text.strip():
#                         row_text.append(cell.text.strip())
#                 if row_text:
#                     full_text.append(" | ".join(row_text))
        
#         return "\n\n".join(full_text)
    
#     def process_pptx(self, file_path):
#         """Process PowerPoint presentations"""
#         pres = pptx.Presentation(file_path)
        
#         slides_text = []
#         for slide_number, slide in enumerate(pres.slides, 1):
#             slide_content = []
#             slide_content.append(f"Slide {slide_number}:")
            
#             # Extract text from shapes
#             for shape in slide.shapes:
#                 if hasattr(shape, "text") and shape.text.strip():
#                     slide_content.append(shape.text.strip())
            
#             slides_text.append("\n".join(slide_content))
        
#         return "\n\n".join(slides_text)
    
#     def process_markdown(self, file_path):
#         """Process Markdown files"""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             md_text = f.read()
        
#         # Convert to HTML first
#         html = markdown.markdown(md_text)
        
#         # Use BeautifulSoup to extract clean text
#         soup = BeautifulSoup(html, 'html.parser')
        
#         # Remove code blocks and preserve them separately
#         code_blocks = []
#         for code in soup.find_all('code'):
#             code_blocks.append(f"Code:\n{code.get_text()}")
#             code.decompose()
        
#         # Get main text
#         text = soup.get_text()
        
#         # Combine text and code blocks
#         if code_blocks:
#             text += "\n\nExtracted Code Blocks:\n" + "\n\n".join(code_blocks)
        
#         return text
    
#     def process_json(self, file_path):
#         """Process JSON files"""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         def flatten_json(data, prefix=''):
#             items = []
#             if isinstance(data, dict):
#                 for key, value in data.items():
#                     new_prefix = f"{prefix}.{key}" if prefix else key
#                     items.extend(flatten_json(value, new_prefix))
#             elif isinstance(data, list):
#                 for i, item in enumerate(data):
#                     new_prefix = f"{prefix}[{i}]"
#                     items.extend(flatten_json(item, new_prefix))
#             else:
#                 items.append(f"{prefix}: {data}")
#             return items
        
#         # Convert JSON to a flattened text representation
#         flattened = flatten_json(data)
#         return "\n".join(flattened)
    
#     def process_csv(self, file_path):
#         """Process CSV files"""
#         df = pd.read_csv(file_path)
        
#         # Convert DataFrame to a text description
#         text_parts = []
        
#         # Add column descriptions
#         text_parts.append("Columns:")
#         for col in df.columns:
#             text_parts.append(f"- {col}: {df[col].dtype}")
        
#         # Add basic statistics
#         text_parts.append("\nBasic Statistics:")
#         text_parts.append(str(df.describe()))
        
#         # Add sample rows
#         text_parts.append("\nSample Rows:")
#         text_parts.append(str(df.head()))
        
#         return "\n".join(text_parts)
    
#     def process_text(self, file_path):
#         """Process plain text files"""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             text = f.read()
        
#         # Basic text cleaning
#         text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
#         text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
#         return text.strip()
    
#     def clean_text_for_llm(self, text, max_length=None):
#         """
#         Clean and format text for LLM input
#         Optionally truncate to max_length
#         """
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove special characters but keep basic punctuation
#         text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
#         # Truncate if needed
#         if max_length and len(text) > max_length:
#             text = text[:max_length-3] + "..."
        
#         return text.strip()

# # Example usage
# if __name__ == "__main__":
#     processor = DocumentProcessor()
    
#     # Process a Word document
#     docx_text = processor.process_file("example.docx")
#     clean_text = processor.clean_text_for_llm(docx_text, max_length=4000)
    
#     # Process a JSON file
#     json_text = processor.process_file("data.json")
#     clean_json = processor.clean_text_for_llm(json_text)
