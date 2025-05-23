import json
import os
import argparse
from pydoc import Doc
import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError

from crawler import Crawler
from processing.process_markitdown import MarkItDownConverter
from processing.process_docling import DoclingConverter

schema1 = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Document Core Properties",
  "description": "Schema defining the fundamental metadata and unique terminology of a document.",
  "type": "object",
  "required": [
    "title",
    "author",
    "date",
    "keywords",
    "unique_words"
  ],
  "properties": {
    "title": {
      "type": "string",
      "maxLength": 2550,
      "description": "The official title of the document. Should be concise and accurately reflect the document's content."
    },
    "author": {
      "type": "array",
      "description": "A list of all individuals or entities responsible for creating the document.",
      "items": {
        "type": "string",
        "maxLength": 2550,
        "description": "The full name of an individual author or the name of an organizational author (e.g., 'John Doe', 'Example Corporation')."
      }
    },
    "date": {
      "type": "integer",
      "description": "The year of the document's official publication or last significant revision. Please enter as a four-digit year (YYYY).",
      "minimum": 1900,
      "maximum": 2100
    },
    "keywords": {
      "type": "array",
      "description": "A list of relevant terms or phrases that categorize the document's subject matter, aiding in search and discovery.",
      "items": {
        "type": "string",
        "maxLength": 5120,
        "description": "An individual keyword or key phrase (e.g., 'artificial intelligence', 'machine learning applications')."
      }
    },
    "unique_words": {
      "type": "array",
      "description": "A list of terms or short phrases from the document that are domain-specific, highly technical, or might not be common knowledge. These words are crucial for a specialized understanding of the document's content.",
      "items": {
        "type": "string",
        "maxLength": 100,
        "description": "A single unique or domain-specific term/phrase."
      },
      "minItems": 0
    }
  }
}
schema2 = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Document Summary Points",
  "description": "Schema defining distinct summary aspects of a document.",
  "type": "object",
  "required": [
    "summary_item_1"
  ],
  "properties": {
    "summary_item_1": {
      "type": "string",
      "maxLength": 15000,
      "description": "A concise summary of the primary topic or a unique, central argument discussed in the document. Focus on the most significant general idea or contribution."
    },
    "summary_item_2": {
      "type": "string",
      "maxLength": 15000,
      "description": "If the document explores a second distinct topic or presents another significant unique aspect, describe it here. This should cover a different core idea than summary_item_1."
    },
    "summary_item_3": {
      "type": "string",
      "maxLength": 15000,
      "description": "If the document addresses a third distinct major theme or offers an additional unique insight, provide that summary here. Ensure it highlights a separate concept from the previous summary items."
    }
  }
}
extra_fields = ["summary_item_1", "summary_item_2", "summary_item_3"]

import json
import time
from typing import Dict, Any, Optional, List, TypeVar, Type
from functools import wraps

T = TypeVar('T')

class MyExtractor:
    """
    Custom extractor class for processing documents with timeout capability.
    """
    def __init__(self, config: dict, llm) -> None:
        self.config = config
        self.converter = MarkItDownConverter(config)
        self.llm = llm
        self.max_retries = 2  # Maximum number of retries after timeout
        self.timeout_seconds = 180  # 3 minutes

    def extract_metadata_with_schema(self, text: str, schema) -> dict:
        """
        Extract metadata from text using an LLM with timeout and retry logic.
        
        Args:
            text: The input text to extract metadata from
            schema: The schema to use for structured output
            
        Returns:
            Dict: Extracted metadata
        """
        s_llm = self.llm.with_structured_output(schema)
        prompt = f"Extract metadata from the following text according to these guidelines:\nExtract the metadata fields from the text following the schema provided.\n\nText excerpt (analyze the full text even if truncated here):\n{text[:100000]}... [text continues]\n\nReturn your analysis in the required JSON format."
        
        retries = 0
        while retries <= self.max_retries:
            try:
                print("Starting LLM request with timeout...")
                llm_response = self._invoke_llm_with_timeout(s_llm, prompt)
                print("LLM request completed.")

                if isinstance(llm_response, dict):
                    return llm_response
                else:
                    return json.loads(llm_response.content.replace("```json", "").replace("```", ""))
            
            except (TimeoutError, FuturesTimeoutError) as e:
                retries += 1
                print(f"LLM request timed out ({self.timeout_seconds}s). Retry {retries}/{self.max_retries}")
                if retries > self.max_retries:
                    print(f"Max retries exceeded. Returning empty metadata.")
                    return {}
            
            except Exception as e:
                print(f"Error parsing LLM metadata response: {e}, response: {llm_response if 'llm_response' in locals() else 'No response'}")
                return {}
    
    def _invoke_llm_with_timeout(self, llm, prompt):
        """
        Invoke the LLM with a timeout using concurrent.futures.
        
        Args:
            llm: The LLM instance to use
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response
            
        Raises:
            TimeoutError: If the LLM call exceeds the timeout
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm.invoke, prompt)
            try:
                print("Invoking LLM inside timeout wrapped function...")
                result = future.result(timeout=self.timeout_seconds)
                return result
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(f"LLM request timed out after {self.timeout_seconds} seconds")

    def extract(self, filepath: str):
        """
        Extract data from a file.
        
        Args:
            filepath: Path to the file to extract data from
            
        Returns:
            List[Dict]: Extracted data
        """
        data = []
        text = self.converter.convert(filepath)
        metadata = {"source": filepath}
        metadata.update(self.extract_metadata_with_schema(text, schema1))
        metadata.update(self.extract_metadata_with_schema(text, schema2))
        chunks = self.converter.chunk_smart(text, 5000)
        for i, chunk in enumerate(chunks):
            meta = metadata.copy()
            meta["text"] = chunk
            meta["chunk_index"] = i
            data.append(meta)
        for i, field in enumerate(extra_fields):
            meta = metadata.copy()
            meta["text"] = meta[field]
            meta["chunk_index"] = len(chunks) + i
            data.append(meta)
        return data
def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--config', '-c', type=str, help='Path to your configuration file')
    args = parser.parse_args()

    # Create and run crawler
    crawler = Crawler(args.config)
    crawler.set_llm()
    crawler.set_extractor(MyExtractor(crawler.config, crawler.llm))
    
    # Process all documents
    docs = []
    for data in crawler.run():
        docs.extend(data)

    with crawler._setup_vector_db() as db:
        db.insert_data(docs)

    print("complete")

if __name__ == "__main__":
    main()