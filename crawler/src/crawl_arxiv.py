import json
import os
import argparse
from pydoc import Doc

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
    },
    "description": {
      "type": "string",
      "maxLength": 15000,
      "description": "A brief overview of the document's content, including its main arguments, findings, or contributions. This should be a high-level summary that captures the essence of the document."
    },
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


class MyExtractor:
    """
    Custom extractor class for processing documents.
    This class is a placeholder and should be implemented with actual logic.
    """
    def __init__(self, config: dict, llm) -> None:
        self.config = config
        # self.converter = DoclingConverter(config)  
        self.converter = MarkItDownConverter(config)
        self.llm = llm

    def extract_metadata_with_schema(self, text: str, schema) -> dict:
        s_llm = self.llm.with_structured_output(schema)
        prompt = f"Extract metadata from the following text according to these guidelines:\nExtract the metadata fields from the text following the schema provided.\n\nText excerpt (analyze the full text even if truncated here):\n{text[:100000]}... [text continues]\n\nReturn your analysis in the required JSON format."
        try:
            llm_response = s_llm.invoke(prompt)
            if isinstance(llm_response, dict):
                return llm_response
            else:
                return json.loads(llm_response.content.replace("```json", "").replace("```", ""))
        except Exception as e:
            print(f"Error parsing LLM metadata response: {e}, response: {llm_response}")
            return {}

    def extract(self, filepath: str):
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
    arxiv_yaml = "/Users/mattsteffen/projects/llm/internal-perplexity/crawler/src/config/directories/arxiv.yaml"

    # Create and run crawler
    arxiv_crawler = Crawler(arxiv_yaml)
    arxiv_crawler.set_llm()
    arxiv_crawler.set_extractor(MyExtractor(arxiv_crawler.config, arxiv_crawler.llm))
    
    # Process all documents
    docs = []
    for data in arxiv_crawler.run():
        docs.extend(data)

    with arxiv_crawler._setup_vector_db() as db:
        db.insert_data(docs)

    print("complete")

if __name__ == "__main__":
    main()