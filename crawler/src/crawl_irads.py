from typing import Any, Dict
from crawler import Crawler
from processing.llm import LLM
from processing.extractor import Extractor, MultiSchemaExtractor

full_schema = {
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


irad_library_description = ""


irad_config = {
    "embeddings": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model": "gemma3:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "milvus": {
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "Milvus",
        "collection": "test_arxiv2",
        "recreate": True,
    },
    "llm": {
        "model": "gemma3",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "utils": {
        "chunk_size": 1000,
        "temp_dir": "/tmp/irads",
    }
}

dir_path = "/home/ubuntu/irads-crawler/data/irads"
short_options = ["/home/ubuntu/irads-crawler/data/irads/test.pdf"]


def main():
    myExtractor = MultiSchemaExtractor(irad_config, [schema1, schema2], irad_library_description)
    mycrawler = Crawler(irad_config, full_schema, None, myExtractor, None, None, None)
    mycrawler.crawl(short_options)


if __name__ == "__main__":
    main()