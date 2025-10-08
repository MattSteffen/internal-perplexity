from crawler import Crawler, CrawlerConfig
from crawler.processing import (
    ExtractorConfig,
    ConverterConfig,
    EmbedderConfig,
    LLMConfig,
)
from crawler.storage import DatabaseClientConfig


# https://user.xmission.com/~research/mormonpdf/index.htm
# Example configuration for crawling church documents

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

# Combine schemas
metadata_schema = {
    "type": "object",
    "required": schema1.get("required", []) + ["summary_item_1", "summary_item_2", "summary_item_3"],
    "properties": {**schema1.get("properties", {}), **{
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
    }},
}

church_config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model_name": "gemma3:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "Milvus",
        "collection": "church_documents",
        "recreate": True,
    },
    "llm": {
        "model_name": "gemma3",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "extractor": {
        "type": "multi_schema",
        "llm": {
            "model_name": "gemma3",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        },
        "metadata_schema": [schema1, {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Document Summary Points",
            "description": "Schema defining distinct summary aspects of a document.",
            "type": "object",
            "required": ["summary_item_1"],
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
                },
            },
        }],
    },
    "converter": {
        "type": "pymupdf",
        "metadata": {
            "preserve_formatting": True,
            "include_page_numbers": True,
            "include_metadata": True,
            "sort_reading_order": True,
            "extract_tables": True,
            "table_strategy": "lines_strict",
            "image_description_prompt": "Describe this image in detail for a religious document.",
            "image_describer": {
                "type": "ollama",
                "model": "gemma3:latest",
                "base_url": "http://localhost:11434",
            },
        }
    },
    "utils": {
        "chunk_size": 1000,
        "temp_dir": "/tmp/church",
    },
    "metadata_schema": metadata_schema,
}

# File paths for processing
dir_path = "/home/ubuntu/irads-crawler/data/irads"
short_options = ["/home/ubuntu/irads-crawler/data/irads/test.pdf"]


def create_church_config() -> CrawlerConfig:
    """Create type-safe configuration for church document processing.
    
    This uses the new Pydantic-based configuration system for better type safety
    and validation.
    """
    # Define schema for the second part (summary items)
    schema2 = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Document Summary Points",
        "description": "Schema defining distinct summary aspects of a document.",
        "type": "object",
        "required": ["summary_item_1"],
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
            },
        },
    }
    
    # Embeddings configuration
    embeddings = EmbedderConfig.ollama(
        model="all-minilm:v2", 
        base_url="http://localhost:11434"
    )
    
    # Vision LLM for image processing
    vision_llm = LLMConfig.ollama(
        model_name="gemma3:latest",
        base_url="http://localhost:11434"
    )
    
    # Main LLM for metadata extraction
    llm = LLMConfig.ollama(
        model_name="gemma3",
        base_url="http://localhost:11434",
        structured_output="tools"
    )
    
    # Database configuration
    database = DatabaseClientConfig.milvus(
        collection="church_documents",
        host="localhost",
        port=19530,
        username="root",
        password="Milvus",
        recreate=True,
    )
    
    # Multi-schema extractor configuration
    extractor = ExtractorConfig.multi_schema(
        schemas=[schema1, schema2],
        llm=llm,
        document_library_context="Religious and church documents"
    )
    
    # PyMuPDF converter configuration with image processing
    converter = ConverterConfig.pymupdf(
        vision_llm=vision_llm,
        metadata={
            "preserve_formatting": True,
            "include_page_numbers": True,
            "include_metadata": True,
            "sort_reading_order": True,
            "extract_tables": True,
            "table_strategy": "lines_strict",
            "image_description_prompt": "Describe this image in detail for a religious document.",
            "image_describer": {
                "type": "ollama",
                "model": "gemma3:latest",
                "base_url": "http://localhost:11434",
            },
        },
    )
    
    # Create the complete crawler configuration
    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunk_size=1000,
        metadata_schema=metadata_schema,
        temp_dir="/tmp/church",
        benchmark=False,
        log_level="INFO",
    )
    
    return config


def main():
    """Main function to run the church document processing pipeline."""
    print("🚀 Starting church document processing with type-safe configuration...")
    
    # Create configuration using the new type-safe approach
    config = create_church_config()
    
    # Alternative: Use dictionary-based configuration for backward compatibility
    # config = CrawlerConfig.from_dict(church_config_dict)
    
    print(f"📊 Configuration created:")
    print(f"   • Collection: {config.database.collection}")
    print(f"   • LLM: {config.llm.model_name}")
    print(f"   • Vision LLM: {config.vision_llm.model_name}")
    print(f"   • Chunk size: {config.chunk_size}")
    
    # Create and run crawler
    mycrawler = Crawler(config)
    print("🔄 Starting document processing...")
    
    # Process documents
    mycrawler.crawl(short_options)
    
    # Run benchmark if enabled
    if config.benchmark:
        print("📊 Running benchmark analysis...")
        mycrawler.benchmark()
    
    print("✅ Church document processing completed successfully!")


if __name__ == "__main__":
    main()