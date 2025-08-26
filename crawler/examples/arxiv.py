from crawler import Crawler, CrawlerConfig
from crawler.processing import MultiSchemaExtractor, OllamaLLM

schema1 = {
    "type": "object",
    "required": ["title", "author", "date", "keywords", "unique_words"],
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 2550,
            "description": "The official title of the document. Should be concise and accurately reflect the document's content.",
        },
        "author": {
            "type": "array",
            "description": "A list of all individuals or entities responsible for creating the document.",
            "items": {
                "type": "string",
                "maxLength": 2550,
                "description": "The full name of an individual author or the name of an organizational author (e.g., 'John Doe', 'Example Corporation').",
            },
        },
        "date": {
            "type": "integer",
            "description": "The year of the document's official publication or last significant revision. Please enter as a four-digit year (YYYY).",
            "minimum": 1900,
            "maximum": 2100,
        },
        "keywords": {
            "type": "array",
            "description": "A list of relevant terms or phrases that categorize the document's subject matter, aiding in search and discovery.",
            "items": {
                "type": "string",
                "maxLength": 5120,
                "description": "An individual keyword or key phrase (e.g., 'artificial intelligence', 'machine learning applications').",
            },
        },
        "unique_words": {
            "type": "array",
            "description": "A list of terms or short phrases from the document that are domain-specific, highly technical, or might not be common knowledge. These words are crucial for a specialized understanding of the document's content.",
            "items": {
                "type": "string",
                "maxLength": 100,
                "description": "A single unique or domain-specific term/phrase.",
            },
            "minItems": 0,
        },
        "description": {
            "type": "string",
            "maxLength": 15000,
            "description": "A brief overview of the document's content, including its main arguments, findings, or contributions. This should be a high-level summary that captures the essence of the document.",
        },
    },
}
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
            "description": "A concise summary of the primary topic or a unique, central argument discussed in the document. Focus on the most significant general idea or contribution.",
        },
        "summary_item_2": {
            "type": "string",
            "maxLength": 15000,
            "description": "If the document explores a second distinct topic or presents another significant unique aspect, describe it here. This should cover a different core idea than summary_item_1.",
        },
        "summary_item_3": {
            "type": "string",
            "maxLength": 15000,
            "description": "If the document addresses a third distinct major theme or offers an additional unique insight, provide that summary here. Ensure it highlights a separate concept from the previous summary items.",
        },
    },
}

# Combine schemas
metadata_schema = {
    "type": "object",
    "required": schema1.get("required", []) + schema2.get("required", []),
    "properties": {**schema1.get("properties", {}), **schema2.get("properties", {})},
}


arxiv_config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model_name": "granite-3.2vision:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "Milvus",
        "collection": "test_arxiv3",
        "recreate": True,
    },
    "llm": {
        "model_name": "qwen3:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "extractor": {
        "type": "multi_schema",
        "llm": {
            "model_name": "qwen3:latest",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        },
        "metadata_schema": [schema1, schema2],
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
            "image_description_prompt": "Describe this image in detail for a technical document.",
            "image_describer": {
                "type": "ollama",
                "model": "granite-3.2vision:latest",
                "base_url": "http://localhost:11434",
            },
        }
    },
    "utils": {
        "chunk_size": 1000,
        "benchmark": True,
    },
    "metadata_schema": metadata_schema,
}


short_options = [
    "/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv/2408.12236v1.pdf",
]
arxiv_dir_path = "/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv"


def main():
    config = CrawlerConfig.from_dict(arxiv_config_dict)
    config.log_level = "INFO"  # Set log level for testing
    mycrawler = Crawler(config)  # Extractor will be created from config
    mycrawler.crawl(short_options)
    # mycrawler.benchmark()  # Comment out benchmark for now


if __name__ == "__main__":
    main()
