"""
X-Midas Data Crawler Example

This module demonstrates how to use the Crawler package to process and index X-Midas data sources.
X-Midas is a specialized programming language and environment designed for signal processing applications.
This crawler handles three types of X-Midas data sources:

1. LearnXM Data (learnxm.json)
   - Learning documentation with subjects, descriptions, tags, and URLs
   - Educational materials teaching X-Midas programming concepts
   - Focuses on syntax, data flow paradigms, and signal processing techniques

2. XM Docs Data (xm_docs.json)
   - X-Midas documentation with subjects, descriptions, tags, and file paths
   - Technical reference documentation for X-Midas functionality
   - Includes help files, documentation, and example code references

3. Q&A Data (processed_xm_qa.json)
   - Question and answer data with context and user information
   - Community discussions and technical support interactions
   - Timestamped conversations about X-Midas usage and troubleshooting

Architecture:
The crawler leverages the Crawler package's modular architecture:
- CrawlerConfig: Centralized configuration management
- BasicExtractor: LLM-powered metadata extraction and enhancement
- Milvus: Vector database for embeddings and semantic search
- Custom preprocessing: Converts JSON data to crawler-compatible format
- Batch processing: Handles large datasets efficiently

Usage Examples:

1. Crawl all data sources:
   python xmidas.py

2. Crawl specific data source:
   from crawler.examples.xmidas import crawl_learnxm
   crawl_learnxm()

3. Programmatic usage:
   from crawler.examples.xmidas import crawl_data_source
   crawl_data_source("learnxm", learnxm_schema, "learnxm_partition")

Configuration Requirements:

1. Data Source Paths:
   Update the DATA_PATHS dictionary to point to your local data files:
   - "/data/Copilot/learnxm.json"
   - "/data/Copilot/xm_docs.json"
   - "/data/Copilot/processed_xm_qa.json"

2. Environment Setup:
   - Ollama server running with appropriate models (mistral-small3.2)
   - Milvus vector database accessible
   - Sufficient disk space for temporary files (temp_dir: /tmp/xm)

3. Model Configuration:
   - Embedding model: nomic-embed-text (4096 dimensions)
   - LLM: mistral-small3.2 for metadata extraction
   - Vision model: mistral-small3.2 for image descriptions

Data Format Requirements:

Input Files:
- Must be JSON arrays of objects
- Each data source has its own JSON schema defined in this module
- Files are preprocessed into individual JSON documents for the crawler

Output:
- Processed documents stored in Milvus vector database
- Metadata enhanced with LLM-extracted information
- Embeddings generated for semantic search capabilities

Available Functions:

Core Functions:
- crawl_data_source(data_type, schema, partition): Generic crawler for any data source
- crawl_learnxm(): Crawl LearnXM learning documentation
- crawl_xm_docs(): Crawl X-Midas technical documentation
- crawl_qa(): Crawl Q&A data
- crawl_all(): Process all data sources sequentially

Preprocessing Functions:
- preprocess_learnxm_data(): Convert LearnXM JSON to crawler format
- preprocess_xm_docs_data(): Convert XM Docs JSON to crawler format
- preprocess_qa_data(): Convert Q&A JSON to crawler format

Configuration:
- xm_config_dict: Complete crawler configuration
- DATA_PATHS: Data source file paths
- Schema definitions: JSON schemas for each data type

Dependencies:
- crawler package (main processing framework)
- json, os, sys (standard library)
- pathlib (standard library)
- typing (standard library)

Error Handling:
- Graceful handling of missing data files
- Continues processing other data sources if one fails
- Comprehensive logging for debugging and monitoring

Performance Considerations:
- Batch processing for large datasets
- Configurable chunk sizes for embedding
- Temporary file cleanup
- Memory-efficient processing for large JSON files

Example Output:
After successful crawling, the system provides:
- Vector embeddings for semantic search
- Enhanced metadata with LLM-extracted insights
- Structured data ready for RAG applications
- Benchmarking results for search performance evaluation
"""

from typing import Any, Dict, List
import json
import os
from pathlib import Path

from crawler import Crawler, JsonDataPreset
from crawler.extractor import MetadataExtractorConfig
from crawler.converter import ConverterConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig


# Schema for LearnXM data (learnxm.json)
learnxm_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "LearnXM Document Schema",
    "description": "Schema for X-Midas learning documentation entries",
    "type": "object",
    "required": ["subject", "description", "tags", "url"],
    "properties": {
        "subject": {
            "type": "string",
            "maxLength": 2550,
            "description": "The title/subject of the learning documentation",
        },
        "description": {
            "type": "string",
            "maxLength": 65000,
            "description": "The detailed description of the X-Midas learning content",
        },
        "url": {
            "type": "string",
            "maxLength": 2048,
            "description": "The URL where the documentation can be found",
        },
        "tags": {
            "type": "array",
            "description": "A list of relevant tags for categorization",
            "items": {
                "type": "string",
                "maxLength": 1024,
                "description": "Individual tag or keyword",
            },
        },
    },
}

# Schema for XM Docs data (xm_docs.json)
xm_docs_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "XM Documentation Schema",
    "description": "Schema for X-Midas documentation entries",
    "type": "object",
    "required": ["subject", "description", "tags", "xm_path"],
    "properties": {
        "subject": {
            "type": "string",
            "maxLength": 2550,
            "description": "The subject/title of the documentation",
        },
        "description": {
            "type": "string",
            "maxLength": 65000,
            "description": "The description of the X-Midas functionality",
        },
        "xm_path": {
            "type": "string",
            "maxLength": 512,
            "description": "The path to the help file, documentation, or example code",
        },
        "tags": {
            "type": "array",
            "description": "A list of relevant terms or phrases that categorize the document",
            "items": {
                "type": "string",
                "maxLength": 255,
                "description": "An individual keyword or key phrase",
            },
        },
    },
}

# Schema for Q&A data (processed_xm_qa.json)
qa_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "X-Midas Q&A Schema",
    "description": "Schema for X-Midas question and answer data",
    "type": "object",
    "required": ["question", "answer", "context", "users", "time"],
    "properties": {
        "question": {
            "type": "string",
            "maxLength": 2550,
            "description": "The question asked by the user",
        },
        "answer": {
            "type": "string",
            "maxLength": 2048,
            "description": "The answer provided",
        },
        "context": {
            "type": "string",
            "maxLength": 65000,
            "description": "The conversation context or additional information",
        },
        "users": {
            "type": "array",
            "description": "List of users involved in the conversation",
            "items": {
                "type": "string",
                "maxLength": 255,
                "description": "Username",
            },
        },
        "time": {
            "type": "string",
            "maxLength": 255,
            "description": "Timestamp of the interaction",
        },
    },
}
learnxm_description = """These documents teach X-Midas, a specialized programming language and environment designed for signal processing applications. When reading them, pay attention to the unique syntax and data flow paradigms that distinguish X-Midas from conventional programming languages. Focus on understanding how X-Midas handles signal data types, particularly its approach to complex data structures and multi-dimensional arrays commonly used in RF and digital signal processing.
Key areas to emphasize while studying include: the command-line interface and interactive programming model, file I/O operations for signal data formats, built-in mathematical functions for frequency domain analysis, and the integration between X-Midas primitives and custom user-defined functions. Pay special attention to memory management concepts, as efficient handling of large signal datasets is crucial for real-world applications.
These materials will guide you through both fundamental concepts and advanced techniques, preparing you to develop signal processing workflows, perform spectral analysis, and create custom processing chains. The documents are structured to build proficiency from basic syntax through complex multi-stage signal processing applications commonly encountered in communications, radar, and electronic warfare domains."""


# Base configuration for X-Midas crawling
xm_config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://ollama.a1.autobahn.rinconres.com",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model_name": "mistral-small3.2:latest",
        "provider": "ollama",
        "base_url": "http://ollama.a1.autobahn.rinconres.com",
    },
    "database": {
        "provider": "milvus",
        "host": "10.43.210.111",
        "port": 19530,
        "username": "root",
        "password": "Milvus",
        "collection": "xmidas",
        "recreate": False,
    },
    "llm": {
        "model_name": "mistral-small3.2",
        "provider": "ollama",
        "base_url": "http://ollama.a1.autobahn.rinconres.com",
    },
    "extractor": {
        "type": "basic",
        "llm": {
            "model_name": "mistral-small3.2",
            "provider": "ollama",
            "base_url": "http://ollama.a1.autobahn.rinconres.com",
        },
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
                "model": "mistral-small3.2:latest",
                "base_url": "http://ollama.a1.autobahn.rinconres.com",
            },
        },
    },
    "utils": {
        "chunk_size": 10000,
        "temp_dir": "/tmp/xm",
    },
}

# Data source paths
DATA_PATHS = {
    "learnxm": "/data/Copilot/learnxm.json",
    "xm_docs": "/data/Copilot/xm_docs.json",
    "qa": "/data/Copilot/processed_xm_qa.json",
}


"""
Preprocessing functions for X-Midas data sources
"""


def preprocess_learnxm_data(data: List[Dict[str, Any]], temp_dir: str) -> None:
    """Preprocess LearnXM data into crawler-compatible format."""
    print("Processing LearnXM data...")
    for i, item in enumerate(data):
        # Create text content from the learning documentation
        text = f"Subject: {item.get('subject', '')}\n\nDescription: {item.get('description', '')}\n\nURL: {item.get('url', '')}"

        # Prepare metadata
        metadata = {
            "subject": item.get("subject", ""),
            "description": item.get("description", ""),
            "url": item.get("url", ""),
            "tags": item.get("tags", []),
            "data_type": "learnxm",
            "source": "learnxm.json",
        }

        # Save as JSON file for crawler
        output_file = f"{temp_dir}/learnxm_{i}.json"
        json.dump({"text": text, "metadata": metadata}, open(output_file, "w"))
        print(f"Processed LearnXM item {i+1}/{len(data)}", end="\r")


def preprocess_xm_docs_data(data: List[Dict[str, Any]], temp_dir: str) -> None:
    """Preprocess XM Docs data into crawler-compatible format."""
    print("Processing XM Docs data...")
    for i, item in enumerate(data):
        # Create text content from the documentation
        text = f"Subject: {item.get('subject', '')}\n\nDescription: {item.get('description', '')}\n\nPath: {item.get('xm_path', '')}"

        # Prepare metadata
        metadata = {
            "subject": item.get("subject", ""),
            "description": item.get("description", ""),
            "xm_path": item.get("xm_path", ""),
            "tags": item.get("tags", []),
            "data_type": "xm_docs",
            "source": "xm_docs.json",
        }

        # Save as JSON file for crawler
        output_file = f"{temp_dir}/xm_docs_{i}.json"
        json.dump({"text": text, "metadata": metadata}, open(output_file, "w"))
        print(f"Processed XM Docs item {i+1}/{len(data)}", end="\r")


def preprocess_qa_data(data: List[Dict[str, Any]], temp_dir: str) -> None:
    """Preprocess Q&A data into crawler-compatible format."""
    print("Processing Q&A data...")
    for i, item in enumerate(data):
        # Create text content from the Q&A
        text = f"Question: {item.get('question', '')}\n\nAnswer: {item.get('answer', '')}\n\nContext: {item.get('context', '')}"

        # Prepare metadata
        metadata = {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "context": item.get("context", ""),
            "users": item.get("users", []),
            "time": item.get("time", ""),
            "data_type": "qa",
            "source": "processed_xm_qa.json",
        }

        # Save as JSON file for crawler
        output_file = f"{temp_dir}/qa_{i}.json"
        json.dump({"text": text, "metadata": metadata}, open(output_file, "w"))
        print(f"Processed Q&A item {i+1}/{len(data)}", end="\r")


def create_xmidas_config(
    partition: str = "default",
    metadata_schema: Dict[str, Any] = None,
    temp_dir: str = "/tmp/xm",
):
    """Create configuration for X-Midas document processing using preset and builder pattern.

    Args:
        partition: Milvus partition name for data organization
        metadata_schema: JSON schema for metadata validation
        temp_dir: Temporary directory for caching

    Returns:
        CrawlerConfig instance
    """
    # Start with JsonDataPreset, then override with X-Midas specific settings
    base_url = "http://ollama.a1.autobahn.rinconres.com"
    
    config = JsonDataPreset.create(
        collection="xmidas",
        llm_base_url=base_url,
        embedder_base_url=base_url,
        llm_model="mistral-small3.2",
        embedder_model="nomic-embed-text",
        chunk_size=10000,
        metadata_schema=metadata_schema or {},
        host="10.43.210.111",
        port=19530,
        username="root",
        password="Milvus",
        recreate=False,
        temp_dir=temp_dir,
        benchmark=False,
        context=learnxm_description,
    )
    
    # Override database to add partition support
    database = DatabaseClientConfig.milvus(
        collection="xmidas",
        host="10.43.210.111",
        port=19530,
        username="root",
        password="Milvus",
        recreate=False,
        partition=partition,
    )
    
    # Override extractor with X-Midas specific context
    extractor = MetadataExtractorConfig(
        json_schema=metadata_schema or {},
        context=learnxm_description,
    )
    
    # Override converter for image processing (even though JSON, may have embedded images)
    vision_llm = LLMConfig.ollama(
        model_name="mistral-small3.2:latest",
        base_url=base_url,
    )
    
    converter = ConverterConfig(
        type="pymupdf4llm",
        vlm_config=vision_llm,
        image_prompt="Describe this image in detail for a technical document.",
        max_workers=4,
        to_markdown_kwargs={
            "page_chunks": False,
            "write_images": True,
        },
    )
    
    # Apply all overrides
    config = config.model_copy(update={
        "database": database,
        "extractor": extractor,
        "converter": converter,
        "vision_llm": vision_llm,
    })
    
    return config


def crawl_data_source(data_type: str, schema: Dict[str, Any], partition: str):
    """Generic function to crawl a specific data source.

    Args:
        data_type: Type of data source (learnxm, xm_docs, qa)
        schema: JSON schema for metadata validation
        partition: Milvus partition name for data organization
    """
    temp_dir = f"/tmp/xm/{data_type}"

    # Create temp directory
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    data_path = DATA_PATHS[data_type]
    if not os.path.exists(data_path):
        print(f"Warning: Data file {data_path} not found. Skipping {data_type}.")
        return

    print(f"📂 Loading data from {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    # Preprocess based on data type
    print(f"🔄 Preprocessing {data_type} data...")
    if data_type == "learnxm":
        preprocess_learnxm_data(data, temp_dir)
    elif data_type == "xm_docs":
        preprocess_xm_docs_data(data, temp_dir)
    elif data_type == "qa":
        preprocess_qa_data(data, temp_dir)

    # Create crawler with preset configuration
    print(f"⚙️  Creating configuration for {data_type}...")
    crawler_config = create_xmidas_config(
        partition=partition, metadata_schema=schema, temp_dir=temp_dir
    )

    print(f"🚀 Starting crawler for {data_type}...")
    print(f"   • Collection: {crawler_config.database.collection}")
    print(f"   • Partition: {partition}")
    print(f"   • LLM: {crawler_config.llm.model_name}")

    # Create crawler - can also use builder pattern for runtime overrides
    # Example: mycrawler = Crawler(crawler_config).with_llm(LLMConfig.ollama(...))
    mycrawler = Crawler(crawler_config)

    # Crawl the data
    mycrawler.crawl(temp_dir)
    print(f"\n✅ Completed crawling {data_type} data")


def crawl_learnxm():
    """Crawl LearnXM data."""
    crawl_data_source("learnxm", learnxm_schema, "learnxm")


def crawl_xm_docs():
    """Crawl XM Docs data."""
    crawl_data_source("xm_docs", xm_docs_schema, "xm_docs")


def crawl_qa():
    """Crawl Q&A data."""
    crawl_data_source("qa", qa_schema, "qa")


def crawl_all():
    """Crawl all X-Midas data sources with comprehensive error handling."""
    print("=" * 80)
    print("🚀 Starting X-Midas Data Crawling Pipeline")
    print("=" * 80)
    print()

    results = {"LearnXM": False, "XM Docs": False, "Q&A": False}

    try:
        print("\n📚 Processing LearnXM Documentation...")
        print("-" * 80)
        crawl_learnxm()
        results["LearnXM"] = True
        print("✅ LearnXM data crawled successfully")
    except Exception as e:
        print(f"❌ Error crawling LearnXM data: {e}")

    try:
        print("\n📖 Processing XM Documentation...")
        print("-" * 80)
        crawl_xm_docs()
        results["XM Docs"] = True
        print("✅ XM Docs data crawled successfully")
    except Exception as e:
        print(f"❌ Error crawling XM Docs data: {e}")

    try:
        print("\n💬 Processing Q&A Data...")
        print("-" * 80)
        crawl_qa()
        results["Q&A"] = True
        print("✅ Q&A data crawled successfully")
    except Exception as e:
        print(f"❌ Error crawling Q&A data: {e}")

    # Print summary
    print()
    print("=" * 80)
    print("📊 Crawling Summary")
    print("=" * 80)
    successful = sum(results.values())
    total = len(results)

    for data_source, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {data_source:.<30} {status}")

    print()
    print(f"   Total: {successful}/{total} data sources crawled successfully")
    print("=" * 80)

    if successful == total:
        print("🎉 All X-Midas data sources crawled successfully!")
    elif successful > 0:
        print(f"⚠️  {total - successful} data source(s) failed to crawl")
    else:
        print("❌ All data sources failed to crawl")


def main():
    """Main function - crawl all X-Midas data sources.

    This function demonstrates the complete X-Midas data processing pipeline,
    including preprocessing, configuration, and crawling multiple data sources.
    """
    crawl_all()
