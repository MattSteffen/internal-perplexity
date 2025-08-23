"""
X-Midas Data Crawler

This module provides functionality to crawl and index three types of X-Midas data:
1. LearnXM data (learnxm.json) - Learning documentation with subjects, descriptions, tags, and URLs
2. XM Docs data (xm_docs.json) - X-Midas documentation with subjects, descriptions, tags, and file paths
3. Q&A data (processed_xm_qa.json) - Question and answer data with context and users

The crawler uses the current crawler architecture with:
- CrawlerConfig for configuration management
- BasicExtractor for metadata extraction using LLMs
- Milvus for vector storage
- Proper preprocessing to convert JSON data to crawler-compatible format

Usage:
    python crawl_xmidas.py              # Crawl all data sources
    python crawl_xmidas.py test         # Run basic tests
    python -c "from crawler.crawl_xmidas import crawl_learnxm; crawl_learnxm()"  # Crawl specific data source

Configuration:
- Update DATA_PATHS dictionary to point to your data files
- Modify xm_config for your specific environment (LLM endpoints, Milvus connection, etc.)
- Adjust temp_dir paths as needed

Data Format Requirements:
- All data files should be JSON arrays of objects
- Each data source has its own schema defined in this module
- The crawler will create individual JSON files for each item in the temp directory
"""

from typing import Any, Dict, List
import json, copy, os, sys
import uuid
from pathlib import Path
from crawler.src import Crawler, CrawlerConfig
from crawler.src.processing import BasicExtractor, PyMuPDFConverter
from crawler.src.processing.llm import LLM
from crawler.src.storage import get_db


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
xm_config = {
    "embeddings": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://ollama.a1.autobahn.rinconres.com",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model": "mistral-small3.2:latest",
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
        "model": "mistral-small3.2",
        "provider": "ollama",
        "base_url": "http://ollama.a1.autobahn.rinconres.com",
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


def crawl_data_source(data_type: str, schema: Dict[str, Any], partition: str):
    """Generic function to crawl a specific data source."""
    config = xm_config.copy()
    config["database"]["partition"] = partition
    temp_dir = f"{config['utils']['temp_dir']}/{data_type}"

    # Create temp directory
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    data_path = DATA_PATHS[data_type]
    if not os.path.exists(data_path):
        print(f"Warning: Data file {data_path} not found. Skipping {data_type}.")
        return

    data = json.load(open(data_path, "r"))

    # Preprocess based on data type
    if data_type == "learnxm":
        preprocess_learnxm_data(data, temp_dir)
    elif data_type == "xm_docs":
        preprocess_xm_docs_data(data, temp_dir)
    elif data_type == "qa":
        preprocess_qa_data(data, temp_dir)

    # Create crawler with proper configuration
    crawler_config = CrawlerConfig.from_dict(config)
    crawler_config.metadata_schema = schema

    # Initialize LLM for extraction
    llm = LLM(crawler_config.llm)

    # Create extractor
    extractor = BasicExtractor(schema, llm, learnxm_description)

    # Create crawler
    mycrawler = Crawler(crawler_config, extractor=extractor)

    # Crawl the data
    mycrawler.crawl(temp_dir)
    print(f"\nCompleted crawling {data_type} data")


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
    """Crawl all X-Midas data sources."""
    print("Starting X-Midas data crawling...")

    try:
        crawl_learnxm()
        print("LearnXM data crawled successfully")
    except Exception as e:
        print(f"Error crawling LearnXM data: {e}")

    try:
        crawl_xm_docs()
        print("XM Docs data crawled successfully")
    except Exception as e:
        print(f"Error crawling XM Docs data: {e}")

    try:
        crawl_qa()
        print("Q&A data crawled successfully")
    except Exception as e:
        print(f"Error crawling Q&A data: {e}")

    print("X-Midas data crawling completed!")


def main():
    """Main function - crawl all data sources."""
    crawl_all()
