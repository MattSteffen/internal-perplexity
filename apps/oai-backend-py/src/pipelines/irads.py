from typing import Any, cast

from crawler import CrawlerConfig
from crawler.chunker import ChunkingConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig

irad_library_description = (
    "You are about to process a collection of internal company research documents focused on "
    "signal processing, machine learning, and development initiatives. These materials contain "
    "proprietary research findings, technical methodologies, experimental results, implementation "
    "strategies, and development protocols specific to our organization's projects and objectives. "
    "The documents span various aspects of signal processing algorithms, machine learning model "
    "architectures, data analysis techniques, software development practices, and applied research "
    "outcomes. Each document represents internal knowledge, technical insights, and research "
    "progress that may include confidential methodologies, performance metrics, and strategic "
    "technical directions relevant to our company's research and development efforts."
)


schema1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Document Core Properties",
    "description": "Schema defining the fundamental metadata and unique terminology of a document.",
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
            "description": (
                "A list of terms or short phrases from the document that are domain-specific, "
                "highly technical, or might not be common knowledge. These words are crucial "
                "for a specialized understanding of the document's content."
            ),
            "items": {
                "type": "string",
                "maxLength": 100,
                "description": "A single unique or domain-specific term/phrase.",
            },
            "minItems": 0,
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
            "description": (
                "If the document explores a second distinct topic or presents another significant " "unique aspect, describe it here. This should cover a different core idea than " "summary_item_1."
            ),
        },
        "summary_item_3": {
            "type": "string",
            "maxLength": 15000,
            "description": (
                "If the document addresses a third distinct major theme or offers an additional "
                "unique insight, provide that summary here. Ensure it highlights a separate "
                "concept from the previous summary items."
            ),
        },
    },
}
# Combine schemas
schema1_required = cast(list[str], schema1.get("required", []) or [])
schema2_required = cast(list[str], schema2.get("required", []) or [])
schema1_props = cast(dict[str, Any], schema1.get("properties", {}) or {})
schema2_props = cast(dict[str, Any], schema2.get("properties", {}) or {})
metadata_schema = {
    "type": "object",
    "required": schema1_required + schema2_required,
    "properties": {**schema1_props, **schema2_props},
}


def create_default_config() -> CrawlerConfig:
    """Create type-safe configuration for default document processing."""
    return CrawlerConfig(
        embeddings=EmbedderConfig.ollama(model="all-minilm:v2", base_url="http://localhost:11434"),
        llm=LLMConfig.ollama(model_name="gpt-oss:20b", base_url="http://localhost:11434"),
        vision_llm=LLMConfig.ollama(model_name="granite3.2-vision:2b", base_url="http://localhost:11434"),
        database=DatabaseClientConfig.milvus(collection="temp", host="localhost", port=19530, username="root", password="Milvus"),
    )


def create_irad_config() -> CrawlerConfig:
    """Create type-safe configuration for IRADS document processing.

    This uses the new Pydantic-based configuration system for better type safety
    and validation.
    """
    # Embeddings configuration
    embeddings = EmbedderConfig.ollama(model="all-minilm:v2", base_url="http://localhost:11434")

    # Vision LLM for image processing
    vision_llm = LLMConfig.ollama(model_name="gpt-oss:20b", base_url="http://localhost:11434")

    # Main LLM for metadata extraction
    llm = LLMConfig.ollama(
        model_name="gpt-oss:20b",
        base_url="http://localhost:11434",
        structured_output="tools",
    )

    # Database configuration
    database = DatabaseClientConfig.milvus(
        collection="irad_documents",
        host="localhost",
        port=19530,
        username="root",
        password="Milvus",
        recreate=True,
    )

    # Multi-schema extractor configuration
    extractor = MetadataExtractorConfig(json_schema=metadata_schema, context=irad_library_description)

    # PyMuPDF4LLM converter configuration with image processing
    converter = PyMuPDF4LLMConfig(
        type="pymupdf4llm",
        vlm_config=vision_llm,
        image_prompt="Describe this image in detail for a technical document.",
        max_workers=4,
        to_markdown_kwargs={
            "page_chunks": False,
            "embed_images": True,
        },
    )

    chunking = ChunkingConfig.create(chunk_size=1000)

    # Create the complete crawler configuration
    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunking=chunking,
        metadata_schema=metadata_schema,
        temp_dir="/tmp/irads",
        benchmark=False,
    )

    return config
