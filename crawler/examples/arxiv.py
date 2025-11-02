from crawler import Crawler, CrawlerConfig
from crawler.extractor import (
    MetadataExtractorConfig,
    MetadataExtractor,
)
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.chunker import ChunkingConfig
from crawler.vector_db import DatabaseClientConfig, MilvusBenchmark

# Schema definitions for ArXiv document metadata extraction
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

# Combine schemas for metadata schema
metadata_schema = {
    "type": "object",
    "required": schema1.get("required", []) + schema2.get("required", []),
    "properties": {**schema1.get("properties", {}), **schema2.get("properties", {})},
}


# Type-safe configuration using new factory methods
def create_arxiv_config():
    """Create type-safe configuration for ArXiv document processing."""

    # Embeddings configuration
    embeddings = EmbedderConfig.ollama(
        model="all-minilm:v2", base_url="http://localhost:11434"
    )

    # Vision LLM for image processing
    vision_llm = LLMConfig.ollama(
        model_name="granite3.2-vision:latest", base_url="http://localhost:11434"
    )

    # Database configuration
    database = DatabaseClientConfig.milvus(
        collection="arxiv3",
        host="localhost",
        port=19530,
        username="matt",
        password="steffen",
        recreate=True,
    )

    # Main LLM for metadata extraction (using tools mode)
    llm = LLMConfig.ollama(
        # model_name="qwen3:latest",
        model_name="gpt-oss:20b",
        base_url="http://localhost:11434",
        # structured_output="response_format",  # Use OpenAI-compatible tools format
        structured_output="tools",  # Use OpenAI-compatible tools format
    )

    # Multi-schema extractor configuration
    extractor = MetadataExtractorConfig(
        json_schema=metadata_schema,
        context="A sequence of papers on topics related to clustering.",
    )

    # PyMuPDF4LLM converter configuration with image processing
    converter = PyMuPDF4LLMConfig(
        type="pymupdf4llm",
        vlm_config=vision_llm,
        image_prompt="Describe this image in detail for a technical document.",
        max_workers=4,
        to_markdown_kwargs={
            "page_chunks": False,
            "write_images": True,
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
        benchmark=True,
        security_groups=["read_only"],
    )

    return config


def search_louvain_clustering():
    """Demonstrate searching for documents related to Louvain clustering."""
    print("\n🔍 Demonstrating search functionality...")
    print("Searching for: 'louvain clustering'")

    # Create search configuration using the same database settings
    search_config = create_arxiv_config()
    search_benchmark = MilvusBenchmark(
        db_config=search_config.database, embed_config=search_config.embeddings
    )

    # Perform search
    search_results = search_benchmark.search(queries=["louvain clustering"], filters=[])

    print(f"\n📊 Search Results: Found {len(search_results)} matches")
    print("-" * 60)

    # Display top results
    for i, result in enumerate(search_results[:5]):  # Show top 5 results
        print(f"\n🔸 Result {i+1}:")
        print(f"   📏 Distance: {result.get('distance', 'N/A'):.4f}")

        # Access the entity data which contains the actual document fields
        entity = result.get("entity", {})

        # Get the actual document data from entity
        if hasattr(entity, "to_dict"):
            entity_data = entity.to_dict()
        else:
            entity_data = entity if isinstance(entity, dict) else {}

        # Get source and other fields (using new prefixed field names)
        source = entity_data.get("default_source") or result.get("default_id", "N/A")
        print(f"   📄 Source: {source}")

        # Try to get text from entity data
        text_content = entity_data.get("default_text")
        if text_content:
            print(f"   📝 Text Preview: {text_content[:200]}...")
        else:
            print("   📝 Text Preview: [Text content not available]")

        # Try to display metadata if available
        metadata_str = entity_data.get("default_metadata", "{}")
        if (
            isinstance(metadata_str, str)
            and metadata_str != "{}"
            and metadata_str != "null"
        ):
            try:
                import json

                metadata_dict = json.loads(metadata_str)
                if "title" in metadata_dict and metadata_dict["title"]:
                    print(f"   📖 Title: {metadata_dict['title']}")
                if "author" in metadata_dict and metadata_dict["author"]:
                    authors = metadata_dict["author"]
                    if isinstance(authors, list):
                        print(
                            f"   👤 Authors: {', '.join(authors[:2])}"
                            + ("..." if len(authors) > 2 else "")
                        )
                    else:
                        print(f"   👤 Author: {authors}")
                if (
                    "summary_item_1" in metadata_dict
                    and metadata_dict["summary_item_1"]
                ):
                    print(f"   📋 Summary: {metadata_dict['summary_item_1'][:150]}...")
            except json.JSONDecodeError:
                print(f"   📋 Metadata: {metadata_str[:100]}...")
        else:
            print("   📋 Metadata: [Not available]")

        # Display some entity fields for debugging
        if entity_data:
            available_fields = [
                k for k in entity_data.keys() if k != "text"
            ]  # Don't show full text
            print(
                f"   🔍 Available entity fields: {available_fields[:5]}"
                + ("..." if len(available_fields) > 5 else "")
            )

    print(
        f"\n✅ Search demo completed! Found {len(search_results)} relevant documents."
    )


# File paths for processing
short_options = [
    "/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv/2408.12236v1.pdf",
]
arxiv_dir_path = "/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv"


def main():
    """Main function to run the ArXiv document processing pipeline."""
    print("🚀 Starting ArXiv document processing with type-safe configuration...")

    # Create configuration using the new type-safe approach
    config = create_arxiv_config()

    print(f"📊 Configuration created:")
    print(f"   • Collection: {config.database.collection}")
    print(f"   • LLM: {config.llm.model_name}")
    print(f"   • Vision LLM: {config.vision_llm.model_name}")
    print(f"   • Chunk size: {config.chunking.chunk_size}")
    print(f"   • Benchmark: {config.benchmark}")

    # Create and run crawler
    mycrawler = Crawler(config)
    print("🔄 Starting document processing...")

    # Process documents
    mycrawler.crawl(short_options)

    # Run benchmark if enabled
    if config.benchmark:
        print("📊 Running benchmark analysis...")
        mycrawler.benchmark()

    # Demonstrate search functionality
    search_louvain_clustering()

    print("✅ ArXiv processing completed successfully!")


if __name__ == "__main__":
    main()
