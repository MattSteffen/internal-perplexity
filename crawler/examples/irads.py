from crawler import Crawler, CrawlerConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig
from crawler.chunker import ChunkingConfig


full_schema = {
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
            "description": "A list of terms or short phrases from the document that are domain-specific, highly technical, or might not be common knowledge. These words are crucial for a specialized understanding of the document's content.",
            "items": {
                "type": "string",
                "maxLength": 100,
                "description": "A single unique or domain-specific term/phrase.",
            },
            "minItems": 0,
        },
        "primary_contribution": {
            "type": "string",
            "maxLength": 15000,
            "description": (
                "A concise summary capturing the document's primary technical contribution. "
                "Focus on the main problem addressed, its motivation, and the core solution "
                "approach. For IRAD, this typically means the central signal-processing "
                "method, machine-learning architecture, or development goal."
            ),
        },
        "methodology_overview": {
            "type": "string",
            "maxLength": 15000,
            "description": (
                "A focused description of the most significant methodology used in the "
                "document. Summarize algorithms, model architectures, experimental setup, "
                "workflow design, or implementation strategy. Should clarify *how* the "
                "research was performed or *how* the system works."
            ),
        },
        "key_findings": {
            "type": "string",
            "maxLength": 15000,
            "description": (
                "A summary of the key findings, results, performance metrics, observed "
                "behaviors, or engineering insights. May include limitations, tradeoffs, "
                "deployment considerations, or recommendations for future work within the "
                "R&D context."
            ),
        },
    },
}

irad_library_description = "You are about to process a collection of internal company research documents focused on signal processing, machine learning, and development initiatives. These materials contain proprietary research findings, technical methodologies, experimental results, implementation strategies, and development protocols specific to our organization's projects and objectives. The documents span various aspects of signal processing algorithms, machine learning model architectures, data analysis techniques, software development practices, and applied research outcomes. Each document represents internal knowledge, technical insights, and research progress that may include confidential methodologies, performance metrics, and strategic technical directions relevant to our company's research and development efforts."

# File paths for processing
dir_path = "/home/ubuntu/irads-crawler/data/irads"
short_options = ["/home/ubuntu/irads-crawler/data/irads/test.pdf"]


def create_irad_config() -> CrawlerConfig:
    """Create type-safe configuration for IRADS document processing.

    This uses the new Pydantic-based configuration system for better type safety
    and validation.
    """
    # Embeddings configuration
    embeddings = EmbedderConfig.ollama(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )

    # Vision LLM for image processing
    vision_llm = LLMConfig.ollama(
        model_name="gemma3:latest", base_url="http://localhost:11434"
    )

    # Main LLM for metadata extraction
    llm = LLMConfig.ollama(
        model_name="gemma3",
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
    extractor = MetadataExtractorConfig(
        json_schema=full_schema, context=irad_library_description
    )

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
        metadata_schema=full_schema,
        temp_dir="/tmp/irads",
        benchmark=False,
    )

    return config


def main():
    """Main function to run the IRADS document processing pipeline."""
    print("🚀 Starting IRADS document processing with type-safe configuration...")

    # Create configuration using the new type-safe approach
    config = create_irad_config()

    # Alternative: Use dictionary-based configuration for backward compatibility
    # config = CrawlerConfig.from_dict(irad_config_dict)

    print(f"📊 Configuration created:")
    print(f"   • Collection: {config.database.collection}")
    print(f"   • LLM: {config.llm.model_name}")
    print(f"   • Vision LLM: {config.vision_llm.model_name}")
    print(f"   • Chunk size: {config.chunking.chunk_size}")

    # Create and run crawler
    mycrawler = Crawler(config)
    print("🔄 Starting document processing...")

    # Process documents
    mycrawler.crawl(short_options)

    # Run benchmark if enabled
    if config.benchmark:
        print("📊 Running benchmark analysis...")
        mycrawler.benchmark()

    print("✅ IRADS document processing completed successfully!")


if __name__ == "__main__":
    main()
