"""
Examples of type-safe configuration for the document crawler.

This file demonstrates how to create configurations using the new type-safe approach
instead of JSON dictionaries. The new approach provides:

1. Full type checking and IDE autocompletion
2. Validation at configuration creation time
3. Factory methods for common configurations
4. Clear, readable configuration code
"""

from .processing import EmbedderConfig, LLMConfig, ConverterConfig, ExtractorConfig
from .storage import DatabaseClientConfig
from .main import CrawlerConfig


def example_basic_ollama_config():
    """Example: Basic configuration using Ollama models."""
    # Create embedding configuration
    embeddings = EmbedderConfig.ollama(
        model="all-minilm:v2", base_url="http://localhost:11434"
    )

    # Create LLM configurations
    llm = LLMConfig.ollama(
        model_name="llama3.2:3b", base_url="http://localhost:11434", ctx_length=32000
    )

    vision_llm = LLMConfig.ollama(
        model_name="llava:latest", base_url="http://localhost:11434"
    )

    # Create database configuration
    database = DatabaseClientConfig.milvus(
        collection="documents", host="localhost", port=19530
    )

    # Create converter and extractor configurations
    converter = ConverterConfig.markitdown(vision_llm=vision_llm)

    extractor = ExtractorConfig.basic(
        llm=llm,
        metadata_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 512},
                "author": {"type": "string", "maxLength": 256},
                "summary": {"type": "string", "maxLength": 2048},
            },
        },
    )

    # Create crawler configuration
    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunk_size=10000,
        log_level="INFO",
    )

    return config


def example_openai_config():
    """Example: Configuration using OpenAI models."""
    # Create configurations with API keys
    embeddings = EmbedderConfig.openai(
        model="text-embedding-3-small", api_key="your-openai-api-key-here"
    )

    llm = LLMConfig.openai(model_name="gpt-4", api_key="your-openai-api-key-here")

    vision_llm = LLMConfig.openai(
        model_name="gpt-4-vision-preview", api_key="your-openai-api-key-here"
    )

    database = DatabaseClientConfig.milvus(
        collection="documents", host="localhost", port=19530
    )

    converter = ConverterConfig.markitdown(vision_llm=vision_llm)
    extractor = ExtractorConfig.basic(llm=llm)

    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
    )

    return config


def example_advanced_config():
    """Example: Advanced configuration with custom settings."""
    # Custom embedding configuration
    embeddings = EmbedderConfig.ollama(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        dimension=768,  # Explicitly set dimension
    )

    # Custom LLM with system prompt
    llm = LLMConfig.ollama(
        model_name="mistral:7b-instruct",
        base_url="http://localhost:11434",
        system_prompt="You are a helpful assistant for document analysis.",
        ctx_length=16000,
        default_timeout=600.0,
    )

    # Vision LLM for document processing
    vision_llm = LLMConfig.ollama(
        model_name="llava:13b", base_url="http://localhost:11434", ctx_length=16000
    )

    # Custom database configuration
    database = DatabaseClientConfig.milvus(
        collection="research_papers",
        partition="2024_q1",
        host="milvus-server.example.com",
        port=19530,
        username="researcher",
        password="secure_password",
        recreate=False,
        collection_description="Research paper collection for Q1 2024",
    )

    # Advanced converter configuration
    converter = ConverterConfig.pymupdf(
        vision_llm=vision_llm,
        metadata={
            "preserve_formatting": True,
            "include_page_numbers": True,
            "extract_tables": True,
            "image_description_prompt": "Describe this image in detail for academic research.",
        },
    )

    # Advanced extractor with multi-schema support
    schemas = [
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 512},
                "authors": {"type": "array", "items": {"type": "string"}},
                "abstract": {"type": "string", "maxLength": 2048},
            },
        },
        {
            "type": "object",
            "properties": {
                "publication_year": {"type": "integer"},
                "journal": {"type": "string", "maxLength": 256},
                "doi": {"type": "string", "maxLength": 128},
            },
        },
    ]

    extractor = ExtractorConfig.multi_schema(schemas=schemas, llm=llm)

    # Complete crawler configuration
    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunk_size=8000,  # Smaller chunks for research papers
        temp_dir="/tmp/crawler_cache/",
        benchmark=True,
        generate_benchmark_questions=True,
        num_benchmark_questions=5,
        log_level="DEBUG",
        log_file="/var/log/crawler.log",
    )

    return config


def example_default_config():
    """Example: Using the default configuration helper."""
    # This creates a complete configuration with sensible defaults
    config = CrawlerConfig.default_ollama(
        collection="my_documents",
        embed_model="all-minilm:v2",
        llm_model="llama3.2:3b",
        vision_model="llava:latest",
        base_url="http://localhost:11434",
        host="localhost",
        port=19530,
        chunk_size=12000,
        benchmark=False,
        log_level="INFO",
    )

    return config


def example_custom_converter_types():
    """Example: Using different converter types."""
    # Base configurations
    llm = LLMConfig.ollama(model_name="llama3.2:3b")
    vision_llm = LLMConfig.ollama(model_name="llava:latest")

    # Different converter types
    markitdown_converter = ConverterConfig.markitdown(vision_llm=vision_llm)

    docling_converter = ConverterConfig.docling(
        vision_llm=vision_llm, metadata={"timeout": 600}
    )

    docling_vlm_converter = ConverterConfig.docling_vlm()

    pymupdf_converter = ConverterConfig.pymupdf(
        vision_llm=vision_llm,
        metadata={
            "extract_tables": True,
            "include_metadata": True,
            "image_description_prompt": "Analyze this image for document understanding.",
        },
    )

    # Use any of these in your crawler config
    config = CrawlerConfig.create(
        embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
        llm=llm,
        vision_llm=vision_llm,
        database=DatabaseClientConfig.milvus(collection="test"),
        converter=pymupdf_converter,  # Choose your preferred converter
        extractor=ExtractorConfig.basic(llm=llm),
    )

    return config


# Example of how to use these configurations
if __name__ == "__main__":
    # Create different configurations
    basic_config = example_basic_ollama_config()
    openai_config = example_openai_config()
    advanced_config = example_advanced_config()
    default_config = example_default_config()

    print("✅ All configurations created successfully!")
    print(f"Basic config collection: {basic_config.database.collection}")
    print(f"OpenAI config provider: {openai_config.llm.provider}")
    print(f"Advanced config chunk size: {advanced_config.chunk_size}")
    print(f"Default config LLM model: {default_config.llm.model_name}")
