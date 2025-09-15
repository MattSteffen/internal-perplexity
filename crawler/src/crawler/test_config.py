"""
Simple test of the new type-safe configuration system.

This module verifies that the new configuration classes work correctly
and provide proper validation without requiring external test frameworks.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from processing import EmbedderConfig, LLMConfig, ConverterConfig, ExtractorConfig
from storage import DatabaseClientConfig
from main import CrawlerConfig


def test_embedder_config():
    """Test EmbedderConfig functionality."""
    print("Testing EmbedderConfig...")

    # Test factory methods
    ollama_config = EmbedderConfig.ollama(
        model="all-minilm:v2", base_url="http://localhost:11434", dimension=384
    )
    assert ollama_config.model == "all-minilm:v2"
    assert ollama_config.provider == "ollama"
    print("✅ Ollama embedder factory works")

    openai_config = EmbedderConfig.openai(
        model="text-embedding-3-small", api_key="test-key"
    )
    assert openai_config.model == "text-embedding-3-small"
    assert openai_config.provider == "openai"
    print("✅ OpenAI embedder factory works")

    # Test validation
    try:
        EmbedderConfig(model="", base_url="http://test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Embedder model cannot be empty" in str(e)
        print("✅ Embedder validation works")


def test_llm_config():
    """Test LLMConfig functionality."""
    print("Testing LLMConfig...")

    # Test factory methods
    ollama_config = LLMConfig.ollama(
        model_name="llama3.2:3b", base_url="http://localhost:11434"
    )
    assert ollama_config.model_name == "llama3.2:3b"
    assert ollama_config.provider == "ollama"
    print("✅ Ollama LLM factory works")

    openai_config = LLMConfig.openai(model_name="gpt-4", api_key="test-key")
    assert openai_config.model_name == "gpt-4"
    assert openai_config.provider == "openai"
    print("✅ OpenAI LLM factory works")

    # Test validation
    try:
        LLMConfig(model_name="", base_url="http://test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "LLM model_name cannot be empty" in str(e)
        print("✅ LLM validation works")


def test_converter_config():
    """Test ConverterConfig functionality."""
    print("Testing ConverterConfig...")

    vision_llm = LLMConfig.ollama(model_name="llava:latest")

    # Test factory methods
    markitdown_config = ConverterConfig.markitdown(vision_llm=vision_llm)
    assert markitdown_config.type == "markitdown"
    print("✅ MarkItDown converter factory works")

    docling_config = ConverterConfig.docling(vision_llm=vision_llm)
    assert docling_config.type == "docling"
    print("✅ Docling converter factory works")

    docling_vlm_config = ConverterConfig.docling_vlm()
    assert docling_vlm_config.type == "docling_vlm"
    print("✅ Docling VLM converter factory works")

    pymupdf_config = ConverterConfig.pymupdf(vision_llm=vision_llm)
    assert pymupdf_config.type == "pymupdf"
    print("✅ PyMuPDF converter factory works")

    # Test validation
    try:
        ConverterConfig(type="markitdown", vision_llm=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires vision_llm configuration" in str(e)
        print("✅ Converter validation works")


def test_extractor_config():
    """Test ExtractorConfig functionality."""
    print("Testing ExtractorConfig...")

    llm = LLMConfig.ollama(model_name="llama3.2:3b")

    # Test factory methods
    basic_config = ExtractorConfig.basic(llm=llm)
    assert basic_config.type == "basic"
    print("✅ Basic extractor factory works")

    schemas = [{"type": "object", "properties": {"title": {"type": "string"}}}]
    multi_config = ExtractorConfig.multi_schema(schemas=schemas, llm=llm)
    assert multi_config.type == "multi_schema"
    print("✅ Multi-schema extractor factory works")

    # Test validation
    try:
        ExtractorConfig(type="basic", llm=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires LLM configuration" in str(e)
        print("✅ Extractor validation works")


def test_database_config():
    """Test DatabaseClientConfig functionality."""
    print("Testing DatabaseClientConfig...")

    # Test factory method
    milvus_config = DatabaseClientConfig.milvus(
        collection="test_collection", host="localhost", port=19530
    )
    assert milvus_config.provider == "milvus"
    assert milvus_config.collection == "test_collection"
    print("✅ Milvus database factory works")

    # Test validation
    try:
        DatabaseClientConfig(provider="milvus", collection="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Database collection cannot be empty" in str(e)
        print("✅ Database validation works")


def test_crawler_config():
    """Test CrawlerConfig functionality."""
    print("Testing CrawlerConfig...")

    # Test create method
    embeddings = EmbedderConfig.ollama(model="all-minilm:v2")
    llm = LLMConfig.ollama(model_name="llama3.2:3b")
    vision_llm = LLMConfig.ollama(model_name="llava:latest")
    database = DatabaseClientConfig.milvus(collection="test")

    config = CrawlerConfig.create(
        embeddings=embeddings, llm=llm, vision_llm=vision_llm, database=database
    )
    assert config.embeddings.model == "all-minilm:v2"
    assert config.database.collection == "test"
    print("✅ CrawlerConfig.create() works")

    # Test default_ollama method
    default_config = CrawlerConfig.default_ollama(collection="default_test")
    assert default_config.database.collection == "default_test"
    assert default_config.llm.model_name == "llama3.2:3b"
    print("✅ CrawlerConfig.default_ollama() works")

    # Test validation
    try:
        CrawlerConfig(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            chunk_size=-1,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Chunk size must be positive" in str(e)
        print("✅ CrawlerConfig validation works")


def test_integration():
    """Test complete configuration integration."""
    print("Testing full integration...")

    # Create a complete configuration
    embeddings = EmbedderConfig.ollama(model="all-minilm:v2")
    llm = LLMConfig.ollama(model_name="llama3.2:3b")
    vision_llm = LLMConfig.ollama(model_name="llava:latest")
    database = DatabaseClientConfig.milvus(collection="integration_test")
    converter = ConverterConfig.markitdown(vision_llm=vision_llm)
    extractor = ExtractorConfig.basic(llm=llm)

    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunk_size=10000,
        benchmark=False,
        log_level="INFO",
    )

    # Verify all components are properly configured
    assert config.embeddings.model == "all-minilm:v2"
    assert config.llm.model_name == "llama3.2:3b"
    assert config.vision_llm.model_name == "llava:latest"
    assert config.database.collection == "integration_test"
    assert config.converter.type == "markitdown"
    assert config.extractor.type == "basic"
    assert config.chunk_size == 10000
    assert config.benchmark is False
    assert config.log_level == "INFO"

    print("✅ Full integration test passed!")


def main():
    """Run all tests."""
    print("🚀 Running type-safe configuration tests...\n")

    try:
        test_embedder_config()
        print()

        test_llm_config()
        print()

        test_converter_config()
        print()

        test_extractor_config()
        print()

        test_database_config()
        print()

        test_crawler_config()
        print()

        test_integration()
        print()

        print(
            "🎉 All tests passed! The type-safe configuration system is working correctly."
        )

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
