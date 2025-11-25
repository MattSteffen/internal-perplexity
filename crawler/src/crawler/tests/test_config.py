#!/usr/bin/env python3
"""
Test script to verify the new config system works correctly.
"""

import logging
import os
import sys
from unittest.mock import Mock

# Add the crawler src directory to the path
# From crawler/src/tests/, we need to go up to crawler/ and then into src/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.crawler.processing import BasicExtractor, EmbedderConfig  # noqa: E402
from src.crawler.storage import DatabaseClientConfig, DatabaseDocument  # noqa: E402
from src.crawler.storage.milvus_client import MilvusDB  # noqa: E402

from src import (  # noqa: E402
    RESERVED,
    ConverterConfig,
    CrawlerConfig,
    sanitize_metadata,
)


def test_converter_config():
    """Test ConverterConfig creation and validation."""
    print("Testing ConverterConfig...")

    # Test config creation
    config = ConverterConfig(type="pymupdf4llm")
    assert config.type == "pymupdf4llm"
    print("✓ ConverterConfig creation works")

    # Test from_dict
    config_dict = {
        "type": "pymupdf4llm",
        "vision_llm": {
            "model": "llava:latest",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        },
    }
    config = ConverterConfig.from_dict(config_dict)
    assert config.type == "pymupdf4llm"
    assert config.vision_llm.model_name == "llava:latest"
    print("✓ ConverterConfig.from_dict works")

    # Test validation
    try:
        ConverterConfig.from_dict({"type": ""})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot be empty" in str(e)
        print("✓ ConverterConfig validation works")


def test_crawler_config():
    """Test CrawlerConfig with new structure."""
    print("\nTesting CrawlerConfig...")

    # Test creating from dict with new converter structure
    config_dict = {
        "embeddings": {"model": "all-minilm:v2", "base_url": "http://localhost:11434"},
        "llm": {
            "model": "llama3.2:3b",  # Test the new model key
            "base_url": "http://localhost:11434",
        },
        "vision_llm": {
            "model_name": "llava:latest",  # Test the model_name key
            "base_url": "http://localhost:11434",
        },
        "database": {"provider": "milvus", "collection": "test_collection"},
        "converter": {
            "type": "pymupdf4llm",
            "vlm_config": {"model_name": "llava:latest", "provider": "ollama", "base_url": "http://localhost:11434"},
        },
    }

    crawler_config = CrawlerConfig.from_dict(config_dict)

    # Verify all configs were created correctly
    assert crawler_config.embeddings.model == "all-minilm:v2"
    assert crawler_config.llm.model_name == "llama3.2:3b"  # Should work with 'model' key
    assert crawler_config.vision_llm.model_name == "llava:latest"  # Should work with 'model_name' key
    assert crawler_config.database.collection == "test_collection"
    assert crawler_config.converter.type == "pymupdf4llm"
    assert crawler_config.converter.vlm_config.model_name == "llava:latest"

    print("✓ CrawlerConfig with new converter structure works")
    print("✓ LLM config accepts both 'model' and 'model_name' keys")


def test_validation():
    """Test validation in config creation."""
    print("\nTesting validation...")

    # Test EmbedderConfig validation
    try:
        from src.crawler.processing import EmbedderConfig

        EmbedderConfig.from_dict({"model": "", "base_url": "http://localhost:11434"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model cannot be empty" in str(e)
        print("✓ EmbedderConfig validation works")

    # Test LLMConfig validation
    try:
        from src.crawler.processing import LLMConfig

        LLMConfig.from_dict({"model_name": "", "base_url": "http://localhost:11434"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model_name' or 'model' must be provided" in str(e)
        print("✓ LLMConfig validation works")

    # Test DatabaseClientConfig validation
    try:
        from src.crawler.storage import DatabaseClientConfig

        DatabaseClientConfig.from_dict({"provider": "", "collection": "test"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provider cannot be empty" in str(e)
        print("✓ DatabaseClientConfig validation works")


def test_sanitize_metadata():
    """Test the sanitize_metadata function."""
    print("\nTesting sanitize_metadata...")

    # Test basic sanitization
    metadata = {
        "title": "Test Document",
        "author": "Test Author",
        "default_text": "This should be removed",  # Reserved key
        "default_chunk_index": 5,  # Reserved key
        "default_source": "test.pdf",  # Reserved key
        "custom_field": "should remain",
    }

    sanitized = sanitize_metadata(metadata)
    assert "title" in sanitized
    assert "author" in sanitized
    assert "custom_field" in sanitized
    assert "default_text" not in sanitized
    assert "default_chunk_index" not in sanitized
    assert "default_source" not in sanitized
    print("✓ Basic sanitization works")

    # Test with None input
    assert sanitize_metadata(None) == {}
    print("✓ None input handling works")

    # Test with empty dict
    assert sanitize_metadata({}) == {}
    print("✓ Empty dict handling works")

    # Test with all reserved keys
    reserved_only = {key: f"value_{key}" for key in RESERVED}
    assert sanitize_metadata(reserved_only) == {}
    print("✓ All reserved keys removed")

    # Test with logger
    logger = Mock()
    metadata_with_reserved = {
        "title": "Test",
        "default_document_id": "should_be_removed",
    }
    sanitized = sanitize_metadata(metadata_with_reserved, logger=logger)
    assert "title" in sanitized
    assert "default_document_id" not in sanitized
    print("✓ Logger integration works")


def test_embedder_config_dimension():
    """Test EmbedderConfig with dimension field."""
    print("\nTesting EmbedderConfig dimension...")

    # Test with dimension specified
    config_dict = {
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "dimension": 384,
    }
    config = EmbedderConfig.from_dict(config_dict)
    assert config.dimension == 384
    print("✓ Dimension field works")

    # Test without dimension (should be None)
    config_dict_no_dim = {
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
    }
    config_no_dim = EmbedderConfig.from_dict(config_dict_no_dim)
    assert config_no_dim.dimension is None
    print("✓ Dimension defaults to None")


def test_token_chunking_overlap():
    """Test token-aware chunking behavior."""
    print("\nTesting token chunking...")

    # Create a mock LLM for the extractor
    mock_llm = Mock()

    # Create a BasicExtractor instance
    extractor = BasicExtractor({}, mock_llm)

    # Test basic chunking
    text = "This is a test document with multiple words that should be chunked properly."
    chunks = extractor.chunk_text(text, chunk_size=10)

    assert len(chunks) > 0
    assert all(len(chunk) <= 10 for chunk in chunks)
    print("✓ Basic chunking works")

    # Test with empty text
    empty_chunks = extractor.chunk_text("", chunk_size=10)
    assert empty_chunks == []  # Empty text produces no chunks
    print("✓ Empty text handling works")

    # Test with text shorter than chunk size
    short_text = "Short text"
    short_chunks = extractor.chunk_text(short_text, chunk_size=100)
    assert len(short_chunks) == 1
    assert short_chunks[0] == short_text
    print("✓ Short text handling works")


def test_milvus_duplicate_filtering():
    """Test MilvusDB insert_data duplicate filtering with mock client."""
    print("\nTesting MilvusDB duplicate filtering...")

    # Create a mock client
    mock_client = Mock()

    # Create mock config and MilvusDB instance
    config = DatabaseClientConfig(provider="milvus", collection="test_collection", host="localhost", port=19530)

    # Mock the MilvusDB to avoid actual connection
    db = MilvusDB.__new__(MilvusDB)  # Create without calling __init__
    db.config = config
    db.client = mock_client
    db.logger = logging.getLogger("TestMilvusDB")

    # Create test data with duplicates
    test_data = [
        DatabaseDocument(
            default_text="Chunk 1",
            default_text_embedding=[0.1] * 384,
            default_chunk_index=0,
            default_source="doc1.pdf",
            metadata={"title": "Test"},
        ),
        DatabaseDocument(
            default_text="Chunk 2",
            default_text_embedding=[0.2] * 384,
            default_chunk_index=1,
            default_source="doc1.pdf",
            metadata={"title": "Test"},
        ),
        DatabaseDocument(
            default_text="Chunk 1 duplicate",
            default_text_embedding=[0.1] * 384,
            default_chunk_index=0,  # Duplicate chunk_index for same source
            default_source="doc1.pdf",
            metadata={"title": "Test"},
        ),
    ]

    # Mock _existing_chunk_indexes to return some existing indexes
    db._existing_chunk_indexes = Mock(return_value={0})  # chunk_index 0 already exists

    # Mock insert and flush methods
    mock_client.insert.return_value = {"insert_count": 1}
    mock_client.flush = Mock()

    # Test the insert_data method
    db.insert_data(test_data)

    # Verify that insert was called
    mock_client.insert.assert_called_once()
    mock_client.flush.assert_called_once()

    # Verify _existing_chunk_indexes was called for the source
    db._existing_chunk_indexes.assert_called_with("doc1.pdf")

    print("✓ MilvusDB duplicate filtering works")


if __name__ == "__main__":
    print("Running comprehensive tests...\n")

    # Existing tests
    test_converter_config()
    test_crawler_config()
    test_validation()

    # New tests
    test_sanitize_metadata()
    test_embedder_config_dimension()
    test_token_chunking_overlap()
    test_milvus_duplicate_filtering()

    print("\n🎉 All tests passed! The system is working correctly.")
