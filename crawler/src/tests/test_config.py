#!/usr/bin/env python3
"""
Test script to verify the new config system works correctly.
"""

import sys
import os

# Add the crawler src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    CrawlerConfig,
    ConverterConfig,
    DEFAULT_OLLAMA_EMBEDDINGS,
    DEFAULT_OLLAMA_LLM,
    DEFAULT_OLLAMA_VISION_LLM,
    DEFAULT_MILVUS_CONFIG,
    DEFAULT_CONVERTER_CONFIG
)

def test_converter_config():
    """Test ConverterConfig creation and validation."""
    print("Testing ConverterConfig...")

    # Test default config
    config = ConverterConfig()
    assert config.type == "markitdown"
    assert config.vision_llm is None
    print("✓ Default ConverterConfig works")

    # Test from_dict
    config_dict = {
        "type": "docling",
        "vision_llm": {
            "model": "llava:latest",
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        }
    }
    config = ConverterConfig.from_dict(config_dict)
    assert config.type == "docling"
    assert config.vision_llm["model"] == "llava:latest"
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
        "embeddings": {
            "model": "all-minilm:v2",
            "base_url": "http://localhost:11434"
        },
        "llm": {
            "model": "llama3.2:3b",  # Test the new model key
            "base_url": "http://localhost:11434"
        },
        "vision_llm": {
            "model_name": "llava:latest",  # Test the model_name key
            "base_url": "http://localhost:11434"
        },
        "database": {
            "provider": "milvus",
            "collection": "test_collection"
        },
        "converter": {
            "type": "markitdown",
            "vision_llm": {
                "model": "llava:latest",
                "provider": "ollama"
            }
        }
    }

    crawler_config = CrawlerConfig.from_dict(config_dict)

    # Verify all configs were created correctly
    assert crawler_config.embeddings.model == "all-minilm:v2"
    assert crawler_config.llm.model_name == "llama3.2:3b"  # Should work with 'model' key
    assert crawler_config.vision_llm.model_name == "llava:latest"  # Should work with 'model_name' key
    assert crawler_config.database.collection == "test_collection"
    assert crawler_config.converter.type == "markitdown"
    assert crawler_config.converter.vision_llm["model"] == "llava:latest"

    print("✓ CrawlerConfig with new converter structure works")
    print("✓ LLM config accepts both 'model' and 'model_name' keys")

def test_validation():
    """Test validation in config creation."""
    print("\nTesting validation...")

    # Test EmbedderConfig validation
    try:
        from src.processing import EmbedderConfig
        EmbedderConfig.from_dict({"model": "", "base_url": "http://localhost:11434"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model cannot be empty" in str(e)
        print("✓ EmbedderConfig validation works")

    # Test LLMConfig validation
    try:
        from src.processing import LLMConfig
        LLMConfig.from_dict({"model_name": "", "base_url": "http://localhost:11434"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model_name' or 'model' must be provided" in str(e)
        print("✓ LLMConfig validation works")

    # Test DatabaseClientConfig validation
    try:
        from src.storage import DatabaseClientConfig
        DatabaseClientConfig.from_dict({"provider": "", "collection": "test"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provider cannot be empty" in str(e)
        print("✓ DatabaseClientConfig validation works")

def test_defaults():
    """Test that default configurations are properly defined."""
    print("\nTesting default configurations...")

    # Test that defaults exist and have expected values
    assert DEFAULT_OLLAMA_EMBEDDINGS.model == "all-minilm:v2"
    assert DEFAULT_OLLAMA_EMBEDDINGS.base_url == "http://localhost:11434"

    assert DEFAULT_OLLAMA_LLM.model_name == "llama3.2:3b"
    assert DEFAULT_OLLAMA_LLM.base_url == "http://localhost:11434"

    assert DEFAULT_OLLAMA_VISION_LLM.model_name == "llava:latest"

    assert DEFAULT_MILVUS_CONFIG.collection == "documents"
    assert DEFAULT_MILVUS_CONFIG.host == "localhost"

    assert DEFAULT_CONVERTER_CONFIG.type == "markitdown"

    print("✓ All default configurations are properly defined")

if __name__ == "__main__":
    print("Running config system tests...\n")

    test_converter_config()
    test_crawler_config()
    test_validation()
    test_defaults()

    print("\n🎉 All config tests passed! The new config system is working correctly.")
