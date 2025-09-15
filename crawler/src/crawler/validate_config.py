#!/usr/bin/env python3
"""
Simple validation script for the new type-safe configuration system.
"""

import sys
import os

# Add the crawler directory to the path
sys.path.insert(0, os.path.dirname(__file__))


def validate_config():
    """Validate that the new configuration classes work correctly."""
    print("🔧 Validating type-safe configuration system...")

    try:
        # Test basic imports
        from processing.embeddings import EmbedderConfig
        from processing.llm import LLMConfig
        from processing.converter import ConverterConfig
        from processing.extractor import ExtractorConfig
        from storage.database_client import DatabaseClientConfig
        from main import CrawlerConfig

        print("✅ All imports successful")

        # Test factory methods
        embeddings = EmbedderConfig.ollama(model="test-model")
        llm = LLMConfig.ollama(model_name="test-model")
        vision_llm = LLMConfig.ollama(model_name="test-vision")
        database = DatabaseClientConfig.milvus(collection="test")
        converter = ConverterConfig.markitdown(vision_llm=vision_llm)
        extractor = ExtractorConfig.basic(llm=llm)

        print("✅ All factory methods work")

        # Test configuration creation
        config = CrawlerConfig.create(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter,
            extractor=extractor,
        )

        print("✅ CrawlerConfig creation successful")

        # Test default configuration
        default_config = CrawlerConfig.default_ollama(collection="test")
        print("✅ Default configuration creation successful")

        # Test validation
        try:
            EmbedderConfig(model="", base_url="http://test")
            print("❌ Validation failed - should have caught empty model")
            return False
        except ValueError:
            print("✅ Validation working correctly")

        print("🎉 Configuration system validation complete!")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
