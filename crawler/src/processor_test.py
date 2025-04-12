import os
from langchain.chat_models import init_chat_model
from processing.extractor import Extractor, VisionLLM
from processing.embeddings import LocalEmbedder
from processing.processor import DocumentProcessor
from config.config_manager import ConfigManager

# Test configs
TEST_CONFIG = {
    "chunking": {
        "length": {
            "enabled": True,
            "chunk_size": 5000,
            "overlap": 100
        },
        "metadata": {
            "enabled": True
        }
    },
    "document_readers": [
        {"type": "pdf", "enabled": True},
        {"type": "json", "enabled": True}
    ],
    "metadata": {
        "schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Document",
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "maxLength": 1024,
                    "description": "Text content of the document chunk."
                },
                "source": {
                    "type": "string",
                    "maxLength": 1024,
                    "description": "Source identifier of the document chunk."
                },
                "title": {
                    "type": "string",
                    "maxLength": 255,
                    "description": "Title of the document."
                },
                "author": {
                    "type": "array",
                    "maxItems": 255,
                    "items": {
                        "type": "string",
                        "description": "An author of the document."
                    },
                    "description": "List of authors of the document."
                },
                "author_role": {
                    "type": "string",
                    "maxLength": 255,
                    "description": "Role of the author in the document (e.g., writer, editor)."
                },
                "url": {
                    "type": "string",
                    "maxLength": 1024,
                    "description": "URL associated with the document."
                },
                "chunk_index": {
                    "type": "integer",
                    "description": "Index of the document chunk."
                }
            }
        }
    }
}

def main():
    # Initialize components
    config = ConfigManager(config_source=TEST_CONFIG).config
    print(config.get("embeddings", {}))
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    vision_llm = VisionLLM()
    embedder = LocalEmbedder(config.get("embeddings", {}))
    
    
    # Initialize processor
    processor = DocumentProcessor(config, llm, vision_llm, embedder)
    
    # Get test files
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    test_files = [
        os.path.join(base_dir, "data/sample/c4611_sample_explain.pdf"),
        os.path.join(base_dir, "data/conference/Simple_Is_the_Doctrine_of_Jesus_Christ.json")
    ]
    
    print("Starting processing...")
    for file_path in test_files:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Process document
        for text, metadata, embedding in processor.process_document(file_path):
            # Print results
            print(f"Text length: {len(text)}")
            print(f"Embedding length: {len(embedding)}")
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print("---")

if __name__ == "__main__":
    main() 