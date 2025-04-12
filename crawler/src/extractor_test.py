import os
from langchain.chat_models import init_chat_model
from processing.extractor import Extractor, VisionLLM

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
    ]
}

TEST_METADATA_CONFIG = {
    "metadata": {
        "extra_embeddings": ["title", "author", "author_role"]
    }
}


sample_schema = {
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


def main():
    # Initialize extractor
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    vision_llm = VisionLLM()
    extractor = Extractor(llm, vision_llm, sample_schema, TEST_METADATA_CONFIG, TEST_CONFIG)
    
    # Get test files
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    test_files = [
        os.path.join(base_dir, "data/sample/c4611_sample_explain.pdf"),
        os.path.join(base_dir, "data/conference/Simple_Is_the_Doctrine_of_Jesus_Christ.json")
    ]
    
    print("Starting extraction...")
    results = list(extractor.extract(test_files))
    
    print(f"\nExtracted {len(results)} chunks/metadata pairs")
    for i, (text, metadata) in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {metadata['source']}")
        print(f"Format: {metadata['format']}")
        if 'chunk_index' in metadata:
            print(f"Chunk Index: {metadata['chunk_index']}")
        print(f"Text length: {len(text)}")
        print("Metadata fields:")
        for key, value in metadata.items():
            if key not in ['source', 'format', 'chunk_index']:
                print(f"  {key}: {value}")
        print("---")

if __name__ == "__main__":
    main()
