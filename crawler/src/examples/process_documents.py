"""
Document Processing Example

This script demonstrates how to use the document processing system to:
1. Process documents from a directory
2. Extract metadata using LLMs
3. Generate embeddings for the documents
4. Save the results to a file
"""

import os
import json
import sys
from typing import Dict, Any, List, Tuple

# Add the parent directory to the path so we can import the processing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.processor import DocumentProcessor


def save_results(results: List[Tuple[str, Dict[str, Any], List[float]]], output_file: str) -> None:
    """Save processing results to a JSON file.
    
    Args:
        results: List of (text, metadata, embedding) tuples
        output_file: Path to output file
    """
    output = []
    for text, metadata, embedding in results:
        # Truncate text for display
        display_text = text[:100] + "..." if len(text) > 100 else text
        
        output.append({
            "text": display_text,
            "metadata": metadata,
            "embedding_dimension": len(embedding),
            # We don't save the full embedding here as it would make the file very large
            "embedding_sample": embedding[:5]  # Just save the first 5 dimensions as a sample
        })
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    # Initialize the document processor
    processor = DocumentProcessor(
        # Optional: Specify a config file
        # config_path="config.yaml",
        
        # Or specify parameters directly
        schema_path="crawler/src/storage/document.json",
        llm_model="llama3.2:1b",
        embedding_model="all-minilm:v2",
        base_url="http://localhost:11434"
    )
    
    # Process a directory of documents
    input_dir = "documents"
    if not os.path.exists(input_dir):
        print(f"Creating example directory: {input_dir}")
        os.makedirs(input_dir)
        print(f"Please add some documents to {input_dir} and run this script again.")
        return
    
    print(f"Processing documents in {input_dir}...")
    
    # Process only certain file types
    extensions = [".pdf", ".txt", ".md", ".html", ".csv", ".json"]
    
    # Collect results
    results = []
    for text, metadata, embedding in processor.process_directory(input_dir, extensions):
        print(f"Processed: {metadata['source']}")
        print(f"  Format: {metadata['format']}")
        print(f"  Embedding dimension: {len(embedding)}")
        print()
        
        results.append((text, metadata, embedding))
    
    # Save results
    if results:
        save_results(results, "document_results.json")
        print(f"Processed {len(results)} documents")
    else:
        print(f"No documents found in {input_dir} with extensions {extensions}")


if __name__ == "__main__":
    main() 