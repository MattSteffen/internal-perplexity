import json
import os
import argparse
from pydoc import Doc

from crawler import Crawler
from processing.process_markitdown import MarkItDownConverter
from processing.process_docling import DoclingConverter

schema1 = {}
schema2 = {}
extra_fields = []


class MyExtractor:
    """
    Custom extractor class for processing documents.
    This class is a placeholder and should be implemented with actual logic.
    """
    def __init__(self, config: dict, llm) -> None:
        self.config = config
        # self.converter = DoclingConverter(config)  
        self.converter = MarkItDownConverter(config)
        self.llm = llm

    def extract_metadata_with_schema(self, text: str, schema) -> dict:
        s_llm = self.llm.with_structured_output(schema)
        prompt = f"Extract metadata from the following text according to these guidelines:\nExtract the metadata fields from the text following the schema provided.\n\nText excerpt (analyze the full text even if truncated here):\n{text[:100000]}... [text continues]\n\nReturn your analysis in the required JSON format."
        try:
            llm_response = s_llm.invoke(prompt)
            if isinstance(llm_response, dict):
                return llm_response
            else:
                return json.loads(llm_response.content.replace("```json", "").replace("```", ""))
        except Exception as e:
            print(f"Error parsing LLM metadata response: {e}, response: {llm_response}")
            return {}

    def extract(self, filepath: str):
        text = self.converter.convert(filepath)
        metadata = {"source": filepath}
        metadata.update(self.extract_metadata_with_schema(text, schema1))
        metadata.update(self.extract_metadata_with_schema(text, schema2))
        chunks = self.converter.chunk_smart(text, 5000)
        for i, chunk in enumerate(chunks):
            meta = metadata.copy()
            meta["text"] = chunk
            meta["chunk_index"] = i
            yield meta
        for i, field in enumerate(extra_fields):
            meta = metadata.copy()
            meta["text"] = meta[field]
            meta["chunk_index"] = len(chunks) + i
            yield meta

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--config', '-c', type=str, help='Path to your configuration file')
    args = parser.parse_args()
    
    # Load config file if provided
    config_dict = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Validate we have a path
    if "path" not in config_dict:
        parser.error("Must specify directory path either via --directory arg or in config file")
    
    # Create and run crawler
    crawler = Crawler(config_dict)
    crawler.set_extractor(MyExtractor(config_dict, crawler.llm))
    
    # Process all documents
    with crawler._setup_vector_db() as db:
        for data in crawler.run():
            db.insert_data(data)

    print("complete")

if __name__ == "__main__":
    main()