import json
import os
from abc import ABC, abstractmethod

# TODO: Make this call the hanlder for each file in the directory individually not necessarily folder-wide


class ExtractorHandler(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    @abstractmethod
    def extract(self):
        """Yields text and metadata from documents."""
        pass

    def files(self, extension: str):
        """Yields paths to files with given extension in the directory."""
        for file_name in os.listdir(self.dir_path):
            if file_name.endswith(extension):
                yield os.path.join(self.dir_path, file_name)

class JSONHandler(ExtractorHandler):
    def extract(self):
        """Yields text and metadata from JSON documents."""
        for i, file_path in enumerate(self.files('.json')):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            text = data.get('content', [])
            metadata = {k: v for k, v in data.items() if k != 'content'}
            metadata['source'] = file_path # TODO: use full path
            metadata['format'] = 'json'
            metadata['chunk_index'] = i
            yield text, metadata

class TXTHandler(ExtractorHandler):
    def extract(self):
        """Yields text and metadata from TXT documents."""
        for file_path in self.files('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
            
            metadata = {
                'source': file_path,
                'format': 'txt'
            }
            yield text, metadata

def get_handler(dir_path: str) -> ExtractorHandler:
    """Factory function to get appropriate handler based on files in directory."""
    files = os.listdir(dir_path)
    if any(f.endswith('.json') for f in files):
        return JSONHandler(dir_path)
    elif any(f.endswith('.txt') for f in files):
        return TXTHandler(dir_path)
    else:
        raise ValueError(f"No supported files found in {dir_path}")

# test / demo
if __name__ == "__main__":
    handler = get_handler("../../../data/conference/")
    for text, metadata in handler.extract():
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")