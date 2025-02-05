import json
from discovery import find_dirs
from processing.extractors import JSONHandler
from processing.chunking import TextSplitter
from processing.embeddings import LocalEmbedder
from storage.vector_db import VectorStorage

def process_file(dir_path: str, vector_db: VectorStorage, splitter: TextSplitter, embedder: LocalEmbedder):
    handler = JSONHandler(dir_path)
    texts, metadatas = [], []
    
    for text, metadata in handler.extract():
        if isinstance(text, str) and text.strip():
            chunks = splitter.split_text(text)
        else:  # text is a list of strings already
            chunks = text
        texts.extend(chunks)
        for i in range(len(chunks)):
            metadata['chunk_index'] = i
            metadatas.append(metadata.copy())

    if texts:
        embeddings = embedder.generate(texts)
        vector_db.insert_data(texts, embeddings, metadatas)

def setup(input_dir: str):
    vector_db = VectorStorage()
    splitter = TextSplitter()
    embedder = LocalEmbedder()

    for json_dir in find_dirs(input_dir):
        print(f"Processing {json_dir}")
        process_file(json_dir, vector_db, splitter, embedder)

    # --- Search Test Without Author Filter ---
    query = "admired great-grandmother"
    embed = embedder.generate([query])[0]
    results = vector_db.search(embed)
    print(f"Results for query '{query}':")
    for result in results:
        print(json.dumps(result, indent=4))
    
    # --- Search Test With Author Filter ---
    # Example: only return results where the author is in the provided list.
    results_with_authors = vector_db.search(embed, filters=["chunk_index > 22"])
    print(f"Results for query '{query}' with filters:")
    for result in results_with_authors:
        print(json.dumps(result, indent=4))
    
    vector_db.close()

if __name__ == "__main__":
    setup("../../data/conference")
