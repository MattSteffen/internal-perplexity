class TextSplitter:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list:
        """Splits text into chunks with overlap."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(' '.join(words[start:end]))
            start = end - self.chunk_overlap
            if start < 0: start = end
        return chunks
    

if __name__ == "__main__":
    splitter = TextSplitter()
    text = "This is a sample text to be split into chunks."
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")