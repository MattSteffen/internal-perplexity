import ollama

# TODO: Download the nomic-embed-text model from Ollama

class LocalEmbedder:
    def __init__(self, model_name: str = 'all-minilm:v2'):
        self.model_name = model_name
    
    def generate(self, texts: list) -> list:
        """Generates embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings

# test / demo
if __name__ == "__main__":
    embedder = LocalEmbedder()
    text = "This is a sample text to be embedded."
    embeddings = embedder.generate([text])
    print(embeddings)