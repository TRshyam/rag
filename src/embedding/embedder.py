from langchain_ollama import OllamaEmbeddings

class Embedder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.embeddings = OllamaEmbeddings(model=model_id)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)
        
    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
