from sentence_transformers import SentenceTransformer
from typing import List


class SentenceEmbeddings:
    def __init__(self, modelPath):
        self.modelPath=modelPath
        self.model = SentenceTransformer(self.modelPath)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
        
    def embed_query(self, texts: str) -> List[float]:
        return self.embed_documents([texts])[0]
