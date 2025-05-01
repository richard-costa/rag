from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer


class HFModels(Enum):
    default = 'all-MiniLM-L6-v2'
    
def embed_query(
        query: str, 
        model: SentenceTransformer = HFModels.default
        )-> list[float]:
    
    model: SentenceTransformer = SentenceTransformer(model.value)
    embeddings: np.ndarray = model.encode(sentences=query).tolist() # shape: (len(sentences), 384)
        
    return embeddings

def generate_transformer_embeddings(
        sentences: list[str], 
        model: SentenceTransformer = HFModels.default
        ) -> list[float]:
    
    model: SentenceTransformer = SentenceTransformer(model.value)
    embeddings: list[float] = model.encode(sentences=sentences).tolist() # shape: (len(sentences), 384)
        
    return embeddings


if __name__ == "__main__":
    query = "I'm a physicist and a Data Scientist"
    embeddings = embed_query(query)
    print(embeddings)