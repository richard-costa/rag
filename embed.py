from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer


class HFModels(Enum):
    default = 'all-MiniLM-L6-v2'

def embed_query(query: str, model: SentenceTransformer = HFModels.default.value) -> np.ndarray:
    model: SentenceTransformer = SentenceTransformer(HFModels.default.value)
    embeddings: np.ndarray = model.encode(sentences=query) # shape: (len(sentences), 384)
        
    return embeddings

