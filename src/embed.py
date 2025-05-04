from enum import Enum

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from ollama_utils import OllamaConfig


class HFModels(Enum):
    default = 'all-MiniLM-L6-v2'

def embed_query(
        query: str, 
        model: SentenceTransformer = HFModels.default
        )-> list[float]:
    
    model: SentenceTransformer = SentenceTransformer(model.value)
    embeddings: np.ndarray = model.encode(sentences=query).tolist() # shape: (len(sentences), 384)
        
    return embeddings

def hf_transformer_embeddings(
        sentences: list[str], 
        model: SentenceTransformer = HFModels.default
        ) -> list[float]:
    
    model: SentenceTransformer = SentenceTransformer(model.value)
    embeddings: list[float] = model.encode(sentences=sentences).tolist() # shape: (len(sentences), 384)
        
    return embeddings

def ollama_embeddings(
        sentences: list[str], 
        model: str  = OllamaConfig.default_embedding_model
        )-> list[float]:
    
    resp = ollama.embed(
                model=model,
                input=sentences,
                )
    embeddings: list[list[float]] = resp.embeddings
    return embeddings


if __name__ == "__main__":
    sentence = "I'm a physicist and a Data Scientist"
    hf_tr_embs = hf_transformer_embeddings([sentence])
    print(len(hf_tr_embs))
    print(len(hf_tr_embs[0]))
    ol_embs = ollama_embeddings([sentence])
    print(len(ol_embs))
    print(len(ol_embs[0]))