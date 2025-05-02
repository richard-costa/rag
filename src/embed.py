from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
import ollama 


class HFModels(Enum):
    default = 'all-MiniLM-L6-v2'
    
class OllamaModels(Enum):
    default = 'mxbai-embed-large'


def ensure_model_pulled(model=OllamaModels.default):
    """Ensure the embedding model is pulled locally using Ollama."""
    existing_models = ollama.list()
    existing_model_names = [m.model.split(':')[0] for m in existing_models['models']]
    print(f"Existing models: {existing_model_names}")
    
    if model.value not in existing_model_names:
        print(f"Pulling embedding model {model.value}…")
        ollama.pull(model.value)
    else:
        print(f"Embedding model {model.value} already present.")

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
        model: OllamaModels = OllamaModels.default
        )-> list[float]:
    
    resp = ollama.embed(
                model=model.value,
                input=sentences,
                )
    embeddings: list[list[float]] = resp.embeddings
    return embeddings


if __name__ == "__main__":
    sentence = "I'm a physicist and a Data Scientist"
    hf_tr_embs = hf_transformer_embeddings([sentence])
    print(len(hf_tr_embs))
    print(len(hf_tr_embs[0]))
    ensure_model_pulled()
    ol_embs = ollama_embeddings([sentence])
    print(len(ol_embs))
    print(len(ol_embs[0]))