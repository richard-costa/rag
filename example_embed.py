import numpy as np
from sentence_transformers import SentenceTransformer

from db_utils import insert_data_into_table
from embed import HFModels, embed_query


def generate_encodings(
        sentences: list, 
        model: SentenceTransformer = HFModels.default.value,
        save_to_file: bool = True, 
        filename: str = 'example_embeddings.npy'
        ) -> np.ndarray:
    
    try:
        embeddings = np.load(filename)
        return embeddings
    except FileNotFoundError:
        print(f"File '{filename}' not found. Generating embeddings...")

    model: SentenceTransformer = SentenceTransformer(HFModels.default.value)
    embeddings: np.ndarray = model.encode(sentences=sentences) # shape: (len(sentences), 384)
    if save_to_file: np.save('example_embeddings.npy', embeddings)
        

    return embeddings

if __name__ == '__main__':

    sentences = ["I'm a physicist and a Data Scientist", "I don't linke the Copenhagen interpretation"]
    embeddings: np.ndarray = generate_encodings(sentences)
    embeddings = embeddings.tolist()
    insert_data_into_table('test_db', sentences, embeddings)

