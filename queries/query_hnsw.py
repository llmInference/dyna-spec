import numpy as np
import faiss
import os

dim = 2560

# load index and lookup

project_root = os.path.join(os.path.dirname(__file__), '../')

index = faiss.read_index(project_root + "/queries/generated/hnsw.index")

lookup = np.load(project_root + "/queries/generated/lookup.npy")

def get_hnsw_similar_words(query, k):
    query_array = np.array([query])
    dis, idx = index.search(query_array, k)
    # idx: index in embed_tokens_weight
    return idx[0]
