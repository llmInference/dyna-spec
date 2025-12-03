import numpy as np
import faiss

dim = 2560

# load index and lookup

index = faiss.read_index("./generated/hnsw.index")

lookup = np.load("./generated/lookup.npy")

def get_hnsw_similar_words(query, k):
    query_array = np.array([query])
    dis, idx = index.search(query_array, k)
    # idx: index in embed_tokens_weight
    return idx[0]
