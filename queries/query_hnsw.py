import numpy as np
import faiss
import os

dim = 2560

# load index and lookup

project_root = os.path.join(os.path.dirname(__file__), '../')

index = faiss.read_index(project_root + "/queries/generated/hnsw.index")

lookup = np.load(project_root + "/queries/generated/lookup.npy")

with open(project_root + "/queries/assets/custom_static_vocab.txt", 'r', encoding='utf-8') as f:
    static_vocab = set(map(int, f))

def get_hnsw_similar_words(query, k):
    query_array = np.array([query])
    dis, idx = index.search(query_array, k)
    # idx: index in embed_tokens_weight
    ans_id = idx[0].tolist()

    # remove all in static_vocab
    for i in range(len(ans_id)-1, 0, -1):
        if ans_id[i] in static_vocab:
            del ans_id[i]
    
    return ans_id

