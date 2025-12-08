import igraph as ig
import numpy as np
import json
import os

project_root = os.path.join(os.path.dirname(__file__), '../')

# lookup = np.load("./generated/lookup.npy")

# with open("./assets/vocab.json", 'r', encoding='utf-8') as f:
#     vocab = json.load(f)

with open(project_root + "/queries/assets/custom_static_vocab.txt", 'r', encoding='utf-8') as f:
    static_vocab = set(map(int, f))

# load graph

g = ig.load(project_root + "/queries/generated/graph.graphml", format="graphml")
    

def get_co_occurrence(query_id):
    ans_id = g.neighbors(vertex=query_id)

    # remove all in static_vocab
    for i in range(0, len(ans_id)):
        if ans_id[i] in static_vocab:
            del ans_id[i]
    
    return ans_id
