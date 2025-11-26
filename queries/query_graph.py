import igraph as ig
import numpy as np
import json

lookup = np.load("./generated/lookup.npy")

with open("./assets/vocab.json", 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# load graph

g = ig.load("./generated/graph.graphml", format="graphml")
    

def get_co_occurrence(query):
    query_id = vocab[query]
    ans_id = g.neighbors(vertex=query_id)
    
    return lookup[ans_id]
