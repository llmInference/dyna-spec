# token_id -> token
# created using vocab.json in Qwen3-4B

import numpy as np
import json

with open("./assets/vocab.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

    lookup = np.empty((151643), dtype='U128')   # vocab.json中最长的token有这么长，我就把所有都设成这么长了，虽然有点占空间

    for string, idx in data.items():
        lookup[idx] = string

    np.save("./generated/lookup.npy", lookup)

