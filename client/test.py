import requests

API_BASE = "http://127.0.0.1:30000"  # 改成你的 api_server 地址

def inspect_hidden_states(prompt: str = "Hello world"):
    url = f"{API_BASE}/v1/hidden_states"
    payload = {
        "prompt": prompt,
        "max_new_tokens": 0,   # 只算 hidden states，不生成新 token
        "temperature": 0.0,
        "sampling_params": {},
        "client_id": None,
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    hidden_states = data.get("hidden_states")
    meta_info = data.get("meta_info")

    print("type(hidden_states):", type(hidden_states))

    if hidden_states is None:
        print("hidden_states is None, meta_info:", meta_info)
        return

    try:
        print("len(hidden_states):", len(hidden_states))
    except TypeError:
        print("hidden_states has no len(), value:", hidden_states)
        return

    # 如果是二维或三维列表，进一步打印前几层长度
    try:
        if len(hidden_states) > 0 and isinstance(hidden_states[0], (list, tuple)):
            print("len(hidden_states[0]):", len(hidden_states[0]))
            if len(hidden_states[0]) > 0 and isinstance(hidden_states[0][0], (list, tuple)):
                print("len(hidden_states[0][0]):", len(hidden_states[0][0]))
    except Exception as e:
        print("Error while inspecting nested lengths:", e)

    print("sample element type:", type(hidden_states[0]))
    print("meta_info keys:", list(meta_info.keys()) if isinstance(meta_info, dict) else meta_info)

if __name__ == "__main__":
    inspect_hidden_states("introduce yourself")