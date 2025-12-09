import os
import uuid

import requests


BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")
ADD_ENDPOINT = "/v1/vocab/add"
REMOVE_ENDPOINT = "/v1/vocab/remove"
GENERATE_ENDPOINT = "/v1/generate"


def post_json(endpoint: str, payload: dict) -> requests.Response:
    url = f"{BASE_URL}{endpoint}"
    response = requests.post(url, json=payload, timeout=10)
    print(f"\nPOST {url}")
    print(f"Payload: {payload}")
    print(f"Status code: {response.status_code}")
    try:
        parsed = response.json()
    except requests.exceptions.JSONDecodeError:
        body_preview = response.text[:512].strip()
        print("Response JSON: <failed to decode>")
        print(f"Raw body preview: {body_preview or '<empty>'}")
        response.raise_for_status()
        raise RuntimeError(
            f"Failed to decode JSON from {url}; status {response.status_code}"
        )
    else:
        print(f"Response JSON: {parsed}")
        response.raise_for_status()
    return response


def main():
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    vocab_words = ["token-alpha", "token-beta", "token-gamma"]

    # Step 1: add a temporary vocab list.
    add_resp = post_json(ADD_ENDPOINT, {"client_id": client_id, "words": vocab_words})
    assert add_resp.json().get("status") == "success", "Failed to add vocabulary"
    print(f"✅ Added vocab for {client_id}")

    # Optional check: try generating once to ensure vocab is recognized.
    gen_payload = {
        "client_id": client_id,
        "prompt": "Testing vocab add/remove flow",
        "max_new_tokens": 1,
    }
    gen_resp = post_json(GENERATE_ENDPOINT, gen_payload)
    print(f"Generation response: {gen_resp.json()}")

    # Step 2: remove the same vocab entries.
    remove_resp = post_json(
        REMOVE_ENDPOINT, {"client_id": client_id, "words": vocab_words}
    )
    assert remove_resp.json().get("status") == "success", "Failed to remove vocabulary"
    print(f"✅ Removed vocab for {client_id}")

    # Final check: generation should now fail because the vocab list is empty.
    try:
        post_json(GENERATE_ENDPOINT, gen_payload)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            print("✅ Backend confirms vocab removal (400: no active vocabulary).")
        else:
            raise
    else:
        raise RuntimeError("Expected generation to fail after vocab removal")


if __name__ == "__main__":
    main()

