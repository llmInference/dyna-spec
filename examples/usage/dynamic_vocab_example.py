import argparse

import requests
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://localhost:30000")
    parser.add_argument("--hidden-size", type=int, default=4096)
    args = parser.parse_args()

    # 1. Add new tokens
    print(f"Adding new tokens to {args.host}...")
    url = f"{args.host}/dynamic_vocab/add"

    # Example: Add 2 new tokens
    # Note: The token IDs should be consistent with what the tokenizer expects or handles.
    # For this demo, we assume the base vocab size is X and we are adding X, X+1.
    # In a real scenario, you would also update the tokenizer on the client side.

    new_token_ids = [32000, 32001]
    new_weights = torch.randn(len(new_token_ids), args.hidden_size).tolist()

    data = {
        "vocab_size": 32000,  # This might be used for validation or offset
        "new_token_ids": new_token_ids,
        "new_weights": new_weights,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print("Add response:", response.json())
    except Exception as e:
        print(f"Failed to add tokens: {e}")
        if "response" in locals():
            print(response.text)

    # 2. Check status
    print(f"\nChecking status from {args.host}...")
    url = f"{args.host}/dynamic_vocab/status"

    try:
        response = requests.post(url, json={})
        response.raise_for_status()
        print("Status response:", response.json())
    except Exception as e:
        print(f"Failed to get status: {e}")
        if "response" in locals():
            print(response.text)


if __name__ == "__main__":
    main()
