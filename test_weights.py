#!/usr/bin/env python3
"""
Test script to verify dynamic vocabulary weights extraction and usage.
"""
import json

import requests

SERVER_URL = "http://localhost:30000"


def test_dynamic_vocab_weights():
    """Check if dynamic vocab weights are extracted correctly."""

    # 1. Add some tokens
    print("Adding tokens to dynamic vocab...")
    response = requests.post(
        f"{SERVER_URL}/dynamic_vocab/add",
        headers={"Content-Type": "application/json"},
        json={
            "vocab_size": 151936,
            "new_token_ids": [100, 200, 300, 400, 500, 1000, 2000, 3000],
        },
    )
    print(f"Add response: {response.json()}")

    # 2. Check status and weights
    print("\nChecking dynamic vocab status...")
    response = requests.post(
        f"{SERVER_URL}/dynamic_vocab/status",
        headers={"Content-Type": "application/json"},
        json={},
    )
    status = response.json()
    print(f"Populated size: {status['populated_size']}")
    print(f"Weights head norm: {status.get('weights_head_norm', 'N/A')}")
    print(f"Weights head mean: {status.get('weights_head_mean', 'N/A')}")

    # 3. Analyze weights
    if "weights_head_norm" in status:
        norms = status["weights_head_norm"]
        means = status["weights_head_mean"]

        print("\nWeight statistics:")
        print(f"  Norm range: [{min(norms):.4f}, {max(norms):.4f}]")
        print(f"  Mean range: [{min(means):.6f}, {max(means):.6f}]")

        # Check if weights look reasonable (non-zero, not too large)
        if all(abs(n) < 0.001 for n in norms):
            print("  ⚠️  WARNING: All weight norms are near zero!")
        elif any(abs(n) > 100 for n in norms):
            print("  ⚠️  WARNING: Some weight norms are very large!")
        else:
            print("  ✓ Weight norms look reasonable")

    # 4. Test generation with high temperature to encourage diversity
    print("\nTesting generation with high temperature...")
    for i in range(3):
        response = requests.post(
            f"{SERVER_URL}/generate",
            headers={"Content-Type": "application/json"},
            json={
                "text": f"Test {i}:",
                "sampling_params": {
                    "max_new_tokens": 20,
                    "temperature": 1.5,  # High temperature
                    "top_k": 50,
                    "top_p": 0.95,
                },
            },
        )
        result = response.json()
        print(f"  Generated IDs: {result['output_ids'][:10]}")


if __name__ == "__main__":
    test_dynamic_vocab_weights()
