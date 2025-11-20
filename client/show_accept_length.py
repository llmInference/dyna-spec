#!/usr/bin/env python3
"""
Utility script to send prompts directly to an SGLang runtime and print
the draft model's per-request speculative decoding metrics (especially
the average acceptance length) along with the generated text.

It also supports real-time subscription to monitor the average acceptance
length from /get_server_info endpoint.

Example:
    # Send prompts and show stats
    python client/show_accept_length.py -p "Hello" -p "Write a haiku"
    
    # Real-time subscription mode
    python client/show_accept_length.py --subscribe
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_RUNTIME_URL = os.environ.get("SGLANG_RUNTIME_URL", "http://127.0.0.1:30000")
DEFAULT_CLIENT_ID = os.environ.get("SGLANG_CLIENT_ID", "show-accept-length-client")


class EndpointNotFoundError(RuntimeError):
    """Raised when the expected /generate endpoint does not exist."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send prompts to an SGLang runtime and display the draft model's "
            "average acceptance length for each request."
        )
    )
    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text. Repeat the flag to send multiple prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        action="append",
        dest="prompt_files",
        help="Path to a UTF-8 text file. Each non-empty line becomes one prompt.",
    )
    parser.add_argument(
        "--runtime-url",
        default=DEFAULT_RUNTIME_URL,
        help=f"Runtime base URL (default: {DEFAULT_RUNTIME_URL}).",
    )
    parser.add_argument(
        "--client-id",
        default=DEFAULT_CLIENT_ID,
        help=(
            "Client identifier used when connecting to the FastAPI dynamic vocab server. "
            f"Default: {DEFAULT_CLIENT_ID}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for sampling (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top-p value.",
    )
    parser.add_argument(
        "--sampling-json",
        type=str,
        default=None,
        help="Extra sampling params as JSON string (merged with other values).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Print the generated text for each prompt.",
    )
    parser.add_argument(
        "--suppress-stats",
        action="store_true",
        help="Only print the acceptance length, hiding other stats.",
    )
    parser.add_argument(
        "--subscribe",
        action="store_true",
        help="Enable real-time subscription mode to monitor avg_spec_accept_length from /get_server_info.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for subscription mode (default: 1.0).",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts:
        prompts.extend([p for p in args.prompts if p is not None])

    if args.prompt_files:
        for file_path in args.prompt_files:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    prompt = line.strip()
                    if prompt:
                        prompts.append(prompt)

    if not prompts:
        try:
            prompt = input("Enter a prompt: ").strip()
        except EOFError:
            prompt = ""
        if prompt:
            prompts.append(prompt)

    if not prompts:
        raise SystemExit("No prompts provided. Use -p/--prompt or --prompt-file.")

    return prompts


def build_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    if args.top_p is not None:
        params["top_p"] = args.top_p

    if args.sampling_json:
        try:
            extra = json.loads(args.sampling_json)
            if not isinstance(extra, dict):
                raise ValueError("sampling_json must decode to a JSON object")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse --sampling-json: {exc}") from exc
        params.update(extra)

    return params


def request_generation(
    runtime_url: str,
    prompt: str,
    sampling_params: Dict[str, Any],
    timeout: float,
    client_id: str,
    prefer_api_endpoint: bool,
) -> Tuple[str, Dict[str, Any], bool]:
    """Send a generation request, falling back to /v1/generate if /generate is unavailable."""

    if not prefer_api_endpoint:
        try:
            text, meta = _request_runtime_generate(
                runtime_url, prompt, sampling_params, timeout
            )
            return text, meta, False
        except EndpointNotFoundError:
            prefer_api_endpoint = True

    text, meta = _request_api_generate(
        runtime_url, prompt, sampling_params, timeout, client_id
    )
    return text, meta, True


def _request_runtime_generate(
    runtime_url: str, prompt: str, sampling_params: Dict[str, Any], timeout: float
) -> Tuple[str, Dict[str, Any]]:
    payload = {"text": prompt, "sampling_params": sampling_params}
    url = runtime_url.rstrip("/") + "/generate"

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    if response.status_code == 404:
        raise EndpointNotFoundError(f"{url} returned 404")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:256].strip()
        raise RuntimeError(f"Runtime responded with {response.status_code}: {snippet}") from exc

    obj: Any = response.json()
    return _extract_text_and_meta(obj)


def _request_api_generate(
    runtime_url: str,
    prompt: str,
    sampling_params: Dict[str, Any],
    timeout: float,
    client_id: str,
) -> Tuple[str, Dict[str, Any]]:
    if not client_id:
        raise RuntimeError(
            "client_id is required when falling back to /v1/generate. "
            "Pass --client-id or set SGLANG_CLIENT_ID."
        )

    url = runtime_url.rstrip("/") + "/v1/generate"
    payload: Dict[str, Any] = {
        "client_id": client_id,
        "prompt": prompt,
        "sampling_params": sampling_params,
    }
    # Preserve backwards compatibility with older API servers that inspect top-level fields
    if "max_new_tokens" in sampling_params:
        payload["max_new_tokens"] = sampling_params["max_new_tokens"]
    if "temperature" in sampling_params:
        payload["temperature"] = sampling_params["temperature"]

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    response.raise_for_status()
    obj: Any = response.json()
    text, meta = _extract_text_and_meta(obj)
    if not meta:
        raise RuntimeError(
            "API server response did not include meta_info. "
            "Please update the API server to the latest version."
        )
    return text, meta


def _extract_text_and_meta(obj: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(obj, dict) and "error" in obj:
        raise RuntimeError(obj["error"].get("message", obj["error"]))

    if isinstance(obj, list):
        if not obj:
            raise RuntimeError("Runtime returned an empty list.")
        obj = obj[0]

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected response payload type: {type(obj)}")

    text = obj.get("text", "")
    if isinstance(text, list):
        text = "".join(text)

    meta = obj.get("meta_info") or {}
    if not isinstance(meta, dict):
        raise RuntimeError("meta_info is missing from runtime response.")

    return text, meta


def compute_accept_length(meta: Dict[str, Any]) -> Optional[float]:
    if "spec_accept_length" in meta and meta["spec_accept_length"] is not None:
        return float(meta["spec_accept_length"])

    spec_verify_ct = meta.get("spec_verify_ct", 0)
    completion_tokens = meta.get("completion_tokens")
    if spec_verify_ct and completion_tokens is not None:
        try:
            return float(completion_tokens) / float(spec_verify_ct)
        except ZeroDivisionError:
            return None
    return None


def get_server_info(runtime_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Fetch server info from /get_server_info endpoint."""
    url = runtime_url.rstrip("/") + "/get_server_info"

    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:256].strip()
        raise RuntimeError(f"Runtime responded with {response.status_code}: {snippet}") from exc

    obj: Any = response.json()
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected response payload type: {type(obj)}")

    return obj


def extract_avg_accept_length(server_info: Dict[str, Any]) -> Optional[float]:
    """Extract avg_spec_accept_length from server info."""
    internal_states = server_info.get("internal_states", [])
    if not internal_states:
        return None

    # Get the first internal state (for single DP) or aggregate if multiple
    for state in internal_states:
        if isinstance(state, dict) and "avg_spec_accept_length" in state:
            value = state["avg_spec_accept_length"]
            if value is not None:
                return float(value)

    return None


def subscribe_accept_length(runtime_url: str, interval: float) -> None:
    """Real-time subscription to monitor avg_spec_accept_length."""
    print(f"Subscribing to avg_spec_accept_length from {runtime_url}/get_server_info")
    print(f"Update interval: {interval:.2f} seconds")
    print("Press Ctrl+C to stop.\n")

    # Track if we should continue
    should_continue = True

    def signal_handler(sig, frame):
        nonlocal should_continue
        should_continue = False
        print("\n\nStopping subscription...")

    signal.signal(signal.SIGINT, signal_handler)

    last_value: Optional[float] = None

    try:
        while should_continue:
            try:
                server_info = get_server_info(runtime_url, timeout=interval)
                avg_length = extract_avg_accept_length(server_info)

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Use ANSI escape codes to update the same line
                if avg_length is not None:
                    status = "✓" if last_value is None or abs(avg_length - last_value) < 0.001 else "↗" if avg_length > last_value else "↘"
                    print(
                        f"\r[{timestamp}] Avg Spec Accept Length: {avg_length:.4f} tokens/verify {status}",
                        end="",
                        flush=True,
                    )
                    last_value = avg_length
                else:
                    print(
                        f"\r[{timestamp}] Avg Spec Accept Length: N/A (no data yet)",
                        end="",
                        flush=True,
                    )

                time.sleep(interval)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"\n[Error] {exc}", file=sys.stderr)
                time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print("\nSubscription stopped.")


def render_stats(
    prompt_idx: int,
    prompt: str,
    meta: Dict[str, Any],
    acc_length: Optional[float],
    show_text: bool,
    text: str,
    suppress_stats: bool,
) -> None:
    header = f"Prompt #{prompt_idx}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    if not suppress_stats:
        stats_lines = [
            f"Completion tokens: {meta.get('completion_tokens', 'n/a')}",
            f"Verify steps: {meta.get('spec_verify_ct', 'n/a')}",
            f"Spec accept rate: {meta.get('spec_accept_rate', 'n/a')}",
            f"E2E latency (s): {meta.get('e2e_latency', 'n/a')}",
        ]
        if acc_length is not None:
            stats_lines.insert(
                0, f"Draft avg accept length: {acc_length:.3f} tokens / verify"
            )
        else:
            stats_lines.insert(0, "Draft avg accept length: unavailable")

        print("\n".join(stats_lines))

    if show_text:
        print("\n--- Generated text ---")
        wrapped = textwrap.fill(text, width=100, replace_whitespace=False)
        print(wrapped)
        print("--- End of text ---")

    print()


def main() -> None:
    args = parse_args()

    # If subscribe mode is enabled, run subscription loop
    if args.subscribe:
        subscribe_accept_length(args.runtime_url, args.interval)
        return

    # Otherwise, run the original prompt-based mode
    prompts = load_prompts(args)
    sampling_params = build_sampling_params(args)

    prefer_api_endpoint = False

    for idx, prompt in enumerate(prompts, start=1):
        try:
            text, meta, used_api = request_generation(
                args.runtime_url,
                prompt,
                sampling_params,
                args.timeout,
                args.client_id,
                prefer_api_endpoint,
            )
            if used_api and not prefer_api_endpoint:
                print(
                    f"[Info] /generate not found at {args.runtime_url}. "
                    "Falling back to /v1/generate (dynamic vocab API server).",
                    file=sys.stderr,
                )
            prefer_api_endpoint = used_api
            acc_length = compute_accept_length(meta)
            render_stats(
                idx,
                prompt,
                meta,
                acc_length,
                args.show_response,
                text,
                args.suppress_stats,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Prompt #{idx}] Failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

