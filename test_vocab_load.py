import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_static_vocab_from_file(path, vocab_size):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return None, None

    indices = []
    seen_tokens = set()
    invalid_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                token_id = int(stripped)
            except ValueError:
                invalid_count += 1
                print(f"Invalid line {line_no}: {stripped}")
                continue

            if token_id < 0 or token_id >= vocab_size:
                invalid_count += 1
                print(f"Out of range line {line_no}: {token_id}")
                continue

            if token_id in seen_tokens:
                continue

            seen_tokens.add(token_id)
            indices.append(token_id)

    return indices, path


vocab_size = 152000
indices, path = _load_static_vocab_from_file("numbers.txt", vocab_size)

if indices:
    print(f"Loaded {len(indices)} indices.")
    if 210 in indices:
        print("210 is in indices.")
    else:
        print("210 is NOT in indices.")

    if 198 in indices:
        print("198 is in indices.")
    else:
        print("198 is NOT in indices.")

    # Check mapping
    inverse_map = {token_id: idx for idx, token_id in enumerate(indices)}
    print(f"Map[198] = {inverse_map.get(198)}")
    print(f"Map[210] = {inverse_map.get(210)}")
else:
    print("Failed to load indices.")
