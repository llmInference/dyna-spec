import threading
from collections import defaultdict, deque
from typing import List


class ActiveVocabManager:
    """Thread-safe LRU-based active vocabulary tracker per client."""

    def __init__(self) -> None:
        # Storage: {client_id: deque([token_id_1, token_id_2, ...])}
        # Deque makes it easy to implement per-client LRU.
        self.client_vocabs = defaultdict(deque)
        # Capacity map, defaulting to 1024 tokens per client.
        self.client_capacities = defaultdict(lambda: 1024)
        self.lock = threading.Lock()

    def add_words(self, client_id: str, token_ids: List[int]) -> None:
        with self.lock:
            vocab_deque = self.client_vocabs[client_id]
            capacity = self.client_capacities[client_id]

            # Deduplicate while preserving order: newest tokens at the right end.
            existing_tokens = set(vocab_deque)
            for token_id in reversed(token_ids):
                if token_id not in existing_tokens:
                    vocab_deque.append(token_id)
                    existing_tokens.add(token_id)

            # Enforce LRU eviction when capacity is exceeded.
            while len(vocab_deque) > capacity:
                vocab_deque.popleft()

    def remove_words(self, client_id: str, token_ids: List[int]) -> None:
        with self.lock:
            vocab_deque = self.client_vocabs[client_id]
            tokens_to_remove = set(token_ids)
            self.client_vocabs[client_id] = deque(
                tid for tid in vocab_deque if tid not in tokens_to_remove
            )

    def get_vocab_list(self, client_id: str) -> List[int]:
        with self.lock:
            return list(self.client_vocabs.get(client_id, []))


# Global singleton shared by the API server.
vocab_manager = ActiveVocabManager()

