import threading
from collections import deque
from typing import List


class ActiveVocabManager:
    """Thread-safe global active vocabulary tracker for draft model.
    
    The vocabulary is initialized with the initial vocabulary (from --init-vocab-size).
    Words can then be added or removed from this vocabulary dynamically.
    The vocabulary represents the current draft model vocabulary, which can grow or shrink
    based on client requests.
    """

    def __init__(self, dyna_space: int = 1024) -> None:
        # Storage: deque([token_id_1, token_id_2, ...])
        # Deque makes it easy to implement LRU.
        self.vocab = deque()
        # Capacity configuration: allow growth beyond the static vocab.
        # Use dyna_space parameter if provided, otherwise default to 1024
        self.capacity_buffer = dyna_space
        self.capacity: int = 0
        # Track initial vocabulary size (from --init-vocab-size)
        self.initial_vocab_size: int = 0
        # Track if vocabulary has been initialized with initial vocab
        self.initialized: bool = False
        self.lock = threading.Lock()

    def initialize_config(self, vocab_size: int) -> None:
        """Initialize vocabulary configuration without populating the vocabulary.
        
        This sets up the vocab_size and capacity but leaves the vocabulary empty.
        Words must be added via add_words().
        """
        with self.lock:
            if not self.initialized:
                self.capacity = vocab_size + self.capacity_buffer
                self.initial_vocab_size = vocab_size
                self.initialized = True
    
    def initialize_static_vocab(self, vocab_size: int) -> None:
        """Initialize the vocabulary with the initial vocabulary (0 to vocab_size-1).
        
        This should be called once before any add/remove operations.
        If called multiple times, it will reset the vocabulary to the initial vocab.
        The vocab_size should be the value from --init-vocab-size parameter.
        """
        with self.lock:
            # Initialize with all token IDs from 0 to vocab_size-1
            static_vocab = list(range(vocab_size))
            self.vocab = deque(static_vocab)
            # Ensure there is room for future dynamic additions.
            if not self.initialized:
                self.capacity = vocab_size + self.capacity_buffer
            else:
                self.capacity = max(self.capacity, vocab_size)
            # Store initial vocabulary size
            self.initial_vocab_size = vocab_size
            self.initialized = True

    def is_initialized(self) -> bool:
        """Check if the vocabulary has been initialized with initial vocabulary."""
        with self.lock:
            return self.initialized

    def add_words(self, token_ids: List[int]) -> None:
        """Add words to the vocabulary.
        
        This adds tokens to the current draft model vocabulary. If a token is not
        already in the vocabulary, it will be added. When the vocabulary exceeds
        the capacity (initial_vocab_size + dyna_space), LRU eviction is applied.
        """
        with self.lock:
            # Deduplicate while preserving order: newest tokens at the right end.
            existing_tokens = set(self.vocab)
            for token_id in reversed(token_ids):
                if token_id not in existing_tokens:
                    self.vocab.append(token_id)
                    existing_tokens.add(token_id)

            # Enforce LRU eviction when capacity is exceeded.
            while len(self.vocab) > self.capacity:
                self.vocab.popleft()

    def remove_words(self, token_ids: List[int]) -> None:
        """Remove words from the vocabulary.
        
        This removes tokens from the current draft model vocabulary. After removal,
        only the remaining tokens will be available for generation.
        """
        with self.lock:
            tokens_to_remove = set(token_ids)
            self.vocab = deque(
                tid for tid in self.vocab if tid not in tokens_to_remove
            )

    def get_vocab_list(self) -> List[int]:
        """Get the current vocabulary list.
        
        Returns an empty list if the vocabulary hasn't been initialized.
        """
        with self.lock:
            return list(self.vocab)
    
    def get_initial_vocab_size(self) -> int:
        """Get the initial vocabulary size.
        
        Returns 0 if the vocabulary hasn't been initialized.
        """
        with self.lock:
            return self.initial_vocab_size
    
    def set_dyna_space(self, dyna_space: int) -> None:
        """Update the dynamic space buffer size.
        
        This updates the capacity buffer (dyna_space) and recalculates the capacity.
        Should be called before initialization or after to update the capacity limit.
        """
        with self.lock:
            self.capacity_buffer = dyna_space
            if self.initialized:
                # Recalculate capacity if already initialized
                self.capacity = self.initial_vocab_size + self.capacity_buffer


# Global singleton shared by the API server.
vocab_manager = ActiveVocabManager()

