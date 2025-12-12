from typing import Dict, List

import torch


class DynamicVocabularyManager:
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.populated_size = 0

        # slot -> token_id. -1 means empty.
        self.slots = torch.full((capacity,), -1, dtype=torch.long, device=device)

        # token_id -> slot.
        self.token_to_slot: Dict[int, int] = {}

        # For simple eviction (FIFO/Round-Robin)
        self.next_evict_slot = 0

        # Populated size as GPU tensor for Triton kernels
        self.populated_size_gpu = torch.zeros(1, dtype=torch.int32, device=device)

    def add(self, token_ids: List[int]) -> List[int]:
        """
        Add tokens to the dynamic vocabulary.
        Returns the list of allocated slots.
        """
        slots = []
        for token_id in token_ids:
            if token_id in self.token_to_slot:
                slots.append(self.token_to_slot[token_id])
                continue

            if self.populated_size < self.capacity:
                slot = self.populated_size
                self.populated_size += 1
            else:
                # Evict
                slot = self.next_evict_slot
                self.next_evict_slot = (self.next_evict_slot + 1) % self.capacity
                evicted_token = self.slots[slot].item()
                if evicted_token != -1:
                    if evicted_token in self.token_to_slot:
                        del self.token_to_slot[evicted_token]

            self.slots[slot] = token_id
            self.token_to_slot[token_id] = slot
            slots.append(slot)

        # Update GPU tensor
        if self.populated_size_gpu is not None:
            self.populated_size_gpu[0] = self.populated_size

        return slots

    def get_status(self):
        return {
            "capacity": self.capacity,
            "populated_size": self.populated_size,
            "slots": self.slots.tolist(),
        }
