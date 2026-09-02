"""Replay-буферы action-only Deep CFR."""
from __future__ import annotations

import numpy as np

from src.core.action_space import NUM_ACTIONS


class AdvantageBuffer:
    """Reservoir буфер ``(state, regrets[6], legal_mask[6], iteration)``."""

    def __init__(self, capacity, state_dim, num_actions=NUM_ACTIONS):
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"AdvantageBuffer требует ровно {NUM_ACTIONS} действий")
        self.capacity = int(capacity)
        self.num_actions = NUM_ACTIONS
        self._states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._regrets = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self._cur_id = 0
        self._size = 0
        self.eviction_count = 0
        self.skip_count = 0

    def add(self, state, regrets, mask, iteration):
        regrets = np.asarray(regrets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if regrets.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
            raise ValueError("Regret и mask должны содержать ровно шесть действий")
        if self._cur_id < self.capacity:
            position, status = self._cur_id, "recorded"
        else:
            position = np.random.randint(0, self._cur_id + 1)
            if position >= self.capacity:
                self._cur_id += 1
                self.skip_count += 1
                return "skipped"
            status = "evicted"
            self.eviction_count += 1
        self._states[position] = state
        self._regrets[position] = regrets
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
        self._cur_id += 1
        self._size = min(self._size + 1, self.capacity)
        return status

    def sample(self, num_samples=-1):
        length = min(self._cur_id, self.capacity)
        if num_samples < 0 or num_samples > length:
            num_samples = length
        if num_samples <= 0:
            return None
        indices = np.random.choice(length, num_samples, replace=False)
        return (
            self._states[indices],
            self._regrets[indices],
            self._masks[indices],
            self._iterations[indices],
        )

    def clear(self):
        self._cur_id = 0
        self._size = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)


class StrategyBuffer:
    """Буфер ``(state, policy[6], legal_mask[6], iteration)`` с FIFO или reservoir."""

    def __init__(self, capacity, state_dim, num_actions=NUM_ACTIONS, reservoir=False):
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"StrategyBuffer требует ровно {NUM_ACTIONS} действий")
        self.capacity = int(capacity)
        self.num_actions = NUM_ACTIONS
        self.reservoir = bool(reservoir)
        self._states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._policies = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self._cur_id = 0
        self.eviction_count = 0
        self.skip_count = 0

    def add(self, state, policy, mask, iteration):
        policy = np.asarray(policy, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if policy.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
            raise ValueError("Policy и mask должны содержать ровно шесть действий")
        if self._cur_id < self.capacity:
            position = self._cur_id
            status = "recorded"
        elif self.reservoir:
            position = np.random.randint(0, self._cur_id + 1)
            if position >= self.capacity:
                self._cur_id += 1
                self.skip_count += 1
                return "skipped"
            status = "evicted"
            self.eviction_count += 1
        else:
            position = self._cur_id % self.capacity
            status = "evicted"
            self.eviction_count += 1
        self._states[position] = state
        self._policies[position] = policy
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
        self._cur_id += 1
        return status

    def sample(self, num_samples=-1):
        length = min(self._cur_id, self.capacity)
        if num_samples < 0 or num_samples > length:
            num_samples = length
        if num_samples <= 0:
            return None
        indices = np.random.choice(length, num_samples, replace=False)
        return (
            self._states[indices],
            self._policies[indices],
            self._masks[indices],
            self._iterations[indices],
        )

    def clear(self):
        self._cur_id = 0
        self.eviction_count = 0
        self.skip_count = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)


__all__ = ["AdvantageBuffer", "StrategyBuffer"]
