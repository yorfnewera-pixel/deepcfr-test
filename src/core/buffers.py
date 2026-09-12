"""Replay-буферы action-only Deep CFR."""
from __future__ import annotations

import numpy as np

from src.core.action_space import NUM_ACTIONS


class _BufferAccounting:
    """Хранит отдельно число валидных строк и число всех увиденных samples."""

    def _initialize_accounting(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Ёмкость replay-буфера должна быть положительной")
        self._size = 0
        self._total_seen = 0

    @property
    def _cur_id(self) -> int:
        """Совместимое read-only имя для внешней диагностики до удаления legacy API."""
        return self._total_seen

    @_cur_id.setter
    def _cur_id(self, value: int) -> None:
        self._total_seen = int(value)

    def _next_reservoir_slot(self) -> tuple[int | None, str]:
        self._total_seen += 1
        if self._size < self.capacity:
            position = self._size
            self._size += 1
            return position, "recorded"

        position = int(np.random.randint(0, self._total_seen))
        if position >= self.capacity:
            return None, "skipped"
        return position, "evicted"

    def _next_fifo_slot(self) -> tuple[int, str]:
        self._total_seen += 1
        if self._size < self.capacity:
            position = self._size
            self._size += 1
            return position, "recorded"
        return (self._total_seen - 1) % self.capacity, "evicted"

    def _clear_samples(self) -> None:
        self._size = 0
        self._total_seen = 0


class AdvantageBuffer(_BufferAccounting):
    """Reservoir буфер ``(state, regrets[6], legal_mask[6], iteration)``."""

    def __init__(self, capacity, state_dim, num_actions=NUM_ACTIONS):
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"AdvantageBuffer требует ровно {NUM_ACTIONS} действий")
        self._initialize_accounting(capacity)
        self.num_actions = NUM_ACTIONS
        self._states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._regrets = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self.eviction_count = 0
        self.skip_count = 0

    def add(self, state, regrets, mask, iteration):
        regrets = np.asarray(regrets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if regrets.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
            raise ValueError("Regret и mask должны содержать ровно шесть действий")
        position, status = self._next_reservoir_slot()
        if position is None:
            self.skip_count += 1
            return status
        if status == "evicted":
            self.eviction_count += 1
        self._states[position] = state
        self._regrets[position] = regrets
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
        return status

    def sample(self, num_samples=-1):
        length = self._size
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
        self._clear_samples()

    def __len__(self):
        return self._size


class DuelingAdvantageBuffer(_BufferAccounting):
    """Reservoir буфер ``(state, Q[6], V, regrets[6], legal_mask[6], iteration)``."""

    def __init__(self, capacity, state_dim, num_actions=NUM_ACTIONS):
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"DuelingAdvantageBuffer требует ровно {NUM_ACTIONS} действий")
        self._initialize_accounting(capacity)
        self.num_actions = NUM_ACTIONS
        self._states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self._action_values = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._state_values = np.empty(self.capacity, dtype=np.float32)
        self._regrets = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self.eviction_count = 0
        self.skip_count = 0

    def add(self, state, action_values, state_value, regrets, mask, iteration):
        state_array = np.asarray(state, dtype=np.float32)
        action_values_array = np.asarray(action_values, dtype=np.float32)
        state_value_array = np.asarray(state_value, dtype=np.float32)
        regrets_array = np.asarray(regrets, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.float32)
        iteration_value = float(iteration)

        expected_state_shape = self._states.shape[1:]
        if state_array.shape != expected_state_shape:
            raise ValueError("State DuelingAdvantageBuffer имеет неверную форму")
        if action_values_array.shape != (NUM_ACTIONS,) or regrets_array.shape != (NUM_ACTIONS,):
            raise ValueError("Q и regrets должны содержать ровно шесть действий")
        if state_value_array.shape != ():
            raise ValueError("V должен быть скалярным значением")
        if mask_array.shape != (NUM_ACTIONS,) or not np.all(np.isin(mask_array, (0.0, 1.0))):
            raise ValueError("legal mask DuelingAdvantageBuffer должен содержать только 0 или 1")
        if not all(
            np.all(np.isfinite(value))
            for value in (state_array, action_values_array, state_value_array, regrets_array, mask_array)
        ):
            raise ValueError("DuelingAdvantageBuffer принимает только конечные значения")
        if not np.isfinite(iteration_value) or iteration_value < 1.0:
            raise ValueError("iteration DuelingAdvantageBuffer должен быть не меньше 1")

        position, status = self._next_reservoir_slot()
        if position is None:
            self.skip_count += 1
            return status
        if status == "evicted":
            self.eviction_count += 1
        self._states[position] = state_array
        self._action_values[position] = action_values_array
        self._state_values[position] = state_value_array
        self._regrets[position] = regrets_array
        self._masks[position] = mask_array
        self._iterations[position] = iteration_value
        return status

    def sample(self, num_samples=-1):
        length = self._size
        if num_samples < 0 or num_samples > length:
            num_samples = length
        if num_samples <= 0:
            return None
        indices = np.random.choice(length, num_samples, replace=False)
        return (
            self._states[indices],
            self._action_values[indices],
            self._state_values[indices],
            self._regrets[indices],
            self._masks[indices],
            self._iterations[indices],
        )

    def clear(self):
        self._clear_samples()

    def __len__(self):
        return self._size


class StrategyBuffer(_BufferAccounting):
    """Буфер ``(state, policy[6], legal_mask[6], iteration)`` с FIFO или reservoir."""

    def __init__(self, capacity, state_dim, num_actions=NUM_ACTIONS, reservoir=False):
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"StrategyBuffer требует ровно {NUM_ACTIONS} действий")
        self._initialize_accounting(capacity)
        self.num_actions = NUM_ACTIONS
        self.reservoir = bool(reservoir)
        self._states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._policies = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self.eviction_count = 0
        self.skip_count = 0

    def add(self, state, policy, mask, iteration):
        policy = np.asarray(policy, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if policy.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
            raise ValueError("Policy и mask должны содержать ровно шесть действий")
        if self.reservoir:
            position, status = self._next_reservoir_slot()
            if position is None:
                self.skip_count += 1
                return status
        else:
            position, status = self._next_fifo_slot()
        if status == "evicted":
            self.eviction_count += 1
        self._states[position] = state
        self._policies[position] = policy
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
        return status

    def sample(self, num_samples=-1):
        length = self._size
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
        self._clear_samples()
        self.eviction_count = 0
        self.skip_count = 0

    def __len__(self):
        return self._size


__all__ = ["AdvantageBuffer", "DuelingAdvantageBuffer", "StrategyBuffer"]
