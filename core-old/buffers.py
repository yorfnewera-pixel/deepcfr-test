"""NumPy-only replay and training buffers extracted from deep_cfr."""

import numpy as np

class PrioritizedMemory:
    def __init__(self, capacity, state_dim, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._action_types = np.empty(capacity, dtype=np.int32)
        self._bet_sizes = np.empty(capacity, dtype=np.float32)
        self._regrets = np.empty(capacity, dtype=np.float32)
        self._priorities = np.empty(capacity, dtype=np.float64)
        self._position = 0
        self._size = 0
        self._max_priority = 1.0

    def add(self, state, action_type, bet_size, regret, priority=None):
        if priority is None:
            priority = self._max_priority
        pos = self._position
        self._states[pos] = state
        self._action_types[pos] = action_type
        self._bet_sizes[pos] = bet_size
        self._regrets[pos] = float(regret)
        self._priorities[pos] = priority ** self.alpha
        self._position = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta=0.4):
        if self._size < batch_size:
            idx = np.arange(self._size)
            return (self._states[:self._size], self._action_types[:self._size],
                    self._bet_sizes[:self._size], self._regrets[:self._size],
                    idx, np.ones(self._size, dtype=np.float32))

        priorities = self._priorities[:self._size]
        probs = priorities / priorities.sum()
        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        sample_probs = probs[indices]
        weights = (self._size * sample_probs) ** -beta
        weights = weights / weights.max()
        return (self._states[indices], self._action_types[indices],
                self._bet_sizes[indices], self._regrets[indices],
                indices, weights.astype(np.float32))

    def update_priority(self, index, priority):
        priority = max(1e-8, priority)
        self._max_priority = max(self._max_priority, priority)
        self._priorities[index] = priority ** self.alpha

    def clear(self):
        self._position = 0
        self._size = 0
        self._max_priority = 1.0

    def __len__(self):
        return self._size

    def get_memory_stats(self):
        if self._size == 0:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}
        raw = self._priorities[:self._size] ** (1 / self.alpha)
        return {
            "min": raw.min(), "max": raw.max(),
            "mean": raw.mean(), "median": np.median(raw),
            "size": self._size
        }

class PolicyGradientMemory:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._z_raws = np.empty(capacity, dtype=np.float32)
        self._regret_raises = np.empty(capacity, dtype=np.float32)
        self._bet_sizes = np.empty(capacity, dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._sources = np.zeros(capacity, dtype=np.int32)
        self._position = 0
        self._size = 0

    def add(self, state, z_raw, regret_raise, bet_size, iteration=-1, source=0):
        pos = self._position
        self._states[pos] = state
        self._z_raws[pos] = float(z_raw)
        self._regret_raises[pos] = float(regret_raise)
        self._bet_sizes[pos] = float(bet_size)
        self._iterations[pos] = float(iteration)
        self._sources[pos] = int(source)
        self._position = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, min_iteration=None):
        if self._size < batch_size:
            return None
        if min_iteration is not None:
            fresh_mask = self._iterations[:self._size] >= min_iteration
            fresh_indices = np.where(fresh_mask)[0]
            if len(fresh_indices) < batch_size:
                if len(fresh_indices) == 0:
                    return None
                indices = fresh_indices
            else:
                indices = np.random.choice(fresh_indices, batch_size, replace=False)
        else:
            indices = np.random.choice(self._size, batch_size, replace=False)
        return (self._states[indices], self._z_raws[indices],
                self._regret_raises[indices], self._bet_sizes[indices],
                self._sources[indices])

    def clear(self):
        self._position = 0
        self._size = 0

    def __len__(self):
        return self._size

class AdvantageBuffer:
    """Reservoir buffer: full-vector storage (state, regrets[num_actions], mask[num_actions], iteration).
    Uniform sampling. Cleared each CFR iteration."""
    def __init__(self, capacity, state_dim, num_actions=4):
        self.capacity = capacity
        self.num_actions = num_actions
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._regrets = np.empty((capacity, num_actions), dtype=np.float32)
        self._masks = np.empty((capacity, num_actions), dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._cur_id = 0
        self._size = 0

    def add(self, state, regrets, mask, iteration):
        regrets = np.asarray(regrets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if self._cur_id < self.capacity:
            pos = self._cur_id
        else:
            idx = np.random.randint(0, self._cur_id + 1)
            if idx >= self.capacity:
                self._cur_id += 1
                return
            pos = idx
        self._states[pos] = state
        self._regrets[pos] = regrets
        self._masks[pos] = mask
        self._iterations[pos] = float(iteration)
        if self._cur_id < self.capacity:
            self._size = min(self._size + 1, self.capacity)
        self._cur_id += 1

    def sample(self, num_samples=-1):
        data_length = min(self._cur_id, self.capacity)
        if num_samples < 0 or num_samples > data_length:
            num_samples = data_length
        if num_samples <= 0:
            return None
        indices = np.random.choice(data_length, num_samples, replace=False)
        return (self._states[indices], self._regrets[indices],
                self._masks[indices], self._iterations[indices])

    def clear(self):
        self._cur_id = 0
        self._size = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)

class StrategyBuffer:
    """Reservoir buffer for strategy data: (state, policy[num_actions], mask[num_actions], iteration, bet_size).
    NOT cleared between iterations — accumulates across all iterations."""
    def __init__(self, capacity, state_dim, num_actions=4):
        self.capacity = capacity
        self.num_actions = num_actions
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._policies = np.empty((capacity, num_actions), dtype=np.float32)
        self._masks = np.empty((capacity, num_actions), dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._bet_sizes = np.zeros(capacity, dtype=np.float32)
        self._cur_id = 0

    def add(self, state, policy, mask, iteration, bet_size=0.0):
        policy = np.asarray(policy, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if self._cur_id < self.capacity:
            pos = self._cur_id
        else:
            idx = np.random.randint(0, self._cur_id + 1)
            if idx >= self.capacity:
                self._cur_id += 1
                return
            pos = idx
        self._states[pos] = state
        self._policies[pos] = policy
        self._masks[pos] = mask
        self._iterations[pos] = float(iteration)
        self._bet_sizes[pos] = float(bet_size)
        self._cur_id += 1

    def sample(self, num_samples):
        data_length = min(self._cur_id, self.capacity)
        if data_length < num_samples:
            return None
        indices = np.random.choice(data_length, num_samples, replace=False)
        return (self._states[indices], self._policies[indices],
                self._masks[indices], self._iterations[indices],
                self._bet_sizes[indices])

    def clear(self):
        self._cur_id = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)

class SizingAdvantageBuffer:
    """Reservoir buffer для regret-вектора фиксированных sizing-анкеров."""

    def __init__(self, capacity, state_dim, num_anchors=4):
        self.capacity = capacity
        self.num_anchors = num_anchors
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._regrets = np.empty((capacity, num_anchors), dtype=np.float32)
        self._masks = np.empty((capacity, num_anchors), dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._cur_id = 0

    def add(self, state, regrets, mask, iteration):
        regrets = np.asarray(regrets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if self._cur_id < self.capacity:
            pos = self._cur_id
        else:
            idx = np.random.randint(0, self._cur_id + 1)
            if idx >= self.capacity:
                self._cur_id += 1
                return
            pos = idx
        self._states[pos] = state
        self._regrets[pos] = regrets
        self._masks[pos] = mask
        self._iterations[pos] = float(iteration)
        self._cur_id += 1

    def sample(self, num_samples=-1):
        data_length = min(self._cur_id, self.capacity)
        if num_samples < 0 or num_samples > data_length:
            num_samples = data_length
        if num_samples <= 0:
            return None
        indices = np.random.choice(data_length, num_samples, replace=False)
        return (
            self._states[indices], self._regrets[indices],
            self._masks[indices], self._iterations[indices],
        )

    def clear(self):
        self._cur_id = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)

class SizingStrategyBuffer:
    """Reservoir buffer для дистилляции sizing-probs по фиксированным анкорам."""

    def __init__(self, capacity, state_dim, num_anchors=4):
        self.capacity = capacity
        self.num_anchors = num_anchors
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._probs = np.empty((capacity, num_anchors), dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._cur_id = 0

    def add(self, state, probs, iteration):
        probs = np.asarray(probs, dtype=np.float32)
        if self._cur_id < self.capacity:
            pos = self._cur_id
        else:
            idx = np.random.randint(0, self._cur_id + 1)
            if idx >= self.capacity:
                self._cur_id += 1
                return
            pos = idx
        self._states[pos] = state
        self._probs[pos] = probs
        self._iterations[pos] = float(iteration)
        self._cur_id += 1

    def sample(self, num_samples):
        data_length = min(self._cur_id, self.capacity)
        if data_length < num_samples:
            return None
        indices = np.random.choice(data_length, num_samples, replace=False)
        return self._states[indices], self._probs[indices], self._iterations[indices]

    def clear(self):
        self._cur_id = 0

    def __len__(self):
        return min(self._cur_id, self.capacity)

class QValueBuffer:
    """Circular buffer для обучения Q-сети variance reduction.
    Хранит (state, action, reward, next_state, next_policy_state, next_mask, is_terminal).
    next_state — perspective traversing_player (Q-value target).
    next_policy_state — perspective current_player (strategy bootstrap).
    НЕ очищается каждую итерацию — Q-сети нужен persistent experience replay."""

    def __init__(self, capacity, state_dim, num_actions=4):
        self.capacity = capacity
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int32)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_policy_states = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_masks = np.empty((capacity, num_actions), dtype=np.float32)
        self._terminals = np.empty(capacity, dtype=np.float32)
        self._next_is_hero = np.zeros(capacity, dtype=np.float32)
        self._next_player_ids = np.full(capacity, -1, dtype=np.int32)
        self._position = 0
        self._size = 0

    def add(self, state, action, reward, next_state, next_policy_state, next_mask, is_terminal, next_is_hero=False, next_player_id=-1):
        pos = self._position
        self._states[pos] = state
        self._actions[pos] = action
        self._rewards[pos] = reward
        self._next_states[pos] = next_state
        self._next_policy_states[pos] = next_policy_state
        self._next_masks[pos] = next_mask
        self._terminals[pos] = float(is_terminal)
        self._next_is_hero[pos] = float(bool(next_is_hero))
        self._next_player_ids[pos] = int(next_player_id)
        self._position = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        if self._size < batch_size:
            return None
        indices = np.random.choice(self._size, batch_size, replace=False)
        return (self._states[indices], self._actions[indices],
                self._rewards[indices], self._next_states[indices],
                self._next_policy_states[indices],
                self._next_masks[indices], self._terminals[indices],
                self._next_is_hero[indices])

    def __len__(self):
        return self._size

class SizingQBuffer:
    """Circular buffer для обучения SizingQNetwork (Bug #48).
    Хранит (state, normalized_size, target_value, iteration, source, anchor_idx).
    target_value = raw_v3 / pot (pot-normalized) или raw_v3 как есть.
    НЕ очищается каждую итерацию — persistent replay для Q-аппроксимации."""

    SELECTED_KIND_TO_ID = {
        'UNKNOWN': 0,
        'NORMAL': 1,
        'MIN_RAISE': 2,
        'ALL_IN': 3,
    }
    SELECTED_ID_TO_KIND = {v: k for k, v in SELECTED_KIND_TO_ID.items()}

    def __init__(self, capacity, state_dim, num_anchors=15):
        self.capacity = capacity
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._norm_sizes = np.empty(capacity, dtype=np.float32)
        self._targets = np.empty(capacity, dtype=np.float32)
        self._iterations = np.empty(capacity, dtype=np.float32)
        self._sources = np.zeros(capacity, dtype=np.int32)
        self._anchor_indices = np.full(capacity, -1, dtype=np.int32)
        self._anchor_counts = np.zeros(num_anchors, dtype=np.int64)
        self._selected_anchor_indices = np.full(capacity, -1, dtype=np.int32)
        self._effective_anchor_indices = np.full(capacity, -1, dtype=np.int32)
        self._selected_kind_ids = np.full(capacity, self.SELECTED_KIND_TO_ID['UNKNOWN'], dtype=np.int32)
        self._version = 0
        self._position = 0
        self._size = 0

    def add(self, state, normalized_size, target_value, iteration, source=0, anchor_idx=-1,
            selected_anchor_idx=-1, effective_anchor_idx=None, selected_kind='UNKNOWN'):
        pos = self._position
        if self._size >= self.capacity:
            old_idx = int(self._anchor_indices[pos])
            if old_idx >= 0:
                self._anchor_counts[old_idx] = max(self._anchor_counts[old_idx] - 1, 0)
        self._states[pos] = state
        self._norm_sizes[pos] = float(normalized_size)
        self._targets[pos] = float(target_value)
        self._iterations[pos] = float(iteration)
        self._sources[pos] = int(source)
        self._anchor_indices[pos] = int(anchor_idx)
        if anchor_idx >= 0:
            self._anchor_counts[anchor_idx] += 1
        kind_id = self.SELECTED_KIND_TO_ID.get(str(selected_kind), self.SELECTED_KIND_TO_ID['UNKNOWN'])
        eff_idx = int(anchor_idx if effective_anchor_idx is None else effective_anchor_idx)
        self._selected_anchor_indices[pos] = int(selected_anchor_idx)
        self._effective_anchor_indices[pos] = eff_idx
        self._selected_kind_ids[pos] = int(kind_id)
        self._version += 1
        self._position = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, return_sources=False, return_selected_kinds=False):
        if self._size < batch_size:
            return None
        indices = np.random.choice(self._size, batch_size, replace=False)
        if return_sources:
            result = (self._states[indices], self._norm_sizes[indices],
                      self._targets[indices], self._sources[indices])
            if return_selected_kinds:
                result = result + (self._selected_kind_ids[indices],)
            return result
        return (self._states[indices], self._norm_sizes[indices],
                self._targets[indices])

    def sample_stratified(self, batch_size, num_anchors, return_sources=False, return_selected_kinds=False):
        if self._size < batch_size:
            return None

        anchors = self._anchor_indices[:self._size]
        present = np.unique(anchors[anchors >= 0])
        n_present = len(present)
        if n_present <= 1:
            return self.sample(batch_size, return_sources=return_sources,
                               return_selected_kinds=return_selected_kinds)

        per_anchor = batch_size // n_present
        remainder = batch_size - per_anchor * n_present
        all_indices = np.arange(self._size)

        chosen = []
        for anchor in present:
            mask = anchors == anchor
            pool = all_indices[mask]
            take = min(per_anchor, len(pool))
            if take == 0:
                continue
            chosen.extend(np.random.choice(pool, size=take, replace=True).tolist())

        rest_pool = np.setdiff1d(all_indices, chosen)
        if remainder > 0 and len(rest_pool) > 0:
            take_rem = min(remainder, len(rest_pool))
            chosen.extend(np.random.choice(rest_pool, size=take_rem, replace=False).tolist())

        if len(chosen) < batch_size:
            fill = batch_size - len(chosen)
            fill_indices = np.random.choice(self._size, fill, replace=False)
            chosen.extend(fill_indices.tolist())

        indices = np.array(chosen[:batch_size], dtype=np.int64)
        np.random.shuffle(indices)
        if return_sources:
            result = (self._states[indices], self._norm_sizes[indices],
                      self._targets[indices], self._sources[indices])
            if return_selected_kinds:
                result = result + (self._selected_kind_ids[indices],)
            return result
        return (self._states[indices], self._norm_sizes[indices],
                self._targets[indices])

    def __len__(self):
        return self._size

    def clear(self):
        """Очищает replay без пересоздания numpy-массивов."""
        self._position = 0
        self._size = 0
        self._anchor_counts.fill(0)
        self._version += 1

    def anchor_counts(self, num_anchors):
        return self._anchor_counts[:num_anchors].copy()

    def selected_effective_target_summary(self, num_anchors):
        """Диагностика: mean/min/max target по (source, selected, effective, kind).

        Возвращает dict вида:
            'source=0|selected=4|effective=1|kind=ALL_IN': {
                'count': N, 'target_mean': ..., 'target_min': ..., 'target_max': ...
            }
        """
        summary = {}
        for idx in range(self._size):
            selected_idx = int(self._selected_anchor_indices[idx])
            effective_idx = int(self._effective_anchor_indices[idx])
            kind_id = int(self._selected_kind_ids[idx])
            source = int(self._sources[idx])

            if selected_idx < 0 or selected_idx >= num_anchors:
                continue
            if effective_idx < 0 or effective_idx >= num_anchors:
                continue

            kind = self.SELECTED_ID_TO_KIND.get(kind_id, 'UNKNOWN')
            key = f'source={source}|selected={selected_idx}|effective={effective_idx}|kind={kind}'
            target = float(self._targets[idx])

            if key not in summary:
                summary[key] = {
                    'count': 0,
                    'target_sum': 0.0,
                    'target_min': target,
                    'target_max': target,
                }

            item = summary[key]
            item['count'] += 1
            item['target_sum'] += target
            item['target_min'] = min(item['target_min'], target)
            item['target_max'] = max(item['target_max'], target)

        for item in summary.values():
            item['target_mean'] = float(item['target_sum']) / max(item['count'], 1)
            del item['target_sum']

        return summary

# Preserve historical pickle identity for checkpoints containing instances.
for _cls in (PrioritizedMemory, PolicyGradientMemory, AdvantageBuffer, StrategyBuffer, SizingAdvantageBuffer, SizingStrategyBuffer, QValueBuffer, SizingQBuffer):
    _cls.__module__ = "src.core.deep_cfr"

__all__ = ['PrioritizedMemory', 'PolicyGradientMemory', 'AdvantageBuffer', 'StrategyBuffer', 'SizingAdvantageBuffer', 'SizingStrategyBuffer', 'QValueBuffer', 'SizingQBuffer']
