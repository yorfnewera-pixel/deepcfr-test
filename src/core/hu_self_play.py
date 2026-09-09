"""Изолированный контур HU self-play с неизменяемым профилем policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import AdvantageBuffer


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class HuTraversalAdapter(Generic[StateT]):
    """Адаптирует игровой движок к независимому HU external-sampling обходу."""

    current_player: Callable[[StateT], int]
    is_terminal: Callable[[StateT], bool]
    legal_mask: Callable[[StateT], np.ndarray]
    encode: Callable[[StateT, int], np.ndarray]
    apply: Callable[[StateT, int], StateT]
    terminal_value: Callable[[StateT, int], float]


class HuStrategyBuffer:
    """Общий reservoir-буфер strategy с явным actor_id для условной сети."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Ёмкость HU strategy-буфера должна быть положительной")
        self.state_dim = int(state_dim)
        self._states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self._actor_ids = np.empty(self.capacity, dtype=np.int64)
        self._policies = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
        self._cur_id = 0
        self.eviction_count = 0
        self.skip_count = 0

    @staticmethod
    def condition_state(state: np.ndarray, actor_id: int) -> np.ndarray:
        """Добавляет one-hot P0/P1 к относительному infoset перед strategy_net."""
        actor = int(actor_id)
        if actor not in (0, 1):
            raise ValueError("HU strategy принимает только actor_id 0 или 1")
        actor_features = np.zeros(2, dtype=np.float32)
        actor_features[actor] = 1.0
        return np.concatenate((np.asarray(state, dtype=np.float32), actor_features))

    def add(self, actor_id: int, state, policy, mask, iteration: int) -> str:
        actor = int(actor_id)
        state = np.asarray(state, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if actor not in (0, 1):
            raise ValueError("HU strategy принимает только actor_id 0 или 1")
        if state.shape != (self.state_dim,):
            raise ValueError("HU strategy state имеет неверный размер")
        if policy.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
            raise ValueError("HU strategy policy и mask должны содержать шесть действий")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(policy)):
            raise ValueError("HU strategy sample содержит NaN или Inf")
        legal = mask > 0.0
        if not legal.any() or np.any(policy < 0.0):
            raise ValueError("HU strategy policy или mask некорректны")
        if not np.allclose(policy[~legal], 0.0, atol=1e-6, rtol=0.0):
            raise ValueError("HU strategy policy содержит вероятность недопустимого действия")
        if not np.isclose(policy[legal].sum(), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("HU strategy policy должна быть нормирована")

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
        self._actor_ids[position] = actor
        self._policies[position] = policy
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
        self._cur_id += 1
        return status

    def sample(self, num_samples: int = -1):
        length = len(self)
        if num_samples < 0 or num_samples > length:
            num_samples = length
        if num_samples <= 0:
            return None
        indices = np.random.choice(length, int(num_samples), replace=False)
        return (
            self._states[indices],
            self._actor_ids[indices],
            self._policies[indices],
            self._masks[indices],
            self._iterations[indices],
        )

    def actor_ids(self) -> np.ndarray:
        return self._actor_ids[:len(self)].copy()

    def sample_conditioned(self, num_samples: int = -1):
        """Возвращает batch, готовый для общей strategy_net с actor one-hot."""
        samples = self.sample(num_samples)
        if samples is None:
            return None
        states, actor_ids, policies, masks, iterations = samples
        conditioned_states = np.concatenate(
            (states, np.eye(2, dtype=np.float32)[actor_ids]), axis=1
        )
        return conditioned_states, policies, masks, iterations

    def clear(self) -> None:
        self._cur_id = 0

    def __len__(self) -> int:
        return min(self._cur_id, self.capacity)


class HuCurrentPolicySelfPlayCoordinator(Generic[StateT]):
    """Координирует две advantage-ноги и одну actor-conditioned strategy-ногу."""

    def __init__(
        self,
        *,
        advantage_nets: list[nn.Module],
        advantage_target_nets: list[nn.Module],
        advantage_optimizers: list[Optimizer],
        advantage_buffers: list[AdvantageBuffer],
        strategy_net: nn.Module,
        strategy_optimizer: Optimizer,
        strategy_buffer: HuStrategyBuffer,
        adapter: HuTraversalAdapter[StateT],
        sampler: Callable[[np.ndarray, np.ndarray], int] | None = None,
        train_advantage: Callable[[int, nn.Module, nn.Module, Optimizer, AdvantageBuffer], None] | None = None,
        train_strategy: Callable[[nn.Module, Optimizer, HuStrategyBuffer], None] | None = None,
    ):
        for collection_name, collection in (
            ("advantage_nets", advantage_nets),
            ("advantage_target_nets", advantage_target_nets),
            ("advantage_optimizers", advantage_optimizers),
            ("advantage_buffers", advantage_buffers),
        ):
            if len(collection) != 2:
                raise ValueError(f"HU требует ровно две коллекции {collection_name}")
        self.advantage_nets = tuple(advantage_nets)
        self.advantage_target_nets = tuple(advantage_target_nets)
        self.advantage_optimizers = tuple(advantage_optimizers)
        self.advantage_buffers = tuple(advantage_buffers)
        self.strategy_net = strategy_net
        self.strategy_optimizer = strategy_optimizer
        self.strategy_buffer = strategy_buffer
        strategy_input_size = self._network_input_size(strategy_net)
        if strategy_input_size is not None and strategy_input_size != strategy_buffer.state_dim + 2:
            raise ValueError(
                "HU strategy_net должен принимать infoset и two-hot actor_id P0/P1"
            )
        self.adapter = adapter
        self.sampler = sampler or self._sample_action
        self.train_advantage = train_advantage
        self.train_strategy = train_strategy
        self.snapshots: tuple[nn.Module, nn.Module] | tuple[()] = ()

    @staticmethod
    def _network_input_size(network: nn.Module) -> int | None:
        """Извлекает размер входа без привязки к PokerNetwork."""
        if hasattr(network, "in_features"):
            return int(network.in_features)
        base = getattr(network, "base", None)
        if base is not None and len(base) and hasattr(base[0], "in_features"):
            return int(base[0].in_features)
        return None

    @staticmethod
    def validate_runtime_configuration(
        *,
        enabled: bool,
        num_players: int,
        num_trainable_players: int,
        opponent_checkpoint_dir=None,
        teacher_strategy_checkpoint=None,
    ) -> None:
        """Отклоняет неявное смешивание HU self-play с внешним opponent pool."""
        if not enabled:
            return
        if int(num_players) != 2 or int(num_trainable_players) != 2:
            raise ValueError(
                "hu_current_policy_self_play требует num_players == num_trainable_players == 2"
            )
        if opponent_checkpoint_dir is not None or teacher_strategy_checkpoint is not None:
            raise ValueError(
                "hu_current_policy_self_play несовместим с внешним opponent pool или teacher checkpoint"
            )

    @staticmethod
    def _regret_matching(advantages: np.ndarray, mask: np.ndarray) -> np.ndarray:
        legal = np.asarray(mask, dtype=np.float32) > 0.0
        positive = np.maximum(np.asarray(advantages, dtype=np.float32), 0.0) * legal
        total = float(positive.sum())
        if total > 0.0:
            return positive / total
        legal_count = int(legal.sum())
        return legal.astype(np.float32) / legal_count if legal_count else np.zeros(NUM_ACTIONS, dtype=np.float32)

    @staticmethod
    def _sample_action(legal_slots: np.ndarray, policy: np.ndarray) -> int:
        return int(np.random.choice(legal_slots, p=policy[legal_slots]))

    def begin_iteration(self) -> None:
        """Создаёт независимые read-only snapshots до запуска любой фазы."""
        snapshots = []
        for network in self.advantage_nets:
            snapshot = deepcopy(network)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            snapshots.append(snapshot)
        self.snapshots = tuple(snapshots)

    def _policy_from_snapshot(self, actor_id: int, encoded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if len(self.snapshots) != 2:
            raise RuntimeError("Перед HU traversal необходимо вызвать begin_iteration")
        snapshot = self.snapshots[int(actor_id)]
        device = next(snapshot.parameters()).device
        state_t = torch.as_tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            advantages = snapshot(state_t)[0].detach().cpu().numpy()
        if advantages.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(advantages)):
            raise ValueError("HU snapshot вернул некорректные advantages")
        policy = self._regret_matching(advantages, mask)
        legal = np.asarray(mask, dtype=np.float32) > 0.0
        if not legal.any() or not np.isclose(policy[legal].sum(), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("HU policy не нормирована по допустимым действиям")
        return policy.astype(np.float32, copy=False)

    def traverse(self, state: StateT, *, traversing_player: int, iteration: int) -> float:
        """Запускает один ES-обход; strategy target пишется до выборки оппонента."""
        traverser = int(traversing_player)
        if traverser not in (0, 1):
            raise ValueError("HU traversing_player должен быть P0 или P1")
        return self._traverse(state, traverser, int(iteration))

    def _traverse(self, state: StateT, traverser: int, iteration: int) -> float:
        if self.adapter.is_terminal(state):
            return float(self.adapter.terminal_value(state, traverser))
        actor_id = int(self.adapter.current_player(state))
        if actor_id not in (0, 1):
            raise ValueError("HU traversal встретил actor вне P0/P1")
        mask = np.asarray(self.adapter.legal_mask(state), dtype=np.float32)
        if mask.shape != (NUM_ACTIONS,) or not np.all(np.isin(mask, (0.0, 1.0))):
            raise ValueError("HU traversal получил некорректный legal mask")
        legal_slots = np.flatnonzero(mask).astype(int)
        if not len(legal_slots):
            raise ValueError("HU traversal встретил узел без допустимых действий")
        encoded = np.asarray(self.adapter.encode(state, actor_id), dtype=np.float32)
        policy = self._policy_from_snapshot(actor_id, encoded, mask)

        if actor_id == traverser:
            action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for slot in legal_slots:
                action_values[slot] = self._traverse(self.adapter.apply(state, int(slot)), traverser, iteration)
            expected_value = float(np.dot(policy, action_values))
            regrets = (action_values - expected_value) * mask
            self.advantage_buffers[actor_id].add(encoded, regrets, mask, iteration)
            return expected_value

        # Одна и та же нормированная policy становится target и входом sampler-а.
        self.strategy_buffer.add(actor_id, encoded, policy, mask, iteration)
        slot = int(self.sampler(legal_slots, policy))
        if slot not in legal_slots:
            raise ValueError("HU sampler выбрал недопустимое действие")
        return self._traverse(self.adapter.apply(state, slot), traverser, iteration)

    def run_iteration(
        self,
        *,
        iteration: int,
        traversals_per_player: int,
        new_initial_state: Callable[[int, int], StateT],
        after_phase: Callable[[], None] | None = None,
    ) -> None:
        """Проводит P0/P1 на одном profile и обучает advantage только после обеих фаз."""
        self.begin_iteration()
        for traverser in (0, 1):
            for traversal_index in range(int(traversals_per_player)):
                self.traverse(
                    new_initial_state(traverser, traversal_index),
                    traversing_player=traverser,
                    iteration=iteration,
                )
            if after_phase is not None:
                after_phase()
        for player_id in (0, 1):
            if self.train_advantage is not None:
                self.train_advantage(
                    player_id,
                    self.advantage_nets[player_id],
                    self.advantage_target_nets[player_id],
                    self.advantage_optimizers[player_id],
                    self.advantage_buffers[player_id],
                )
            self.advantage_target_nets[player_id].load_state_dict(
                self.advantage_nets[player_id].state_dict()
            )
        if self.train_strategy is not None:
            self.train_strategy(self.strategy_net, self.strategy_optimizer, self.strategy_buffer)


__all__ = [
    "HuCurrentPolicySelfPlayCoordinator",
    "HuStrategyBuffer",
    "HuTraversalAdapter",
]
