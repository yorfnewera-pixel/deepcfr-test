"""Изолированный контур HU self-play с неизменяемым профилем policies."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import AdvantageBuffer, DuelingAdvantageBuffer, _BufferAccounting
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext


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
    normalize_regrets: Callable[[StateT, np.ndarray, np.ndarray], np.ndarray] | None = None
    normalise_d2cfr_targets: Callable[
        [StateT, np.ndarray, float, np.ndarray], tuple[np.ndarray, np.float32, np.ndarray]
    ] | None = None


class HuStrategyBuffer(_BufferAccounting):
    """Общий reservoir-буфер strategy с явным actor_id для условной сети."""

    def __init__(self, capacity: int, state_dim: int):
        self._initialize_accounting(capacity)
        self.state_dim = int(state_dim)
        self._states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self._actor_ids = np.empty(self.capacity, dtype=np.int64)
        self._policies = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.empty((self.capacity, NUM_ACTIONS), dtype=np.float32)
        self._iterations = np.empty(self.capacity, dtype=np.float32)
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

        position, status = self._next_reservoir_slot()
        if position is None:
            self.skip_count += 1
            return status
        if status == "evicted":
            self.eviction_count += 1
        self._states[position] = state
        self._actor_ids[position] = actor
        self._policies[position] = policy
        self._masks[position] = mask
        self._iterations[position] = float(iteration)
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
        self._clear_samples()

    def __len__(self) -> int:
        return self._size


class HuCurrentPolicySelfPlayCoordinator(Generic[StateT]):
    """Координирует две advantage-ноги и одну actor-conditioned strategy-ногу."""

    def __init__(
        self,
        *,
        advantage_nets: list[nn.Module],
        advantage_target_nets: list[nn.Module] | None,
        advantage_optimizers: list[Optimizer],
        advantage_buffers: list[AdvantageBuffer | DuelingAdvantageBuffer],
        strategy_net: nn.Module,
        strategy_optimizer: Optimizer,
        strategy_buffer: HuStrategyBuffer,
        adapter: HuTraversalAdapter[StateT],
        sampler: Callable[[np.ndarray, np.ndarray], int] | None = None,
        train_advantage: Callable[[int, nn.Module, nn.Module | None, Optimizer, AdvantageBuffer | DuelingAdvantageBuffer], None] | None = None,
        train_strategy: Callable[[nn.Module, Optimizer, HuStrategyBuffer], None] | None = None,
        d2cfr_enabled: bool = False,
    ):
        self.d2cfr_enabled = bool(d2cfr_enabled)
        for collection_name, collection in (
            ("advantage_nets", advantage_nets),
            ("advantage_optimizers", advantage_optimizers),
            ("advantage_buffers", advantage_buffers),
        ):
            if len(collection) != 2:
                raise ValueError(f"HU требует ровно две коллекции {collection_name}")
            if id(collection[0]) == id(collection[1]):
                raise ValueError(f"HU P0/P1 требуют независимые {collection_name}")
        if self.d2cfr_enabled:
            if advantage_target_nets is not None:
                raise ValueError("HU D2CFR не использует advantage target-сети")
            if adapter.normalise_d2cfr_targets is None:
                raise ValueError("HU D2CFR требует normalise_d2cfr_targets adapter")
            if not all(isinstance(buffer, DuelingAdvantageBuffer) for buffer in advantage_buffers):
                raise ValueError("HU D2CFR требует DuelingAdvantageBuffer для P0/P1")
        else:
            if advantage_target_nets is None or len(advantage_target_nets) != 2:
                raise ValueError("HU требует ровно две коллекции advantage_target_nets")
        if advantage_target_nets is not None and id(advantage_target_nets[0]) == id(advantage_target_nets[1]):
            raise ValueError("HU P0/P1 требуют независимые advantage_target_nets")
        all_networks = [*advantage_nets, *(advantage_target_nets or [])]
        if len({id(network) for network in all_networks}) != len(all_networks):
            raise ValueError("HU advantage-сети и target-сети должны быть независимыми")
        advantage_parameter_ids = [self._parameter_ids(network) for network in advantage_nets]
        if advantage_parameter_ids[0] & advantage_parameter_ids[1]:
            raise ValueError("HU advantage-сети P0/P1 не должны разделять параметры")
        if advantage_target_nets is not None:
            target_parameter_ids = [self._parameter_ids(network) for network in advantage_target_nets]
            if target_parameter_ids[0] & target_parameter_ids[1]:
                raise ValueError("HU target-сети P0/P1 не должны разделять параметры")
            if (advantage_parameter_ids[0] | advantage_parameter_ids[1]) & (
                target_parameter_ids[0] | target_parameter_ids[1]
            ):
                raise ValueError("HU advantage-сети и target-сети не должны разделять параметры")
        for player_id, (network, optimizer) in enumerate(
            zip(advantage_nets, advantage_optimizers, strict=True)
        ):
            self._validate_optimizer_ownership(player_id, network, optimizer)
        self.advantage_nets = tuple(advantage_nets)
        self.advantage_target_nets = (
            None if advantage_target_nets is None else tuple(advantage_target_nets)
        )
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
        self._pending_advantage_samples: list[list[tuple]] | None = None
        self._pending_strategy_samples: list[tuple] | None = None
        self._active_actor_id: int | None = None
        self._active_depth = 0
        self._active_action_trace: tuple[str, ...] = ()

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
    def _parameter_ids(network: nn.Module) -> set[int]:
        """Возвращает идентификаторы параметров, включая разделяемые между Module."""
        return {id(parameter) for parameter in network.parameters()}

    @staticmethod
    def _validate_optimizer_ownership(
        player_id: int,
        network: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        """Не допускает, чтобы обновление P0/P1 меняло чужие параметры."""
        network_parameter_ids = HuCurrentPolicySelfPlayCoordinator._parameter_ids(network)
        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        if network_parameter_ids != optimizer_parameter_ids:
            raise ValueError(
                f"HU optimizer P{player_id} должен принадлежать только advantage-сети P{player_id}"
            )

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
        return self._traverse(state, traverser, int(iteration), depth=0, action_trace=())

    def _record_advantage(self, actor_id, state, regrets, mask, iteration) -> None:
        sample = (state.copy(), regrets.copy(), mask.copy(), int(iteration))
        if self._pending_advantage_samples is None:
            self.advantage_buffers[actor_id].add(*sample)
            return
        self._pending_advantage_samples[actor_id].append(sample)

    def _record_d2cfr_advantage(
        self, actor_id, state, action_values, state_value, regrets, mask, iteration
    ) -> None:
        sample = (
            state.copy(),
            action_values.copy(),
            np.float32(state_value),
            regrets.copy(),
            mask.copy(),
            int(iteration),
        )
        if self._pending_advantage_samples is None:
            self.advantage_buffers[actor_id].add(*sample)
            return
        self._pending_advantage_samples[actor_id].append(sample)

    @staticmethod
    def _validate_d2cfr_targets(action_values, state_value, regrets, mask) -> None:
        action_values = np.asarray(action_values, dtype=np.float32)
        regrets = np.asarray(regrets, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        state_value = np.asarray(state_value, dtype=np.float32)
        if (
            action_values.shape != (NUM_ACTIONS,)
            or regrets.shape != (NUM_ACTIONS,)
            or mask.shape != (NUM_ACTIONS,)
            or state_value.shape != ()
        ):
            raise ValueError("HU D2CFR normalizer вернул некорректную форму targets")
        if not all(np.all(np.isfinite(value)) for value in (action_values, regrets, state_value)):
            raise ValueError("HU D2CFR normalizer вернул NaN или Inf")
        legal = mask == 1.0
        if not np.allclose(
            regrets[legal], action_values[legal] - state_value, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("HU D2CFR regrets должны быть равны Q - V")

    def _record_strategy(self, actor_id, state, policy, mask, iteration) -> None:
        sample = (int(actor_id), state.copy(), policy.copy(), mask.copy(), int(iteration))
        if self._pending_strategy_samples is None:
            self.strategy_buffer.add(*sample)
            return
        self._pending_strategy_samples.append(sample)

    def _traverse(
        self,
        state: StateT,
        traverser: int,
        iteration: int,
        *,
        depth: int,
        action_trace: tuple[str, ...],
    ) -> float:
        self._active_actor_id = None
        self._active_depth = int(depth)
        self._active_action_trace = action_trace
        if self.adapter.is_terminal(state):
            return float(self.adapter.terminal_value(state, traverser))
        actor_id = int(self.adapter.current_player(state))
        self._active_actor_id = actor_id
        self._active_depth = int(depth)
        self._active_action_trace = action_trace
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
                next_trace = (*action_trace, f"P{actor_id}:slot {slot}")
                self._active_actor_id = actor_id
                self._active_depth = int(depth)
                self._active_action_trace = next_trace
                action_values[slot] = self._traverse(
                    self.adapter.apply(state, int(slot)),
                    traverser,
                    iteration,
                    depth=depth + 1,
                    action_trace=next_trace,
                )
            expected_value = float(np.dot(policy, action_values))
            if self.d2cfr_enabled:
                assert self.adapter.normalise_d2cfr_targets is not None
                action_targets, state_target, regret_targets = self.adapter.normalise_d2cfr_targets(
                    state, action_values, expected_value, mask
                )
                self._validate_d2cfr_targets(
                    action_targets, state_target, regret_targets, mask
                )
                self._record_d2cfr_advantage(
                    actor_id,
                    encoded,
                    np.asarray(action_targets, dtype=np.float32),
                    np.float32(state_target),
                    np.asarray(regret_targets, dtype=np.float32),
                    mask,
                    iteration,
                )
            else:
                regrets = (action_values - expected_value) * mask
                if self.adapter.normalize_regrets is not None:
                    regrets = np.asarray(
                        self.adapter.normalize_regrets(state, regrets, mask),
                        dtype=np.float32,
                    )
                if regrets.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(regrets)):
                    raise ValueError("HU normalizer вернул некорректные regrets")
                self._record_advantage(actor_id, encoded, regrets, mask, iteration)
            return expected_value

        # Одна и та же нормированная policy становится target и входом sampler-а.
        self._record_strategy(actor_id, encoded, policy, mask, iteration)
        slot = int(self.sampler(legal_slots, policy))
        if slot not in legal_slots:
            raise ValueError("HU sampler выбрал недопустимое действие")
        next_trace = (*action_trace, f"P{actor_id}:slot {slot}")
        self._active_actor_id = actor_id
        self._active_depth = int(depth)
        self._active_action_trace = next_trace
        return self._traverse(
            self.adapter.apply(state, slot),
            traverser,
            iteration,
            depth=depth + 1,
            action_trace=next_trace,
        )

    def run_iteration(
        self,
        *,
        iteration: int,
        traversals_per_player: int,
        new_initial_state: Callable[[int, int], StateT],
        after_phase: Callable[[], None] | None = None,
        traversal_context: Callable[[], object] | None = None,
        on_traversal_attempt: Callable[[], None] | None = None,
        on_traversal_success: Callable[[], None] | None = None,
        handle_traversal_failure: Callable[[TraversalFailure], bool] | None = None,
        train_strategy_due: bool = True,
    ) -> None:
        """Проводит атомарную HU-итерацию с обработкой ошибок отдельных обходов."""
        if self._pending_advantage_samples is not None:
            raise RuntimeError("HU transaction уже активна")
        self._pending_advantage_samples = [[], []]
        self._pending_strategy_samples = []
        try:
            self.begin_iteration()
            for traverser in (0, 1):
                with traversal_context() if traversal_context is not None else nullcontext():
                    for traversal_index in range(int(traversals_per_player)):
                        mark = (
                            len(self._pending_advantage_samples[0]),
                            len(self._pending_advantage_samples[1]),
                            len(self._pending_strategy_samples),
                        )
                        if on_traversal_attempt is not None:
                            on_traversal_attempt()
                        self._active_actor_id = None
                        self._active_depth = 0
                        self._active_action_trace = ()
                        try:
                            self.traverse(
                                new_initial_state(traverser, traversal_index),
                                traversing_player=traverser,
                                iteration=iteration,
                            )
                        except Exception as error:
                            del self._pending_advantage_samples[0][mark[0]:]
                            del self._pending_advantage_samples[1][mark[1]:]
                            del self._pending_strategy_samples[mark[2]:]
                            failure = self._as_traversal_failure(
                                error, iteration, traversal_index, traverser
                            )
                            if handle_traversal_failure is not None and handle_traversal_failure(failure):
                                continue
                            if failure is error:
                                raise
                            raise failure from error
                        if on_traversal_success is not None:
                            on_traversal_success()
                if after_phase is not None:
                    after_phase()
            self._commit_pending_samples()
            for player_id in (0, 1):
                if self.train_advantage is not None:
                    self.train_advantage(
                        player_id,
                        self.advantage_nets[player_id],
                        None if self.advantage_target_nets is None else self.advantage_target_nets[player_id],
                        self.advantage_optimizers[player_id],
                        self.advantage_buffers[player_id],
                    )
                if self.advantage_target_nets is not None:
                    self.advantage_target_nets[player_id].load_state_dict(
                        self.advantage_nets[player_id].state_dict()
                    )
            if train_strategy_due and self.train_strategy is not None:
                self.train_strategy(self.strategy_net, self.strategy_optimizer, self.strategy_buffer)
        finally:
            self._pending_advantage_samples = None
            self._pending_strategy_samples = None

    def _as_traversal_failure(
        self,
        error: Exception,
        iteration: int,
        traversal_index: int,
        traversing_player: int,
    ) -> TraversalFailure:
        if isinstance(error, TraversalFailure):
            return error
        return TraversalFailure(
            TraversalFailureContext(
                iteration=int(iteration),
                traversal_index=int(traversal_index),
                traversing_player=int(traversing_player),
                acting_player=self._active_actor_id,
                depth=self._active_depth,
                reason=f"HU traversal завершился ошибкой: {error}",
                action_trace=self._active_action_trace,
                details={"exception_type": type(error).__name__},
            ),
            error,
        )

    def _commit_pending_samples(self) -> None:
        """Фиксирует samples только после успешного завершения обеих traversal-фаз."""
        assert self._pending_advantage_samples is not None
        assert self._pending_strategy_samples is not None
        for player_id, samples in enumerate(self._pending_advantage_samples):
            for sample in samples:
                self.advantage_buffers[player_id].add(*sample)
        for sample in self._pending_strategy_samples:
            self.strategy_buffer.add(*sample)


__all__ = [
    "HuCurrentPolicySelfPlayCoordinator",
    "HuStrategyBuffer",
    "HuTraversalAdapter",
]
