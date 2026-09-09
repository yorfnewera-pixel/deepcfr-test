"""Циклы обучения Deep CFR для фиксированных шести действий."""
from __future__ import annotations

import os
import random
import re
import tempfile
import time
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pokers as pkrs
import torch

from src.agents.random_agent import RandomAgent
from src.core.action_space import ACTION_SPACE_VERSION
from src.core.deep_cfr import DeepCFRAgent
from src.core.traversal_errors import TraversalFailure
from src.utils.config import (
    cfg_get,
    cfg_training_error_mode,
    cfg_training_max_failed_traversals_per_iteration,
)


_HEAVY_CHECKPOINT_PREFIX = "multi_checkpoint_iter_"
_LIGHT_CHECKPOINT_PREFIX = "light_checkpoint_iter_"
_OPPONENT_RECENT_CHECKPOINTS = 11
_OPPONENT_HISTORICAL_CHECKPOINTS = 2


def _checkpoint_iteration(path: Path, prefix: str = _HEAVY_CHECKPOINT_PREFIX) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _current_strategy_opponent_count(iteration: int, checkpoint_every: int) -> int:
    """Определяет число временных слотов общей strategy до заполнения пула checkpoint."""
    completed_slots = max(0, (int(iteration) - 1) // max(1, int(checkpoint_every)))
    return max(0, 5 - min(completed_slots, 5))


def _transition_checkpoint_iterations(iteration: int, checkpoint_every: int) -> list[int]:
    """Возвращает последние собственные checkpoint в период плавного перехода."""
    checkpoint_every = max(1, int(checkpoint_every))
    completed_slots = (int(iteration) - 1) // checkpoint_every
    own_count = min(completed_slots, 4)
    if completed_slots >= 5:
        return []
    return [slot * checkpoint_every for slot in range(completed_slots, 0, -1)][:own_count]


def _assign_checkpoint_opponents(
    traversing_player: int,
    num_players: int,
    strategy_count: int,
    checkpoint_paths: list[Path],
) -> tuple[tuple[int, ...], dict[int, Path]]:
    """Перемешивает общую strategy и исторические checkpoint по местам оппонентов."""
    opponent_ids = [player_id for player_id in range(num_players) if player_id != int(traversing_player)]
    if strategy_count < 0 or strategy_count + len(checkpoint_paths) != len(opponent_ids):
        raise ValueError("Состав смешанного opponent pool не заполняет все позиции")
    entries: list[Path | None] = [None] * strategy_count + list(checkpoint_paths)
    random.shuffle(entries)
    strategy_positions = tuple(
        player_id for player_id, entry in zip(opponent_ids, entries, strict=True) if entry is None
    )
    checkpoint_by_player = {
        player_id: entry
        for player_id, entry in zip(opponent_ids, entries, strict=True)
        if entry is not None
    }
    return strategy_positions, checkpoint_by_player


def _heavy_checkpoints(directory: str | Path) -> dict[int, Path]:
    """Возвращает полные checkpoint для продолжения обучения."""
    return _checkpoint_paths(directory, _HEAVY_CHECKPOINT_PREFIX)


def _latest_strategy_checkpoint_path(iteration: int, checkpoint_dir: str | Path) -> Path:
    """Возвращает последний full checkpoint, предшествующий текущей итерации."""
    checkpoints = {
        number: path
        for number, path in _heavy_checkpoints(checkpoint_dir).items()
        if number < int(iteration)
    }
    if not checkpoints:
        raise ValueError(
            "Для игры против общей strategy нужен предыдущий full checkpoint"
        )
    return checkpoints[max(checkpoints)]


def _checkpoint_paths(directory: str | Path, prefix: str) -> dict[int, Path]:
    """Сканирует checkpoint заданного формата без смешивания артефактов."""
    result: dict[int, Path] = {}
    for path in Path(directory).glob(f"{prefix}*.pt"):
        iteration = _checkpoint_iteration(path, prefix)
        if iteration is not None:
            result[iteration] = path
    return result


def _checkpoint_save_due(iteration: int, every: int) -> bool:
    """Возвращает, следует ли сохранить checkpoint после итерации."""
    return int(iteration) > 0 and int(iteration) % max(1, int(every)) == 0


def _prune_full_checkpoints(directory: str | Path, historical_every: int) -> None:
    """Оставляет 11 свежих и два последних исторических full checkpoint."""
    checkpoints = _heavy_checkpoints(directory)
    historical_every = max(1, int(historical_every))
    historical = sorted(
        iteration for iteration in checkpoints if iteration % historical_every == 0
    )[-_OPPONENT_HISTORICAL_CHECKPOINTS:]
    historical_set = set(historical)
    ordinary = sorted(
        (iteration for iteration in checkpoints if iteration not in historical_set),
        reverse=True,
    )[:_OPPONENT_RECENT_CHECKPOINTS]
    retained = historical_set | set(ordinary)
    for iteration, path in checkpoints.items():
        if iteration not in retained:
            path.unlink(missing_ok=True)


def _prune_light_checkpoints(directory: str | Path) -> None:
    """Сохраняет light-checkpoint только если есть парный тяжёлый checkpoint."""
    retained_heavy = set(_heavy_checkpoints(directory))
    for path in Path(directory).glob(f"{_LIGHT_CHECKPOINT_PREFIX}*.pt"):
        iteration = _checkpoint_iteration(path, _LIGHT_CHECKPOINT_PREFIX)
        if iteration is None or iteration not in retained_heavy:
            path.unlink(missing_ok=True)


def _load_full_checkpoint_strategy_state(path: Path) -> dict[str, torch.Tensor]:
    """Извлекает strategy-веса из full checkpoint и проверяет игровой контракт."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Некорректный full checkpoint: {path}")
    version = checkpoint.get("action_space_version")
    if version != ACTION_SPACE_VERSION:
        raise ValueError(
            f"В full checkpoint {path} action-space '{version}', "
            f"ожидался '{ACTION_SPACE_VERSION}'"
        )
    state_dict = checkpoint.get("strategy_net")
    if not isinstance(state_dict, dict):
        raise ValueError(f"В full checkpoint {path} отсутствует strategy_net")
    return state_dict


def _resolve_checkpoint_path(
    requested_iteration: int,
    checkpoints: dict[int, Path],
) -> Path:
    """Берёт точный checkpoint или ближайший доступный более старый."""
    if not checkpoints:
        raise ValueError("Нет full checkpoint для загрузки весов оппонентов")
    if requested_iteration in checkpoints:
        return checkpoints[requested_iteration]
    older = [iteration for iteration in checkpoints if iteration <= requested_iteration]
    if older:
        return checkpoints[max(older)]
    return checkpoints[min(checkpoints)]


def _sample_checkpoint_iterations(candidates: list[int], count: int) -> list[int]:
    """Выбирает без повторов, а при нехватке доступных весов дублирует старые."""
    if not candidates:
        return []
    if len(candidates) >= count:
        return random.sample(candidates, count)
    result = list(candidates)
    result.extend(random.choices(candidates, k=count - len(result)))
    return result


def _opponent_checkpoint_paths(
    iteration: int,
    checkpoint_dir: str | Path,
    num_opponents: int,
    checkpoint_every: int,
    historical_every: int,
) -> list[Path]:
    """Строит warm-up и устойчивую смесь прошлых full checkpoint."""
    if not 1 <= int(num_opponents) <= 5:
        raise ValueError("Число checkpoint-оппонентов должно быть от 1 до 5")
    checkpoint_every = max(1, int(checkpoint_every))
    checkpoints = {
        number: path for number, path in _heavy_checkpoints(checkpoint_dir).items()
        if number < int(iteration)
    }
    if not checkpoints:
        return []

    latest_slot_index = (int(iteration) - 1) // checkpoint_every
    latest_slot = latest_slot_index * checkpoint_every
    if latest_slot_index <= 6:
        requested = [
            slot * checkpoint_every
            for slot in range(latest_slot_index, max(1, latest_slot_index - 4) - 1, -1)
        ]
        requested = requested[:num_opponents]
        requested.extend([checkpoint_every] * (num_opponents - len(requested)))
    else:
        recent_candidates = [
            slot * checkpoint_every
            for slot in range(max(1, latest_slot_index - 10), latest_slot_index)
        ]
        historical = sorted(
            number for number in checkpoints
            if number % max(1, int(historical_every)) == 0
        )
        active_historical = historical[-2] if len(historical) >= 2 else (
            historical[-1] if historical else None
        )
        requested = [latest_slot]
        needs_separate_historical_slot = (
            active_historical is not None
            and active_historical != latest_slot
            and num_opponents > 1
        )
        requested += _sample_checkpoint_iterations(
            recent_candidates,
            num_opponents - 1 - int(needs_separate_historical_slot),
        )
        if needs_separate_historical_slot:
            requested.append(active_historical)

    return [_resolve_checkpoint_path(number, checkpoints) for number in requested]


class OpponentPoolSchedule:
    """Фиксирует состав opponent pool до следующего checkpoint-слота."""

    def __init__(self, checkpoint_dir: str | Path, checkpoint_every: int, historical_every: int):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every = max(1, int(checkpoint_every))
        self._historical_every = max(1, int(historical_every))
        self._slot: int | None = None
        self._paths: tuple[Path, ...] = ()

    def paths_for_iteration(self, iteration: int, num_opponents: int) -> list[Path]:
        slot = (int(iteration) - 1) // self._checkpoint_every
        if slot != self._slot:
            self._paths = tuple(_opponent_checkpoint_paths(
                iteration=iteration,
                checkpoint_dir=self._checkpoint_dir,
                num_opponents=num_opponents,
                checkpoint_every=self._checkpoint_every,
                historical_every=self._historical_every,
            ))
            self._slot = slot
        return list(self._paths)


def _configure_opponent_pool(
    agent: DeepCFRAgent,
    iteration: int,
    checkpoint_dir: str | Path,
    traversing_player: int,
    state_cache: dict[Path, dict[str, torch.Tensor]] | None = None,
    selected_paths: list[Path] | None = None,
) -> list[Path]:
    paths = list(selected_paths) if selected_paths is not None else _opponent_checkpoint_paths(
        iteration=iteration,
        checkpoint_dir=checkpoint_dir,
        num_opponents=agent.num_players - 1,
        checkpoint_every=int(cfg_get("checkpoint_save_every", 1000)),
        historical_every=int(cfg_get("checkpoint_keep_every", 50000)),
    )
    if not paths:
        agent.clear_opponent_advantage_states()
        return []
    random.shuffle(paths)
    if state_cache is None:
        state_cache = {}
    for path in dict.fromkeys(paths):
        resolved_path = path.resolve()
        if resolved_path not in state_cache:
            state_cache[resolved_path] = _load_full_checkpoint_strategy_state(resolved_path)
    agent.set_opponent_strategy_states(
        [state_cache[path.resolve()] for path in paths],
        traversing_player=traversing_player,
    )
    return paths


def _configure_strategy_opponent_pool(
    agent: DeepCFRAgent,
    checkpoint_by_player: dict[int, Path],
    strategy_positions: tuple[int, ...],
    current_strategy_path: Path | None,
    traversing_player: int,
    state_cache: dict[Path, dict[str, torch.Tensor]],
) -> list[Path]:
    """Назначает актуальную strategy и исторические strategy checkpoint по позициям."""
    states_by_player: dict[int, dict[str, torch.Tensor]] = {}
    if strategy_positions:
        if current_strategy_path is None:
            current_strategy_state = agent.strategy_net.state_dict()
        else:
            resolved_current_strategy_path = current_strategy_path.resolve()
            if resolved_current_strategy_path not in state_cache:
                state_cache[resolved_current_strategy_path] = _load_full_checkpoint_strategy_state(
                    resolved_current_strategy_path
                )
            current_strategy_state = state_cache[resolved_current_strategy_path]
        states_by_player = {
            player_id: current_strategy_state for player_id in strategy_positions
        }
    for player_id, path in checkpoint_by_player.items():
        resolved_path = path.resolve()
        if resolved_path not in state_cache:
            state_cache[resolved_path] = _load_full_checkpoint_strategy_state(resolved_path)
        states_by_player[player_id] = state_cache[resolved_path]
    agent.set_opponent_strategy_states_by_player(states_by_player, traversing_player)
    return list(checkpoint_by_player.values())


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Записывает checkpoint атомарно, чтобы не оставить битый файл при остановке."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_iteration_checkpoint(
    agent: DeepCFRAgent,
    save_dir: str | Path,
    iteration: int,
    prefix: str = "multi_checkpoint_iter_",
    seed: int | None = None,
) -> Path:
    path = Path(save_dir) / f"{prefix}{int(iteration)}.pt"
    _atomic_torch_save(agent._build_checkpoint(seed=seed), path)
    if prefix == _HEAVY_CHECKPOINT_PREFIX:
        _prune_full_checkpoints(save_dir, int(cfg_get("checkpoint_keep_every", 50000)))
    return path


def _save_iteration_light_checkpoint(
    agent: DeepCFRAgent,
    save_dir: str | Path,
    iteration: int,
    prefix: str = _LIGHT_CHECKPOINT_PREFIX,
    seed: int | None = None,
) -> Path:
    """Сохраняет только усреднённую стратегию для инференса."""
    path = Path(save_dir) / f"{prefix}{int(iteration)}.pt"
    _atomic_torch_save(agent.build_light_checkpoint(seed=seed), path)
    _prune_light_checkpoints(save_dir)
    return path


def _get_process_rss_mb() -> float | None:
    """Возвращает RSS процесса, если psutil доступен."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _apply_process_priority(priority: str) -> None:
    """Применяет приоритет Windows и явно сообщает результат в консоль."""
    if priority == "normal":
        print("Приоритет процесса: normal (не запрашивалось повышение).")
        return
    if priority != "above_normal":
        raise ValueError("process_priority должен быть normal или above_normal.")
    if os.name != "nt":
        print("Приоритет процесса: НЕ ПРИМЕНЁН (above_normal поддержан только в Windows).")
        return

    try:
        import psutil
    except ImportError:
        print("Приоритет процесса: НЕ ПРИМЕНЁН (не установлен psutil).")
        return

    try:
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        print("Приоритет процесса: ПРИМЕНЁН (above_normal).")
    except psutil.Error as error:
        print(f"Приоритет процесса: НЕ ПРИМЕНЁН (Windows отклонил запрос: {error}).")


def _format_fanout_histogram(histogram: dict[int, int]) -> str:
    return ", ".join(f"{fanout}:{count}" for fanout, count in sorted(histogram.items())) or "—"


def _print_depth_histogram(depth_histogram: dict[int, int]) -> None:
    print("  Глубина: " + _format_fanout_histogram(depth_histogram))


def _print_opponent_checkpoints(
    paths: list[Path],
    random_opponents: int = 0,
) -> None:
    """Выводит фактически загруженные advantage-checkpoint оппонентов."""
    if random_opponents > 0:
        print(f"  Играет против: {random_opponents} RandomAgent")
        return
    if not paths:
        print("  Играет против: холодный старт (advantage checkpoint ещё не созданы)")
        return
    print("  Играет против:")
    for path in paths:
        print(f"    {path.name}")


def _print_current_strategy_opponents(count: int) -> None:
    print(f"  Играет против актуальной strategy: {count}")


def _format_iteration_summary(
    iteration_elapsed: float,
    traversal_elapsed: float,
    advantage_loss: float,
    strategy_loss: float,
    opponent_setup_elapsed: float = 0.0,
) -> str:
    return (
        f"Time/Iteration={iteration_elapsed:.1f}s | "
        f"Time/Traversal={traversal_elapsed:.1f}s | "
        f"Time/OpponentPoolSetup={opponent_setup_elapsed:.1f}s | "
        f"Loss/Advantage={advantage_loss:.6f} | "
        f"Loss/Strategy={strategy_loss:.6f}"
    )


@contextmanager
def _traversal_thread_limit(enabled: bool):
    """Ограничивает PyTorch одним потоком только на мелких forward обхода."""
    previous_threads = torch.get_num_threads()
    if enabled:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if enabled:
            torch.set_num_threads(previous_threads)


@contextmanager
def _training_thread_limit(num_threads: int | None):
    """Задаёт число intra-op потоков только для loss/backprop фазы."""
    if num_threads is None:
        yield
        return

    requested_threads = int(num_threads)
    if requested_threads <= 0:
        raise ValueError("training_torch_threads должен быть положительным.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(requested_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _log_multi_cfr_diagnostics(agent, writer, iteration, traversing_player, traversals_per_iteration):
    """Пишет только метрики обхода, независимые от представления действий."""
    del traversing_player, traversals_per_iteration
    stats = agent.get_traversal_stats()
    print(
        f"  Обходы: узлы={stats['nodes']}, макс_глубина={stats['max_depth']}, "
        f"попытки={stats.get('attempted', 0)}, успешные={stats.get('successful', 0)}, "
        f"ошибки={stats.get('failed', 0)}, отменённые_образцы={stats.get('cancelled_samples', 0)}, "
        f"лимит_глубины={stats.get('depth_limit_hits', 0)}"
    )
    if writer is None:
        return stats
    for key in (
        "nodes",
        "terminal_nodes",
        "max_depth",
        "recorded_nodes",
        "buffer_skip_ratio",
        "attempted",
        "successful",
        "failed",
        "cancelled_samples",
        "depth_limit_hits",
    ):
        writer.add_scalar(f"Traversal/{key}", stats[key], iteration)
    writer.add_scalar("Memory/Advantage", len(agent.advantage_buffer), iteration)
    writer.add_scalar("Memory/Strategy", len(agent.strategy_buffer), iteration)
    return stats


def _create_writer(log_dir: str | Path | None):
    """Создаёт TensorBoard writer только когда указан каталог логов."""
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard недоступен: метрики будут выведены только в консоль.")
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _new_hand(num_players: int, seed: int) -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=num_players,
        button=seed % num_players,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=seed,
    )


def evaluate_against_random(
    agent: DeepCFRAgent,
    num_games: int = 100,
    num_players: int = 6,
    return_stats: bool = False,
) -> float | dict[str, float]:
    """Оценивает agent против случайных оппонентов без смены смысла raise-слотов."""
    rewards: list[float] = []
    decisions = raises = 0
    opponents = [RandomAgent(player) for player in range(num_players)]
    for game in range(int(num_games)):
        state = _new_hand(num_players, game)
        while not state.final_state:
            player = int(state.current_player)
            action = agent.choose_action(state, player_id=player) if player == agent.player_id else opponents[player].choose_action(state)
            if player == agent.player_id:
                decisions += 1
                raises += int(action.action == pkrs.ActionEnum.Raise)
            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                break
        rewards.append(float(state.players_state[agent.player_id].reward))
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    if not return_stats:
        return mean_reward
    return {
        "mean_reward": mean_reward,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "raise_frequency": raises / decisions if decisions else 0.0,
        "games": float(len(rewards)),
    }


class _PerspectiveAgentWrapper:
    """Совместимый адаптер для оценочного кода с разными позициями игрока."""

    def __init__(self, agent: DeepCFRAgent, player_id: int):
        self.agent = agent
        self.player_id = int(player_id)

    def choose_action(self, state):
        return self.agent.choose_action(state, player_id=self.player_id)


def evaluate_against_agent(agent: DeepCFRAgent, opponent_agent, num_games: int = 100) -> float:
    """Возвращает среднюю награду первого игрока в heads-up оценке."""
    rewards: list[float] = []
    for game in range(int(num_games)):
        state = _new_hand(2, game)
        while not state.final_state:
            current = int(state.current_player)
            policy = agent if current == agent.player_id else opponent_agent
            action = policy.choose_action(state)
            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                break
        rewards.append(float(state.players_state[agent.player_id].reward))
    return float(np.mean(rewards)) if rewards else 0.0


def evaluate_against_checkpoint_agents(agent: DeepCFRAgent, opponent_agents, num_games: int = 100) -> float:
    if not opponent_agents:
        return evaluate_against_random(agent, num_games=num_games, num_players=agent.num_players)
    return evaluate_against_agent(agent, opponent_agents[0], num_games)


def train_self_play_multi(
    num_iterations: int = 1_000,
    traversals_per_iteration: int = 100,
    save_dir: str | Path = "models",
    evaluate_every: int = 10,
    evaluation_games: int = 500,
    num_players: int = 6,
    device: str = "cpu",
    seed: int | None = None,
    initial_checkpoint: str | None = None,
    log_dir: str | Path | None = None,
    opponent_checkpoint_dir: str | Path | None = None,
    trainable_players: int | None = None,
    teacher_strategy_checkpoint: str | Path | None = None,
    **_unused_options,
) -> DeepCFRAgent:
    """Обучает один общий action-only агент external-sampling Deep CFR."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    agent = DeepCFRAgent(player_id=0, num_players=num_players, device=device)
    if trainable_players is not None:
        agent.num_trainable_players = max(1, min(int(trainable_players), agent.num_players))
    if teacher_strategy_checkpoint is None:
        teacher_strategy_checkpoint = cfg_get("teacher_strategy_checkpoint", None)
    if teacher_strategy_checkpoint:
        agent.load_teacher_strategy_checkpoint(teacher_strategy_checkpoint)
    opponent_state_cache: dict[Path, dict[str, torch.Tensor]] = {}
    if initial_checkpoint:
        checkpoint = agent.load_model(initial_checkpoint)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("strategy_net"), dict):
            opponent_state_cache[Path(initial_checkpoint).resolve()] = checkpoint["strategy_net"]
        if opponent_checkpoint_dir is None and not _heavy_checkpoints(save_dir):
            save_dir = Path(initial_checkpoint).parent
    opponent_checkpoint_dir = Path(opponent_checkpoint_dir) if opponent_checkpoint_dir is not None else Path(save_dir)

    checkpoint_every = int(cfg_get("checkpoint_save_every", 1000))
    opponent_pool_schedule = OpponentPoolSchedule(
        opponent_checkpoint_dir,
        checkpoint_every=checkpoint_every,
        historical_every=int(cfg_get("checkpoint_keep_every", 50000)),
    )

    start_iteration = agent.iteration_count + 1
    writer = _create_writer(log_dir)
    try:
        print(
            f"Старт обучения: итераций={num_iterations}, "
            f"обходов/итерацию={traversals_per_iteration}, device={device}"
        )
        for iteration in range(start_iteration, start_iteration + int(num_iterations)):
            iteration_started = time.perf_counter()
            agent.iteration_count = iteration
            print(f"\nИтерация {iteration}:")
            opponent_setup_started = time.perf_counter()
            trainable_player_count = max(1, int(getattr(agent, "num_trainable_players", 1)))
            traversing_players = range(trainable_player_count)
            opponent_setup_elapsed = time.perf_counter() - opponent_setup_started
            traversal_started = time.perf_counter()
            with _traversal_thread_limit(bool(cfg_get("traversal_single_thread", True))):
                failed_traversals = 0
                agent.reset_traversal_stats()
                for traversing_player in traversing_players:
                    max_opponents = max(0, agent.num_players - 1)
                    strategy_count = min(
                        _current_strategy_opponent_count(iteration, checkpoint_every),
                        max_opponents,
                    )
                    transition_iterations = _transition_checkpoint_iterations(iteration, checkpoint_every)
                    if transition_iterations:
                        available_checkpoints = _heavy_checkpoints(opponent_checkpoint_dir)
                        selected_paths = [
                            _resolve_checkpoint_path(number, available_checkpoints)
                            for number in transition_iterations[:max(0, max_opponents - strategy_count)]
                        ]
                    else:
                        checkpoint_opponent_count = max_opponents - strategy_count
                        selected_paths = (
                            opponent_pool_schedule.paths_for_iteration(
                                iteration,
                                num_opponents=checkpoint_opponent_count,
                            )
                            if checkpoint_opponent_count > 0
                            else []
                        )
                    strategy_positions, checkpoint_by_player = _assign_checkpoint_opponents(
                        traversing_player=traversing_player,
                        num_players=agent.num_players,
                        strategy_count=strategy_count,
                        checkpoint_paths=selected_paths,
                    )
                    opponent_checkpoints = _configure_strategy_opponent_pool(
                        agent,
                        checkpoint_by_player=checkpoint_by_player,
                        strategy_positions=strategy_positions,
                        current_strategy_path=None,
                        traversing_player=traversing_player,
                        state_cache=opponent_state_cache,
                    )
                    if strategy_count:
                        _print_current_strategy_opponents(strategy_count)
                    if opponent_checkpoints:
                        _print_opponent_checkpoints(opponent_checkpoints, traversing_player)
                    agent.prepare_iteration(iteration, traversing_player=traversing_player)
                    if trainable_player_count == 1:
                        print(f"  Запускаю {traversals_per_iteration} обходов...")
                    else:
                        print(
                            f"  Игрок {traversing_player}: запускаю "
                            f"{traversals_per_iteration} обходов..."
                        )
                    for traversal in range(int(traversals_per_iteration)):
                        state_seed = (
                            (seed or 0)
                            + iteration * max(1, int(traversals_per_iteration)) * trainable_player_count
                            + traversing_player * max(1, int(traversals_per_iteration))
                            + traversal
                        )
                        try:
                            agent.record_traversal_attempt()
                            agent.cfr_traverse_multi(
                                _new_hand(num_players, state_seed),
                                iteration,
                                traversing_player=traversing_player,
                                random_agent=None,
                                traversal_index=traversal,
                            )
                            agent.record_traversal_success()
                        except TraversalFailure as error:
                            agent.record_traversal_failure(error)
                            if cfg_training_error_mode() != "skip_traversal":
                                raise
                            failed_traversals += 1
                            if failed_traversals >= cfg_training_max_failed_traversals_per_iteration():
                                raise RuntimeError(
                                    "Превышен training_max_failed_traversals_per_iteration"
                                ) from error
                    _log_multi_cfr_diagnostics(agent, writer, iteration, traversing_player, traversals_per_iteration)
            traversal_elapsed = time.perf_counter() - traversal_started

            training_threads = cfg_get("training_torch_threads")
            with _training_thread_limit(training_threads):
                advantage_loss = agent.train_advantage_network_multi()
                strategy_loss = agent.train_strategy_network()
            if writer is not None:
                writer.add_scalar("Loss/Advantage", advantage_loss, iteration)
                writer.add_scalar("Loss/Strategy", strategy_loss, iteration)
                writer.add_scalar("Time/Traversal", traversal_elapsed, iteration)
                writer.add_scalar("Time/OpponentPoolSetup", opponent_setup_elapsed, iteration)
                writer.add_scalar("Train/AdvantageLearningRate", agent.optimizer.param_groups[0]["lr"], iteration)
            if evaluate_every and iteration % int(evaluate_every) == 0:
                evaluation = evaluate_against_random(
                    agent, evaluation_games, num_players, return_stats=True
                )
                score = evaluation["mean_reward"]
                raise_frequency = evaluation["raise_frequency"]
                print(
                    f"  Оценка против random: reward={score:.4f}, "
                    f"raise_freq={raise_frequency:.3f}, игр={int(evaluation['games'])}"
                )
                if writer is not None:
                    writer.add_scalar("Evaluation/RandomMeanReward", score, iteration)
                    writer.add_scalar("Evaluation/RandomRaiseFrequency", raise_frequency, iteration)
            if _checkpoint_save_due(iteration, int(cfg_get("checkpoint_save_every", 1000))):
                last_checkpoint = _save_iteration_checkpoint(
                    agent, save_dir, iteration, seed=seed
                )
                light_checkpoint = _save_iteration_light_checkpoint(
                    agent, save_dir, iteration, seed=seed
                )
                print(f"  Checkpoint: {last_checkpoint}")
                print(f"  Light checkpoint: {light_checkpoint}")
            iteration_elapsed = time.perf_counter() - iteration_started
            if writer is not None:
                writer.add_scalar("Time/Iteration", iteration_elapsed, iteration)
            print(_format_iteration_summary(
                iteration_elapsed, traversal_elapsed, advantage_loss, strategy_loss, opponent_setup_elapsed
            ))
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    return agent


def train_deep_cfr(*args, **kwargs) -> DeepCFRAgent:
    return train_self_play_multi(*args, **kwargs)


def continue_training(checkpoint_path: str, additional_iterations: int = 1_000, **kwargs) -> DeepCFRAgent:
    return train_self_play_multi(
        num_iterations=additional_iterations,
        initial_checkpoint=checkpoint_path,
        **kwargs,
    )


def train_against_checkpoint(checkpoint_path: str, additional_iterations: int = 1_000, **kwargs) -> DeepCFRAgent:
    """Совместимое имя: старый checkpoint будет отвергнут строгой проверкой формата."""
    return continue_training(checkpoint_path, additional_iterations, **kwargs)


def train_with_mixed_checkpoints(*_args, **_kwargs):
    raise NotImplementedError("Смешивание старых sizing/Q checkpoint удалено; используйте six_fixed_v2.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-only Deep CFR")
    parser.add_argument("--self-play-multi", action="store_true", help="Запустить self-play обучение")
    parser.add_argument("--iterations", type=int, default=1_000, help="Число итераций CFR")
    parser.add_argument("--traversals", type=int, default=100, help="Обходов на итерацию")
    parser.add_argument("--save-dir", default="models", help="Каталог checkpoint-файлов")
    parser.add_argument("--log-dir", default=None, help="Каталог TensorBoard-логов")
    parser.add_argument(
        "--opponent-checkpoint-dir",
        default=None,
        help="Каталог full checkpoint-файлов для opponent pool",
    )
    parser.add_argument(
        "--trainable-players",
        type=int,
        default=int(cfg_get("num_trainable_players", 1)),
        help="Сколько мест собирать как traversing players за итерацию",
    )
    parser.add_argument(
        "--teacher-strategy-checkpoint",
        default=cfg_get("teacher_strategy_checkpoint", None),
        help="Checkpoint strategy-сети teacher-а для policy distillation",
    )
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=500)
    parser.add_argument("--num-players", type=int, default=int(cfg_get("num_players", 6)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.iterations <= 0 or args.traversals <= 0:
        raise ValueError("--iterations и --traversals должны быть положительными")
    if args.num_players < 2:
        raise ValueError("--num-players должен быть не меньше 2")
    _apply_process_priority(str(cfg_get("process_priority", "normal")))
    if not args.self_play_multi:
        print("Режим не указан; запускаю self-play. Для явности используйте --self-play-multi.")
    train_self_play_multi(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        evaluate_every=args.evaluate_every,
        evaluation_games=args.evaluation_games,
        num_players=args.num_players,
        device=args.device,
        seed=args.seed,
        initial_checkpoint=args.initial_checkpoint,
        opponent_checkpoint_dir=args.opponent_checkpoint_dir,
        trainable_players=args.trainable_players,
        teacher_strategy_checkpoint=args.teacher_strategy_checkpoint,
    )


if __name__ == "__main__":
    main()
