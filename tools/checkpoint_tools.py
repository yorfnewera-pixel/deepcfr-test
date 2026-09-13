"""Проверка checkpoint формата six_fixed_v2."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pokers as pkrs
import torch

from policy_runtime.adapters.pokers import action_to_pokers, wrap_state
from policy_runtime.core import PolicyRuntimeAgent
from src.agents.random_agent import RandomAgent
from src.core.action_space import ACTION_LABELS, ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core.deep_cfr import (
    CHECKPOINT_FORMAT_VERSION,
    DeepCFRAgent,
    full_checkpoint_network_spec,
)
from src.core.model import encoder_input_size
from src.evaluation import FrozenBlueprintPolicy, evaluate_paired


_STREET_LABELS = ("preflop", "flop", "turn", "river")
_OPPONENT_RESPONSE_LABELS = ("fold", "check", "call", "raise")


def _invalid_tensors(value, path="") -> list[str]:
    if torch.is_tensor(value):
        return [path] if torch.is_floating_point(value) and not torch.isfinite(value).all() else []
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _invalid_tensors(child, f"{path}.{key}".strip("."))]
    return []


def run_weight_sanity(checkpoint_path: str | Path) -> dict:
    """Проверяет контракт checkpoint и отсутствие NaN/Inf в обязательных сетях."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    errors: list[str] = []
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        errors.append(f"требуется checkpoint_format_version={CHECKPOINT_FORMAT_VERSION}")
    if payload.get("action_space_version") != ACTION_SPACE_VERSION:
        errors.append(f"требуется action_space_version={ACTION_SPACE_VERSION}")
    if int(payload.get("num_actions", -1)) != NUM_ACTIONS:
        errors.append(f"требуется num_actions={NUM_ACTIONS}")
    encoding_version = payload.get("encoding_version")
    checkpoint_config = payload.get("config", {})
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint содержит некорректный config")
        checkpoint_config = {}
    use_multi_agent = bool(checkpoint_config.get("use_multi_agent_advantage", False))
    try:
        expected_input_size = encoder_input_size(
            int(payload.get("num_players", -1)),
            encoding_version,
            use_multi_agent,
        )
    except (TypeError, ValueError):
        errors.append("checkpoint содержит неизвестную версию encoder")
    else:
        if int(payload.get("encoder_input_size", -1)) != expected_input_size:
            errors.append(f"требуется encoder_input_size={expected_input_size}")
    network_keys = ("strategy_net",)
    if payload.get("checkpoint_kind") not in {"strategy_only", "hu_strategy_only"}:
        network_keys = ("advantage_net", "advantage_target_net", "strategy_net")
    for key in network_keys:
        if key not in payload:
            errors.append(f"отсутствует {key}")
        else:
            errors.extend(f"{key}.{path}" for path in _invalid_tensors(payload[key]))
    return {"ok": not errors, "errors": errors, "iteration": payload.get("iteration")}


def _play_game(
    agent: PolicyRuntimeAgent,
    seed: int,
    player_id: int = 0,
    num_players: int | None = None,
) -> float:
    num_players = int(agent.num_players if num_players is None else num_players)
    state = pkrs.State.from_seed(
        n_players=num_players, button=seed % num_players, sb=1.0, bb=2.0, stake=200.0, seed=seed
    )
    opponents = [RandomAgent(player) for player in range(num_players)]
    while not state.final_state:
        player = int(state.current_player)
        if player == player_id:
            slot = agent.choose_action(wrap_state(state, player_id), player_id=player_id)
            action = action_to_pokers(slot, state)
        else:
            action = opponents[player].choose_action(state)
        state = state.apply_action(action)
        if state.status != pkrs.StateStatus.Ok:
            raise RuntimeError(f"Движок отклонил действие: {state.status}")
    return float(state.players_state[player_id].reward)


def evaluate_checkpoint(checkpoint_path: str | Path, games: int = 100, seed: int = 0) -> dict:
    agent = PolicyRuntimeAgent(str(checkpoint_path))
    rewards = np.asarray([_play_game(agent, seed + game) for game in range(int(games))])
    return {
        "games": int(games),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "std_reward": float(rewards.std()) if rewards.size else 0.0,
    }


def _paired_statistics(differences: np.ndarray, big_blind: float) -> dict:
    samples = int(differences.size)
    mean_difference = float(differences.mean()) if samples else 0.0
    standard_error = float(differences.std(ddof=1) / np.sqrt(samples)) if samples > 1 else 0.0
    confidence_delta = 1.96 * standard_error
    scale = 100.0 / big_blind
    return {
        "samples": samples,
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "ci95_low": mean_difference - confidence_delta,
        "ci95_high": mean_difference + confidence_delta,
        "bb_per_100": mean_difference * scale,
        "standard_error_bb_per_100": standard_error * scale,
        "ci95_low_bb_per_100": (mean_difference - confidence_delta) * scale,
        "ci95_high_bb_per_100": (mean_difference + confidence_delta) * scale,
    }


def evaluate_paired_checkpoints(
    checkpoint_path: str | Path,
    opponent_path: str | Path,
    games: int = 100,
    seed: int = 0,
) -> dict:
    """Сравнивает candidate и baseline на дублированных раздачах с CRN."""
    baseline = FrozenBlueprintPolicy.from_checkpoint(opponent_path, device="cpu")
    candidate = FrozenBlueprintPolicy.from_checkpoint(checkpoint_path, device="cpu")
    evaluation = evaluate_paired(
        baseline,
        candidate,
        num_deals=int(games),
        seed=int(seed),
        num_players=baseline.num_players,
        rotate_seats=True,
    )
    deal_differences = evaluation.differences.reshape(evaluation.deals, evaluation.seats).mean(axis=1)
    seat_statistics = _paired_statistics(evaluation.differences, evaluation.bb)
    deal_statistics = _paired_statistics(deal_differences, evaluation.bb)
    return {
        "games": int(games),
        "seats": evaluation.seats,
        "seat_samples": int(evaluation.differences.size),
        "baseline": str(opponent_path),
        "candidate": str(checkpoint_path),
        **deal_statistics,
        "seat_statistics": seat_statistics,
        "deal_statistics": deal_statistics,
    }


def _street_label(stage: int) -> str:
    return _STREET_LABELS[stage] if 0 <= stage < len(_STREET_LABELS) else f"stage_{stage}"


def _engine_action_label(action: pkrs.Action) -> str:
    return _OPPONENT_RESPONSE_LABELS[int(action.action)]


def _profile_game(
    agent: PolicyRuntimeAgent,
    seed: int,
    player_id: int = 0,
    num_players: int | None = None,
) -> dict:
    num_players = int(agent.num_players if num_players is None else num_players)
    state = pkrs.State.from_seed(
        n_players=num_players, button=seed % num_players, sb=1.0, bb=2.0, stake=200.0, seed=seed
    )
    opponents = [RandomAgent(player) for player in range(num_players)]
    actions: list[tuple[str, int]] = []
    flop_paths = {"raise": False, "three_bet": False, "all_in": False}
    opponent_responses: Counter[str] = Counter()
    awaiting_flop_response = False
    while not state.final_state:
        player = int(state.current_player)
        stage = int(state.stage)
        if player == player_id:
            had_flop_raise = stage == 1 and any(
                getattr(item, "last_stage_action", None) == pkrs.ActionEnum.Raise
                for item in state.players_state
            )
            slot = agent.choose_action(wrap_state(state, player_id), player_id=player_id)
            action = action_to_pokers(slot, state)
            actions.append((_street_label(stage), slot))
            if stage == 1 and slot in (3, 4, 5):
                flop_paths["raise"] = True
                flop_paths["three_bet"] |= had_flop_raise
                flop_paths["all_in"] |= slot == 5
                awaiting_flop_response = True
        else:
            action = opponents[player].choose_action(state)
            if awaiting_flop_response and stage == 1:
                opponent_responses[_engine_action_label(action)] += 1
            elif stage != 1:
                awaiting_flop_response = False
        state = state.apply_action(action)
        if state.status != pkrs.StateStatus.Ok:
            raise RuntimeError(f"Движок отклонил действие: {state.status}")
    return {
        "reward": float(state.players_state[player_id].reward),
        "actions": actions,
        "flop_paths": flop_paths,
        "opponent_responses": opponent_responses,
    }


def _profile_path_summary(rewards: list[float], responses: Counter[str]) -> dict:
    values = np.asarray(rewards, dtype=np.float64)
    return {
        "hands": len(rewards),
        "mean_reward": float(values.mean()) if values.size else 0.0,
        "std_reward": float(values.std()) if values.size else 0.0,
        "opponent_responses": {label: int(responses[label]) for label in _OPPONENT_RESPONSE_LABELS},
    }


def profile_checkpoint(checkpoint_path: str | Path, games: int = 100, seed: int = 0) -> dict:
    """Профилирует решения hero и исходы раздач после флоповой агрессии."""
    agent = PolicyRuntimeAgent(str(checkpoint_path))
    action_counts = {street: Counter() for street in _STREET_LABELS}
    path_rewards = {name: [] for name in ("raise", "three_bet", "all_in")}
    path_responses = {name: Counter() for name in path_rewards}
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        for game in range(int(games)):
            result = _profile_game(agent, int(seed) + game)
            for street, slot in result["actions"]:
                if street in action_counts:
                    action_counts[street][ACTION_LABELS[slot]] += 1
            for name, happened in result["flop_paths"].items():
                if happened:
                    path_rewards[name].append(result["reward"])
                    path_responses[name].update(result["opponent_responses"])
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)
    return {
        "games": int(games),
        "checkpoint": str(checkpoint_path),
        "actions_by_street": {
            street: {
                label: {"count": int(action_counts[street][label])}
                for label in ACTION_LABELS
            }
            for street in _STREET_LABELS
        },
        "flop_paths": {
            name: _profile_path_summary(path_rewards[name], path_responses[name])
            for name in path_rewards
        },
    }


def _collect_random_flop_states(games: int, seed: int, player_id: int = 0, num_players: int = 6) -> list[pkrs.State]:
    """Собирает model-independent состояния hero на флопе из random rollout."""
    states: list[pkrs.State] = []
    opponents = [RandomAgent(player) for player in range(num_players)]
    for game in range(int(games)):
        state = pkrs.State.from_seed(
            n_players=num_players,
            button=(seed + game) % num_players,
            sb=1.0,
            bb=2.0,
            stake=200.0,
            seed=seed + game,
        )
        while not state.final_state:
            player = int(state.current_player)
            if player == player_id and int(state.stage) == 1:
                states.append(state)
            action = opponents[player].choose_action(state)
            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                raise RuntimeError(f"Движок отклонил действие: {state.status}")
    return states


def probe_flop_probabilities(
    checkpoint_path: str | Path,
    opponent_path: str | Path,
    games: int = 100,
    seed: int = 0,
) -> dict:
    """Сравнивает policy distribution на одних и тех же флоповых состояниях."""
    if games <= 0:
        raise ValueError("games должен быть положительным")
    baseline = FrozenBlueprintPolicy.from_checkpoint(opponent_path, device="cpu")
    candidate = FrozenBlueprintPolicy.from_checkpoint(checkpoint_path, device="cpu")
    if baseline.num_players != candidate.num_players:
        raise ValueError("Checkpoint-ы используют разное число игроков")
    random_state = random.getstate()
    random.seed(seed)
    try:
        states = _collect_random_flop_states(games, seed, num_players=candidate.num_players)
    finally:
        random.setstate(random_state)

    if not states:
        raise ValueError("Не найдено флоповых состояний; увеличьте --games")
    baseline_probabilities = np.stack([baseline.probabilities(state) for state in states])
    candidate_probabilities = np.stack([candidate.probabilities(state) for state in states])
    differences = candidate_probabilities - baseline_probabilities
    return {
        "games": int(games),
        "samples": len(states),
        "baseline": str(opponent_path),
        "candidate": str(checkpoint_path),
        "mean_l1_distance": float(np.abs(differences).sum(axis=1).mean()),
        "argmax_changes": int((baseline_probabilities.argmax(axis=1) != candidate_probabilities.argmax(axis=1)).sum()),
        "actions": {
            label: {
                "baseline_probability": float(baseline_probabilities[:, slot].mean()),
                "candidate_probability": float(candidate_probabilities[:, slot].mean()),
                "difference": float(differences[:, slot].mean()),
            }
            for slot, label in enumerate(ACTION_LABELS)
        },
    }


def _load_full_checkpoint_agent(path: str | Path) -> tuple[DeepCFRAgent, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("advantage_net", "advantage_target_net", "strategy_net")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Для full-probe нужны полные checkpoint-веса: отсутствуют {missing}")
    architecture, _, hidden_size = full_checkpoint_network_spec(payload)
    agent = DeepCFRAgent(
        player_id=0,
        num_players=int(payload.get("num_players", 6)),
        device="cpu",
        network_architecture=architecture,
        hidden_size=hidden_size,
    )
    agent.load_model(str(path))
    agent.advantage_net.eval()
    agent.strategy_net.eval()
    return agent, payload


def _network_outputs(agent: DeepCFRAgent, states: list[pkrs.State]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = np.stack([agent._encode_state(state, 0) for state in states]).astype(np.float32, copy=False)
    masks = np.stack([agent.get_legal_action_mask(state) for state in states]).astype(np.float32, copy=False)
    states_t = torch.from_numpy(encoded).to(agent.device)
    masks_t = torch.from_numpy(masks).to(agent.device)
    with torch.inference_mode():
        advantages = agent.advantage_net(states_t).cpu().numpy()
        strategy = agent._masked_softmax(agent.strategy_net(states_t), masks_t).cpu().numpy()
    return advantages, strategy, masks


def _legal_action_comparison(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    masks: np.ndarray,
    baseline_key: str,
    candidate_key: str,
) -> dict:
    result = {}
    for slot, label in enumerate(ACTION_LABELS):
        legal = masks[:, slot] > 0.0
        baseline_mean = float(baseline_values[legal, slot].mean()) if legal.any() else 0.0
        candidate_mean = float(candidate_values[legal, slot].mean()) if legal.any() else 0.0
        result[label] = {
            "legal_samples": int(legal.sum()),
            baseline_key: baseline_mean,
            candidate_key: candidate_mean,
            "difference": candidate_mean - baseline_mean,
        }
    return result


def _buffer_summary(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {"present": False, "count": 0}
    count = int(payload.get("count", 0))
    values = np.asarray(payload.get("values"))
    masks = np.asarray(payload.get("masks"))
    if count <= 0 or values.shape != (count, NUM_ACTIONS) or masks.shape != (count, NUM_ACTIONS):
        return {"present": True, "count": 0, "valid": False}
    return {
        "present": True,
        "valid": True,
        "count": count,
        "actions": {
            label: {
                "legal_samples": int((masks[:, slot] > 0.0).sum()),
                "mean_value": float(values[masks[:, slot] > 0.0, slot].mean()) if (masks[:, slot] > 0.0).any() else 0.0,
            }
            for slot, label in enumerate(ACTION_LABELS)
        },
    }


def probe_full_checkpoints(
    checkpoint_path: str | Path,
    opponent_path: str | Path,
    games: int = 100,
    seed: int = 0,
) -> dict:
    """Сопоставляет strategy/advantage и сохранённые buffers полных checkpoint-ов."""
    if games <= 0:
        raise ValueError("games должен быть положительным")
    baseline, baseline_payload = _load_full_checkpoint_agent(opponent_path)
    candidate, candidate_payload = _load_full_checkpoint_agent(checkpoint_path)
    random_state = random.getstate()
    random.seed(seed)
    try:
        states = _collect_random_flop_states(games, seed)
    finally:
        random.setstate(random_state)
    if not states:
        raise ValueError("Не найдено флоповых состояний; увеличьте --games")

    baseline_advantages, baseline_strategy, masks = _network_outputs(baseline, states)
    candidate_advantages, candidate_strategy, candidate_masks = _network_outputs(candidate, states)
    if not np.array_equal(masks, candidate_masks):
        raise RuntimeError("Legal mask различается на одинаковых игровых состояниях")
    strategy_difference = candidate_strategy - baseline_strategy
    return {
        "games": int(games),
        "samples": len(states),
        "baseline": str(opponent_path),
        "candidate": str(checkpoint_path),
        "strategy": {
            "mean_l1_distance": float(np.abs(strategy_difference).sum(axis=1).mean()),
            "argmax_changes": int((baseline_strategy.argmax(axis=1) != candidate_strategy.argmax(axis=1)).sum()),
            "actions": _legal_action_comparison(
                baseline_strategy, candidate_strategy, masks, "baseline_probability", "candidate_probability"
            ),
        },
        "advantages": _legal_action_comparison(
            baseline_advantages, candidate_advantages, masks, "baseline_output", "candidate_output"
        ),
        "buffers": {
            "advantage": {
                "present": "advantage_buffer" in baseline_payload and "advantage_buffer" in candidate_payload,
                "baseline": _buffer_summary(baseline_payload.get("advantage_buffer")),
                "candidate": _buffer_summary(candidate_payload.get("advantage_buffer")),
            },
            "strategy": {
                "present": "strategy_buffer" in baseline_payload and "strategy_buffer" in candidate_payload,
                "baseline": _buffer_summary(baseline_payload.get("strategy_buffer")),
                "candidate": _buffer_summary(candidate_payload.get("strategy_buffer")),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка action-only Deep CFR checkpoint")
    parser.add_argument("mode", choices=("diagnose", "eval", "paired", "profile", "flop-probe", "full-probe", "full"))
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", help="Baseline checkpoint для paired, flop-probe и full-probe")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    result = {"contract": {"version": ACTION_SPACE_VERSION, "actions": ACTION_LABELS}}
    if args.mode in ("diagnose", "full"):
        result["diagnose"] = run_weight_sanity(args.checkpoint)
    if args.mode in ("eval", "full"):
        result["evaluation"] = evaluate_checkpoint(args.checkpoint, args.games, args.seed)
    if args.mode == "paired":
        if not args.opponent:
            parser.error("режим paired требует --opponent")
        result["paired"] = evaluate_paired_checkpoints(args.checkpoint, args.opponent, args.games, args.seed)
    if args.mode == "profile":
        result["profile"] = profile_checkpoint(args.checkpoint, args.games, args.seed)
    if args.mode == "flop-probe":
        if not args.opponent:
            parser.error("режим flop-probe требует --opponent")
        result["flop_probe"] = probe_flop_probabilities(args.checkpoint, args.opponent, args.games, args.seed)
    if args.mode == "full-probe":
        if not args.opponent:
            parser.error("режим full-probe требует --opponent")
        result["full_probe"] = probe_full_checkpoints(args.checkpoint, args.opponent, args.games, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
