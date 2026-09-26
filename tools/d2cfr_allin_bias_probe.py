"""Read-only диагностика: сравнение D2CFR Q/advantage по фиксированным action slots.

Утилита загружает полный HU checkpoint, собирает свежие postflop infoset'ы,
повторяет ES-traversal из каждого такого состояния и сохраняет JSON-отчёт.
Она не запускает optimizer и не меняет checkpoint или replay на диске.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pokers as pkrs

from src.core.action_space import ACTION_LABELS, ActionSlot, legal_action_mask, resolve_action
from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod


_REQUIRED_SLOTS = frozenset((
    int(ActionSlot.RAISE_HALF_POT),
    int(ActionSlot.RAISE_POT),
    int(ActionSlot.ALL_IN),
))


def _finite_number(row: dict[str, Any], name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"{name} должен быть конечным числом")
    return value


def summarise_action_comparison(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Сводит action-level target/prediction без смешивания разных слотов."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row["action_label"])
        if label not in ACTION_LABELS:
            raise ValueError(f"неизвестный слот действия: {label}")
        grouped[label].append(row)

    by_action: dict[str, dict[str, float | int]] = {}
    for label, action_rows in sorted(grouped.items()):
        target_q = np.asarray([_finite_number(row, "target_q") for row in action_rows])
        predicted_q = np.asarray([_finite_number(row, "predicted_q") for row in action_rows])
        target_regret = np.asarray([_finite_number(row, "target_regret") for row in action_rows])
        predicted_regret = np.asarray([_finite_number(row, "predicted_regret") for row in action_rows])
        q_error = predicted_q - target_q
        sign_flips = (target_regret > 0.0) != (predicted_regret > 0.0)
        by_action[label] = {
            "count": int(len(action_rows)),
            "q_target_mean": float(target_q.mean()),
            "q_prediction_mean": float(predicted_q.mean()),
            "q_bias_mean": float(q_error.mean()),
            "q_mae": float(np.abs(q_error).mean()),
            "regret_sign_flip_rate": float(sign_flips.mean()),
        }
        if all("target_q_std" in row for row in action_rows):
            target_q_std = np.asarray([_finite_number(row, "target_q_std") for row in action_rows])
            q_noise_ratios = np.asarray([
                _finite_number(row, "q_error_to_noise_ratio") for row in action_rows
            ])
            by_action[label].update({
                "target_q_std_mean": float(target_q_std.mean()),
                "q_error_to_noise_ratio_mean": float(q_noise_ratios.mean()),
            })

    all_in = by_action.get("all_in")
    half_pot = by_action.get("raise_0.5pot")
    return {
        "by_action": by_action,
        "all_in_minus_half_pot_q_bias": (
            float(all_in["q_bias_mean"] - half_pot["q_bias_mean"])
            if all_in is not None and half_pot is not None
            else None
        ),
    }


def _regret_matching(regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Возвращает policy regret matching только по разрешённым слотам."""
    legal = np.asarray(legal_mask, dtype=np.float64) == 1.0
    if regrets.shape != legal.shape or regrets.shape != (len(ACTION_LABELS),):
        raise ValueError("Regret и legal mask должны содержать все action slots")
    if not legal.any():
        raise ValueError("Нужен хотя бы один допустимый action slot")
    positive = np.maximum(np.asarray(regrets, dtype=np.float64), 0.0) * legal
    total = float(positive.sum())
    if total > 0.0:
        return positive / total
    return legal.astype(np.float64) / float(legal.sum())


def describe_resampled_targets(
    *,
    q_samples: np.ndarray,
    v_samples: np.ndarray,
    regret_samples: np.ndarray,
    predicted_q: np.ndarray,
    predicted_v: float,
    predicted_regret: np.ndarray,
    legal_mask: np.ndarray,
    noise_floor: float = 0.01,
) -> dict[str, Any]:
    """Сравнивает сеть со средним frozen-policy target и его MC-разбросом."""
    q_samples = np.asarray(q_samples, dtype=np.float64)
    v_samples = np.asarray(v_samples, dtype=np.float64)
    regret_samples = np.asarray(regret_samples, dtype=np.float64)
    predicted_q = np.asarray(predicted_q, dtype=np.float64)
    predicted_regret = np.asarray(predicted_regret, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=np.float64)
    sample_count = q_samples.shape[0]
    expected_vector_shape = (sample_count, len(ACTION_LABELS))
    if sample_count <= 0 or q_samples.shape != expected_vector_shape:
        raise ValueError("q_samples должен иметь форму (repeats, NUM_ACTIONS)")
    if regret_samples.shape != expected_vector_shape or v_samples.shape != (sample_count,):
        raise ValueError("Q, V и regret samples должны иметь согласованные формы")
    if predicted_q.shape != (len(ACTION_LABELS),) or predicted_regret.shape != predicted_q.shape:
        raise ValueError("Прогнозы сети должны содержать все action slots")
    values_to_check = (q_samples, v_samples, regret_samples, predicted_q, predicted_regret)
    if not all(np.all(np.isfinite(value)) for value in values_to_check) or not np.isfinite(predicted_v):
        raise ValueError("Resampling targets и прогнозы сети должны быть конечными")
    if not np.isfinite(noise_floor) or noise_floor <= 0.0:
        raise ValueError("noise_floor должен быть положительным конечным числом")

    target_q_mean = q_samples.mean(axis=0)
    target_q_std = q_samples.std(axis=0, ddof=0)
    target_regret_mean = regret_samples.mean(axis=0)
    target_regret_std = regret_samples.std(axis=0, ddof=0)
    centered_q = q_samples - target_q_mean
    centered_regret = regret_samples - target_regret_mean
    target_q_covariance = centered_q.T @ centered_q / float(sample_count)
    target_regret_covariance = centered_regret.T @ centered_regret / float(sample_count)
    target_policy = _regret_matching(target_regret_mean, legal_mask)
    predicted_policy = _regret_matching(predicted_regret, legal_mask)
    by_action: dict[str, dict[str, float]] = {}
    for slot in np.flatnonzero(legal_mask == 1.0):
        label = ACTION_LABELS[int(slot)]
        q_abs_error = abs(float(predicted_q[slot] - target_q_mean[slot]))
        by_action[label] = {
            "target_q_mean": float(target_q_mean[slot]),
            "target_q_std": float(target_q_std[slot]),
            "predicted_q": float(predicted_q[slot]),
            "q_abs_error": q_abs_error,
            "q_error_to_noise_ratio": float(q_abs_error / max(float(target_q_std[slot]), noise_floor)),
            "target_regret_mean": float(target_regret_mean[slot]),
            "target_regret_std": float(target_regret_std[slot]),
            "predicted_regret": float(predicted_regret[slot]),
        }
    return {
        "sample_count": int(sample_count),
        "noise_floor": float(noise_floor),
        "target_v_mean": float(v_samples.mean()),
        "target_v_std": float(v_samples.std(ddof=0)),
        "predicted_v": float(predicted_v),
        "v_abs_error": abs(float(predicted_v) - float(v_samples.mean())),
        "target_rm_policy": target_policy.tolist(),
        "predicted_rm_policy": predicted_policy.tolist(),
        "rm_l1": float(np.abs(target_policy - predicted_policy).sum()),
        "target_q_covariance": target_q_covariance.tolist(),
        "target_regret_covariance": target_regret_covariance.tolist(),
        "by_action": by_action,
    }


def summarise_all_in_traces(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Сводит ответы оппонента и terminal reward после принудительного all-in."""
    trace_rows = list(rows)
    if not trace_rows:
        raise ValueError("Нужна хотя бы одна all-in трасса")
    responses = [str(row["opponent_response"]) for row in trace_rows]
    rewards = np.asarray([_finite_number(row, "terminal_reward") for row in trace_rows])
    counts = {response: responses.count(response) for response in sorted(set(responses))}
    return {
        "samples": len(trace_rows),
        "opponent_response_counts": counts,
        "terminal_reward_mean": float(rewards.mean()),
        "terminal_reward_std": float(rewards.std(ddof=0)),
    }


def card_code(card: Any) -> str:
    """Возвращает стабильный покерный код карты, а не адрес Python-объекта."""
    rank_name = str(card.rank).split(".")[-1]
    suit_name = str(card.suit).split(".")[-1]
    if not rank_name.startswith("R") or len(rank_name) != 2:
        raise ValueError(f"Неизвестный rank карты: {rank_name}")
    rank = rank_name[1]
    suit_codes = {"Clubs": "c", "Diamonds": "d", "Hearts": "h", "Spades": "s"}
    try:
        suit = suit_codes[suit_name]
    except KeyError as error:
        raise ValueError(f"Неизвестный suit карты: {suit_name}") from error
    return f"{rank}{suit}"


def deal_key(state: Any) -> tuple[tuple[str, ...], ...]:
    """Идентифицирует private deal, общий для всех улиц одной раздачи."""
    return tuple(
        tuple(card_code(card) for card in player.hand)
        for player in state.players_state
    )


def _new_hand(seed: int) -> pkrs.State:
    return train_mod._new_hand(2, seed)


def _state_summary(state: pkrs.State) -> dict[str, Any]:
    player = state.players_state[int(state.current_player)]
    return {
        "stage": str(state.stage),
        "current_player": int(state.current_player),
        "pot": float(state.pot),
        "min_bet": float(state.min_bet),
        "hands": {
            str(index): [card_code(card) for card in player_state.hand]
            for index, player_state in enumerate(state.players_state)
        },
        "public_cards": [card_code(card) for card in state.public_cards],
        "stake": float(player.stake),
        "bet_chips": float(player.bet_chips),
    }


def _network_components(agent: DeepCFRAgent, state: pkrs.State, player_id: int) -> tuple[np.ndarray, float, np.ndarray]:
    network = agent.hu_advantage_nets[player_id]
    encoded = agent._encode_state(state, player_id)
    state_t = torch.as_tensor(encoded, dtype=torch.float32, device=agent.device).unsqueeze(0)
    network.eval()
    with torch.inference_mode():
        components = network.forward_components(state_t)
    return (
        components.action_values[0].detach().cpu().numpy().astype(np.float64),
        float(components.state_values[0, 0].detach().cpu().item()),
        components.regrets[0].detach().cpu().numpy().astype(np.float64),
    )


def _is_candidate(state: pkrs.State, traverser: int, mask: np.ndarray) -> bool:
    return (
        state.stage not in (pkrs.Stage.Preflop, pkrs.Stage.Showdown)
        and int(state.current_player) == traverser
        and _REQUIRED_SLOTS.issubset(set(np.flatnonzero(mask).astype(int).tolist()))
    )


def _collect_candidate_states(
    coordinator: Any,
    *,
    roots: int,
    seed: int,
    target_count: int,
    traverser: int,
    iteration: int,
) -> list[pkrs.State]:
    captured: list[pkrs.State] = []
    captured_deals: set[tuple[tuple[str, ...], ...]] = set()
    original = coordinator.adapter.normalise_d2cfr_targets
    if original is None:
        raise RuntimeError("HU coordinator не настроен для D2CFR targets")

    def capture(state, action_values, state_value, mask):
        if len(captured) < target_count and _is_candidate(state, traverser, np.asarray(mask)):
            key = deal_key(state)
            if key not in captured_deals:
                # pokers.State неизменяем при apply_action: дочернее состояние возвращается отдельно.
                captured.append(state)
                captured_deals.add(key)
        return original(state, action_values, state_value, mask)

    coordinator.adapter = replace(coordinator.adapter, normalise_d2cfr_targets=capture)
    try:
        coordinator.begin_iteration()
        for index in range(roots):
            if len(captured) >= target_count:
                break
            coordinator.traverse(
                _new_hand(seed + index),
                traversing_player=traverser,
                iteration=iteration,
            )
    finally:
        coordinator.adapter = replace(coordinator.adapter, normalise_d2cfr_targets=original)
    return captured


def _sample_state_targets(
    coordinator: Any,
    state: pkrs.State,
    *,
    repeats: int,
    traverser: int,
    iteration: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    samples: list[tuple[np.ndarray, float, np.ndarray]] = []
    original = coordinator.adapter.normalise_d2cfr_targets
    if original is None:
        raise RuntimeError("HU coordinator не настроен для D2CFR targets")

    for _ in range(repeats):
        root = state

        def capture(current, action_values, state_value, mask):
            result = original(current, action_values, state_value, mask)
            if current is root:
                q, value, regrets = result
                samples.append((
                    np.asarray(q, dtype=np.float64),
                    float(value),
                    np.asarray(regrets, dtype=np.float64),
                ))
            return result

        coordinator.adapter = replace(coordinator.adapter, normalise_d2cfr_targets=capture)
        try:
            coordinator.traverse(root, traversing_player=traverser, iteration=iteration)
        finally:
            coordinator.adapter = replace(coordinator.adapter, normalise_d2cfr_targets=original)
    if len(samples) != repeats:
        raise RuntimeError("Не удалось снять target корневого postflop-состояния")
    return samples


def _trace_all_in_rollouts(
    coordinator: Any,
    state: pkrs.State,
    *,
    samples: int,
    traverser: int,
    max_depth: int = 128,
) -> list[dict[str, Any]]:
    """Делает on-policy continuations после all-in без записи в training buffers."""
    if samples <= 0:
        raise ValueError("samples должно быть положительным")
    traces: list[dict[str, Any]] = []
    for _ in range(samples):
        current = coordinator.adapter.apply(state, int(ActionSlot.ALL_IN))
        actions = [{"actor": traverser, "action": "all_in"}]
        response = "terminal_after_all_in"
        if not coordinator.adapter.is_terminal(current):
            actor = int(coordinator.adapter.current_player(current))
            if actor != 1 - traverser:
                raise RuntimeError("После all-in ожидался ответ оппонента")
            mask = legal_action_mask(current)
            encoded = coordinator.adapter.encode(current, actor)
            policy = coordinator._policy_from_snapshot(actor, encoded, mask)
            legal_slots = np.flatnonzero(mask).astype(int)
            slot = int(coordinator.sampler(legal_slots, policy))
            response = ACTION_LABELS[slot]
            actions.append({"actor": actor, "action": response})
            current = coordinator.adapter.apply(current, slot)

        for _depth in range(max_depth):
            if coordinator.adapter.is_terminal(current):
                break
            actor = int(coordinator.adapter.current_player(current))
            mask = legal_action_mask(current)
            encoded = coordinator.adapter.encode(current, actor)
            policy = coordinator._policy_from_snapshot(actor, encoded, mask)
            legal_slots = np.flatnonzero(mask).astype(int)
            slot = int(coordinator.sampler(legal_slots, policy))
            actions.append({"actor": actor, "action": ACTION_LABELS[slot]})
            current = coordinator.adapter.apply(current, slot)
        else:
            raise RuntimeError(f"all-in trace превысил max_depth={max_depth}")
        traces.append({
            "opponent_response": response,
            "terminal_reward": float(coordinator.adapter.terminal_value(current, traverser)),
            "terminal_rewards": {
                str(player.player): float(player.reward)
                for player in current.players_state
            },
            "showdown": {
                "hands": {
                    str(player.player): [card_code(card) for card in player.hand]
                    for player in current.players_state
                },
                "public_cards": [card_code(card) for card in current.public_cards],
            },
            "actions": actions,
        })
    return traces


def run_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    states: int,
    repeats: int,
    roots: int,
    seed: int,
    device: str = "cpu",
    trace_all_in: bool = False,
) -> dict[str, Any]:
    """Выполняет изолированную диагностику и атомарно сохраняет JSON-отчёт."""
    if states <= 0 or repeats <= 0 or roots <= 0:
        raise ValueError("states, repeats и roots должны быть положительными")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=device)
    coordinator = train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not bool(agent.d2cfr_enabled):
        raise ValueError("Нужен D2CFR HU checkpoint")

    probe_iteration = max(int(checkpoint["iteration"]), 1)
    candidates = _collect_candidate_states(
        coordinator, roots=roots, seed=seed, target_count=states, traverser=0,
        iteration=probe_iteration,
    )
    if not candidates:
        raise RuntimeError("Не найдено postflop-состояний с half-pot, pot и all-in")

    flat_rows: list[dict[str, Any]] = []
    state_reports: list[dict[str, Any]] = []
    resampling_reports: list[dict[str, Any]] = []
    for state_index, state in enumerate(candidates):
        q_prediction, v_prediction, regret_prediction = _network_components(agent, state, 0)
        target_samples = _sample_state_targets(
            coordinator, state, repeats=repeats, traverser=0, iteration=probe_iteration,
        )
        q_samples = np.stack([sample[0] for sample in target_samples])
        v_samples = np.asarray([sample[1] for sample in target_samples])
        regret_samples = np.stack([sample[2] for sample in target_samples])
        legal_mask = legal_action_mask(state)
        resampling = describe_resampled_targets(
            q_samples=q_samples,
            v_samples=v_samples,
            regret_samples=regret_samples,
            predicted_q=q_prediction,
            predicted_v=v_prediction,
            predicted_regret=regret_prediction,
            legal_mask=legal_mask,
        )
        resampling_reports.append(resampling)
        per_action: dict[str, Any] = {}
        for slot in sorted(_REQUIRED_SLOTS):
            label = ACTION_LABELS[slot]
            action_resampling = resampling["by_action"][label]
            target_q = float(action_resampling["target_q_mean"])
            target_regret = float(action_resampling["target_regret_mean"])
            row = {
                "state_index": state_index,
                "action_label": label,
                "target_q": target_q,
                "predicted_q": float(q_prediction[slot]),
                "target_regret": target_regret,
                "predicted_regret": float(regret_prediction[slot]),
                "target_q_std": float(action_resampling["target_q_std"]),
                "q_error_to_noise_ratio": float(action_resampling["q_error_to_noise_ratio"]),
            }
            flat_rows.append(row)
            per_action[label] = {
                **row,
                "target_regret_std": float(action_resampling["target_regret_std"]),
            }
        state_reports.append({
            "state_index": state_index,
            "deal_key": [list(hand) for hand in deal_key(state)],
            "state": _state_summary(state),
            "target_v_mean": float(v_samples.mean()),
            "target_v_std": float(v_samples.std(ddof=0)),
            "predicted_v": v_prediction,
            "actions": per_action,
            "resampling": resampling,
            "target_samples": {
                "q": q_samples.tolist(),
                "v": v_samples.tolist(),
                "regret": regret_samples.tolist(),
            },
            **({
                "all_in_trace": {
                    "summary": summarise_all_in_traces(trace_rows := _trace_all_in_rollouts(
                        coordinator, state, samples=repeats, traverser=0,
                    )),
                    "samples": trace_rows,
                },
            } if trace_all_in else {}),
        })

    summary = summarise_action_comparison(flat_rows)
    summary.update({
        "rm_l1_mean": float(np.mean([item["rm_l1"] for item in resampling_reports])),
        "target_v_std_mean": float(np.mean([item["target_v_std"] for item in resampling_reports])),
        "v_abs_error_mean": float(np.mean([item["v_abs_error"] for item in resampling_reports])),
    })
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "config": str(config_path),
        "seed": int(seed),
        "states_requested": int(states),
        "states_collected": len(candidates),
        "roots_examined": int(roots),
        "repeats_per_state": int(repeats),
        "summary": summary,
        "states": state_reports,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Путь к полному HU checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Конфигурация, совместимая с checkpoint")
    parser.add_argument("--output", required=True, help="Куда сохранить JSON-отчёт")
    parser.add_argument("--states", type=int, default=16, help="Число postflop-состояний")
    parser.add_argument("--repeats", type=int, default=64, help="ES-прогонов на состояние")
    parser.add_argument("--roots", type=int, default=512, help="Максимум стартовых раздач для поиска состояний")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--trace-all-in",
        action="store_true",
        help="Сохранить ответы P1 и terminal reward в отдельных all-in rollouts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        states=args.states,
        repeats=args.repeats,
        roots=args.roots,
        seed=args.seed,
        device=args.device,
        trace_all_in=args.trace_all_in,
    )
    print(f"Отчёт сохранён: {args.output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
