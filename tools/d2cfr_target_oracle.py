"""Oracle-проверка D2CFR target construction на контролируемом game tree.

Проверка строит одношаговое дерево:

root infoset traverser-а -> legal actions -> terminal payoff.

Exact Q/V/R считаются независимо, а production `DeepCFRAgent.cfr_traverse_multi`
должен записать тот же D2CFR sample в replay buffer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pokers as pkrs

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import DeepCFRAgent
from src.utils import config as config_mod


@dataclass(frozen=True)
class OraclePlayerState:
    reward: float = 0.0
    stake: float = 200.0


class OracleState:
    """Минимальный state-протокол, нужный production traversal."""

    def __init__(
        self,
        *,
        current_player: int,
        final_state: bool,
        payoffs_by_action: dict[int, float] | None = None,
        traversing_player: int = 0,
        reward: float = 0.0,
        num_players: int = 2,
    ) -> None:
        self.current_player = int(current_player)
        self.final_state = bool(final_state)
        self.status = pkrs.StateStatus.Ok
        self.pot = 0.0
        self._payoffs_by_action = dict(payoffs_by_action or {})
        self._traversing_player = int(traversing_player)
        self.players_state = [
            OraclePlayerState(reward=float(reward) if player == traversing_player else 0.0)
            for player in range(int(num_players))
        ]

    def apply_action(self, action: int) -> "OracleState":
        slot = int(action)
        if slot not in self._payoffs_by_action:
            raise ValueError(f"OracleState получил неизвестный action slot={slot}")
        return OracleState(
            current_player=self.current_player,
            final_state=True,
            traversing_player=self._traversing_player,
            reward=float(self._payoffs_by_action[slot]),
            num_players=len(self.players_state),
        )


class FixedAdvantageNet(torch.nn.Module):
    """Возвращает фиксированные advantages для root infoset."""

    def __init__(self, advantages: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "_advantages",
            torch.as_tensor(advantages, dtype=torch.float32).reshape(1, NUM_ACTIONS),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        advantages = cast(torch.Tensor, getattr(self, "_advantages"))
        return advantages.repeat(int(states.shape[0]), 1)


def _dense_vector(values_by_slot: dict[int, float]) -> np.ndarray:
    vector = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for slot, value in values_by_slot.items():
        slot = int(slot)
        if slot < 0 or slot >= NUM_ACTIONS:
            raise ValueError(f"slot вне диапазона 0..{NUM_ACTIONS - 1}: {slot}")
        vector[slot] = np.float32(value)
    return vector


def _legal_mask(slots: list[int]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[np.asarray(slots, dtype=np.intp)] = 1.0
    return mask


def _regret_matching(advantages: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(advantages, dtype=np.float32), 0.0) * mask
    total = float(positive.sum())
    if total > 0.0:
        return positive / total
    legal_count = float(mask.sum())
    return mask / legal_count if legal_count > 0.0 else np.zeros(NUM_ACTIONS, dtype=np.float32)


def _expected_scale(agent: DeepCFRAgent) -> float:
    norm = str(getattr(agent, "advantage_regret_norm", "none"))
    if norm != "none":
        raise ValueError(
            "target oracle сейчас поддерживает только advantage_regret_norm=none, "
            f"получено {norm!r}"
        )
    scale = float(getattr(agent, "advantage_reward_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Некорректный advantage_reward_scale={scale}")
    return scale


def _expected_targets(
    *,
    agent: DeepCFRAgent,
    legal_payoffs: dict[int, float],
    advantages: dict[int, float],
) -> dict[str, Any]:
    legal_slots = sorted(int(slot) for slot in legal_payoffs)
    mask = _legal_mask(legal_slots)
    raw_action_values = _dense_vector(legal_payoffs)
    advantage_vector = _dense_vector(advantages)
    strategy = _regret_matching(advantage_vector, mask)
    raw_state_value = float(
        sum(float(strategy[slot]) * float(raw_action_values[slot]) for slot in legal_slots)
    )
    raw_regrets = (raw_action_values - np.float32(raw_state_value)) * mask
    scale = _expected_scale(agent)
    return {
        "legal_slots": legal_slots,
        "scale": scale,
        "strategy": strategy.astype(np.float32).tolist(),
        "raw_action_values": raw_action_values.astype(np.float32).tolist(),
        "raw_state_value": raw_state_value,
        "raw_regrets": raw_regrets.astype(np.float32).tolist(),
        "action_values": (raw_action_values / scale).astype(np.float32).tolist(),
        "state_value": float(np.float32(raw_state_value / scale)),
        "regrets": (raw_regrets / scale).astype(np.float32).tolist(),
        "mask": mask.astype(np.float32).tolist(),
    }


def _first_d2cfr_sample(agent: DeepCFRAgent) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    buffer = cast(Any, agent.d2cfr_buffer)
    if buffer is None or len(buffer) != 1:
        raise AssertionError(f"Oracle ожидал ровно один D2CFR sample, получено {len(buffer) if buffer else 0}")
    return (
        buffer._states[0].copy(),
        buffer._action_values[0].copy(),
        float(buffer._state_values[0]),
        buffer._regrets[0].copy(),
        buffer._masks[0].copy(),
        float(buffer._iterations[0]),
    )


def _max_legal_error(recorded: np.ndarray, expected: list[float], mask: np.ndarray) -> float:
    expected_array = np.asarray(expected, dtype=np.float32)
    return float(np.max(np.abs((recorded - expected_array) * mask)))


def run_one_step_target_oracle(
    *,
    config_path: str | Path | None,
    legal_payoffs: dict[int, float],
    advantages: dict[int, float],
    iteration: int = 1,
    traversing_player: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Сравнивает production traversal target с exact reference."""
    if config_path is not None:
        config_mod.load_config(config_path)
    num_players = int(config_mod.cfg_get("num_players", 2) or 2)
    agent = DeepCFRAgent(
        player_id=int(traversing_player),
        num_players=num_players,
        device=str(device),
    )
    if not agent.d2cfr_enabled:
        raise ValueError("target oracle требует d2cfr_enabled=true")
    if set(advantages) != set(legal_payoffs):
        raise ValueError("advantages и legal_payoffs должны описывать один набор legal slots")

    legal_slots = sorted(int(slot) for slot in legal_payoffs)
    mask = _legal_mask(legal_slots)
    advantage_vector = _dense_vector(advantages)
    expected = _expected_targets(
        agent=agent,
        legal_payoffs=legal_payoffs,
        advantages=advantages,
    )

    encoded = np.zeros(int(agent.input_size), dtype=np.float32)
    encoded[0] = 1.0

    agent_any = cast(Any, agent)
    agent_any.advantage_net = FixedAdvantageNet(advantage_vector).to(agent.device)
    agent_any.get_legal_action_mask = lambda state: mask.copy()
    agent_any.action_type_to_pokers_action = lambda slot, state: int(slot)
    agent_any._encode_state = lambda state, player_id: encoded.copy()
    agent.reset_traversal_stats()

    root = OracleState(
        current_player=int(traversing_player),
        final_state=False,
        payoffs_by_action={int(slot): float(value) for slot, value in legal_payoffs.items()},
        traversing_player=int(traversing_player),
        num_players=num_players,
    )
    root_return = float(agent.cfr_traverse_multi(root, int(iteration), int(traversing_player)))
    state, action_values, state_value, regrets, recorded_mask, recorded_iteration = _first_d2cfr_sample(agent)

    max_abs_errors = {
        "action_value": _max_legal_error(action_values, expected["action_values"], mask),
        "state_value": abs(float(state_value) - float(expected["state_value"])),
        "regret": _max_legal_error(regrets, expected["regrets"], mask),
        "mask": float(np.max(np.abs(recorded_mask - mask))),
        "iteration": abs(float(recorded_iteration) - float(iteration)),
        "root_return": abs(float(root_return) - float(expected["raw_state_value"])),
    }
    passed = all(value <= 1e-6 for value in max_abs_errors.values())
    return {
        "mode": "one_step_target_oracle",
        "passed": bool(passed),
        "config": str(config_path) if config_path is not None else None,
        "iteration": int(iteration),
        "traversing_player": int(traversing_player),
        "num_players": int(num_players),
        "root_return": root_return,
        "expected": expected,
        "recorded": {
            "encoded_nonzero_count": int(np.count_nonzero(state)),
            "action_values": action_values.astype(np.float32).tolist(),
            "state_value": float(state_value),
            "regrets": regrets.astype(np.float32).tolist(),
            "mask": recorded_mask.astype(np.float32).tolist(),
            "iteration": float(recorded_iteration),
        },
        "max_abs_errors": max_abs_errors,
        "traversal_stats": agent.get_traversal_stats(),
    }


def _parse_slot_values(items: list[str], *, name: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"{name} должен иметь формат slot:value, получено {item!r}")
        slot_text, value_text = item.split(":", 1)
        slot = int(slot_text)
        if slot in result:
            raise ValueError(f"{name} содержит duplicate slot={slot}")
        result[slot] = float(value_text)
    if not result:
        raise ValueError(f"{name} не должен быть пустым")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--payoff", action="append", default=None)
    parser.add_argument("--advantage", action="append", default=None)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--traversing-player", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_one_step_target_oracle(
        config_path=args.config,
        legal_payoffs=_parse_slot_values(args.payoff or ["1:-20", "3:40", "5:80"], name="payoff"),
        advantages=_parse_slot_values(args.advantage or ["1:0", "3:1", "5:3"], name="advantage"),
        iteration=args.iteration,
        traversing_player=args.traversing_player,
        device=args.device,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
