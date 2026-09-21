"""Oracle-проверки настоящего `pokers.State` на простых независимых сценариях."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pokers as pkrs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask
from src.core.deep_cfr import DeepCFRAgent
from src.runtime_search.cards import parse_card, remaining_deck
from src.utils import config as config_mod
from tools.d2cfr_target_oracle import FixedAdvantageNet


def _blind_fold_expected_rewards(state: Any, folder: int) -> list[float]:
    """Независимая HU-арифметика фолда: каждый получает net от своего взноса."""
    if len(state.players_state) != 2:
        raise ValueError("fold oracle сейчас поддерживает только HU")
    folder = int(folder)
    winner = 1 - folder
    folder_contribution = float(state.players_state[folder].bet_chips) + float(
        state.players_state[folder].pot_chips
    )
    winner_contribution = float(state.players_state[winner].bet_chips) + float(
        state.players_state[winner].pot_chips
    )
    rewards = [0.0, 0.0]
    rewards[folder] = -folder_contribution
    rewards[winner] = folder_contribution
    pot = folder_contribution + winner_contribution
    if abs(float(state.pot) - pot) > 1e-6:
        raise ValueError(f"Неожиданный pot: state.pot={state.pot}, contributions={pot}")
    return rewards


def _single_slot_mask(slot: int) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[int(slot)] = 1.0
    return mask


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


def _max_legal_error(recorded: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.abs((recorded - expected) * mask)))


def _make_agent(config_path: str | Path | None, traversing_player: int, device: str | torch.device) -> DeepCFRAgent:
    if config_path is not None:
        config_mod.load_config(config_path)
    return DeepCFRAgent(
        player_id=int(traversing_player),
        num_players=2,
        device=str(device),
    )


def _hu_river_known_winner_state() -> pkrs.State:
    """HU river: player0 имеет пару тузов, player1 — high-card; pot split impossible."""
    hole_cards = [
        (parse_card("Ac"), parse_card("Ah")),
        (parse_card("Kd"), parse_card("Qd")),
    ]
    public_cards = [parse_card(card) for card in ("2c", "7d", "9h", "Jc", "3s")]
    known_cards = [card for hand in hole_cards for card in hand] + public_cards
    return pkrs.State.from_mid_hand(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        deck=list(remaining_deck(known_cards)),
        hole_cards=hole_cards,
        public_cards=public_cards,
        stage=pkrs.Stage.River,
        pot=20.0,
        bet_chips=[0.0, 0.0],
        pot_chips=[10.0, 10.0],
        active=[True, True],
        last_stage_action=[None, None],
        current_player=0,
        last_raise_increment=2.0,
        verbose=False,
    )


def _hu_river_allin_response_state() -> pkrs.State:
    """HU river: player1 решает fold/call против all-in, player0 выигрывает showdown."""
    hole_cards = [
        (parse_card("Ac"), parse_card("Ah")),
        (parse_card("Kd"), parse_card("Qd")),
    ]
    public_cards = [parse_card(card) for card in ("2c", "7d", "9h", "Jc", "3s")]
    known_cards = [card for hand in hole_cards for card in hand] + public_cards
    return pkrs.State.from_mid_hand(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        deck=list(remaining_deck(known_cards)),
        hole_cards=hole_cards,
        public_cards=public_cards,
        stage=pkrs.Stage.River,
        pot=200.0,
        bet_chips=[200.0, 0.0],
        pot_chips=[0.0, 0.0],
        active=[True, True],
        last_stage_action=[pkrs.ActionEnum.Raise, None],
        current_player=1,
        last_raise_increment=200.0,
        verbose=False,
    )


def run_preflop_fold_oracle(
    *,
    config_path: str | Path | None,
    seed: int,
    button: int,
    sb: float,
    bb: float,
    stake: float,
    traversing_player: int,
    iteration: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Проверяет HU preflop fold payoff и D2CFR target на настоящем engine state."""
    agent = _make_agent(config_path, traversing_player, device)
    if not agent.d2cfr_enabled:
        raise ValueError("engine oracle требует d2cfr_enabled=true")

    root = pkrs.State.from_seed(
        n_players=2,
        button=int(button),
        sb=float(sb),
        bb=float(bb),
        stake=float(stake),
        seed=int(seed),
    )
    if int(root.current_player) != int(traversing_player):
        raise ValueError(
            "preflop fold oracle ожидает, что traversing_player совпадает с current_player; "
            f"current={root.current_player}, traversing={traversing_player}"
        )
    full_mask = legal_action_mask(root)
    if full_mask[int(ActionSlot.FOLD)] != 1.0:
        raise ValueError("Fold должен быть legal в выбранном root state")
    fold_action = agent.action_type_to_pokers_action(int(ActionSlot.FOLD), root)
    terminal = root.apply_action(fold_action)
    expected_rewards = _blind_fold_expected_rewards(root, int(root.current_player))
    actual_rewards = [float(player.reward) for player in terminal.players_state]
    engine_errors = [abs(actual - expected) for actual, expected in zip(actual_rewards, expected_rewards)]

    scale = float(getattr(agent, "advantage_reward_scale", 1.0))
    if str(getattr(agent, "advantage_regret_norm", "none")) != "none":
        raise ValueError("engine oracle сейчас поддерживает только advantage_regret_norm=none")
    encoded = np.zeros(int(agent.input_size), dtype=np.float32)
    encoded[0] = 1.0
    fold_mask = _single_slot_mask(int(ActionSlot.FOLD))
    expected_q = np.zeros(NUM_ACTIONS, dtype=np.float32)
    expected_q[int(ActionSlot.FOLD)] = np.float32(expected_rewards[int(traversing_player)] / scale)
    expected_v = np.float32(expected_rewards[int(traversing_player)] / scale)
    expected_r = np.zeros(NUM_ACTIONS, dtype=np.float32)

    agent_any = cast(Any, agent)
    agent_any.advantage_net = FixedAdvantageNet(np.zeros(NUM_ACTIONS, dtype=np.float32)).to(agent.device)
    agent_any.get_legal_action_mask = lambda state: fold_mask.copy()
    agent_any._encode_state = lambda state, player_id: encoded.copy()
    agent.reset_traversal_stats()
    root_return = float(agent.cfr_traverse_multi(root, int(iteration), int(traversing_player)))
    _, action_values, state_value, regrets, recorded_mask, recorded_iteration = _first_d2cfr_sample(agent)

    target_errors = {
        "action_value": _max_legal_error(action_values, expected_q, fold_mask),
        "state_value": abs(float(state_value) - float(expected_v)),
        "regret": _max_legal_error(regrets, expected_r, fold_mask),
        "mask": float(np.max(np.abs(recorded_mask - fold_mask))),
        "iteration": abs(float(recorded_iteration) - float(iteration)),
        "root_return": abs(float(root_return) - float(expected_rewards[int(traversing_player)])),
    }
    passed = (
        bool(terminal.final_state)
        and terminal.status == pkrs.StateStatus.Ok
        and max(engine_errors) <= 1e-6
        and all(value <= 1e-6 for value in target_errors.values())
    )
    return {
        "mode": "pokers_preflop_fold_oracle",
        "passed": bool(passed),
        "config": str(config_path) if config_path is not None else None,
        "root": {
            "seed": int(seed),
            "button": int(button),
            "sb": float(sb),
            "bb": float(bb),
            "stake": float(stake),
            "current_player": int(root.current_player),
            "pot": float(root.pot),
            "legal_mask": full_mask.astype(np.float32).tolist(),
            "fold_action": repr(fold_action),
        },
        "engine": {
            "terminal": bool(terminal.final_state),
            "status": str(terminal.status),
            "expected_rewards": expected_rewards,
            "actual_rewards": actual_rewards,
            "reward_abs_errors": engine_errors,
        },
        "target": {
            "root_return": root_return,
            "expected": {
                "action_values": expected_q.astype(np.float32).tolist(),
                "state_value": float(expected_v),
                "regrets": expected_r.astype(np.float32).tolist(),
                "mask": fold_mask.astype(np.float32).tolist(),
            },
            "recorded": {
                "action_values": action_values.astype(np.float32).tolist(),
                "state_value": float(state_value),
                "regrets": regrets.astype(np.float32).tolist(),
                "mask": recorded_mask.astype(np.float32).tolist(),
                "iteration": float(recorded_iteration),
            },
            "max_abs_errors": target_errors,
        },
        "traversal_stats": agent.get_traversal_stats(),
    }


def run_river_showdown_oracle(
    *,
    config_path: str | Path | None,
    traversing_player: int,
    iteration: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Проверяет river check-check showdown и D2CFR target на настоящем engine state."""
    agent = _make_agent(config_path, traversing_player, device)
    if not agent.d2cfr_enabled:
        raise ValueError("engine oracle требует d2cfr_enabled=true")
    root = _hu_river_known_winner_state()
    if int(root.current_player) != int(traversing_player):
        raise ValueError("showdown oracle ожидает current_player == traversing_player")

    check_action = cast(pkrs.Action, agent.action_type_to_pokers_action(int(ActionSlot.CHECK), root))
    after_hero_check = root.apply_action(check_action)
    opponent_check = cast(pkrs.Action, pkrs.Action(pkrs.ActionEnum.Check))
    terminal = after_hero_check.apply_action(opponent_check)
    expected_rewards = [10.0, -10.0]
    actual_rewards = [float(player.reward) for player in terminal.players_state]
    engine_errors = [abs(actual - expected) for actual, expected in zip(actual_rewards, expected_rewards)]

    scale = float(getattr(agent, "advantage_reward_scale", 1.0))
    if str(getattr(agent, "advantage_regret_norm", "none")) != "none":
        raise ValueError("engine oracle сейчас поддерживает только advantage_regret_norm=none")
    encoded = np.zeros(int(agent.input_size), dtype=np.float32)
    encoded[0] = 1.0
    check_mask = _single_slot_mask(int(ActionSlot.CHECK))
    expected_q = np.zeros(NUM_ACTIONS, dtype=np.float32)
    expected_q[int(ActionSlot.CHECK)] = np.float32(expected_rewards[int(traversing_player)] / scale)
    expected_v = np.float32(expected_rewards[int(traversing_player)] / scale)
    expected_r = np.zeros(NUM_ACTIONS, dtype=np.float32)

    agent_any = cast(Any, agent)
    agent_any.advantage_net = FixedAdvantageNet(np.zeros(NUM_ACTIONS, dtype=np.float32)).to(agent.device)
    agent_any.get_legal_action_mask = lambda state: check_mask.copy()
    agent_any._encode_state = lambda state, player_id: encoded.copy()
    agent.reset_traversal_stats()
    root_return = float(agent.cfr_traverse_multi(root, int(iteration), int(traversing_player)))
    _, action_values, state_value, regrets, recorded_mask, recorded_iteration = _first_d2cfr_sample(agent)

    target_errors = {
        "action_value": _max_legal_error(action_values, expected_q, check_mask),
        "state_value": abs(float(state_value) - float(expected_v)),
        "regret": _max_legal_error(regrets, expected_r, check_mask),
        "mask": float(np.max(np.abs(recorded_mask - check_mask))),
        "iteration": abs(float(recorded_iteration) - float(iteration)),
        "root_return": abs(float(root_return) - float(expected_rewards[int(traversing_player)])),
    }
    passed = (
        bool(terminal.final_state)
        and terminal.status == pkrs.StateStatus.Ok
        and max(engine_errors) <= 1e-6
        and all(value <= 1e-6 for value in target_errors.values())
    )
    return {
        "mode": "pokers_river_showdown_oracle",
        "passed": bool(passed),
        "config": str(config_path) if config_path is not None else None,
        "root": {
            "stage": str(root.stage),
            "public_cards": [repr(card) for card in root.public_cards],
            "current_player": int(root.current_player),
            "pot": float(root.pot),
            "legal_mask": legal_action_mask(root).astype(np.float32).tolist(),
        },
        "engine": {
            "terminal": bool(terminal.final_state),
            "status": str(terminal.status),
            "expected_rewards": expected_rewards,
            "actual_rewards": actual_rewards,
            "reward_abs_errors": engine_errors,
        },
        "target": {
            "root_return": root_return,
            "expected": {
                "action_values": expected_q.astype(np.float32).tolist(),
                "state_value": float(expected_v),
                "regrets": expected_r.astype(np.float32).tolist(),
                "mask": check_mask.astype(np.float32).tolist(),
            },
            "recorded": {
                "action_values": action_values.astype(np.float32).tolist(),
                "state_value": float(state_value),
                "regrets": regrets.astype(np.float32).tolist(),
                "mask": recorded_mask.astype(np.float32).tolist(),
                "iteration": float(recorded_iteration),
            },
            "max_abs_errors": target_errors,
        },
        "traversal_stats": agent.get_traversal_stats(),
    }


def run_river_allin_response_oracle(
    *,
    config_path: str | Path | None,
    traversing_player: int,
    iteration: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Проверяет exact multi-branch Q/V/R для real river fold/call spot."""
    agent = _make_agent(config_path, traversing_player, device)
    if not agent.d2cfr_enabled:
        raise ValueError("engine oracle требует d2cfr_enabled=true")
    root = _hu_river_allin_response_state()
    if int(root.current_player) != int(traversing_player):
        raise ValueError("all-in response oracle ожидает current_player == traversing_player")
    full_mask = legal_action_mask(root)
    expected_mask = _single_slot_mask(int(ActionSlot.FOLD)) + _single_slot_mask(int(ActionSlot.CALL))
    if not np.array_equal(full_mask, expected_mask):
        raise ValueError(f"Ожидались только fold/call, получена mask={full_mask.tolist()}")

    fold_action = agent.action_type_to_pokers_action(int(ActionSlot.FOLD), root)
    call_action = agent.action_type_to_pokers_action(int(ActionSlot.CALL), root)
    fold_terminal = root.apply_action(cast(pkrs.Action, fold_action))
    call_terminal = root.apply_action(cast(pkrs.Action, call_action))
    raw_action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
    raw_action_values[int(ActionSlot.FOLD)] = np.float32(
        fold_terminal.players_state[int(traversing_player)].reward
    )
    raw_action_values[int(ActionSlot.CALL)] = np.float32(
        call_terminal.players_state[int(traversing_player)].reward
    )

    advantages = np.zeros(NUM_ACTIONS, dtype=np.float32)
    advantages[int(ActionSlot.FOLD)] = 1.0
    advantages[int(ActionSlot.CALL)] = 3.0
    strategy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    strategy[int(ActionSlot.FOLD)] = 0.25
    strategy[int(ActionSlot.CALL)] = 0.75
    raw_state_value = float(
        strategy[int(ActionSlot.FOLD)] * raw_action_values[int(ActionSlot.FOLD)]
        + strategy[int(ActionSlot.CALL)] * raw_action_values[int(ActionSlot.CALL)]
    )
    raw_regrets = (raw_action_values - np.float32(raw_state_value)) * expected_mask

    scale = float(getattr(agent, "advantage_reward_scale", 1.0))
    if str(getattr(agent, "advantage_regret_norm", "none")) != "none":
        raise ValueError("engine oracle сейчас поддерживает только advantage_regret_norm=none")
    expected_q = (raw_action_values / scale).astype(np.float32)
    expected_v = np.float32(raw_state_value / scale)
    expected_r = (raw_regrets / scale).astype(np.float32)

    agent_any = cast(Any, agent)
    agent_any.advantage_net = FixedAdvantageNet(advantages).to(agent.device)
    encoded = np.zeros(int(agent.input_size), dtype=np.float32)
    encoded[0] = 1.0
    agent_any._encode_state = lambda state, player_id: encoded.copy()
    agent.reset_traversal_stats()
    root_return = float(agent.cfr_traverse_multi(root, int(iteration), int(traversing_player)))
    _, action_values, state_value, regrets, recorded_mask, recorded_iteration = _first_d2cfr_sample(agent)

    max_abs_errors = {
        "action_value": _max_legal_error(action_values, expected_q, expected_mask),
        "state_value": abs(float(state_value) - float(expected_v)),
        "regret": _max_legal_error(regrets, expected_r, expected_mask),
        "mask": float(np.max(np.abs(recorded_mask - expected_mask))),
        "iteration": abs(float(recorded_iteration) - float(iteration)),
        "root_return": abs(float(root_return) - raw_state_value),
    }
    passed = (
        bool(fold_terminal.final_state)
        and bool(call_terminal.final_state)
        and fold_terminal.status == pkrs.StateStatus.Ok
        and call_terminal.status == pkrs.StateStatus.Ok
        and all(value <= 1e-6 for value in max_abs_errors.values())
    )
    return {
        "mode": "pokers_river_allin_response_oracle",
        "passed": bool(passed),
        "config": str(config_path) if config_path is not None else None,
        "root": {
            "stage": str(root.stage),
            "current_player": int(root.current_player),
            "pot": float(root.pot),
            "legal_mask": full_mask.astype(np.float32).tolist(),
        },
        "branches": {
            "fold_rewards": [float(player.reward) for player in fold_terminal.players_state],
            "call_rewards": [float(player.reward) for player in call_terminal.players_state],
        },
        "expected": {
            "strategy": strategy.astype(np.float32).tolist(),
            "raw_action_values": raw_action_values.astype(np.float32).tolist(),
            "raw_state_value": raw_state_value,
            "raw_regrets": raw_regrets.astype(np.float32).tolist(),
            "action_values": expected_q.astype(np.float32).tolist(),
            "state_value": float(expected_v),
            "regrets": expected_r.astype(np.float32).tolist(),
            "mask": expected_mask.astype(np.float32).tolist(),
        },
        "recorded": {
            "action_values": action_values.astype(np.float32).tolist(),
            "state_value": float(state_value),
            "regrets": regrets.astype(np.float32).tolist(),
            "mask": recorded_mask.astype(np.float32).tolist(),
            "iteration": float(recorded_iteration),
        },
        "root_return": root_return,
        "max_abs_errors": max_abs_errors,
        "traversal_stats": agent.get_traversal_stats(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--button", type=int, default=0)
    parser.add_argument("--sb", type=float, default=1.0)
    parser.add_argument("--bb", type=float, default=2.0)
    parser.add_argument("--stake", type=float, default=200.0)
    parser.add_argument("--traversing-player", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("preflop-fold", "river-showdown", "river-allin-response"),
        default="preflop-fold",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "river-showdown":
        report = run_river_showdown_oracle(
            config_path=args.config,
            traversing_player=args.traversing_player,
            iteration=args.iteration,
            device=args.device,
        )
    elif args.mode == "river-allin-response":
        report = run_river_allin_response_oracle(
            config_path=args.config,
            traversing_player=args.traversing_player,
            iteration=args.iteration,
            device=args.device,
        )
    else:
        report = run_preflop_fold_oracle(
            config_path=args.config,
            seed=args.seed,
            button=args.button,
            sb=args.sb,
            bb=args.bb,
            stake=args.stake,
            traversing_player=args.traversing_player,
            iteration=args.iteration,
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
