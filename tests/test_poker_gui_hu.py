"""Регрессии двухместного режима графического клиента."""
from __future__ import annotations

import random

import numpy as np

from scripts.poker_gui import (
    create_game_state,
    is_policy_runtime_checkpoint,
    parse_arguments,
    preserve_application_rng,
    table_player_ids,
)


def test_hu_table_shows_only_human_and_one_opponent() -> None:
    """Падёт, если HU-режим случайно создаст дополнительные места."""

    player_ids = table_player_ids(2)

    assert player_ids == (1, 0)


def test_hu_gui_starts_hand_with_fixed_two_player_contract() -> None:
    """Падёт, если GUI передаст HU-чекпоинту 6-max или другой стек."""
    state = create_game_state(num_players=2, seed=41, stake=1_000.0, sb=5.0, bb=10.0)

    assert len(state.players_state) == 2
    assert state.sb == 1.0
    assert state.bb == 2.0
    assert all(player.stake + player.bet_chips == 200.0 for player in state.players_state)


def test_hu_flag_selects_two_player_launch_mode(monkeypatch) -> None:
    """Падёт, если командная строка не может включить HU GUI."""
    monkeypatch.setattr("sys.argv", ["poker_gui", "--hu"])

    args = parse_arguments()

    assert args.hu is True


def test_advantage_flag_selects_regret_matching_mode(monkeypatch) -> None:
    """Падёт, если GUI перестанет принимать явный выбор advantage policy."""
    monkeypatch.setattr("sys.argv", ["poker_gui", "--hu", "--policy-source", "advantage"])

    args = parse_arguments()

    assert args.policy_source == "advantage"


def test_checkpoint_load_does_not_reset_gui_random_sequence() -> None:
    """Падёт, если загрузка checkpoint снова задаст одинаковую первую раздачу GUI."""
    random.seed(123)
    np.random.seed(456)
    expected_python = random.Random(123).randint(0, 10_000)
    expected_numpy = np.random.RandomState(456).randint(0, 10_000)

    def loader() -> None:
        random.seed(1)
        np.random.seed(2)

    preserve_application_rng(loader)

    assert random.randint(0, 10_000) == expected_python
    assert np.random.randint(0, 10_000) == expected_numpy


def test_hu_light_checkpoint_uses_policy_runtime_without_monolithic_weights() -> None:
    """Падёт, если GUI отправит card-context HU light в legacy-загрузчик."""
    checkpoint = {
        "checkpoint_kind": "hu_strategy_only",
        "strategy_net": {"card_encoder.0.weight": object()},
    }

    assert is_policy_runtime_checkpoint(checkpoint) is True
