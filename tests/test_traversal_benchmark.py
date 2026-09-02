"""Контракты воспроизводимого traversal benchmark без тяжёлого checkpoint-прогона."""

from __future__ import annotations

from pathlib import Path

from tools.benchmark_traversal import (
    REQUIRED_METRICS,
    RootSeed,
    build_metrics,
    generate_root_bank,
    load_root_bank,
    run_roots,
    save_root_bank,
)


def test_root_bank_generation_is_deterministic() -> None:
    assert generate_root_bank(4, seed=1) == generate_root_bank(4, seed=1)


def test_root_bank_round_trip_preserves_contents(tmp_path: Path) -> None:
    original = [RootSeed(button=4, deal_seed=381294), RootSeed(button=0, deal_seed=42)]
    path = tmp_path / "roots.json"

    save_root_bank(path, original)

    assert load_root_bank(path) == original


def test_metrics_contain_required_fields() -> None:
    metrics = build_metrics(
        roots=2,
        wall_seconds=0.5,
        stats={
            "nodes": 10,
            "terminal_nodes": 4,
            "traversing_decision_nodes": 2,
            "opponent_decision_nodes": 4,
        },
        rss_before_mb=10.0,
        rss_after_mb=11.0,
        profile={},
    )

    assert set(REQUIRED_METRICS).issubset(metrics)
    assert metrics["decision_nodes"] == 6
    assert metrics["microseconds_per_node"] == 50_000.0


def test_run_roots_calls_only_traversal_api() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.states = []
            self.training_calls = 0

        def cfr_traverse_multi(self, state, iteration, traversing_player) -> None:
            self.states.append((state, iteration, traversing_player))

        def train(self) -> None:
            self.training_calls += 1

    agent = FakeAgent()
    roots = [RootSeed(button=1, deal_seed=10), RootSeed(button=2, deal_seed=20)]

    run_roots(
        agent,
        roots,
        iteration=3,
        traversing_player=0,
        state_factory=lambda root: (root.button, root.deal_seed),
    )

    assert agent.states == [((1, 10), 3, 0), ((2, 20), 3, 0)]
    assert agent.training_calls == 0
