"""Инварианты вложенных секций диагностического профайлера."""

from __future__ import annotations

import time

from src.utils.traversal_profiler import TraversalProfiler, profile_section


def test_profiler_nested_sections_preserve_total_and_self_time() -> None:
    profiler = TraversalProfiler(enabled=True, level="full")

    with profile_section(profiler, "parent"):
        time.sleep(0.001)
        with profile_section(profiler, "child"):
            time.sleep(0.001)

    parent = profiler.get_stats("parent")
    child = profiler.get_stats("child")
    profiler.assert_balanced()

    assert parent["total_ms"] >= child["total_ms"]
    assert parent["exclusive_ms"] >= 0.0
    assert parent["total_ms"] >= parent["exclusive_ms"] + child["total_ms"] - 0.2
