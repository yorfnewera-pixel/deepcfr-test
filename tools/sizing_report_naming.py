"""Shared naming helpers for sizing diagnostic reports."""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ITERATION_KEYS = ("iteration", "checkpoint_iteration", "iter", "current_iteration")
_ITERATION_PATTERN = re.compile(r"(?:^|[_-])iter_?(\d+)(?=$|[_\-.])", re.IGNORECASE)


def extract_iteration(metadata: Any = None, source: str | Path | None = None) -> int | None:
    """Prefer checkpoint/report metadata, then infer an iteration from a filename."""
    if isinstance(metadata, Mapping):
        for key in _ITERATION_KEYS:
            value = metadata.get(key)
            if value is not None and not isinstance(value, bool):
                try:
                    iteration = int(value)
                except (TypeError, ValueError):
                    continue
                if iteration >= 0:
                    return iteration
        nested = metadata.get("metadata")
        if isinstance(nested, Mapping):
            iteration = extract_iteration(nested)
            if iteration is not None:
                return iteration

    if source is not None:
        match = _ITERATION_PATTERN.search(Path(source).name)
        if match:
            return int(match.group(1))
    return None


def report_path(
    default_path: str | Path,
    *,
    iteration: int | None,
    explicit_output: str | Path | None = None,
) -> Path:
    """Return an explicit path unchanged, otherwise add ``_iterN`` to the default stem."""
    if explicit_output is not None:
        return Path(explicit_output)
    path = Path(default_path)
    if iteration is None:
        return path
    stem = re.sub(r"_iter\d+$", "", path.stem, flags=re.IGNORECASE)
    return path.with_name(f"{stem}_iter{iteration}{path.suffix}")
