"""CLI для offline построения HU card abstraction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Скрипт поддерживает запуск через ``python tools/...`` из корня проекта.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.card_abstraction.pipeline import build_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Строит HU card abstraction offline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        build_all(arguments.output, sample_count=5, holdout_count=3, clusters=2)
        return 0
    build_all(arguments.output, sample_count=100_000, holdout_count=20_000, clusters=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
