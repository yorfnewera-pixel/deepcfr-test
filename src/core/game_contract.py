"""Неизменяемый игровой контракт активного HU-обучения."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Final


@dataclass(frozen=True)
class HuGameContract:
    """Фиксированные ставки и глубина для HU current-policy self-play."""

    small_blind: float = 1.0
    big_blind: float = 2.0
    starting_stack: float = 200.0

    @property
    def stack_depth_bb(self) -> float:
        return self.starting_stack / self.big_blind

    def metadata(self) -> dict[str, float]:
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "stack_depth_bb": self.stack_depth_bb,
        }

    def description(self) -> str:
        return (
            f"SB={self.small_blind:g}, BB={self.big_blind:g}, "
            f"stack={self.starting_stack:g} ({self.stack_depth_bb:g} BB)"
        )


FIXED_HU_GAME_CONTRACT: Final = HuGameContract()


def validate_fixed_hu_game_contract(metadata: object, artifact_name: str) -> None:
    """Не допускает resume или evaluation с неизвестными параметрами HU-игры."""
    expected_metadata = FIXED_HU_GAME_CONTRACT.metadata()
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected_metadata):
        raise _incompatible_game_contract_error(artifact_name)
    for key, expected_value in expected_metadata.items():
        value = metadata[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value != expected_value
        ):
            raise _incompatible_game_contract_error(artifact_name)


def _incompatible_game_contract_error(artifact_name: str) -> ValueError:
    return ValueError(
        f"{artifact_name} имеет несовместимый игровой контракт; "
        f"поддерживается только {FIXED_HU_GAME_CONTRACT.description()}"
    )
