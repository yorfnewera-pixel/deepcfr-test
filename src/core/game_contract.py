"""Неизменяемый игровой контракт активного HU-обучения."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class HuGameContract:
    """Фиксированные ставки и глубина для HU current-policy self-play."""

    small_blind: float = 1.0
    big_blind: float = 2.0
    starting_stack: float = 200.0

    @property
    def depth_big_blinds(self) -> float:
        return self.starting_stack / self.big_blind

    def metadata(self) -> dict[str, float]:
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "depth_big_blinds": self.depth_big_blinds,
        }

    def description(self) -> str:
        return (
            f"SB={self.small_blind:g}, BB={self.big_blind:g}, "
            f"stack={self.starting_stack:g} ({self.depth_big_blinds:g} BB)"
        )


FIXED_HU_GAME_CONTRACT: Final = HuGameContract()


def validate_fixed_hu_game_contract(metadata: object, artifact_name: str) -> None:
    """Не допускает resume или evaluation с неизвестными параметрами HU-игры."""
    if metadata != FIXED_HU_GAME_CONTRACT.metadata():
        raise ValueError(
            f"{artifact_name} имеет несовместимый игровой контракт; "
            f"поддерживается только {FIXED_HU_GAME_CONTRACT.description()}"
        )
