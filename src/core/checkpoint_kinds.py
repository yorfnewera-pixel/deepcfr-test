"""Общие идентификаторы публичных форматов checkpoint без зависимостей runtime."""

STRATEGY_ONLY_CHECKPOINT_KIND = "strategy_only"
HU_STRATEGY_ONLY_CHECKPOINT_KIND = "hu_strategy_only"
HU_CURRENT_POLICY_SELF_PLAY_CHECKPOINT_KIND = "hu_current_policy_self_play"
HU_FULL_CHECKPOINT_VERSION = 4


__all__ = [
    "HU_CURRENT_POLICY_SELF_PLAY_CHECKPOINT_KIND",
    "HU_FULL_CHECKPOINT_VERSION",
    "HU_STRATEGY_ONLY_CHECKPOINT_KIND",
    "STRATEGY_ONLY_CHECKPOINT_KIND",
]
