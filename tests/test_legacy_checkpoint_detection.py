from pathlib import Path

import torch

from src.agents.legacy_starting_opponent import is_legacy_starting_opponent_checkpoint
from scripts.poker_gui import create_playing_agent


CHECKPOINT = (
    Path(__file__).parents[1]
    / "models"
    / "starting_opponent"
    / "mixed_checkpoint_iter_11200.pt"
)


def test_detects_starting_opponent_checkpoint_by_legacy_network_shape():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)

    assert is_legacy_starting_opponent_checkpoint(checkpoint)


def test_does_not_treat_six_fixed_checkpoint_as_legacy_when_legacy_keys_are_absent():
    checkpoint = {
        "strategy_net": {
            "base.0.weight": torch.zeros((256, 157)),
            "action_head.weight": torch.zeros((6, 256)),
        }
    }

    assert not is_legacy_starting_opponent_checkpoint(checkpoint)


def test_gui_creates_legacy_adapter_for_starting_opponent_checkpoint():
    opponent = create_playing_agent(CHECKPOINT, player_id=1, device="cpu")

    assert opponent.__class__.__name__ == "LegacyStartingOpponent"
