"""Случайный оппонент в том же дискретном пространстве действий."""

import random

from src.core.action_space import legal_action_mask, resolve_action


class RandomAgent:
    def __init__(self, player_id):
        self.player_id = int(player_id)
        self.name = f"RandomAgent_{self.player_id}"

    def choose_action(self, state):
        slots = legal_action_mask(state).nonzero()[0].tolist()
        if not slots:
            raise ValueError("У случайного агента нет легального действия")
        return resolve_action(random.choice(slots), state).action
