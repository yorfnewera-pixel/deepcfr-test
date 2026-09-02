"""Адаптер pokers.State к независимому policy runtime."""

import pokers as pkrs

from policy_runtime.core import PolicyRuntimeAgent
from src.core.action_space import ACTION_LABELS, resolve_action


class PokersPlayerState:
    __slots__ = ("hand", "active", "bet_chips", "pot_chips", "stake", "last_stage_action")

    def __init__(self, pokers_ps):
        self.hand = pokers_ps.hand
        self.active = pokers_ps.active
        self.bet_chips = float(pokers_ps.bet_chips)
        self.pot_chips = float(pokers_ps.pot_chips)
        self.stake = float(pokers_ps.stake)
        self.last_stage_action = getattr(pokers_ps, "last_stage_action", None)


class PokersGameState:
    __slots__ = (
        "bb", "pot", "min_bet", "button", "current_player", "stage",
        "legal_actions", "public_cards", "from_action", "players_state",
        "last_raise_increment",
    )

    def __init__(self, state, player_id=0, bb=2.0):
        self.bb = float(getattr(state, "bb", bb))
        self.pot = float(state.pot)
        self.min_bet = float(state.min_bet)
        self.button = int(state.button)
        self.current_player = int(state.current_player)
        self.stage = int(state.stage)
        self.legal_actions = list(state.legal_actions)
        self.public_cards = list(state.public_cards)
        self.from_action = getattr(state, "from_action", None)
        self.players_state = [PokersPlayerState(ps) for ps in state.players_state]
        if getattr(state, "last_raise_increment", 0):
            self.last_raise_increment = float(state.last_raise_increment)
        elif self.from_action is not None and int(self.from_action.action.action) == 3:
            self.last_raise_increment = float(self.from_action.action.amount)
        else:
            self.last_raise_increment = self.bb


def wrap_state(pokers_state, player_id: int = 0, bb: float = 2.0) -> PokersGameState:
    return PokersGameState(pokers_state, player_id, bb=bb)


def action_to_pokers(action_type: int, state: pkrs.State) -> pkrs.Action:
    """Преобразует только легальный фиксированный слот без изменения его смысла."""
    return resolve_action(action_type, state).action


def play_hand(
    agent: PolicyRuntimeAgent,
    seed: int = 0,
    player_id: int = 0,
    deterministic: bool = False,
):
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=seed
    )

    step = 0
    while not state.final_state:
        cp = state.current_player
        if cp == player_id:
            action_type = agent.choose_action(
                wrap_state(state, player_id, bb=2.0),
                player_id=player_id,
                deterministic=deterministic,
            )
            action = action_to_pokers(action_type, state)
            print(f"  Шаг {step}: AI -> {ACTION_LABELS[action_type]}")
        elif pkrs.ActionEnum.Check in state.legal_actions:
            action = pkrs.Action(pkrs.ActionEnum.Check)
        elif pkrs.ActionEnum.Call in state.legal_actions:
            action = pkrs.Action(pkrs.ActionEnum.Call)
        else:
            action = pkrs.Action(pkrs.ActionEnum.Fold)

        state = state.apply_action(action)
        step += 1

    reward = state.players_state[player_id].reward
    print(f"\n  Итог: reward={reward:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep CFR Policy Runtime (pokers)")
    parser.add_argument("checkpoint", help="Путь к checkpoint формата six_fixed_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    agent = PolicyRuntimeAgent(args.checkpoint)
    for error in agent.validate():
        print(f"  Валидация: {error}")
    agent.export_spec(args.checkpoint.replace(".pt", "_spec.json"))
    play_hand(agent, seed=args.seed, player_id=args.player, deterministic=args.deterministic)
