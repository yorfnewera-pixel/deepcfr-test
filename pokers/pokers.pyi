from typing import Optional
from enum import Enum

# visualization.rs ------------------------------------------------------------
def visualize_state(state: State) -> str: ...
def visualize_trace(trace: list[State]) -> str: ...

# parallel.rs -----------------------------------------------------------------
def parallel_apply_action(
    states: list[State], actions: list[Action]
) -> list[State]: ...

def compare_showdown(
    hero: tuple[Card, Card], opponent: tuple[Card, Card], board: list[Card]
) -> int: ...

# state.rs --------------------------------------------------------------------

class State:
    current_player: int
    players_state: list[PlayerState]
    public_cards: list[Card]
    stage: Stage
    button: int
    from_action: Optional[ActionRecord]
    action_history: list[PublicActionRecord]
    action_history_complete: bool
    legal_actions: list[ActionEnum]
    deck: list[Card]
    pot: float
    min_bet: float
    final_state: bool
    status: StateStatus
    verbose: bool
    bb: float
    ante: float
    min_raise: float
    chip_unit: float

    @staticmethod
    def from_seed(
        n_players: int,
        button: int,
        sb: float,
        bb: float,
        stake: float,
        seed: int,
        verbose: bool = False,
        chip_unit: float = 0.01,
        stakes: Optional[list[float]] = None,
        ante: float = 0.0,
    ) -> State: ...
    @staticmethod
    def from_deck(
        n_players: int,
        button: int,
        sb: float,
        bb: float,
        stake: float,
        deck: list[Card],
        verbose: bool = False,
        chip_unit: float = 0.01,
        stakes: Optional[list[float]] = None,
        ante: float = 0.0,
    ) -> State: ...
    def apply_action(self, action: Action) -> State: ...
    def __str__(self) -> str: ...

class PlayerState:
    player: int
    hand: tuple[Card, Card]
    bet_chips: float
    pot_chips: float
    stake: float
    reward: float
    active: bool
    def __str__(self) -> str: ...

class StateStatus(Enum):
    Ok = 0
    IllegalAction = 1
    LowBet = 2
    HighBet = 3
    InvalidAmount = 4

    def __int__(self): ...

# action.rs -------------------------------------------------------------------

class ActionRecord:
    player: int
    stage: Stage
    action: Action
    legal_actions: list[ActionEnum]

class PublicActionRecord:
    actor_id: int
    street: Stage
    requested_action: Action
    paid_amount: float
    applied_raise_increment: float
    is_effective_raise: bool

class ActionEnum(Enum):
    Fold = 0
    Check = 1
    Call = 2
    Raise = 3

    def __int__(self): ...

class Action:
    action: ActionEnum
    amount: float
    def __new__(cls, action: ActionEnum, amount: float = 0) -> None: ...

# card.rs ---------------------------------------------------------------------

class Card:
    suit: CardSuit
    rank: CardRank
    @staticmethod
    def from_string(string: str) -> Card | None: ...
    @staticmethod
    def collect() -> list[Card]: ...

class CardSuit(Enum):
    Clubs = 0
    Diamonds = 1
    Hearts = 2
    Spades = 3

    def __int__(self): ...

class CardRank(Enum):
    R2 = 0
    R3 = 1
    R4 = 2
    R5 = 3
    R6 = 4
    R7 = 5
    R8 = 6
    R9 = 7
    RT = 8
    RJ = 9
    RQ = 10
    RK = 11
    RA = 12

    def __int__(self): ...

# stage.rs --------------------------------------------------------------------

class Stage(Enum):
    Preflop = 0
    Flop = 1
    Turn = 2
    River = 3
    Showdown = 4

    def __int__(self): ...
