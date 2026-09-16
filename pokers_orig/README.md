# Pokers

[![CI](https://github.com/dberweger2017/pokers/actions/workflows/CI.yml/badge.svg)](https://github.com/dberweger2017/pokers/actions/workflows/CI.yml)

A small Rust no-limit Texas Hold'em simulator with Python bindings. Apply an action to a state to get a new state; the original remains available for another branch.

This is the maintained fork used by [the Deep CFR poker project](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players), based on [Reinforcement-Poker/pokers](https://github.com/Reinforcement-Poker/pokers). Version 0.2 corrects betting, all-ins, side pots, heads-up order, and hand ranking, and uses integer chip accounting.

Read [RULES.md](RULES.md) for the supported cash-game profile, breaking changes, reference comparisons, and remaining scope. The engine handles a single hand with 2–10 players; table seating and changes between hands belong in a session layer.

## Installation

Use Python 3.10 or 3.11 and a Rust toolchain with the current PyO3 binding. This fork is installed from Git, not the upstream PyPI package:

```bash
pip install "pokers @ git+https://github.com/dberweger2017/pokers.git@main"
```

Applications and training manifests should pin a full commit hash instead of `main`.

## Usage

```python
import pokers

state = pokers.State.from_seed(
    n_players=6, button=0, sb=1, bb=2, stake=200, seed=42, chip_unit=1,
)
# Seat 3 opens to 10: call 2, then raise by 8.
next_state = state.apply_action(pokers.Action(pokers.ActionEnum.Raise, 8))
assert next_state.status == pokers.StateStatus.Ok
assert next_state.min_bet == 10
assert next_state.min_raise == 8  # Next minimum raise-to: 18.
```

`stakes=[...]` overrides the common starting stack with one amount per seat. `State.from_deck` accepts a full, distinct 52-card deck for replaying a deal. [pokers.pyi](pokers.pyi) describes the Python interface.

The simulator state contains **all private cards and the undealt deck**. A playing agent needs a separate player-observation interface. Do not pass this state directly to a policy that must only receive human-visible information.

### Actions and errors

`Raise.amount` is the extra amount after calling, not the absolute raise-to total. Only exact multiples of `chip_unit` are accepted. `min_raise` is the last full increment; a smaller raise is legal only as an exact all-in.

`legal_actions` lists action types. Calls and checks are distinct. The engine validates raise amounts when applying them. Invalid actions produce a terminal error state with status `IllegalAction`, `LowBet`, `HighBet`, or `InvalidAmount`, and move no chips. Check `status` before treating any terminal state as a completed hand. Applying another action to an already terminal state leaves it unchanged.

Final stacks include winnings and refunds; reward is final minus starting stack. The pot and committed amounts are zero after settlement.

### Batches and inspection

`pokers.parallel_apply_action(states, actions)` applies one action per state using Rayon and preserves input order. Different list lengths are rejected. Completed states can remain in a batch while other hands finish.

`pokers.visualize_state(state)` and `pokers.visualize_trace(states)` return diagnostic text, including privileged simulator information.

## Development

```bash
pip install ".[dev]"
cargo fmt --check
cargo test --locked
pytest -q
POKERS_REFERENCE_SEEDS=1000 pytest tests/test_rules.py -q
```

Set `PYO3_PYTHON` to the virtual environment's interpreter before running Rust tests if the machine has multiple Python installations. The PokerKit comparison requires Python 3.11; core and fixture tests also run on 3.10.

Tests include exact rule scenarios, generated hands with unequal stacks, and all 9,908 bundled Pluribus hands in serial and parallel. [RULES.md](RULES.md#verification) documents the reference differences and what these checks establish.
