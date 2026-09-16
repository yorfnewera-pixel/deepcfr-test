"""Rules checks use chip-sized integers; Python amounts are table-unit projections."""

import random
import os
import sys

import pokers as p
import pytest

if sys.version_info >= (3, 11):
    from pokerkit import (
        Automation,
        ChipsPushing,
        Folding,
        Mode,
        NoLimitTexasHoldem,
        StandardHighHand,
    )


def state(n=6, stacks=None, seed=0, button=0, **kwargs):
    return p.State.from_seed(
        n, button, 1, 2, 200, seed, chip_unit=1, stakes=stacks, **kwargs
    )


def act(s, action, amount=0):
    result = s.apply_action(p.Action(action, amount))
    assert result.status == p.StateStatus.Ok
    return result


def test_minimum_raise_and_rejected_amounts():
    s = act(state(), p.ActionEnum.Raise, 8)
    assert s.min_bet == 10
    assert s.min_raise == 8
    assert act(s, p.ActionEnum.Raise, 8).min_bet == 18
    for amount in [0, 1, 7]:
        assert (
            s.apply_action(p.Action(p.ActionEnum.Raise, amount)).status
            == p.StateStatus.LowBet
        )
    for amount in [-1, float("nan"), float("inf"), 8.5]:
        assert (
            s.apply_action(p.Action(p.ActionEnum.Raise, amount)).status
            == p.StateStatus.InvalidAmount
        )
    assert (
        s.min_bet == 10
    )  # Branches, including rejected actions, leave the parent alone.


def test_heads_up_blinds_and_postflop_order():
    s = state(n=2)
    assert [ps.bet_chips for ps in s.players_state] == [1, 2]
    assert s.current_player == 0
    s = act(s, p.ActionEnum.Call)
    assert s.current_player == 1
    assert p.ActionEnum.Check in s.legal_actions
    s = act(s, p.ActionEnum.Check)
    assert s.stage == p.Stage.Flop
    assert s.current_player == 1
    assert s.min_raise == 2


def test_short_all_in_does_not_reopen_but_cumulative_raises_do():
    # Seat 3 raises to 10, seat 4 calls, seat 5 goes all-in for 14.
    s = state(stacks=[200, 200, 200, 200, 200, 14])
    for action, amount in [
        (p.ActionEnum.Raise, 8),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Raise, 4),
    ]:
        s = act(s, action, amount)
    for _ in range(3):
        s = act(s, p.ActionEnum.Call)
    assert s.current_player == 3
    assert p.ActionEnum.Raise not in s.legal_actions
    assert p.ActionEnum.Call in s.legal_actions

    s = state(stacks=[18, 200, 200, 200, 200, 14])
    for action, amount in [
        (p.ActionEnum.Raise, 8),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Raise, 4),
        (p.ActionEnum.Raise, 4),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Call, 0),
    ]:
        s = act(s, action, amount)
    assert s.current_player == 3
    assert p.ActionEnum.Raise in s.legal_actions
    assert s.min_raise == 8


def test_short_call_runs_out_without_negative_stacks():
    s = state(n=3, stacks=[200, 7, 20])
    s = act(s, p.ActionEnum.Raise, 198)
    s = act(s, p.ActionEnum.Call)
    s = act(s, p.ActionEnum.Call)
    assert s.final_state and s.stage == p.Stage.Showdown
    assert s.pot == 0
    assert sum(ps.stake for ps in s.players_state) == 227
    assert sum(ps.reward for ps in s.players_state) == 0
    assert min(ps.stake for ps in s.players_state) >= 0


def test_state_is_read_only_from_python():
    s = state()
    with pytest.raises(AttributeError):
        s.min_bet = 0
    with pytest.raises(AttributeError):
        s.players_state[0].stake = 1000


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_players=1),
        dict(n_players=11),
        dict(button=6),
        dict(sb=0),
        dict(bb=float("nan")),
        dict(stake=float("inf")),
        dict(chip_unit=0),
        dict(stakes=[200]),
        dict(stakes=[0] * 6),
    ],
)
def test_bad_setup_is_rejected(kwargs):
    args = dict(n_players=6, button=0, sb=1, bb=2, stake=200, seed=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        p.State.from_seed(**args)


def test_deck_must_be_complete_and_unique():
    deck = p.Card.collect()
    for cards in [deck[:-1], [deck[0]] * 52]:
        with pytest.raises(ValueError):
            p.State.from_deck(6, 0, 1, 2, 200, cards)


def card_string(card):
    return "23456789TJQKA"[int(card.rank)] + "cdhs"[int(card.suit)]


def reference(s, stacks):
    order = [(s.button + i + 1) % len(stacks) for i in range(len(stacks))]
    auto = (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.RUNOUT_COUNT_SELECTION,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )
    ref = NoLimitTexasHoldem.create_state(
        auto,
        True,
        0,
        (1, 2),
        2,
        [stacks[i] for i in order],
        len(stacks),
        mode=Mode.CASH_GAME,
    )
    for index, seat in enumerate(order):
        ref.deal_hole(
            "".join(card_string(c) for c in s.players_state[seat].hand), index
        )
    return ref, order


def finish_dealing(ref, board):
    while ref.status:
        if ref.actor_index is not None:
            able = sum(
                active and stack > 0 for active, stack in zip(ref.statuses, ref.stacks)
            )
            if able <= 1 and ref.checking_or_calling_amount == 0:
                ref.check_or_call()  # Forced check before an uncontested runout.
                continue
            break
        if ref.can_burn_card():
            ref.burn_card("??")
        elif ref.can_deal_board():
            count = ref.board_dealing_count
            ref.deal_board("".join(board.pop(0) for _ in range(count)))
        else:
            raise AssertionError(
                f"Unprocessed reference operation: {ref.operations[-1]}"
            )


def reference_stacks(ref, initial, stacks, order):
    """Use reference investments and hand values with this profile's odd chips.

    PokerKit merges pots after killing losing hands and gives all spare chips
    to one winner. Reconstruct pot boundaries before hand killing and distribute
    one spare per winner. All action/accounting data comes from the reference.
    """
    n = len(order)
    pushed = [
        sum(op.amounts[i] for op in ref.operations if isinstance(op, ChipsPushing))
        for i in range(n)
    ]
    invested = [
        stacks[seat] - ref.stacks[i] + pushed[i] for i, seat in enumerate(order)
    ]
    folded = {op.player_index for op in ref.operations if isinstance(op, Folding)}
    live = [i for i in range(n) if i not in folded]
    if len(live) == 1:
        return ref.stacks
    board = "".join(card_string(c) for c in initial.deck[:5])
    ranks = {
        i: StandardHighHand.from_game(
            "".join(card_string(c) for c in initial.players_state[order[i]].hand), board
        )
        for i in live
    }
    result = [stacks[seat] - invested[i] for i, seat in enumerate(order)]
    previous = 0
    for cap in sorted({invested[i] for i in live}):
        amount = sum(min(value, cap) - min(value, previous) for value in invested)
        previous = cap
        eligible = [i for i in live if invested[i] >= cap]
        best = max(ranks[i] for i in eligible)
        winners = [i for i in eligible if ranks[i] == best]
        share, spare = divmod(int(amount), len(winners))
        for place, i in enumerate(winners):
            result[i] += share + (place < spare)
    # Differences from PokerKit's direct settlement must be bounded rounding
    # differences, never a different pot winner or missing stack contribution.
    assert all(abs(a - b) < n for a, b in zip(result, ref.stacks))
    return result


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PokerKit requires Python 3.11")
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_generated_hands_against_pokerkit(n):
    for seed in range(int(os.environ.get("POKERS_REFERENCE_SEEDS", "100"))):
        rng = random.Random(seed * 10 + n)
        stacks = [rng.randint(1, 200) for _ in range(n)]
        # PokerKit prices a short big blind at the amount actually posted.
        # Keep the reference within the shared profile; test short blinds below.
        bb_seat = (seed % n + (1 if n == 2 else 2)) % n
        stacks[bb_seat] = max(2, stacks[bb_seat])
        s = state(n=n, stacks=stacks, seed=seed, button=seed % n)
        # Construct the reference from the original deal, before actions are chosen.
        initial = p.State.from_seed(n, seed % n, 1, 2, 200, seed, chip_unit=1)
        ref, order = reference(initial, stacks)
        board = [card_string(c) for c in initial.deck[:5]]
        trace = []
        for _ in range(500):
            finish_dealing(ref, board)
            context = (n, seed, stacks, trace)
            assert s.status == p.StateStatus.Ok, context
            assert s.final_state == (not ref.status), context
            if s.final_state:
                expected = reference_stacks(ref, initial, stacks, order)
                assert [s.players_state[i].stake for i in order] == expected, context
                assert [s.players_state[i].reward for i in order] == [
                    value - stacks[seat] for value, seat in zip(expected, order)
                ], context
                break
            assert s.current_player == order[ref.actor_index], context
            ps = s.players_state[s.current_player]
            assert [s.players_state[i].stake for i in order] == ref.stacks, context
            assert [s.players_state[i].bet_chips for i in order] == ref.bets, context
            can_raise = p.ActionEnum.Raise in s.legal_actions
            reference_can_raise = ref.can_complete_bet_or_raise_to()
            if can_raise != reference_can_raise:
                # PokerKit 0.7.5 includes an earlier full all-in in its cumulative
                # short-raise total, even for a caller who acted after that bet.
                # Rule 47 applies the increment faced by this specific player.
                # See RULES.md and the explicit full-all-in regression below.
                assert not can_raise and reference_can_raise, context
                assert 0 < s.min_bet - ps.bet_chips < s.min_raise, context
                earlier_full_all_in = (
                    max(ref.consecutive_all_in_completion_betting_or_raising_amounts)
                    >= s.min_raise
                )
                blind_increment_missing = (
                    s.stage == p.Stage.Preflop
                    and s.min_raise == s.bb
                    and ref.completion_betting_or_raising_amount < s.bb
                )
                assert earlier_full_all_in or blind_increment_missing, context
            assert (p.ActionEnum.Check in s.legal_actions) == (
                ref.checking_or_calling_amount == 0
            ), context
            if can_raise and reference_can_raise and rng.random() < 0.45:
                minimum = ref.min_completion_betting_or_raising_to_amount
                maximum = ref.max_completion_betting_or_raising_to_amount
                target = rng.choice([minimum, maximum, rng.randint(minimum, maximum)])
                amount = target - s.min_bet
                trace.append((s.current_player, "raise-to", target))
                ref.complete_bet_or_raise_to(target)
                s = act(s, p.ActionEnum.Raise, amount)
            elif ref.checking_or_calling_amount > 0 and rng.random() < 0.18:
                trace.append((s.current_player, "fold"))
                ref.fold()
                s = act(s, p.ActionEnum.Fold)
            else:
                trace.append((s.current_player, "call/check"))
                action = (
                    p.ActionEnum.Check
                    if ref.checking_or_calling_amount == 0
                    else p.ActionEnum.Call
                )
                ref.check_or_call()
                s = act(s, action)
        else:
            pytest.fail(f"Hand did not terminate: {context}")


def rigged(hands, board, stacks, button=0):
    n = len(hands)
    order = [(button + i + 1) % n for i in range(n)]
    cards = [hands[i][r] for r in range(2) for i in order] + board
    deck = [p.Card.from_string(c[1].upper() + c[0]) for c in cards]
    used = set(cards)
    deck += [c for c in p.Card.collect() if card_string(c) not in used]
    return p.State.from_deck(n, button, 1, 2, 200, deck, chip_unit=1, stakes=stacks)


def all_in(s):
    for _ in range(100):
        if s.final_state:
            return s
        ps = s.players_state[s.current_player]
        if p.ActionEnum.Raise in s.legal_actions:
            s = act(s, p.ActionEnum.Raise, ps.stake - (s.min_bet - ps.bet_chips))
        else:
            s = act(
                s,
                p.ActionEnum.Call
                if p.ActionEnum.Call in s.legal_actions
                else p.ActionEnum.Check,
            )
    pytest.fail("All-in hand did not settle")


def test_side_pot_eligibility_and_uncalled_excess():
    s = rigged(
        [["Ac", "Ad"], ["Kc", "Kd"], ["Qc", "Qd"]],
        ["2c", "3d", "7h", "8s", "9c"],
        [50, 100, 200],
        button=2,
    )
    s = all_in(s)
    assert [ps.stake for ps in s.players_state] == [150, 100, 100]
    assert [ps.reward for ps in s.players_state] == [100, 0, -100]


def test_odd_chip_goes_to_first_winner_left_of_button():
    s = rigged(
        [["Ac", "Ad"], ["Ah", "As"], ["Kc", "Kd"]],
        ["2c", "3d", "7h", "8s", "9c"],
        [5, 5, 5],
        button=2,
    )
    assert [ps.stake for ps in all_in(s).players_state] == [8, 7, 0]


@pytest.mark.parametrize(
    "hands,board",
    [
        ([["Ac", "2d"], ["6s", "2s"]], ["3c", "4d", "5h", "9s", "Tc"]),
        ([["Ac", "Ah"], ["6c", "6h"]], ["2c", "3c", "4c", "5c", "Kd"]),
    ],
)
def test_wheel_loses_to_six_high(hands, board):
    assert [
        ps.stake for ps in all_in(rigged(hands, board, [20, 20])).players_state
    ] == [0, 40]


def test_full_all_in_does_not_reopen_a_later_short_raise_for_callers():
    s = state(n=5, stacks=[79, 192, 172, 11, 100], button=2)
    for action, amount in [
        (p.ActionEnum.Raise, 77),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Raise, 21),
    ]:
        s = act(s, action, amount)
    assert s.current_player == 1
    assert s.min_raise == 77
    assert s.min_bet - s.players_state[1].bet_chips == 21
    assert p.ActionEnum.Raise not in s.legal_actions


def test_check_before_short_open_preserves_raise_option():
    s = state(n=3, stacks=[48, 12, 5], button=2)
    for a, amount in [
        (p.ActionEnum.Raise, 2),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Check, 0),
        (p.ActionEnum.Check, 0),
        (p.ActionEnum.Raise, 1),
    ]:
        s = act(s, a, amount)
    assert s.current_player == 0
    assert p.ActionEnum.Raise in s.legal_actions
    assert act(s, p.ActionEnum.Raise, 2).min_bet == 3


def test_three_way_tie_distributes_two_odd_chips_to_different_players():
    # Main pot 80 split three ways: 27, 27, 26, followed by side pots.
    s = state(n=5, stacks=[60, 97, 20, 169, 169], seed=61, button=1)
    for a, amount in [
        (p.ActionEnum.Raise, 167),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Fold, 0),
        (p.ActionEnum.Call, 0),
        (p.ActionEnum.Call, 0),
    ]:
        s = act(s, a, amount)
    assert [ps.stake for ps in s.players_state] == [86, 97, 27, 0, 305]


def test_short_big_blind_keeps_the_nominal_bring_in():
    s = state(n=3, stacks=[70, 1, 186], button=2)
    assert s.current_player == 2
    assert s.min_bet == 2
    assert s.min_raise == 2
    assert act(s, p.ActionEnum.Raise, 2).min_bet == 4
    s = act(s, p.ActionEnum.Call)
    assert s.players_state[2].stake == 184
    assert p.ActionEnum.Call in s.legal_actions
    s = act(s, p.ActionEnum.Call)
    assert s.stage == p.Stage.Flop
    assert s.pot == 5


def test_lone_caller_only_matches_the_short_blind():
    s = state(n=3, stacks=[70, 1, 186], button=2)
    s = act(s, p.ActionEnum.Fold)
    assert s.final_state
    assert sum(ps.stake for ps in s.players_state) == 257


def test_short_big_blind_raise_does_not_reopen_limpers():
    s = state(n=4, stacks=[91, 3, 92, 136], button=3)
    for action in [p.ActionEnum.Fold, p.ActionEnum.Call, p.ActionEnum.Call]:
        s = act(s, action)
    s = act(s, p.ActionEnum.Raise, 1)
    assert s.current_player == 3
    assert s.min_raise == 2
    assert p.ActionEnum.Raise not in s.legal_actions
    s = act(s, p.ActionEnum.Call)
    assert p.ActionEnum.Raise not in s.legal_actions


def test_parallel_batch_rejects_missing_actions():
    with pytest.raises(ValueError, match="one action per state"):
        p.parallel_apply_action([state()], [])
