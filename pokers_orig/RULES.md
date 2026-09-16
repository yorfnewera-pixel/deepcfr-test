# Hold'em rules and engine contract

This fork implements a single hand of no-limit Texas Hold'em: 2–10 dealt-in players, one 52-card deck, one board, table stakes, no rake, no ante, and no straddles. Four-, five-, and six-handed play are the main training targets. This is a defined research cash-game profile, not a claim to implement every casino procedure.

## Betting

- Seats increase clockwise. With three or more players, the small and big blinds sit immediately left of the button. Heads-up, the button posts the small blind and acts first preflop; the big blind acts first after the flop.
- Every non-all-in player must respond to a wager. The big blind retains its option after limps. Checks are legal when the player owes nothing; calls pay the amount owed or the remaining stack, whichever is smaller.
- The opening minimum bet is the big blind. A full raise increases the current wager by at least the last full raise increment. A raise from 2 to 10 establishes an increment of 8, so the next minimum raise-to is 18.
- A smaller increase is legal only when it commits the player's entire remaining stack. Such an increase does not replace the last full raise increment.
- A player who already called or bet can raise again only when facing at least a full raise since that action. Multiple short all-ins can cumulatively reopen that player's action; evaluate each player separately. Checking before the opening wager preserves the option to raise a short opening all-in.
- A short opening all-in can be raised by at least one big blind above its amount. A 1-chip opening all-in with a 2-chip big blind permits a raise to 3.
- Raising is unavailable if no other live player can contest chips above the current wager. Once all outstanding calls are resolved and at most one live player has chips, run out the remaining board without artificial check actions.

These betting choices follow [PokerStars' Hold'em examples](https://www.pokerstars.com/poker/learn/lesson/texas-holdem-rules/). We explicitly adopt the per-player cumulative reopening rule described in [TDA rule 47 and its examples](https://www.pokertda.com/view-poker-tda-rules/); tournament procedures outside this betting rule are not implied.

## Settlement and chips

- Chip arithmetic is integer arithmetic. `chip_unit` is the table's smallest denomination, defaulting to 0.01. Python amounts are expressed in table currency units and converted at the boundary. Amounts must be finite, nonnegative multiples of the denomination. Floating conversion noise within 0.0001 chip is tolerated; arbitrary fractional-chip wagers are rejected.
- Each initial stack and input amount is limited to 10^12 chips. This keeps conversion and total accounting within the supported numerical range. Starting stacks must be positive and may be smaller than a blind. Blinds are capped by the posting player's stack. A short big blind does not reduce the nominal preflop bring-in while two players can still bet; a lone remaining caller only matches the actual all-in wager.
- Build main and side pots from contributions and eligibility. Folded money remains in the pot, but folded players cannot win. Folded contribution levels do not create extra pots when eligibility is unchanged.
- Return an unmatched contribution to its owner. Split each actual pot separately. Divide into whole chips, then distribute at most one spare chip per tied winner, clockwise from the button. This follows the button-based allocation in [PokerStars Live cash rule 32](https://www.pokerstarslive.com/poker/cashgamerules/) with the multiple-spare convention made explicit in the [BARGE rulebook, ties and odd chips](https://www.barge.org/rulebook.pdf).
- At settlement, stacks include winnings and refunds, committed amounts and pot are zero, and reward is final stack minus starting stack. The integer sum of rewards is zero for this unraked profile.
- Evaluate the best five of seven cards. An ace can be low only in A-2-3-4-5; that straight loses to a six-high straight. Suits never break a Hold'em showdown tie.

## API and dealing

`State.from_seed(..., chip_unit=0.01, stakes=None)` uses a seeded shuffled deck. `stakes`, when provided, contains a starting stack for each seat and overrides the scalar `stake`. `State.from_deck` requires all 52 distinct cards. Cards are dealt in two rounds, starting left of the button. Board cards follow directly from the deck; physical burn-card handling is omitted in this digital simulation. With a uniformly shuffled hidden deck, this has the same distribution of player-visible cards as burning unknown cards.

`Action(Raise, amount)` retains the library's additional-raise convention: call first, then add `amount`. For an absolute raise-to target, use `amount = target - state.min_bet`. `state.min_raise` exposes the current full increment; an exact all-in may be smaller. `state.bb` and `state.chip_unit` are explicit. Call/check/fold ignore the amount field.

`apply_action` returns a new state. Invalid actions return a terminal error state, with no chips moved and no legal actions. The parent remains usable. Game fields are read-only through Python; callers cannot fabricate a legal-action list or overwrite stacks. `verbose` remains configurable.

The changed deal order, finite stacks, chip denominations, and corrected rules intentionally break legacy traces that relied on the previous bugs. Version 0.2 is not checkpoint-compatible in the sense of preserving the game a model was trained against.

## Information boundary and remaining scope

`State` is a simulator object. It contains every hole card and the undealt deck, including after folds; read-only access does not make this safe to give to a playing policy. A separate player-observation/event interface is required in the bot project. This release does not claim to enforce that boundary or implement public showdown/muck disclosure.

Seats are fixed during a hand. Session-level joins, sit-outs, missed blinds, button movement between hands, top-ups, and player identities belong in the bot's session layer. Rake, antes, straddles, multiple runouts, live-dealer irregularities, and tournament payouts are not supported by this profile.

## Verification

`cargo test` checks generated legal hands for termination and integer chip conservation. `pytest` adds targeted betting, short-stack, side-pot, odd-chip, input-validation, and hand-ranking cases; replays the 9,908 bundled Pluribus hands through serial and parallel execution; and compares generated hands against PokerKit 0.7.5. Set `POKERS_REFERENCE_SEEDS=1000` to run 1,000 deals for each table size from two to six.

The Pluribus fixture adapter supplies finite 10,000-chip starting stacks, interleaves hole-card rounds, fills unused deck cards, and translates a zero-cost recorded call to a check. Fixture rewards remain unchanged.

The independent comparison checks action order, stacks, outstanding bets, legal check/call choices, raise availability, and settlement. Three reference differences are explicit rather than silently inherited:

1. PokerKit 0.7.5 can include an earlier full all-in in its cumulative reopening total for a player who called after that bet. A rise from 79 to 100 is only 21 for that caller, even if the earlier full increment was 77. Our regression follows rule 47. It also initializes its tracked raise increment below the big blind, so an opening short all-in raise from 2 to 3 can incorrectly reopen limpers. Generated comparisons recognize these two cases and proceed with a common legal call or fold; fixed regressions check the correct choices.
2. PokerKit gives the entire rounding remainder to one winner and can merge pots after eliminating losing hands. Our comparison uses its recorded investments and independent hand evaluator, then applies this profile's pot boundaries and spare-chip rule. It also bounds the difference from PokerKit's direct payouts to rounding-sized amounts. Fixed expected-payout cases test these rules separately.

3. PokerKit prices a short big blind at the amount posted instead of the nominal big blind. Generated reference hands therefore give the big blind at least a full blind; fixed short-blind tests cover our bring-in rule separately. This follows the [Kontenders league rulebook, all-in rules](https://kontenderspoker.com/docs/KontendersPokerRules.pdf).

The independent comparison runs on Python 3.11; PokerKit 0.7.5 does not support Python 3.10. Core and fixture tests run on both supported versions.

No finite suite proves every hand correct. New disagreements must become replayable regressions before changing the default engine used for training.
