use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, SeedableRng};

use crate::state::action::{Action, ActionEnum, ActionRecord};
use crate::state::card::Card;
use crate::state::stage::Stage;
use crate::state::{PlayerState, State, StateStatus};

// Bound conversion error and leave ample room for exact sums across ten seats.
const MAX_CHIPS: u64 = 1_000_000_000_000;

fn chips(amount: f64, unit: f64) -> Option<u64> {
    if !amount.is_finite() || amount < 0.0 {
        return None;
    }
    let scaled = amount / unit;
    if !scaled.is_finite() || scaled > MAX_CHIPS as f64 || (scaled - scaled.round()).abs() > 1e-4 {
        return None;
    }
    Some(scaled.round() as u64)
}

#[pymethods]
impl State {
    #[staticmethod]
    #[pyo3(signature = (n_players, button, sb, bb, stake, seed, verbose=false, chip_unit=0.01, stakes=None))]
    pub fn from_seed(
        n_players: u64,
        button: u64,
        sb: f64,
        bb: f64,
        stake: f64,
        seed: u64,
        verbose: bool,
        chip_unit: f64,
        stakes: Option<Vec<f64>>,
    ) -> PyResult<State> {
        let mut deck = Card::collect();
        deck.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
        Self::from_deck(
            n_players, button, sb, bb, stake, deck, verbose, chip_unit, stakes,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (n_players, button, sb, bb, stake, deck, verbose=false, chip_unit=0.01, stakes=None))]
    pub fn from_deck(
        n_players: u64,
        button: u64,
        sb: f64,
        bb: f64,
        stake: f64,
        mut deck: Vec<Card>,
        verbose: bool,
        chip_unit: f64,
        stakes: Option<Vec<f64>>,
    ) -> PyResult<State> {
        if !(2..=10).contains(&n_players) || button >= n_players {
            return Err(PyValueError::new_err(
                "Expected 2–10 players and an occupied button seat",
            ));
        }
        if !chip_unit.is_finite() || chip_unit <= 0.0 {
            return Err(PyValueError::new_err(
                "chip_unit must be finite and positive",
            ));
        }
        if deck.len() != 52 || deck.iter().enumerate().any(|(i, c)| deck[..i].contains(c)) {
            return Err(PyValueError::new_err(
                "Expected a complete deck of 52 distinct cards",
            ));
        }
        let convert = |value| {
            chips(value, chip_unit).ok_or_else(||
            PyValueError::new_err("Amounts must be finite nonnegative multiples of chip_unit within the supported range"))
        };
        let sb = convert(sb)?;
        let bb = convert(bb)?;
        if sb == 0 || bb < sb {
            return Err(PyValueError::new_err(
                "Expected 0 < small blind <= big blind",
            ));
        }
        let amounts = stakes.unwrap_or_else(|| vec![stake; n_players as usize]);
        if amounts.len() != n_players as usize {
            return Err(PyValueError::new_err(
                "stakes must contain one starting stack per seat",
            ));
        }
        let amounts: Vec<u64> = amounts.into_iter().map(convert).collect::<PyResult<_>>()?;
        if amounts.contains(&0) {
            return Err(PyValueError::new_err(
                "Every dealt-in player must have a positive stack",
            ));
        }
        let n = n_players as usize;
        let button = button as usize;
        let sb_seat = if n == 2 { button } else { (button + 1) % n };
        let bb_seat = (sb_seat + 1) % n;
        // Deal one card per seat per round, starting left of the button.
        let mut hands = vec![(deck[0], deck[0]); n];
        for round in 0..2 {
            for offset in 1..=n {
                let seat = (button + offset) % n;
                let card = deck.remove(0);
                if round == 0 {
                    hands[seat].0 = card;
                } else {
                    hands[seat].1 = card;
                }
            }
        }
        let players_state: Vec<_> = (0..n)
            .map(|seat| {
                let blind = if seat == sb_seat {
                    sb
                } else if seat == bb_seat {
                    bb
                } else {
                    0
                };
                let posted = blind.min(amounts[seat]);
                PlayerState {
                    player: seat as u64,
                    hand: hands[seat],
                    bet_chips: posted,
                    pot_chips: 0,
                    stake: amounts[seat] - posted,
                    initial_stake: amounts[seat],
                    reward: 0,
                    chip_unit,
                    active: true,
                    pending: amounts[seat] > posted,
                    acted_at: None,
                }
            })
            .collect();
        let pot = players_state.iter().map(|p| p.bet_chips).sum();
        let mut state = State {
            current_player: bb_seat as u64,
            players_state,
            public_cards: vec![],
            stage: Stage::Preflop,
            button: button as u64,
            from_action: None,
            legal_actions: vec![],
            deck,
            pot,
            min_bet: bb,
            bb,
            last_full_raise: bb,
            chip_unit,
            final_state: false,
            status: StateStatus::Ok,
            verbose,
        };
        state.advance(bb_seat);
        Ok(state)
    }

    #[getter]
    pub fn pot(&self) -> f64 {
        self.pot as f64 * self.chip_unit
    }
    #[getter]
    pub fn min_bet(&self) -> f64 {
        self.min_bet as f64 * self.chip_unit
    }
    #[getter]
    pub fn bb(&self) -> f64 {
        self.bb as f64 * self.chip_unit
    }
    #[getter]
    pub fn min_raise(&self) -> f64 {
        self.last_full_raise as f64 * self.chip_unit
    }

    pub fn apply_action(&self, action: Action) -> State {
        if self.status != StateStatus::Ok || self.final_state {
            return self.clone();
        }
        let mut next = self.clone();
        next.from_action = Some(ActionRecord {
            player: self.current_player,
            action,
            stage: self.stage,
            legal_actions: self.legal_actions.clone(),
        });
        if !self.legal_actions.contains(&action.action) {
            return next.rejected(StateStatus::IllegalAction);
        }
        let seat = self.current_player as usize;
        let player = self.players_state[seat];
        let to_call = self.min_bet.saturating_sub(player.bet_chips);
        match action.action {
            ActionEnum::Fold => {
                next.players_state[seat].active = false;
            }
            ActionEnum::Check => {}
            ActionEnum::Call => {
                next.commit(seat, to_call.min(player.stake));
            }
            ActionEnum::Raise => {
                let amount = match chips(action.amount, self.chip_unit) {
                    Some(value) => value,
                    None => return next.rejected(StateStatus::InvalidAmount),
                };
                let available = player.stake.saturating_sub(to_call);
                if amount > available {
                    return next.rejected(StateStatus::HighBet);
                }
                if amount == 0 || (amount < self.last_full_raise && amount != available) {
                    return next.rejected(StateStatus::LowBet);
                }
                next.commit(seat, to_call + amount);
                next.min_bet = next.players_state[seat].bet_chips;
                if amount >= self.last_full_raise {
                    next.last_full_raise = amount;
                }
                for (index, other) in next.players_state.iter_mut().enumerate() {
                    if index != seat
                        && other.active
                        && other.stake > 0
                        && other.bet_chips < next.min_bet
                    {
                        other.pending = true;
                    }
                }
            }
        }
        next.players_state[seat].pending = false;
        // Checking before the first wager does not surrender the right to raise
        // an opening all-in that is smaller than the minimum bet.
        if action.action != ActionEnum::Check {
            next.players_state[seat].acted_at = Some(next.min_bet);
        }
        next.advance(seat);
        next
    }

    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl State {
    fn rejected(mut self, status: StateStatus) -> Self {
        self.status = status;
        self.final_state = true;
        self.legal_actions.clear();
        self
    }

    fn commit(&mut self, seat: usize, amount: u64) {
        let player = &mut self.players_state[seat];
        player.stake -= amount;
        player.bet_chips += amount;
        self.pot += amount;
    }

    fn advance(&mut self, after: usize) {
        let n = self.players_state.len();
        let active = self.players_state.iter().filter(|p| p.active).count();
        if active == 1 {
            self.settle();
            return;
        }
        let able: Vec<_> = (0..n)
            .filter(|&i| self.players_state[i].active && self.players_state[i].stake > 0)
            .collect();
        if able.len() <= 1 {
            if let Some(&seat) = able.first() {
                // No side betting against all-in players. Return unmatched excess at settlement.
                let actual_bet = self
                    .players_state
                    .iter()
                    .filter(|p| p.active)
                    .map(|p| p.bet_chips)
                    .max()
                    .unwrap();
                self.min_bet = actual_bet;
                self.players_state[seat].pending = self.players_state[seat].bet_chips < actual_bet;
            }
        }
        if let Some(seat) = (1..=n).map(|offset| (after + offset) % n).find(|&i| {
            let p = self.players_state[i];
            p.active && p.stake > 0 && p.pending
        }) {
            self.current_player = seat as u64;
            self.legal_actions = self.available_actions();
            return;
        }
        if able.len() <= 1 {
            while self.stage != Stage::Showdown {
                self.next_street();
            }
            self.settle();
        } else {
            self.next_street();
            if self.stage == Stage::Showdown {
                self.settle();
            } else {
                self.advance(self.button as usize);
            }
        }
    }

    fn available_actions(&self) -> Vec<ActionEnum> {
        if self.final_state {
            return vec![];
        }
        let p = self.players_state[self.current_player as usize];
        let to_call = self.min_bet.saturating_sub(p.bet_chips);
        let mut actions = vec![ActionEnum::Fold];
        actions.push(if to_call == 0 {
            ActionEnum::Check
        } else {
            ActionEnum::Call
        });
        let reopened = p.acted_at.map_or(true, |previous| {
            self.min_bet.saturating_sub(previous) >= self.last_full_raise
        });
        let opponent_can_bet = self.players_state.iter().any(|other| {
            other.player != p.player && other.active && other.stake + other.bet_chips > self.min_bet
        });
        if reopened && opponent_can_bet && p.stake > to_call {
            actions.push(ActionEnum::Raise);
        }
        actions
    }

    fn next_street(&mut self) {
        self.stage = match self.stage {
            Stage::Preflop => Stage::Flop,
            Stage::Flop => Stage::Turn,
            Stage::Turn => Stage::River,
            _ => Stage::Showdown,
        };
        let count = match self.stage {
            Stage::Flop => 3,
            Stage::Turn | Stage::River => 1,
            _ => 0,
        };
        for _ in 0..count {
            self.public_cards.push(self.deck.remove(0));
        }
        for p in &mut self.players_state {
            p.pot_chips += p.bet_chips;
            p.bet_chips = 0;
            p.acted_at = None;
            p.pending = p.active && p.stake > 0;
        }
        self.min_bet = 0;
        self.last_full_raise = self.bb;
    }

    fn settle(&mut self) {
        let contributions: Vec<_> = self
            .players_state
            .iter()
            .map(|p| p.pot_chips + p.bet_chips)
            .collect();
        let mut levels = contributions.clone();
        levels.sort_unstable();
        levels.dedup();
        let survivors: Vec<_> = self
            .players_state
            .iter()
            .enumerate()
            .filter(|(_, p)| p.active)
            .map(|(i, _)| i)
            .collect();
        let ranks: Vec<_> = self
            .players_state
            .iter()
            .map(|p| {
                if survivors.len() > 1 && p.active {
                    rank_hand(p.hand, &self.public_cards)
                } else {
                    (0, 0, 0)
                }
            })
            .collect();
        let mut pots: Vec<(Vec<usize>, u64)> = Vec::new();
        let mut previous = 0;
        for level in levels {
            let contributors: Vec<_> = contributions
                .iter()
                .enumerate()
                .filter(|(_, &c)| c >= level)
                .map(|(i, _)| i)
                .collect();
            let amount = (level - previous) * contributors.len() as u64;
            previous = level;
            if amount == 0 {
                continue;
            }
            if contributors.len() == 1 {
                self.players_state[contributors[0]].stake += amount;
                continue;
            }
            let eligible: Vec<_> = contributors
                .iter()
                .copied()
                .filter(|&i| self.players_state[i].active)
                .collect();
            // Folded contributions add dead money; they must not split a pot
            // into extra rounding events when eligibility has not changed.
            if let Some((last_eligible, last_amount)) = pots.last_mut() {
                if *last_eligible == eligible {
                    *last_amount += amount;
                    continue;
                }
            }
            pots.push((eligible, amount));
        }
        for (eligible, amount) in pots {
            let winners = if survivors.len() == 1 {
                survivors.clone()
            } else {
                let best = eligible
                    .iter()
                    .map(|&i| ranks[i])
                    .min()
                    .expect("A contested pot must have an eligible player");
                eligible
                    .into_iter()
                    .filter(|&i| ranks[i] == best)
                    .collect::<Vec<_>>()
            };
            let share = amount / winners.len() as u64;
            let mut remainder = amount % winners.len() as u64;
            for offset in 1..=self.players_state.len() {
                let seat = (self.button as usize + offset) % self.players_state.len();
                if winners.contains(&seat) {
                    self.players_state[seat].stake += share + u64::from(remainder > 0);
                    remainder = remainder.saturating_sub(1);
                }
            }
        }
        for p in &mut self.players_state {
            p.reward = p.stake as i64 - p.initial_stake as i64;
            p.bet_chips = 0;
            p.pot_chips = 0;
            p.pending = false;
        }
        self.pot = 0;
        self.final_state = true;
        self.legal_actions.clear();
    }
}

fn rank_hand(private: (Card, Card), public: &[Card]) -> (u8, u64, u64) {
    public
        .iter()
        .copied()
        .chain([private.0, private.1])
        .combinations(5)
        .map(|cards| rank_five(&cards))
        .min()
        .unwrap()
}

fn rank_five(cards: &[Card]) -> (u8, u64, u64) {
    let mut ranks: Vec<_> = cards.iter().map(|c| c.rank as u64).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    let flush = cards.iter().all(|c| c.suit == cards[0].suit);
    let wheel = ranks == [12, 3, 2, 1, 0];
    let straight = wheel || ranks.windows(2).all(|w| w[0] == w[1] + 1);
    let high = if wheel { 3 } else { ranks[0] };
    let mut groups: Vec<(usize, u64)> = ranks.iter().copied().dedup_with_count().collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));
    let score = |values: Vec<u64>| {
        values
            .into_iter()
            .fold(0, |acc, rank| acc * 13 + (12 - rank))
    };
    let kickers = score(ranks.clone());
    if straight && flush {
        return (0, 12 - high, 0);
    }
    if groups[0].0 == 4 {
        return (1, 12 - groups[0].1, kickers);
    }
    if groups[0].0 == 3 && groups[1].0 == 2 {
        return (2, 12 - groups[0].1, kickers);
    }
    if flush {
        return (3, kickers, 0);
    }
    if straight {
        return (4, 12 - high, 0);
    }
    if groups[0].0 == 3 {
        return (5, 12 - groups[0].1, kickers);
    }
    if groups[0].0 == 2 && groups[1].0 == 2 {
        return (6, score(vec![groups[0].1, groups[1].1]), kickers);
    }
    if groups[0].0 == 2 {
        return (7, 12 - groups[0].1, kickers);
    }
    (8, kickers, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn legal_hands_terminate_and_conserve_chips(seed: u64, stacks in prop::collection::vec(1u64..=200, 2..=10), choice in prop::collection::vec(any::<usize>(), 1..300)) {
            let n = stacks.len() as u64;
            let total: u64 = stacks.iter().sum();
            let amounts = stacks.iter().map(|&s| s as f64).collect();
            let mut state = State::from_seed(n, seed % n, 1., 2., 200., seed, false, 1., Some(amounts)).unwrap();
            for turn in 0..500 {
                if state.final_state { break; }
                let actions = state.legal_actions.clone();
                let action = actions[choice[turn % choice.len()] % actions.len()];
                let p = state.players_state[state.current_player as usize];
                let maximum = p.stake.saturating_sub(state.min_bet.saturating_sub(p.bet_chips));
                let minimum = state.last_full_raise.min(maximum);
                let amount = minimum + choice[turn % choice.len()] as u64 % (maximum - minimum + 1);
                state = state.apply_action(Action { action, amount: amount as f64 });
                prop_assert_eq!(state.status, StateStatus::Ok);
                prop_assert_eq!(state.players_state.iter().map(|p| p.stake + p.bet_chips + p.pot_chips).sum::<u64>(), total);
            }
            prop_assert!(state.final_state);
            prop_assert_eq!(state.players_state.iter().map(|p| p.reward).sum::<i64>(), 0);
        }
    }
}
