// game_logic.rs
use itertools::Itertools;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use rand::{seq::SliceRandom, SeedableRng};
use strum::IntoEnumIterator;

use crate::state::action::{Action, ActionEnum, ActionRecord, PublicActionRecord};
use crate::state::card::{Card, CardRank, CardSuit};
use crate::state::stage::Stage;
use crate::state::{PlayerState, State, StateStatus};

// Define a macro for verbose printing
macro_rules! verbose_println {
    ($state:expr, $($arg:tt)*) => {
        if $state.verbose {
            println!($($arg)*);
        }
    };
}

const CHIP_EPSILON: f64 = 1e-9;

#[pyfunction]
pub fn compare_showdown(
    hero: (Card, Card),
    opponent: (Card, Card),
    board: Vec<Card>,
) -> PyResult<i8> {
    if board.len() != 5 {
        return Err(PyValueError::new_err("Showdown требует ровно пять карт board"));
    }

    let cards = vec![
        hero.0,
        hero.1,
        opponent.0,
        opponent.1,
        board[0],
        board[1],
        board[2],
        board[3],
        board[4],
    ];
    let unique_cards: std::collections::HashSet<(CardRank, CardSuit)> = cards
        .iter()
        .map(|card| (card.rank, card.suit))
        .collect();
    if unique_cards.len() != cards.len() {
        return Err(PyValueError::new_err("Карты showdown содержат дубликат"));
    }

    let hero_rank = rank_cards(hero, &board);
    let opponent_rank = rank_cards(opponent, &board);
    Ok(match hero_rank.cmp(&opponent_rank) {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => -1,
    })
}

pub struct InitStateError {
    msg: String,
}

impl std::convert::From<InitStateError> for PyErr {
    fn from(err: InitStateError) -> PyErr {
        PyOSError::new_err(err.msg)
    }
}

#[pymethods]
impl State {
    /// Return an independent Rust clone for Python's copy protocol.
    pub fn __copy__(&self) -> State {
        self.clone()
    }

    /// `memo` is accepted for compatibility with `copy.deepcopy`.
    pub fn __deepcopy__(&self, _memo: &PyAny) -> State {
        self.clone()
    }

    /// Переносит полную public history при реконструкции скрытых карт состояния.
    pub fn copy_public_history_from(&mut self, source: &State) -> PyResult<()> {
        if !source.action_history_complete {
            return Err(PyValueError::new_err(
                "Нельзя копировать неполную публичную историю",
            ));
        }
        self.action_history = source.action_history.clone();
        self.action_history_complete = true;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (n_players, button, sb, bb, stake, seed, verbose=false))]
    pub fn from_seed(
        n_players: u64,
        button: u64,
        sb: f64,
        bb: f64,
        stake: f64,
        seed: u64,
        verbose: bool,
    ) -> Result<State, InitStateError> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut deck: Vec<Card> = Card::collect();
        deck.shuffle(&mut rng);

        State::from_deck(n_players, button, sb, bb, stake, deck, verbose)
    }

    #[staticmethod]
    #[pyo3(signature = (n_players, button, sb, bb, stake, deck, verbose=false))]
    pub fn from_deck(
        n_players: u64,
        button: u64,
        sb: f64,
        bb: f64,
        stake: f64,
        mut deck: Vec<Card>,
        verbose: bool,
    ) -> Result<State, InitStateError> {
        if n_players < 2 {
            return Err(InitStateError {
                msg: "The number of players must be 2 or more".to_owned(),
            });
        }

        if button >= n_players {
            return Err(InitStateError {
                msg: "The button must be between the players".to_owned(),
            });
        }

        if deck.len() < 2 * n_players as usize {
            return Err(InitStateError {
                msg: "The number of cards in the deck must be at least 2*n_players".to_owned(),
            });
        }

        if sb < 0.0 {
            return Err(InitStateError {
                msg: "The small blind must be greater than 0".to_owned(),
            });
        }

        if bb < sb {
            return Err(InitStateError {
                msg: "The small blind must be smaller or equal than the big blind".to_owned(),
            });
        }

        if stake < bb {
            return Err(InitStateError {
                msg: "The stake must be greater or equal than the big blind".to_owned(),
            });
        }

        let (small_blind_player, big_blind_player, preflop_player) = if n_players == 2 {
            (button, (button + 1) % n_players, button)
        } else {
            (
                (button + 1) % n_players,
                (button + 2) % n_players,
                (button + 3) % n_players,
            )
        };

        let mut players_state: Vec<PlayerState> = Vec::new();
        for i in 0..n_players {
            let player = (button + i + 1) % n_players;
            let chips = match player {
                _ if player == small_blind_player => sb,
                _ if player == big_blind_player => bb,
                _ => 0.0,
            };

            let p_state = PlayerState {
                player: player,
                hand: (deck.remove(0), deck.remove(0)),
                bet_chips: chips,
                pot_chips: 0.0,
                stake: stake - chips,
                reward: 0.0,
                active: true,
                last_stage_action: None,
            };
            players_state.push(p_state);
        }

        players_state.sort_by_key(|ps| ps.player);

        let mut state = State {
            current_player: preflop_player,
            players_state: players_state,
            public_cards: Vec::new(),
            stage: Stage::Preflop,
            button: button,
            from_action: None,
            action_history: Vec::new(),
            action_history_complete: true,
            legal_actions: Vec::new(),
            deck: deck,
            final_state: false,
            pot: sb + bb,
            min_bet: bb,
            sb: sb,
            bb: bb,
            last_raise_increment: bb,
            status: StateStatus::Ok,
            verbose: verbose,
        };

        state.legal_actions = legal_actions(&state);
        Ok(state)
    }

    #[staticmethod]
    #[pyo3(signature = (
        n_players, button, sb, bb, stake,
        deck, hole_cards, public_cards, stage,
        pot, bet_chips, pot_chips, active, last_stage_action,
        current_player=None, last_raise_increment=None, verbose=false
    ))]
    pub fn from_mid_hand(
        n_players: u64,
        button: u64,
        sb: f64,
        bb: f64,
        stake: f64,
        deck: Vec<Card>,
        hole_cards: Vec<(Card, Card)>,
        public_cards: Vec<Card>,
        stage: Stage,
        pot: f64,
        bet_chips: Vec<f64>,
        pot_chips: Vec<f64>,
        active: Vec<bool>,
        last_stage_action: Vec<Option<ActionEnum>>,
        current_player: Option<u64>,
        last_raise_increment: Option<f64>,
        verbose: bool,
    ) -> Result<State, InitStateError> {
        if n_players < 2 {
            return Err(InitStateError {
                msg: "The number of players must be 2 or more".to_owned(),
            });
        }

        if button >= n_players {
            return Err(InitStateError {
                msg: "The button must be between the players".to_owned(),
            });
        }

        if stage == Stage::Showdown {
            return Err(InitStateError {
                msg: "Cannot restore unresolved Showdown state; settlement is not supported".to_owned(),
            });
        }

        if hole_cards.len() != n_players as usize {
            return Err(InitStateError {
                msg: format!("hole_cards length {} must equal n_players {}", hole_cards.len(), n_players),
            });
        }

        if bet_chips.len() != n_players as usize {
            return Err(InitStateError {
                msg: format!("bet_chips length {} must equal n_players {}", bet_chips.len(), n_players),
            });
        }

        if pot_chips.len() != n_players as usize {
            return Err(InitStateError {
                msg: format!("pot_chips length {} must equal n_players {}", pot_chips.len(), n_players),
            });
        }

        if active.len() != n_players as usize {
            return Err(InitStateError {
                msg: format!("active length {} must equal n_players {}", active.len(), n_players),
            });
        }

        if last_stage_action.len() != n_players as usize {
            return Err(InitStateError {
                msg: format!("last_stage_action length {} must equal n_players {}", last_stage_action.len(), n_players),
            });
        }

        let expected_public = match stage {
            Stage::Preflop => 0,
            Stage::Flop => 3,
            Stage::Turn => 4,
            Stage::River => 5,
            Stage::Showdown => 5,
        };
        if public_cards.len() != expected_public {
            return Err(InitStateError {
                msg: format!("public_cards length {} must match stage {:?} (expected {})", public_cards.len(), stage, expected_public),
            });
        }

        if sb < 0.0 {
            return Err(InitStateError {
                msg: "The small blind must be greater than 0".to_owned(),
            });
        }

        if bb < sb {
            return Err(InitStateError {
                msg: "The small blind must be smaller or equal than the big blind".to_owned(),
            });
        }

        if stake < bb {
            return Err(InitStateError {
                msg: "The stake must be greater or equal than the big blind".to_owned(),
            });
        }

        // All input vectors are indexed by the canonical player id (0..n_players),
        // not by table position after the button.  Keep that identity intact.
        let mut players_state: Vec<PlayerState> = Vec::new();
        for player in 0..n_players {
            let i = player as usize;
            let committed = bet_chips[i] + pot_chips[i];
            if committed > stake + CHIP_EPSILON {
                return Err(InitStateError {
                    msg: format!("player {} has committed more chips than stake", player),
                });
            }
            players_state.push(PlayerState {
                player,
                hand: hole_cards[i],
                bet_chips: bet_chips[i],
                pot_chips: pot_chips[i],
                stake: (stake - committed).max(0.0),
                reward: 0.0,
                active: active[i],
                last_stage_action: last_stage_action[i],
            });
        }

        if !players_state.iter().any(|ps| ps.active) {
            return Err(InitStateError {
                msg: "At least one active player required".to_owned(),
            });
        }

        let current_player = current_player.ok_or_else(|| InitStateError {
            msg: "current_player must be provided for mid-hand reconstruction".to_owned(),
        })?;
        if current_player >= n_players {
            return Err(InitStateError {
                msg: "current_player must be in range".to_owned(),
            });
        }
        if !players_state[current_player as usize].active {
            return Err(InitStateError {
                msg: "current_player must identify an active player".to_owned(),
            });
        }

        let mut state = State {
            current_player,
            players_state,
            public_cards,
            stage,
            button,
            from_action: None,
            action_history: Vec::new(),
            action_history_complete: false,
            legal_actions: Vec::new(),
            deck,
            final_state: false,
            pot,
            min_bet: bet_chips.iter().cloned().fold(0.0, f64::max),
            sb,
            bb,
            last_raise_increment: {
                let increment = last_raise_increment.ok_or_else(|| InitStateError {
                    msg: "last_raise_increment must be provided for mid-hand reconstruction".to_owned(),
                })?;
                if !increment.is_finite() || increment + CHIP_EPSILON < bb {
                    return Err(InitStateError {
                        msg: "last_raise_increment must be finite and at least bb".to_owned(),
                    });
                }
                increment
            },
            status: StateStatus::Ok,
            verbose,
        };

        state.legal_actions = legal_actions(&state);
        Ok(state)
    }

    /// Apply an action using normal engine rules and real stack information.
    pub fn apply_action(&self, action: Action) -> State {
        self.apply_action_internal(action)
    }

    fn apply_action_internal(&self, action: Action) -> State {
        match self.status {
            StateStatus::Ok => (),
            _ => return self.clone(),
        }

        if self.final_state {
            return self.clone();
        }

        let mut new_state = self.clone();
        new_state.from_action = Some(ActionRecord {
            player: self.current_player,
            action: action,
            stage: self.stage,
            legal_actions: self.legal_actions.clone(),
        });

        let player = self.current_player as usize;
        let call_amount = (self.min_bet - self.players_state[player].bet_chips).max(0.0);

        if !self.legal_actions.contains(&action.action) {
            return State {
                status: StateStatus::IllegalAction,
                final_state: true,
                ..new_state
            };
        }

        let mut paid_amount = 0.0;
        let mut applied_raise_increment = 0.0;
        let mut is_effective_raise = false;
        match action.action {
            ActionEnum::Fold => {
                new_state.players_state[player].active = false;
                new_state.players_state[player].pot_chips += self.players_state[player].bet_chips;
                new_state.players_state[player].bet_chips = 0.0;
                new_state.players_state[player].reward =
                    -(new_state.players_state[player].pot_chips as f64);
            }

            ActionEnum::Call => {
                // Calls are capped by the remaining stack.  This makes a short
                // all-in call legal instead of creating a negative stack.
                let call_amount = (self.min_bet - self.players_state[player].bet_chips).max(0.0);
                let paid = call_amount.min(self.players_state[player].stake);
                paid_amount = paid;
                new_state.players_state[player].bet_chips += paid;
                new_state.players_state[player].stake =
                    (self.players_state[player].stake - paid).max(0.0);
                new_state.pot += paid;
            }

            ActionEnum::Raise => {
                let available = self.players_state[player].stake;
                let requested_bet = call_amount + action.amount;
                let bet = requested_bet.min(available);
                let is_all_in = available - bet <= CHIP_EPSILON;
                let actual_raise_increment = (bet - call_amount).max(0.0);
                paid_amount = bet;
                applied_raise_increment = actual_raise_increment;
                let min_raise_increment = self.last_raise_increment.max(self.bb);

                if !is_all_in && actual_raise_increment + CHIP_EPSILON < min_raise_increment {
                    return State {
                        status: StateStatus::IllegalAction,
                        final_state: true,
                        ..new_state
                    };
                }

                new_state.players_state[player].bet_chips += bet;
                new_state.players_state[player].stake = (available - bet).max(0.0);
                new_state.pot += bet;
                if new_state.players_state[player].bet_chips > self.min_bet + CHIP_EPSILON {
                    is_effective_raise = true;
                    new_state.min_bet = new_state.players_state[player].bet_chips;
                    // A short all-in does not redefine the minimum full raise.
                    if actual_raise_increment + CHIP_EPSILON >= min_raise_increment {
                        new_state.last_raise_increment = actual_raise_increment;
                    }
                }
            }

            ActionEnum::Check => (),
        };

        new_state.action_history.push(PublicActionRecord {
            actor_id: self.current_player,
            street: self.stage,
            requested_action: action,
            paid_amount,
            applied_raise_increment,
            is_effective_raise,
        });

        new_state.players_state[player].last_stage_action = Some(action.action);

        new_state.current_player = (self.current_player + 1) % self.players_state.len() as u64;
        while !new_state.players_state[new_state.current_player as usize].active {
            new_state.current_player =
                (new_state.current_player + 1) % self.players_state.len() as u64;
        }

        // The betting round ends if:
        // Theres two or more active players
        let active_players: Vec<PlayerState> = new_state
            .players_state
            .iter()
            .copied()
            .filter(|ps| ps.active)
            .collect();
        let multiple_active = active_players.len() >= 2;
        // Every active player has done an action
        let is_last_player = active_players.iter().all(|ps| ps.last_stage_action != None);
        // And every active player has matched the bet or is all-in
        let all_same_bet = active_players
            .iter()
            .all(|ps| ps.bet_chips >= new_state.min_bet - CHIP_EPSILON || ps.stake <= CHIP_EPSILON);

        let all_broke = active_players.iter().all(|ps| ps.stake <= CHIP_EPSILON);
        let round_ended = multiple_active && (all_broke || (is_last_player && all_same_bet));

        if round_ended {
            new_state.to_next_stage();
            runout_forced_checkdown(&mut new_state);
        }

        // The game ends if the players have reached the showdown or every player except one has folded
        if active_players.len() == 1 {
            new_state.set_winners(vec![active_players[0].player]);
        }

        if new_state.stage == Stage::Showdown {
            let ranks: Vec<(u64, u64, u64)> = active_players
                .iter()
                .map(|ps| rank_hand(&new_state, ps.hand, &new_state.public_cards))
                .collect();
            let min_rank = ranks.iter().copied().min().unwrap();
            
            // Use verbose_println! macro instead of println!
            verbose_println!(&new_state, "Ranks: {:?}", ranks);
            
            let winners_indices: Vec<usize> = ranks
                .iter()
                .enumerate()
                .filter(|(_, &r)| r == min_rank)
                .map(|(i, _)| i)
                .collect();
                
            // Use verbose_println! macro instead of println!    
            verbose_println!(&new_state, "Winner id: {:?}", winners_indices);
            
            new_state.set_winners(
                winners_indices
                    .iter()
                    .map(|&i| active_players[i].player)
                    .collect(),
            );
        }

        new_state.legal_actions = legal_actions(&new_state);
        new_state
    }

    fn set_winners(&mut self, winners: Vec<u64>) {
        assert!(winners.iter().all(|&p| p < self.players_state.len() as u64));

        let winner_reward = self
            .players_state
            .iter()
            .filter(|&&ps| !winners.contains(&ps.player))
            .map(|ps| ps.pot_chips + ps.bet_chips)
            .fold(0.0, |c1, c2| c1 + c2)
            / winners.len() as f64;

        self.players_state = self
            .players_state
            .iter()
            .map(|ps| PlayerState {
                pot_chips: 0.0,
                bet_chips: 0.0,
                reward: if winners.contains(&ps.player) {
                    winner_reward
                } else {
                    -(ps.pot_chips + ps.bet_chips as f64)
                },
                active: false,
                ..*ps
            })
            .collect();

        self.final_state = true;
    }

    fn to_next_stage(&mut self) {
        self.stage = match self.stage {
            Stage::Preflop => Stage::Flop,
            Stage::Flop => Stage::Turn,
            Stage::Turn => Stage::River,
            _ => Stage::Showdown,
        };
        let n_deal_cards = match self.stage {
            Stage::Flop => 3,
            Stage::Turn | Stage::River => 1,
            _ => 0,
        };
        for _ in 0..n_deal_cards {
            self.public_cards.push(self.deck.remove(0))
        }
        self.players_state = self
            .players_state
            .iter()
            .map(|ps| PlayerState {
                pot_chips: ps.pot_chips + ps.bet_chips,
                bet_chips: 0.0,
                last_stage_action: None,
                ..*ps
            })
            .collect();

        self.min_bet = 0.0;
        self.last_raise_increment = self.bb;

        self.current_player = (self.button + 1) % self.players_state.len() as u64;
        while !self.players_state[self.current_player as usize].active {
            self.current_player = (self.current_player + 1) % self.players_state.len() as u64;
        }
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[pyfunction]
fn legal_actions(state: &State) -> Vec<ActionEnum> {
    let mut illegal_actions: Vec<ActionEnum> = Vec::new();
    match state.stage {
        Stage::Showdown => illegal_actions.append(&mut ActionEnum::iter().collect()),
        Stage::Preflop => {}
        _ => (),
    }

    if state.final_state {
        illegal_actions.append(&mut ActionEnum::iter().collect());
    }

    if is_forced_checkdown(state) {
        return vec![ActionEnum::Check];
    }

    let player_state = state.players_state[state.current_player as usize];
    let call_amount = f64::max(0.0, state.min_bet - player_state.bet_chips);
    if call_amount <= CHIP_EPSILON {
        illegal_actions.push(ActionEnum::Fold);
    }

    if call_amount <= CHIP_EPSILON {
        illegal_actions.push(ActionEnum::Call);
    } else {
        illegal_actions.push(ActionEnum::Check);
    }

    // A raise is only available when the player can first cover the call and
    // still add chips. Short all-in calls use the normal Call action.
    if !current_player_can_raise(state) {
        illegal_actions.push(ActionEnum::Raise);
    }

    let legal_actions: Vec<ActionEnum> = ActionEnum::iter()
        .filter(|a| !illegal_actions.contains(a))
        .collect();
    legal_actions
}

fn current_player_can_raise(state: &State) -> bool {
    let player_state = state.players_state[state.current_player as usize];
    let call_amount = f64::max(0.0, state.min_bet - player_state.bet_chips);
    let remaining_stake_after_call = player_state.stake - call_amount;

    remaining_stake_after_call > CHIP_EPSILON
}

fn is_forced_checkdown(state: &State) -> bool {
    let active_players: Vec<PlayerState> = state
        .players_state
        .iter()
        .copied()
        .filter(|ps| ps.active)
        .collect();

    if active_players.len() < 2 {
        return false;
    }

    let current_player = state.players_state[state.current_player as usize];
    let call_amount = state.min_bet - current_player.bet_chips;
    if call_amount > CHIP_EPSILON {
        return false;
    }

    active_players
        .iter()
        .filter(|ps| ps.stake > CHIP_EPSILON)
        .count()
        <= 1
}

fn runout_forced_checkdown(state: &mut State) {
    while state.stage != Stage::Showdown && is_forced_checkdown(state) {
        state.to_next_stage();
    }
}

// Modified to accept state parameter for verbose control
fn rank_hand(_state: &State, private_cards: (Card, Card), public_cards: &Vec<Card>) -> (u64, u64, u64) {
    rank_cards(private_cards, public_cards)
}

fn rank_cards(private_cards: (Card, Card), public_cards: &Vec<Card>) -> (u64, u64, u64) {
    let mut cards = public_cards.clone();
    cards.append(&mut vec![private_cards.0, private_cards.1]);

    let min_rank = cards
        .iter()
        .copied()
        .combinations(5)
        .map(|comb| rank_card_combination(comb))
        .min()
        .unwrap();

    min_rank
}

fn rank_card_combination(cards: Vec<Card>) -> (u64, u64, u64) {
    let mut ordered_cards = cards.clone();
    ordered_cards.sort_by_key(|c| c.rank);
    let suits: Vec<CardSuit> = ordered_cards.iter().map(|c| c.suit).collect();
    let ranks: Vec<CardRank> = ordered_cards.iter().map(|c| c.rank).collect();

    let suit_duplicates: Vec<(usize, CardSuit)> = suits
        .iter()
        .copied()
        .dedup_with_count()
        .sorted_by_key(|(n, _)| n.clone())
        .rev()
        .collect();

    let rank_duplicates: Vec<(usize, CardRank)> = ranks
        .iter()
        .copied()
        .dedup_with_count()
        .sorted_by_key(|(n, _)| n.clone())
        .rev()
        .collect();

    let ranks_in_sequence = ranks
        .windows(2)
        .map(|x| x[1] as i32 - x[0] as i32)
        .all(|d| d == 1)
        || ranks
            == vec![
                CardRank::R2,
                CardRank::R3,
                CardRank::R4,
                CardRank::R5,
                CardRank::RA,
            ];

    // Royal flush: A, K, Q, J, 10, all the same suit.
    if ranks[..]
        == [
            CardRank::RT,
            CardRank::RJ,
            CardRank::RQ,
            CardRank::RK,
            CardRank::RA,
        ]
        && suit_duplicates[0].0 == 5
    {
        return (1, 0, 0_u64);
    }
    // Straight flush: Five cards in a sequence, all in the same suit.
    if ranks_in_sequence && suit_duplicates[0].0 == 5 {
        return (2, high_card_value(&vec![straight_high_rank(&ranks)]), 0_u64);
    }
    // 3. Four of a kind: All four cards of the same rank.
    if rank_duplicates[0].0 == 4 {
        let relevant_ranks = vec![rank_duplicates[0].1];
        return (3, high_card_value(&relevant_ranks), high_card_value(&ranks));
    }
    // 4. Full house: Three of a kind with a pair.
    if rank_duplicates[0].0 == 3 && rank_duplicates[1].0 == 2 {
        let relevant_ranks = vec![rank_duplicates[0].1];
        return (4, high_card_value(&relevant_ranks), high_card_value(&ranks));
    }
    // 5. Flush: Any five cards of the same suit, but not in a sequence.
    if suit_duplicates[0].0 == 5 {
        return (5, high_card_value(&ranks), 0_u64);
    }
    // 6. Straight: Five cards in a sequence, but not of the same suit.
    if ranks_in_sequence {
        return (6, high_card_value(&vec![straight_high_rank(&ranks)]), 0_u64);
    }
    // 7. Three of a kind: Three cards of the same rank.
    if rank_duplicates[0].0 == 3 {
        let relevant_ranks = vec![rank_duplicates[0].1];
        return (7, high_card_value(&relevant_ranks), high_card_value(&ranks));
    }
    // 8. Two pair: Two different pairs.
    if rank_duplicates[0].0 == 2 && rank_duplicates[1].0 == 2 {
        let relevant_ranks = vec![rank_duplicates[0].1, rank_duplicates[1].1];
        return (8, high_card_value(&relevant_ranks), high_card_value(&ranks));
    }
    // 9. Pair: Two cards of the same rank.
    if rank_duplicates[0].0 == 2 {
        let relevant_ranks = vec![rank_duplicates[0].1];
        return (9, high_card_value(&relevant_ranks), high_card_value(&ranks));
    }

    // 10. High Card: When you haven't made any of the hands above, the highest card plays.
    (10, high_card_value(&ranks), 0_u64)
}

fn straight_high_rank(ranks: &[CardRank]) -> CardRank {
    if ranks
        == [
            CardRank::R2,
            CardRank::R3,
            CardRank::R4,
            CardRank::R5,
            CardRank::RA,
        ]
    {
        CardRank::R5
    } else {
        *ranks.iter().max().expect("straight требует пять карт")
    }
}

fn high_card_value(ranks: &Vec<CardRank>) -> u64 {
    let mut value: u64 = 0;
    for (i, &r) in ranks.iter().sorted().enumerate() {
        value += (13_u64.pow(i as u32)) * (12 - r as u64);
    }
    value
}

mod tests {
    #[cfg(test)]
    use super::*;
    #[cfg(test)]
    use proptest::prelude::*;

    #[cfg(test)]
    proptest! {
        #[test]
        fn from_deck_doesnt_crash(n_players in 0..10000, deck: Vec<Card>, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions: Vec<Action>) {
            let initial_state = State::from_deck(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, deck, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                    }
                },
                Err(_) => return Ok(())
            };

        }

        #[test]
        fn zero_sum_game(n_players in 2..26, seed: u64, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions in prop::collection::vec(Action::arbitrary_with(((), ())).prop_filter("Raise abs amount bellow 1e12",
        |a| a.amount.abs() < 1e12), 1..100)) {
            let initial_state = State::from_seed(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, seed, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                        if s.final_state {
                            let sum_rewards = s.players_state.iter().map(|ps| ps.reward).fold(0_f64, |r1, r2| r1 + r2);
                            println!("sum_rewards = {sum_rewards}");
                            prop_assert!(sum_rewards < 1e-12);
                        }
                    }
                },
                Err(err) => {
                    println!("{}", err.msg);
                    prop_assert!(false);
                }
            };
        }

        #[test]
        fn call_and_check_no_legal_at_same_time(n_players in 2..26, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions in prop::collection::vec(Action::arbitrary_with(((), ())), 1..100)) {
            let initial_state = State::from_seed(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, 1234, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                        if s.final_state {
                            prop_assert!(!(s.legal_actions.contains(&ActionEnum::Check) && s.legal_actions.contains(&ActionEnum::Call)));
                        }
                    }
                },
                Err(err) => {
                    println!("{}", err.msg);
                    prop_assert!(false);
                }
            };
        }

        #[test]
        fn illegal_raise_own_call(n_players in 2..26, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions in prop::collection::vec(Action::arbitrary_with(((), ())), 1..100)) {
            let initial_state = State::from_seed(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, 1234, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                        let call_done = s.players_state[s.current_player as usize].last_stage_action == Some(ActionEnum::Call);
                        let not_other_raise = !s.players_state.iter().filter(|ps| ps.active).any(|ps| ps.last_stage_action == Some(ActionEnum::Raise));
                        if call_done && not_other_raise {
                            prop_assert!(!s.legal_actions.contains(&ActionEnum::Raise));
                        }
                    }
                },
                Err(err) => {
                    println!("{}", err.msg);
                    prop_assert!(false);
                }
            };
        }

        #[test]
        fn illegal_call_zero_bets(n_players in 2..26, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions in prop::collection::vec(Action::arbitrary_with(((), ())), 1..100)) {
            let initial_state = State::from_seed(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, 1234, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                        prop_assert!(!(s.legal_actions.contains(&ActionEnum::Call) && s.min_bet == 0.0));
                    }
                },
                Err(err) => {
                    println!("{}", err.msg);
                    prop_assert!(false);
                }
            };
        }

        #[test]
        fn from_action_not_none(n_players in 2..26, sb in 0.5_f64..100.0_f64, bb_mult in 2..5, stake_mult in 100..1000, actions in prop::collection::vec(Action::arbitrary_with(((), ())), 1..100)) {
            let initial_state = State::from_seed(n_players as u64, 0, sb, sb * bb_mult as f64, sb * stake_mult as f64, 1234, false);
            match initial_state {
                Ok(mut s) => {
                    for a in actions {
                        s = s.apply_action(a);
                        prop_assert!(s.from_action != None);
                    }
                },
                Err(err) => {
                    println!("{}", err.msg);
                    prop_assert!(false);
                }
            };
        }
    }

    #[cfg(test)]
    fn card(suit: CardSuit, rank: CardRank) -> Card {
        Card { suit, rank }
    }

    #[cfg(test)]
    fn heads_up_state(button: u64) -> State {
        match State::from_seed(2, button, 1.0, 2.0, 100.0, 17, false) {
            Ok(state) => state,
            Err(error) => panic!("не удалось создать HU-состояние: {}", error.msg),
        }
    }

    #[cfg(test)]
    fn reconstruct_state(source: &State, current_player: Option<u64>) -> Result<State, InitStateError> {
        let stake = source.players_state[0].stake
            + source.players_state[0].bet_chips
            + source.players_state[0].pot_chips;

        State::from_mid_hand(
            source.players_state.len() as u64,
            source.button,
            source.sb,
            source.bb,
            stake,
            source.deck.clone(),
            source.players_state.iter().map(|player| player.hand).collect(),
            source.public_cards.clone(),
            source.stage,
            source.pot,
            source.players_state.iter().map(|player| player.bet_chips).collect(),
            source.players_state.iter().map(|player| player.pot_chips).collect(),
            source.players_state.iter().map(|player| player.active).collect(),
            source.players_state.iter().map(|player| player.last_stage_action).collect(),
            current_player,
            Some(source.last_raise_increment),
            source.verbose,
        )
    }

    #[cfg(test)]
    fn expect_reconstructed_state(result: Result<State, InitStateError>) -> State {
        match result {
            Ok(state) => state,
            Err(error) => panic!("не удалось восстановить состояние: {}", error.msg),
        }
    }

    #[cfg(test)]
    #[test]
    fn from_mid_hand_rejects_unresolved_showdown() {
        let showdown = State {
            stage: Stage::Showdown,
            public_cards: vec![
                card(CardSuit::Clubs, CardRank::R2),
                card(CardSuit::Diamonds, CardRank::R3),
                card(CardSuit::Hearts, CardRank::R4),
                card(CardSuit::Spades, CardRank::R5),
                card(CardSuit::Clubs, CardRank::R6),
            ],
            ..heads_up_state(0)
        };

        let error = reconstruct_state(&showdown, Some(0)).unwrap_err();

        assert_eq!(
            error.msg,
            "Cannot restore unresolved Showdown state; settlement is not supported"
        );
    }

    #[cfg(test)]
    #[test]
    fn from_mid_hand_requires_an_explicit_active_in_range_current_player() {
        let state = heads_up_state(0);

        let missing_actor = reconstruct_state(&state, None).unwrap_err();
        assert_eq!(
            missing_actor.msg,
            "current_player must be provided for mid-hand reconstruction"
        );

        let out_of_range_actor = reconstruct_state(&state, Some(2)).unwrap_err();
        assert_eq!(out_of_range_actor.msg, "current_player must be in range");

        let folded_actor_state = State {
            players_state: vec![
                state.players_state[0].clone(),
                PlayerState {
                    active: false,
                    ..state.players_state[1].clone()
                },
            ],
            ..state
        };
        let inactive_actor = reconstruct_state(&folded_actor_state, Some(1)).unwrap_err();
        assert_eq!(
            inactive_actor.msg,
            "current_player must identify an active player"
        );
    }

    #[cfg(test)]
    #[test]
    fn from_mid_hand_preserves_explicit_hu_and_six_max_actors() {
        for button in 0..2 {
            let initial = heads_up_state(button);
            let big_blind = (button + 1) % 2;
            assert_eq!(
                expect_reconstructed_state(reconstruct_state(&initial, Some(button))).current_player,
                button
            );

            let after_raise = initial.apply_action(Action::new(ActionEnum::Raise, 4.0));
            assert_eq!(
                expect_reconstructed_state(reconstruct_state(&after_raise, Some(big_blind)))
                    .current_player,
                big_blind
            );

            let flop = after_raise.apply_action(Action::new(ActionEnum::Call, 0.0));
            assert_eq!(flop.stage, Stage::Flop);
            assert_eq!(
                expect_reconstructed_state(reconstruct_state(&flop, Some(big_blind))).current_player,
                big_blind
            );
        }

        let six_max = match State::from_seed(6, 0, 1.0, 2.0, 100.0, 17, false) {
            Ok(state) => state,
            Err(error) => panic!("не удалось создать six-max состояние: {}", error.msg),
        };
        assert_eq!(
            expect_reconstructed_state(reconstruct_state(&six_max, Some(3))).current_player,
            3
        );
    }

    #[cfg(test)]
    #[test]
    fn heads_up_button_posts_small_blind_and_acts_first_preflop() {
        for button in 0..2 {
            let state = heads_up_state(button);
            let other_player = (button + 1) % 2;

            assert_eq!(state.players_state[button as usize].bet_chips, state.sb);
            assert_eq!(state.players_state[other_player as usize].bet_chips, state.bb);
            assert_eq!(state.current_player, button);
        }
    }

    #[cfg(test)]
    #[test]
    fn heads_up_big_blind_checks_after_button_limp_and_acts_first_postflop() {
        for button in 0..2 {
            let state = heads_up_state(button);
            let big_blind = (button + 1) % 2;
            let after_limp = state.apply_action(Action::new(ActionEnum::Call, 0.0));

            assert_eq!(after_limp.current_player, big_blind);
            assert!(after_limp.legal_actions.contains(&ActionEnum::Check));
            assert!(!after_limp.legal_actions.contains(&ActionEnum::Call));

            let flop = after_limp.apply_action(Action::new(ActionEnum::Check, 0.0));
            assert_eq!(flop.stage, Stage::Flop);
            assert_eq!(flop.current_player, big_blind);
        }
    }

    #[cfg(test)]
    #[test]
    fn six_max_blinds_and_preflop_actor_remain_unchanged() {
        let state = match State::from_seed(6, 0, 1.0, 2.0, 100.0, 17, false) {
            Ok(state) => state,
            Err(error) => panic!("не удалось создать six-max состояние: {}", error.msg),
        };

        assert_eq!(state.players_state[1].bet_chips, state.sb);
        assert_eq!(state.players_state[2].bet_chips, state.bb);
        assert_eq!(state.current_player, 3);
    }

    #[cfg(test)]
    #[test]
    fn wheel_is_weaker_than_six_high_straight_and_straight_flush() {
        let wheel = vec![
            card(CardSuit::Clubs, CardRank::R2),
            card(CardSuit::Diamonds, CardRank::R3),
            card(CardSuit::Hearts, CardRank::R4),
            card(CardSuit::Spades, CardRank::R5),
            card(CardSuit::Clubs, CardRank::RA),
        ];
        let six_high = vec![
            card(CardSuit::Clubs, CardRank::R2),
            card(CardSuit::Diamonds, CardRank::R3),
            card(CardSuit::Hearts, CardRank::R4),
            card(CardSuit::Spades, CardRank::R5),
            card(CardSuit::Clubs, CardRank::R6),
        ];
        let wheel_flush = vec![
            card(CardSuit::Hearts, CardRank::R2),
            card(CardSuit::Hearts, CardRank::R3),
            card(CardSuit::Hearts, CardRank::R4),
            card(CardSuit::Hearts, CardRank::R5),
            card(CardSuit::Hearts, CardRank::RA),
        ];
        let six_high_flush = vec![
            card(CardSuit::Spades, CardRank::R2),
            card(CardSuit::Spades, CardRank::R3),
            card(CardSuit::Spades, CardRank::R4),
            card(CardSuit::Spades, CardRank::R5),
            card(CardSuit::Spades, CardRank::R6),
        ];

        assert!(rank_card_combination(wheel) > rank_card_combination(six_high));
        assert!(rank_card_combination(wheel_flush) > rank_card_combination(six_high_flush));
    }
}
