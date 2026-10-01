use pyo3::prelude::*;
pub mod action;
pub mod card;
pub mod stage;
use action::{ActionEnum, ActionRecord, PublicActionRecord};
use card::Card;
use stage::Stage;

/// Simulator state. This contains private information; agents need a separate observation.
#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    #[pyo3(get)]
    pub current_player: u64,
    #[pyo3(get)]
    pub players_state: Vec<PlayerState>,
    #[pyo3(get)]
    pub public_cards: Vec<Card>,
    #[pyo3(get)]
    pub stage: Stage,
    #[pyo3(get)]
    pub button: u64,
    #[pyo3(get)]
    pub from_action: Option<ActionRecord>,
    /// Полная публичная последовательность успешно применённых действий раздачи.
    #[pyo3(get)]
    pub action_history: Vec<PublicActionRecord>,
    /// Новая раздача всегда начинается с полной публичной истории.
    #[pyo3(get)]
    pub action_history_complete: bool,
    #[pyo3(get)]
    pub legal_actions: Vec<ActionEnum>,
    #[pyo3(get)]
    pub deck: Vec<Card>,
    pub pot: u64,
    pub min_bet: u64,
    pub bb: u64,
    pub ante: u64,
    pub last_full_raise: u64,
    #[pyo3(get)]
    pub chip_unit: f64,
    #[pyo3(get)]
    pub final_state: bool,
    #[pyo3(get)]
    pub status: StateStatus,
    #[pyo3(get, set)]
    pub verbose: bool,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct PlayerState {
    #[pyo3(get)]
    pub player: u64,
    #[pyo3(get)]
    pub hand: (Card, Card),
    pub bet_chips: u64,
    pub pot_chips: u64,
    pub stake: u64,
    pub reward: i64,
    pub initial_stake: u64,
    pub chip_unit: f64,
    #[pyo3(get)]
    pub active: bool,
    // A short raise requires a response but does not necessarily reopen raising.
    pub pending: bool,
    pub acted_at: Option<u64>,
}

#[pymethods]
impl PlayerState {
    #[getter]
    pub fn stake(&self) -> f64 {
        self.stake as f64 * self.chip_unit
    }
    #[getter]
    pub fn bet_chips(&self) -> f64 {
        self.bet_chips as f64 * self.chip_unit
    }
    #[getter]
    pub fn pot_chips(&self) -> f64 {
        self.pot_chips as f64 * self.chip_unit
    }
    #[getter]
    pub fn reward(&self) -> f64 {
        self.reward as f64 * self.chip_unit
    }
    pub fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateStatus {
    Ok,
    IllegalAction,
    LowBet,
    HighBet,
    InvalidAmount,
}
