// state/action.rs
#![allow(unused)]
use crate::state::stage::Stage;
#[cfg(test)]
use proptest_derive::Arbitrary;
use pyo3::prelude::*;
use strum_macros::EnumIter;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Action {
    #[pyo3(get, set)]
    pub action: ActionEnum,

    #[pyo3(get, set)]
    pub amount: f64,
}

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (action, amount=0.0))]
    pub fn new(action: ActionEnum, amount: f64) -> Action {
        Action {
            action: action,
            amount: amount,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, EnumIter, PartialEq, Eq)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum ActionEnum {
    Fold,
    Check,
    Call,
    Raise,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct ActionRecord {
    #[pyo3(get, set)]
    pub player: u64,

    #[pyo3(get, set)]
    pub stage: Stage,

    #[pyo3(get, set)]
    pub action: Action,

    #[pyo3(get, set)]
    pub legal_actions: Vec<ActionEnum>,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct PublicActionRecord {
    #[pyo3(get)]
    pub actor_id: u64,
    #[pyo3(get)]
    pub street: Stage,
    #[pyo3(get)]
    pub requested_action: Action,
    #[pyo3(get)]
    pub paid_amount: f64,
    #[pyo3(get)]
    pub applied_raise_increment: f64,
    #[pyo3(get)]
    pub is_effective_raise: bool,
}
