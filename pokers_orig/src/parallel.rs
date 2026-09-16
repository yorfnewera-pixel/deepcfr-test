// parallel.rs
use crate::state::action::Action;
use crate::state::State;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn parallel_apply_action(states: Vec<State>, actions: Vec<Action>) -> PyResult<Vec<State>> {
    if states.len() != actions.len() {
        return Err(PyValueError::new_err("Expected one action per state"));
    }
    Ok(states
        .par_iter()
        .zip(actions)
        .map(|(s, a)| s.apply_action(a))
        .collect())
}
