"""Лабораторная проверка: может ли D2CFR-сеть выучить frozen mean targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ActionSlot, legal_action_mask
from tools import d2cfr_accumulation_probe as accumulation_probe
from tools import d2cfr_allin_bias_probe as allin_probe
from tools import d2cfr_signal_probe as signal_probe


_PAIRWISE_ACTIONS = (
    ("all_in_vs_raise_1pot", int(ActionSlot.ALL_IN), int(ActionSlot.RAISE_POT)),
    ("raise_1pot_vs_raise_0.5pot", int(ActionSlot.RAISE_POT), int(ActionSlot.RAISE_HALF_POT)),
)


def pairwise_ordering_metrics(
    action_targets: torch.Tensor,
    action_predictions: torch.Tensor,
    masks: torch.Tensor,
    regret_targets: torch.Tensor | None = None,
    regret_predictions: torch.Tensor | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Считает, сохраняет ли сеть порядок value действий на legal парах."""
    report: dict[str, dict[str, float | int | None]] = {}
    for label, first, second in _PAIRWISE_ACTIONS:
        legal = (masks[:, first] > 0.0) & (masks[:, second] > 0.0)
        target_gap = action_targets[:, first] - action_targets[:, second]
        predicted_gap = action_predictions[:, first] - action_predictions[:, second]
        comparable = legal & (target_gap.abs() > 1e-8)
        count = int(legal.sum().detach().cpu().item())
        comparable_count = int(comparable.sum().detach().cpu().item())
        row: dict[str, float | int | None] = {
            "legal_pairs": count,
            "comparable_pairs": comparable_count,
            "q_ordering_accuracy": (
                float(
                    ((target_gap[comparable] > 0.0) == (predicted_gap[comparable] > 0.0))
                    .to(dtype=torch.float32).mean().detach().cpu().item()
                )
                if comparable_count
                else None
            ),
            "q_gap_abs_error_mean": (
                float((predicted_gap[comparable] - target_gap[comparable]).abs().mean().detach().cpu().item())
                if comparable_count
                else None
            ),
        }
        if regret_targets is not None and regret_predictions is not None:
            target_regret_gap = regret_targets[:, first] - regret_targets[:, second]
            predicted_regret_gap = regret_predictions[:, first] - regret_predictions[:, second]
            comparable_regret = legal & (target_regret_gap.abs() > 1e-8)
            regret_count = int(comparable_regret.sum().detach().cpu().item())
            row["regret_comparable_pairs"] = regret_count
            row["regret_ordering_accuracy"] = (
                float(
                    ((target_regret_gap[comparable_regret] > 0.0)
                     == (predicted_regret_gap[comparable_regret] > 0.0))
                    .to(dtype=torch.float32).mean().detach().cpu().item()
                )
                if regret_count
                else None
            )
        report[label] = row
    return report


def _evaluate_with_ordering(network: Any, batch: signal_probe.ProbeBatch) -> dict[str, Any]:
    report = accumulation_probe.evaluate_prediction_snapshot(network, batch)
    was_training = bool(getattr(network, "training", False))
    network.eval()
    with torch.no_grad():
        components = network.forward_components(batch.states)
        report["pairwise_ordering"] = pairwise_ordering_metrics(
            batch.action_targets,
            components.action_values,
            batch.masks,
            batch.regret_targets,
            components.regrets,
        )
    if was_training:
        network.train()
    return report


def _fit_arm(
    *,
    name: str,
    network_factory: Any,
    training_batch: signal_probe.ProbeBatch,
    reference_batch: signal_probe.ProbeBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None,
) -> dict[str, Any]:
    device = training_batch.states.device
    indices_batches = accumulation_probe._random_index_batches(
        sample_count=int(training_batch.states.shape[0]),
        batch_size=int(batch_size),
        steps=int(steps),
        seed=int(seed),
        device=device,
    )
    torch.manual_seed(int(seed))
    network = network_factory().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    initial = _evaluate_with_ordering(network, reference_batch)
    network.train()
    for indices in indices_batches:
        batch = accumulation_probe._slice_probe_batch(training_batch, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = accumulation_probe._weighted_d2cfr_loss(
            network,
            batch,
            torch.ones_like(batch.iterations),
            state_value_loss_weight=state_value_loss_weight,
        )
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float(max_grad_norm))
        optimizer.step()
    final = _evaluate_with_ordering(network, reference_batch)
    return {
        "name": name,
        "training_samples": int(training_batch.states.shape[0]),
        "initial": initial,
        "final": final,
        "final_by_action": accumulation_probe._action_prediction_metrics(network, reference_batch),
        "final_pairwise_ordering": final["pairwise_ordering"],
    }


def run_noisy_vs_mean_fit(
    *,
    network_factory: Any,
    mean_batch: signal_probe.ProbeBatch,
    noisy_batch: signal_probe.ProbeBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None = 1.0,
) -> dict[str, Any]:
    """Сравнивает direct fit mean targets и fit отдельных noisy labels."""
    if min(int(mean_batch.states.shape[0]), int(noisy_batch.states.shape[0])) < 1:
        raise ValueError("Mean и noisy batch не должны быть пустыми")
    if mean_batch.states.device != noisy_batch.states.device:
        raise ValueError("Mean и noisy batch должны быть на одном device")
    if steps < 0 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size и learning_rate должны быть положительными")
    shared = {
        "network_factory": network_factory,
        "reference_batch": mean_batch,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "state_value_loss_weight": float(state_value_loss_weight),
        "seed": int(seed),
        "max_grad_norm": max_grad_norm,
    }
    return {
        "reference_samples": int(mean_batch.states.shape[0]),
        "noisy_training_samples": int(noisy_batch.states.shape[0]),
        "arms": {
            "mean_targets": _fit_arm(name="mean_targets", training_batch=mean_batch, **shared),
            "noisy_targets": _fit_arm(name="noisy_targets", training_batch=noisy_batch, **shared),
        },
    }


def run_mean_target_train_validation_fit(
    *,
    network_factory: Any,
    training_batch: signal_probe.ProbeBatch,
    validation_batch: signal_probe.ProbeBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None = 1.0,
) -> dict[str, Any]:
    """Учит на frozen mean train-state и отдельно оценивает невиданные validation-state."""
    if min(
        int(training_batch.states.shape[0]),
        int(validation_batch.states.shape[0]),
    ) < 1:
        raise ValueError("Train и validation batch не должны быть пустыми")
    if training_batch.states.device != validation_batch.states.device:
        raise ValueError("Train и validation batch должны быть на одном device")
    if steps < 0 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size и learning_rate должны быть положительными")

    device = training_batch.states.device
    index_batches = accumulation_probe._random_index_batches(
        sample_count=int(training_batch.states.shape[0]),
        batch_size=int(batch_size),
        steps=int(steps),
        seed=int(seed),
        device=device,
    )
    torch.manual_seed(int(seed))
    network = network_factory().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))

    def evaluate() -> dict[str, Any]:
        return {
            "train": _evaluate_with_ordering(network, training_batch),
            "validation": _evaluate_with_ordering(network, validation_batch),
        }

    initial = evaluate()
    network.train()
    for indices in index_batches:
        batch = accumulation_probe._slice_probe_batch(training_batch, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = accumulation_probe._weighted_d2cfr_loss(
            network,
            batch,
            torch.ones_like(batch.iterations),
            state_value_loss_weight=state_value_loss_weight,
        )
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float(max_grad_norm))
        optimizer.step()
    final = evaluate()
    return {
        "training_samples": int(training_batch.states.shape[0]),
        "validation_samples": int(validation_batch.states.shape[0]),
        "initial": initial,
        "final": final,
        "final_by_action": {
            "train": accumulation_probe._action_prediction_metrics(network, training_batch),
            "validation": accumulation_probe._action_prediction_metrics(network, validation_batch),
        },
    }


def run_checkpoint_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    states: int = 64,
    repeats: int = 64,
    roots: int = 4_096,
    fit_steps: int = 5_000,
    fit_batch_size: int = 64,
    validation_states: int = 0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Загружает checkpoint и сохраняет direct mean-vs-noisy fit report."""
    from src.core.deep_cfr import DeepCFRAgent
    from src.training import train as train_mod
    from src.utils import config as config_mod

    if min(states, repeats, roots, fit_batch_size) < 1 or fit_steps < 0:
        raise ValueError("Размеры probe должны быть положительными, fit_steps — неотрицательным")
    if validation_states < 0 or validation_states >= states:
        raise ValueError("validation_states должен быть неотрицательным и меньше states")
    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=str(device))
    coordinator = train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled:
        raise ValueError("Нужен D2CFR HU checkpoint")

    checkpoint_iteration = int(checkpoint["iteration"])
    iteration = max(1, checkpoint_iteration)
    frozen_states = allin_probe._collect_candidate_states(
        coordinator, roots=int(roots), seed=int(seed), target_count=int(states), traverser=0, iteration=iteration,
    )
    if len(frozen_states) != int(states):
        raise RuntimeError(f"Собрано только {len(frozen_states)} из {states} postflop-состояний; увеличьте --roots")
    mean_samples: list[tuple] = []
    noisy_samples: list[tuple] = []
    max_consistency_error = 0.0
    for state in frozen_states:
        target_samples = allin_probe._sample_state_targets(
            coordinator, state, repeats=int(repeats), traverser=0, iteration=iteration,
        )
        q_values = np.stack([sample[0] for sample in target_samples], axis=0)
        state_values = np.asarray([sample[1] for sample in target_samples], dtype=np.float64)
        regrets = np.stack([sample[2] for sample in target_samples], axis=0)
        mean_q = q_values.mean(axis=0).astype(np.float32)
        mean_v = np.float32(state_values.mean())
        mean_r = regrets.mean(axis=0).astype(np.float32)
        mask = legal_action_mask(state).astype(np.float32)
        legal = mask > 0.0
        max_consistency_error = max(
            max_consistency_error,
            float(np.max(np.abs(mean_r[legal] - (mean_q[legal] - mean_v)))) if np.any(legal) else 0.0,
        )
        encoded = agent._encode_state(state, 0)
        mean_samples.append((encoded, mean_q, mean_v, mean_r, mask, np.float32(iteration)))
        if not validation_states:
            for q, value, regret in target_samples:
                noisy_samples.append((
                    encoded, np.asarray(q, dtype=np.float32), np.float32(value),
                    np.asarray(regret, dtype=np.float32), mask, np.float32(iteration),
                ))
    mean_batch = signal_probe.build_probe_batch(mean_samples, device=device)
    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration,
        "traversal_iteration": iteration,
        "config": str(config_path),
        "seed": int(seed),
        "frozen_targets": {
            "states": int(states),
            "repeats_per_state": int(repeats),
            "roots_examined": int(roots),
            "mean_target_consistency_max_abs_error": max_consistency_error,
        },
        "fit": {
            "loss_function": str(agent.d2cfr_loss_function),
            "loss_mode": str(agent.d2cfr_loss_mode),
            "gradient_clip_norm": 1.0 if agent.d2cfr_loss_mode == "anchored" else None,
            "steps": int(fit_steps),
            "batch_size": int(fit_batch_size),
        },
    }
    if validation_states:
        split_indices = np.random.default_rng(int(seed)).permutation(int(states))
        validation_indices = split_indices[:int(validation_states)]
        training_indices = split_indices[int(validation_states):]
        training_batch = signal_probe.build_probe_batch(
            [mean_samples[int(index)] for index in training_indices], device=device
        )
        validation_batch = signal_probe.build_probe_batch(
            [mean_samples[int(index)] for index in validation_indices], device=device
        )
        report["fit"]["train_validation"] = {
            "split_seed": int(seed),
            "mean_target_fit": run_mean_target_train_validation_fit(
                network_factory=agent._new_advantage_network,
                training_batch=training_batch,
                validation_batch=validation_batch,
                steps=int(fit_steps),
                batch_size=int(fit_batch_size),
                learning_rate=float(agent.advantage_lr),
                state_value_loss_weight=float(agent.d2cfr_state_value_loss_weight),
                seed=int(seed),
                max_grad_norm=1.0 if agent.d2cfr_loss_mode == "anchored" else None,
            ),
        }
    else:
        noisy_batch = signal_probe.build_probe_batch(noisy_samples, device=device)
        report["fit"]["comparison"] = run_noisy_vs_mean_fit(
            network_factory=agent._new_advantage_network,
            mean_batch=mean_batch,
            noisy_batch=noisy_batch,
            steps=int(fit_steps),
            batch_size=int(fit_batch_size),
            learning_rate=float(agent.advantage_lr),
            state_value_loss_weight=float(agent.d2cfr_state_value_loss_weight),
            seed=int(seed),
            max_grad_norm=1.0 if agent.d2cfr_loss_mode == "anchored" else None,
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--roots", type=int, default=4096)
    parser.add_argument("--fit-steps", type=int, default=5000)
    parser.add_argument("--fit-batch-size", type=int, default=64)
    parser.add_argument("--validation-states", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checkpoint_probe(
        args.checkpoint, config_path=args.config, output_path=args.output,
        states=args.states, repeats=args.repeats, roots=args.roots,
        fit_steps=args.fit_steps, fit_batch_size=args.fit_batch_size,
        validation_states=args.validation_states,
        seed=args.seed, device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
