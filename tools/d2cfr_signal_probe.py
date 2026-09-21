"""Диагностика прохождения сигнала через D2CFR advantage-сеть."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import dueling_checkpoint_network_spec
from src.core.model import DuelingRegretNetwork


@dataclass(frozen=True)
class ProbeBatch:
    states: torch.Tensor
    action_targets: torch.Tensor
    state_value_targets: torch.Tensor
    regret_targets: torch.Tensor
    masks: torch.Tensor
    iterations: torch.Tensor


__all__ = [
    "ProbeBatch",
    "analyse_targets",
    "build_probe_batch",
    "build_signal_report",
    "collect_live_probe",
    "describe_sample_tensors",
    "load_checkpoint_probe",
    "load_live_probe",
    "run_microfit_signal_probe",
    "run_one_step_signal_probe",
    "samples_from_dueling_buffer",
]


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def build_probe_batch(samples: Iterable[tuple], device: str | torch.device = "cpu") -> ProbeBatch:
    """Собирает D2CFR samples в batch без обращения к replay buffer."""
    rows = list(samples)
    if not rows:
        raise ValueError("samples не должен быть пустым")
    states, action_targets, state_targets, regret_targets, masks, iterations = zip(*rows, strict=True)
    target_device = torch.device(device)
    batch = ProbeBatch(
        states=_as_float_tensor(np.asarray(states, dtype=np.float32), target_device),
        action_targets=_as_float_tensor(np.asarray(action_targets, dtype=np.float32), target_device),
        state_value_targets=_as_float_tensor(np.asarray(state_targets, dtype=np.float32), target_device),
        regret_targets=_as_float_tensor(np.asarray(regret_targets, dtype=np.float32), target_device),
        masks=_as_float_tensor(np.asarray(masks, dtype=np.float32), target_device),
        iterations=_as_float_tensor(np.asarray(iterations, dtype=np.float32), target_device),
    )
    _validate_batch_shapes(batch)
    return batch


def _validate_batch_shapes(batch: ProbeBatch) -> None:
    sample_count = int(batch.states.shape[0])
    if batch.action_targets.shape != (sample_count, NUM_ACTIONS):
        raise ValueError("action_targets должны иметь форму [N, 6]")
    if batch.regret_targets.shape != (sample_count, NUM_ACTIONS):
        raise ValueError("regret_targets должны иметь форму [N, 6]")
    if batch.masks.shape != (sample_count, NUM_ACTIONS):
        raise ValueError("masks должны иметь форму [N, 6]")
    if batch.state_value_targets.shape != (sample_count,):
        raise ValueError("state_value_targets должны иметь форму [N]")
    if batch.iterations.shape != (sample_count,):
        raise ValueError("iterations должны иметь форму [N]")
    tensors = (
        batch.states,
        batch.action_targets,
        batch.state_value_targets,
        batch.regret_targets,
        batch.masks,
        batch.iterations,
    )
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("batch содержит NaN или Inf")
    if not torch.all((batch.masks == 0.0) | (batch.masks == 1.0)).item():
        raise ValueError("masks должны содержать только 0 или 1")


def _stats(values: torch.Tensor, zero_atol: float) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "zero_fraction": 0.0,
        }
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "zero_fraction": float((flat.abs() <= float(zero_atol)).float().mean().item()),
    }


def analyse_targets(
    batch: ProbeBatch,
    *,
    identity_atol: float = 1e-5,
    zero_atol: float = 1e-8,
) -> dict[str, Any]:
    """Проверяет, есть ли target-сигнал и согласованы ли Q, V, R."""
    legal = batch.masks.bool()
    q_minus_v = batch.action_targets - batch.state_value_targets.unsqueeze(1)
    legal_errors = (q_minus_v - batch.regret_targets).abs()[legal]
    max_abs_error = float(legal_errors.max().item()) if legal_errors.numel() else 0.0
    return {
        "samples": int(batch.states.shape[0]),
        "legal_slots": int(legal.sum().item()),
        "q_minus_v_ok": bool(max_abs_error <= float(identity_atol)),
        "q_minus_v_max_abs_error": max_abs_error,
        "action_value_legal": _stats(batch.action_targets[legal], zero_atol),
        "regret_legal": _stats(batch.regret_targets[legal], zero_atol),
        "state_value": _stats(batch.state_value_targets, zero_atol),
        "iterations": _stats(batch.iterations, zero_atol),
    }


def _elementwise_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_function: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_function == "mse":
        return (predictions - targets).pow(2)
    if loss_function == "huber":
        return F.huber_loss(predictions, targets, reduction="none", delta=float(huber_delta))
    raise ValueError(f"Неизвестная loss-функция D2CFR: {loss_function}")


def _loss_components(
    network: DuelingRegretNetwork,
    batch: ProbeBatch,
    *,
    state_value_loss_weight: float,
    loss_function: str,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = network.forward_components(batch.states)
    regret_element_loss = _elementwise_loss(
        output.regrets,
        batch.regret_targets,
        loss_function,
        huber_delta,
    )
    regret_loss = (regret_element_loss * batch.masks).sum(dim=1).mean()
    predicted_state_values = output.state_values.squeeze(-1)
    state_value_loss = _elementwise_loss(
        predicted_state_values,
        batch.state_value_targets,
        loss_function,
        huber_delta,
    ).mean()
    total_loss = regret_loss + float(state_value_loss_weight) * state_value_loss
    return total_loss, {
        "regret_loss": float(regret_loss.detach().item()),
        "state_value_loss": float(state_value_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }


def _module_name(parameter_name: str) -> str:
    return parameter_name.split(".", 1)[0]


def _module_norms(network: torch.nn.Module) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, parameter in network.named_parameters():
        module = _module_name(name)
        current = result.setdefault(module, {"grad_norm_sq": 0.0, "update_norm_sq": 0.0})
        if parameter.grad is not None:
            current["grad_norm_sq"] += float(parameter.grad.detach().pow(2).sum().item())
    return result


def run_one_step_signal_probe(
    network: DuelingRegretNetwork,
    batch: ProbeBatch,
    *,
    learning_rate: float = 1e-3,
    state_value_loss_weight: float = 0.5,
    loss_function: str = "mse",
    huber_delta: float = 1.0,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Делает один SGD-шаг на клоне сети и возвращает loss/grad/update."""
    probe_network: DuelingRegretNetwork = copy.deepcopy(network).to(batch.states.device)
    probe_network.eval()
    before_params = {
        name: parameter.detach().clone()
        for name, parameter in probe_network.named_parameters()
    }
    optimizer = torch.optim.SGD(probe_network.parameters(), lr=float(learning_rate))

    with torch.no_grad():
        before_total, before_losses = _loss_components(
            probe_network,
            batch,
            state_value_loss_weight=state_value_loss_weight,
            loss_function=loss_function,
            huber_delta=huber_delta,
        )
        before_output = probe_network.forward_components(batch.states)

    optimizer.zero_grad(set_to_none=True)
    train_loss, _ = _loss_components(
        probe_network,
        batch,
        state_value_loss_weight=state_value_loss_weight,
        loss_function=loss_function,
        huber_delta=huber_delta,
    )
    train_loss.backward()
    module_norms = _module_norms(probe_network)
    grad_norm_total = _total_grad_norm(probe_network)
    clipped_grad_norm = None
    if max_grad_norm is not None:
        clipped_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(probe_network.parameters(), float(max_grad_norm)).item()
        )
    optimizer.step()

    with torch.no_grad():
        after_total, after_losses = _loss_components(
            probe_network,
            batch,
            state_value_loss_weight=state_value_loss_weight,
            loss_function=loss_function,
            huber_delta=huber_delta,
        )
        after_output = probe_network.forward_components(batch.states)

    update_norm_total = 0.0
    for name, parameter in probe_network.named_parameters():
        module = _module_name(name)
        update_norm_sq = float((parameter.detach() - before_params[name]).pow(2).sum().item())
        module_norms.setdefault(module, {"grad_norm_sq": 0.0, "update_norm_sq": 0.0})
        module_norms[module]["update_norm_sq"] += update_norm_sq
        update_norm_total += update_norm_sq

    modules = {
        module: {
            "grad_norm": float(values["grad_norm_sq"] ** 0.5),
            "update_norm": float(values["update_norm_sq"] ** 0.5),
        }
        for module, values in sorted(module_norms.items())
    }
    alignment = _regret_alignment(before_output.regrets, after_output.regrets, batch)
    return {
        "loss_before": float(before_total.item()),
        "loss_after": float(after_total.item()),
        "loss_delta": float(after_total.item() - before_total.item()),
        "losses_before": before_losses,
        "losses_after": after_losses,
        "grad_norm_total": grad_norm_total,
        "clipped_grad_norm": clipped_grad_norm,
        "update_norm_total": float(update_norm_total ** 0.5),
        "modules": modules,
        "per_sample_regret_alignment": alignment,
    }


def run_microfit_signal_probe(
    network: DuelingRegretNetwork,
    batch: ProbeBatch,
    *,
    steps: int,
    learning_rate: float,
    state_value_loss_weight: float = 0.5,
    loss_function: str = "mse",
    huber_delta: float = 1.0,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Обучает клон на фиксированном batch и пишет историю loss/grad."""
    step_count = int(steps)
    if step_count <= 0:
        raise ValueError("steps должно быть положительным")
    probe_network: DuelingRegretNetwork = copy.deepcopy(network).to(batch.states.device)
    probe_network.eval()
    optimizer = torch.optim.SGD(probe_network.parameters(), lr=float(learning_rate))
    history: list[dict[str, Any]] = []
    max_module_grad_norms: dict[str, float] = {}
    initial_loss: float | None = None
    final_loss = 0.0

    for step in range(1, step_count + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_losses = _loss_components(
            probe_network,
            batch,
            state_value_loss_weight=state_value_loss_weight,
            loss_function=loss_function,
            huber_delta=huber_delta,
        )
        if initial_loss is None:
            initial_loss = float(train_loss.detach().item())
        train_loss.backward()
        module_norms = _module_norms(probe_network)
        grad_norm_total = _total_grad_norm(probe_network)
        clipped_grad_norm = None
        if max_grad_norm is not None:
            clipped_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    probe_network.parameters(),
                    float(max_grad_norm),
                ).item()
            )
        optimizer.step()

        with torch.no_grad():
            after_loss, after_losses = _loss_components(
                probe_network,
                batch,
                state_value_loss_weight=state_value_loss_weight,
                loss_function=loss_function,
                huber_delta=huber_delta,
            )
        final_loss = float(after_loss.item())
        modules = {
            module: {
                "grad_norm": float(values["grad_norm_sq"] ** 0.5),
            }
            for module, values in sorted(module_norms.items())
        }
        for module, values in modules.items():
            max_module_grad_norms[module] = max(
                max_module_grad_norms.get(module, 0.0),
                float(values["grad_norm"]),
            )
        history.append(
            {
                "step": step,
                "loss_before": float(train_loss.detach().item()),
                "loss_after": final_loss,
                "losses_before": train_losses,
                "losses_after": after_losses,
                "grad_norm_total": grad_norm_total,
                "clipped_grad_norm": clipped_grad_norm,
                "modules": modules,
            }
        )

    return {
        "steps": step_count,
        "learning_rate": float(learning_rate),
        "initial_loss": float(initial_loss if initial_loss is not None else 0.0),
        "final_loss": final_loss,
        "loss_delta": final_loss - float(initial_loss if initial_loss is not None else 0.0),
        "max_module_grad_norms": max_module_grad_norms,
        "history": history,
    }


def _total_grad_norm(network: DuelingRegretNetwork) -> float:
    total = 0.0
    for parameter in network.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


def _regret_alignment(
    regrets_before: torch.Tensor,
    regrets_after: torch.Tensor,
    batch: ProbeBatch,
) -> dict[str, float | int]:
    update = (regrets_after - regrets_before) * batch.masks
    residual = (batch.regret_targets - regrets_before) * batch.masks
    dots = (update * residual).sum(dim=1)
    return {
        "samples": int(dots.numel()),
        "mean_dot": float(dots.mean().item()) if dots.numel() else 0.0,
        "positive_fraction": float((dots > 0.0).float().mean().item()) if dots.numel() else 0.0,
        "negative_fraction": float((dots < 0.0).float().mean().item()) if dots.numel() else 0.0,
    }


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]


def _top_abs_features(state: torch.Tensor, max_features: int) -> list[dict[str, float | int]]:
    flat = state.detach().cpu().float().reshape(-1)
    nonzero = torch.nonzero(flat != 0.0, as_tuple=False).reshape(-1)
    if nonzero.numel() == 0:
        return []
    ranked = sorted(
        ((int(index), float(flat[index].item())) for index in nonzero.tolist()),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return [
        {"index": index, "value": value}
        for index, value in ranked[: max(0, int(max_features))]
    ]


def _legal_target_rows(
    q_values: torch.Tensor,
    regrets: torch.Tensor,
    legal_slots: list[int],
) -> list[dict[str, float | int]]:
    return [
        {
            "slot": int(slot),
            "q": float(q_values[slot].detach().cpu().item()),
            "regret": float(regrets[slot].detach().cpu().item()),
        }
        for slot in legal_slots
    ]


def describe_sample_tensors(
    network: DuelingRegretNetwork,
    batch: ProbeBatch,
    *,
    sample_limit: int,
    max_features: int,
) -> dict[str, Any]:
    """Показывает конкретные входы, targets и предсказания сети по samples."""
    limit = min(max(0, int(sample_limit)), int(batch.states.shape[0]))
    with torch.no_grad():
        output = network.forward_components(batch.states)
    samples = []
    for index in range(limit):
        legal_slots = [
            int(slot)
            for slot in torch.nonzero(batch.masks[index] > 0.0, as_tuple=False).reshape(-1).tolist()
        ]
        predicted_q = output.action_values[index]
        predicted_regrets = output.regrets[index]
        target_q = batch.action_targets[index]
        target_regrets = batch.regret_targets[index]
        samples.append(
            {
                "sample_index": index,
                "legal_slots": legal_slots,
                "input": {
                    "nonzero_count": int((batch.states[index] != 0.0).sum().item()),
                    "top_abs_features": _top_abs_features(batch.states[index], max_features),
                },
                "targets": {
                    "action_values": _float_list(target_q),
                    "state_value": float(batch.state_value_targets[index].detach().cpu().item()),
                    "regrets": _float_list(target_regrets),
                    "legal": _legal_target_rows(target_q, target_regrets, legal_slots),
                },
                "network_before": {
                    "action_values": _float_list(predicted_q),
                    "state_value": float(output.state_values[index].squeeze().detach().cpu().item()),
                    "regrets": _float_list(predicted_regrets),
                    "legal": _legal_target_rows(predicted_q, predicted_regrets, legal_slots),
                },
            }
        )
    return {
        "input_size": int(batch.states.shape[1]),
        "dumped_samples": len(samples),
        "max_features_per_sample": int(max_features),
        "samples": samples,
    }


def _samples_from_checkpoint_buffer(
    buffer_payload: dict[str, Any],
    *,
    sample_count: int,
    seed: int,
) -> list[tuple]:
    required = ("states", "action_values", "state_values", "regrets", "masks", "iterations")
    if not isinstance(buffer_payload, dict) or any(key not in buffer_payload for key in required):
        raise ValueError("checkpoint не содержит D2CFR advantage_buffer")
    size = int(buffer_payload.get("size", 0))
    if size <= 0:
        raise ValueError("D2CFR advantage_buffer пуст")
    take = min(int(sample_count), size)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(size, take, replace=False)
    arrays = {key: np.asarray(buffer_payload[key]) for key in required}
    return [
        (
            arrays["states"][index],
            arrays["action_values"][index],
            arrays["state_values"][index],
            arrays["regrets"][index],
            arrays["masks"][index],
            arrays["iterations"][index],
        )
        for index in indices
    ]


def samples_from_dueling_buffer(
    buffer: Any,
    *,
    sample_count: int,
    seed: int,
) -> list[tuple]:
    """Копирует samples из живого D2CFR replay-buffer без изменения RNG буфера."""
    size = len(buffer)
    if size <= 0:
        raise ValueError("D2CFR advantage_buffer пуст")
    take = min(int(sample_count), size)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(size, take, replace=False)
    required_arrays = (
        "_states",
        "_action_values",
        "_state_values",
        "_regrets",
        "_masks",
        "_iterations",
    )
    try:
        arrays = [getattr(buffer, name) for name in required_arrays]
    except AttributeError as error:
        raise ValueError("live buffer не похож на DuelingAdvantageBuffer") from error
    return [
        (
            arrays[0][index].copy(),
            arrays[1][index].copy(),
            np.float32(arrays[2][index]),
            arrays[3][index].copy(),
            arrays[4][index].copy(),
            np.float32(arrays[5][index]),
        )
        for index in indices
    ]


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _set_eval_mode(agent: Any) -> None:
    for name in ("advantage_net", "advantage_target_net", "strategy_net"):
        network = getattr(agent, name, None)
        if network is not None:
            network.eval()


def collect_live_probe(
    agent: Any,
    *,
    roots: Iterable[Any],
    iteration: int,
    traversing_player: int,
    sample_count: int,
    seed: int,
    device: str | torch.device,
    state_factory: Any,
) -> tuple[DuelingRegretNetwork, ProbeBatch, dict[str, Any]]:
    """Собирает D2CFR samples через production traversal, не запуская обучение."""
    if not bool(getattr(agent, "d2cfr_enabled", False)):
        raise ValueError("live-зонд требует d2cfr_enabled=true")
    buffer = getattr(agent, "d2cfr_buffer", None)
    if buffer is None:
        raise ValueError("agent не содержит d2cfr_buffer")
    roots = list(roots)
    if not roots:
        raise ValueError("roots не должен быть пустым")
    _seed_everything(seed)
    _set_eval_mode(agent)
    agent.prepare_iteration(int(iteration), int(traversing_player))
    agent.reset_traversal_stats()
    for root in roots:
        agent.cfr_traverse_multi(state_factory(root), int(iteration), int(traversing_player))
    samples = samples_from_dueling_buffer(
        buffer,
        sample_count=sample_count,
        seed=seed,
    )
    return (
        agent.advantage_net,
        build_probe_batch(samples, device=device),
        {
            "mode": "live_traversal",
            "traversals": len(roots),
            "buffer_size": len(buffer),
            "iteration": int(iteration),
            "traversing_player": int(traversing_player),
            "samples_requested": int(sample_count),
            "samples_used": int(len(samples)),
        },
    )


def load_checkpoint_probe(
    checkpoint_path: str | Path,
    *,
    sample_count: int,
    seed: int,
    device: str | torch.device,
) -> tuple[DuelingRegretNetwork, ProbeBatch, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("algorithm_variant") != "d2cfr_dueling_v1":
        raise ValueError("checkpoint не является D2CFR dueling checkpoint")
    architecture, input_size, hidden_size = dueling_checkpoint_network_spec(checkpoint)
    network = DuelingRegretNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        architecture=architecture,
    )
    network.load_state_dict(checkpoint["advantage_net"])
    network.to(torch.device(device))
    network.eval()
    samples = _samples_from_checkpoint_buffer(
        checkpoint.get("advantage_buffer"),
        sample_count=sample_count,
        seed=seed,
    )
    return network, build_probe_batch(samples, device=device), checkpoint


def load_live_probe(
    config_path: str | Path | None,
    *,
    collect_traversals: int,
    sample_count: int,
    seed: int,
    iteration: int,
    traversing_player: int,
    num_players: int | None,
    device: str | torch.device,
) -> tuple[DuelingRegretNetwork, ProbeBatch, dict[str, Any], dict[str, Any]]:
    from src.core.deep_cfr import DeepCFRAgent
    from src.utils import config as config_mod
    from tools.benchmark_traversal import generate_root_bank, make_state

    if collect_traversals <= 0:
        raise ValueError("collect-traversals должно быть положительным")
    if config_path is not None:
        config_mod.load_config(config_path)
    configured_players = config_mod.cfg_get("num_players", 6)
    player_count = int(num_players) if num_players is not None else int(configured_players or 6)
    agent = DeepCFRAgent(
        player_id=int(traversing_player),
        num_players=player_count,
        device=str(device),
    )
    roots = generate_root_bank(int(collect_traversals), int(seed), player_count)
    network, batch, metadata = collect_live_probe(
        agent,
        roots=roots,
        iteration=iteration,
        traversing_player=traversing_player,
        sample_count=sample_count,
        seed=seed,
        device=device,
        state_factory=lambda root: make_state(root, player_count),
    )
    config = {
        "d2cfr_state_value_loss_weight": float(config_mod.cfg_get("d2cfr_state_value_loss_weight", 0.5)),
        "d2cfr_loss_function": str(config_mod.cfg_get("d2cfr_loss_function", "mse")),
        "d2cfr_huber_delta": float(config_mod.cfg_get("d2cfr_huber_delta", 1.0)),
    }
    metadata["num_players"] = player_count
    if config_path is not None:
        metadata["config"] = str(config_path)
    return network, batch, config, metadata


def build_signal_report(
    network: DuelingRegretNetwork,
    batch: ProbeBatch,
    *,
    learning_rate: float,
    state_value_loss_weight: float,
    loss_function: str,
    huber_delta: float,
    max_grad_norm: float | None,
    sample_dump_limit: int = 0,
    max_features: int = 32,
    microfit_steps: int = 0,
    microfit_learning_rate: float | None = None,
) -> dict[str, Any]:
    report = {
        "targets": analyse_targets(batch),
        "one_step": run_one_step_signal_probe(
            network,
            batch,
            learning_rate=learning_rate,
            state_value_loss_weight=state_value_loss_weight,
            loss_function=loss_function,
            huber_delta=huber_delta,
            max_grad_norm=max_grad_norm,
        ),
    }
    if sample_dump_limit > 0:
        report["sample_tensors"] = describe_sample_tensors(
            network,
            batch,
            sample_limit=sample_dump_limit,
            max_features=max_features,
        )
    if microfit_steps > 0:
        report["microfit"] = run_microfit_signal_probe(
            network,
            batch,
            steps=microfit_steps,
            learning_rate=(
                float(microfit_learning_rate)
                if microfit_learning_rate is not None
                else float(learning_rate)
            ),
            state_value_loss_weight=state_value_loss_weight,
            loss_function=loss_function,
            huber_delta=huber_delta,
            max_grad_norm=max_grad_norm,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--collect-traversals", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--traversing-player", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--state-value-loss-weight", type=float, default=None)
    parser.add_argument("--loss-function", choices=("mse", "huber"), default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--dump-samples", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--microfit-steps", type=int, default=0)
    parser.add_argument("--microfit-learning-rate", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples должно быть положительным")
    checkpoint_mode = args.checkpoint is not None
    live_mode = args.collect_traversals > 0
    if checkpoint_mode == live_mode:
        raise ValueError("Укажи ровно один режим: --checkpoint или --collect-traversals")
    metadata_key = "checkpoint"
    if checkpoint_mode:
        network, batch, checkpoint = load_checkpoint_probe(
            args.checkpoint,
            sample_count=args.samples,
            seed=args.seed,
            device=args.device,
        )
        config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
        metadata = {
            "path": str(args.checkpoint),
            "iteration": checkpoint.get("iteration"),
            "samples_requested": int(args.samples),
            "samples_used": int(batch.states.shape[0]),
        }
    else:
        network, batch, config, metadata = load_live_probe(
            args.config,
            collect_traversals=args.collect_traversals,
            sample_count=args.samples,
            seed=args.seed,
            iteration=args.iteration,
            traversing_player=args.traversing_player,
            num_players=args.num_players,
            device=args.device,
        )
        metadata_key = "live_collection"
    report = build_signal_report(
        network,
        batch,
        learning_rate=args.learning_rate,
        state_value_loss_weight=(
            float(args.state_value_loss_weight)
            if args.state_value_loss_weight is not None
            else float(config.get("d2cfr_state_value_loss_weight", 0.5))
        ),
        loss_function=str(args.loss_function or config.get("d2cfr_loss_function", "mse")),
        huber_delta=(
            float(args.huber_delta)
            if args.huber_delta is not None
            else float(config.get("d2cfr_huber_delta", 1.0))
        ),
        max_grad_norm=args.max_grad_norm,
        sample_dump_limit=args.dump_samples,
        max_features=args.max_features,
        microfit_steps=args.microfit_steps,
        microfit_learning_rate=args.microfit_learning_rate,
    )
    report[metadata_key] = metadata
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
