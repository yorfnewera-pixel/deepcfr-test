"""Парная диагностика: видит ли D2CFR-сеть различия между poker states."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pokers as pkrs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import NUM_ACTIONS, legal_action_mask


@dataclass(frozen=True)
class TraversalTarget:
    encoded: Any
    action_values: Any
    state_value: Any
    regrets: Any
    mask: Any
    root_return: float


def _as_1d_float_tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).detach().cpu().reshape(-1)


def vector_delta(left: Any, right: Any, *, top_k: int = 12, eps: float = 1e-12) -> dict[str, Any]:
    """Сравнивает два вектора без знания их семантики."""
    left_t = _as_1d_float_tensor(left)
    right_t = _as_1d_float_tensor(right)
    if left_t.shape != right_t.shape:
        raise ValueError(f"Векторы имеют разные формы: {tuple(left_t.shape)} vs {tuple(right_t.shape)}")
    delta = right_t - left_t
    left_norm = float(torch.linalg.vector_norm(left_t).item())
    right_norm = float(torch.linalg.vector_norm(right_t).item())
    denom = max(left_norm * right_norm, eps)
    cosine = float(torch.dot(left_t, right_t).item() / denom) if denom > eps else 0.0
    changed = torch.nonzero(delta.abs() > 1e-7, as_tuple=False).reshape(-1)
    order = torch.argsort(delta.abs(), descending=True)[: max(int(top_k), 0)]
    return {
        "size": int(left_t.numel()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "l1": float(delta.abs().sum().item()),
        "mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "cosine": max(-1.0, min(1.0, cosine)),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "changed_count": int(changed.numel()),
        "top_abs_deltas": [
            {
                "index": int(index.item()),
                "left": float(left_t[index].item()),
                "right": float(right_t[index].item()),
                "delta": float(delta[index].item()),
            }
            for index in order
            if float(delta[index].abs().item()) > 0.0
        ],
    }


def regret_matching_policy(regrets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    legal_mask = mask.to(dtype=torch.float32)
    positive = torch.clamp(regrets.to(dtype=torch.float32), min=0.0) * legal_mask
    positive_sum = positive.sum()
    if float(positive_sum.detach().cpu().item()) > 0.0:
        return positive / positive_sum
    legal_count = legal_mask.sum()
    if float(legal_count.detach().cpu().item()) <= 0.0:
        return torch.zeros_like(legal_mask)
    return legal_mask / legal_count


def _network_snapshot(network: Any, encoded: torch.Tensor, legal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    state_t = encoded.to(dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        hidden = network._encode(state_t)[0].detach().cpu()
        components = network.forward_components(state_t)
        q = components.action_values[0].detach().cpu()
        v = components.state_values.squeeze(1)[0].detach().cpu().reshape(1)
        regrets = components.regrets[0].detach().cpu()
        rm = regret_matching_policy(regrets, legal_mask.detach().cpu())
    return {
        "hidden": hidden,
        "q": q,
        "v": v,
        "regrets": regrets,
        "rm_policy": rm,
    }


def compare_encoded_pair(
    network: Any,
    left_encoded: Any,
    right_encoded: Any,
    *,
    label: str,
    legal_mask: Any,
    top_k: int = 12,
) -> dict[str, Any]:
    """Сравнивает representation/output для пары уже закодированных states."""
    left_t = _as_1d_float_tensor(left_encoded)
    right_t = _as_1d_float_tensor(right_encoded)
    mask_t = _as_1d_float_tensor(legal_mask)
    if mask_t.numel() != NUM_ACTIONS:
        raise ValueError(f"legal_mask должен иметь {NUM_ACTIONS} слотов")
    left = _network_snapshot(network, left_t, mask_t)
    right = _network_snapshot(network, right_t, mask_t)
    return {
        "label": str(label),
        "encoded": vector_delta(left_t, right_t, top_k=top_k),
        "hidden": vector_delta(left["hidden"], right["hidden"], top_k=top_k),
        "outputs": {
            "q": vector_delta(left["q"], right["q"], top_k=top_k),
            "v": vector_delta(left["v"], right["v"], top_k=top_k),
            "regrets": vector_delta(left["regrets"], right["regrets"], top_k=top_k),
            "rm_policy": vector_delta(left["rm_policy"], right["rm_policy"], top_k=top_k),
        },
        "left_outputs": {
            "q": [float(item) for item in left["q"].tolist()],
            "v": float(left["v"][0].item()),
            "regrets": [float(item) for item in left["regrets"].tolist()],
            "rm_policy": [float(item) for item in left["rm_policy"].tolist()],
        },
        "right_outputs": {
            "q": [float(item) for item in right["q"].tolist()],
            "v": float(right["v"][0].item()),
            "regrets": [float(item) for item in right["regrets"].tolist()],
            "rm_policy": [float(item) for item in right["rm_policy"].tolist()],
        },
    }


def _target_snapshot(target: TraversalTarget) -> dict[str, torch.Tensor]:
    regrets = _as_1d_float_tensor(target.regrets)
    mask = _as_1d_float_tensor(target.mask)
    return {
        "encoded": _as_1d_float_tensor(target.encoded),
        "q": _as_1d_float_tensor(target.action_values),
        "v": _as_1d_float_tensor(target.state_value),
        "regrets": regrets,
        "rm_policy": regret_matching_policy(regrets, mask),
        "mask": mask,
    }


def compare_target_prediction_pair(
    network: Any,
    left: TraversalTarget,
    right: TraversalTarget,
    *,
    label: str,
    top_k: int = 12,
) -> dict[str, Any]:
    """Сравнивает, насколько prediction реагирует относительно target-diff."""
    left_target = _target_snapshot(left)
    right_target = _target_snapshot(right)
    left_prediction = _network_snapshot(network, left_target["encoded"], left_target["mask"])
    right_prediction = _network_snapshot(network, right_target["encoded"], right_target["mask"])
    legal_union_mask = torch.clamp(left_target["mask"] + right_target["mask"], min=0.0, max=1.0)

    target_regret_delta = vector_delta(left_target["regrets"], right_target["regrets"], top_k=top_k)
    prediction_regret_delta = vector_delta(
        left_prediction["regrets"],
        right_prediction["regrets"],
        top_k=top_k,
    )
    target_q_delta = vector_delta(left_target["q"], right_target["q"], top_k=top_k)
    prediction_q_delta = vector_delta(left_prediction["q"], right_prediction["q"], top_k=top_k)
    target_legal_regret_delta = vector_delta(
        left_target["regrets"] * legal_union_mask,
        right_target["regrets"] * legal_union_mask,
        top_k=top_k,
    )
    prediction_legal_regret_delta = vector_delta(
        left_prediction["regrets"] * legal_union_mask,
        right_prediction["regrets"] * legal_union_mask,
        top_k=top_k,
    )
    target_legal_q_delta = vector_delta(
        left_target["q"] * legal_union_mask,
        right_target["q"] * legal_union_mask,
        top_k=top_k,
    )
    prediction_legal_q_delta = vector_delta(
        left_prediction["q"] * legal_union_mask,
        right_prediction["q"] * legal_union_mask,
        top_k=top_k,
    )

    def ratio(predicted: dict[str, Any], target: dict[str, Any]) -> float:
        denominator = float(target["l2"])
        if denominator <= 1e-12:
            return 0.0
        return float(predicted["l2"] / denominator)

    return {
        "label": str(label),
        "target": {
            "q": target_q_delta,
            "legal_q": target_legal_q_delta,
            "v": vector_delta(left_target["v"], right_target["v"], top_k=top_k),
            "regrets": target_regret_delta,
            "legal_regrets": target_legal_regret_delta,
            "rm_policy": vector_delta(left_target["rm_policy"], right_target["rm_policy"], top_k=top_k),
            "mask": vector_delta(left_target["mask"], right_target["mask"], top_k=top_k),
        },
        "prediction": {
            "q": prediction_q_delta,
            "legal_q": prediction_legal_q_delta,
            "v": vector_delta(left_prediction["v"], right_prediction["v"], top_k=top_k),
            "regrets": prediction_regret_delta,
            "legal_regrets": prediction_legal_regret_delta,
            "rm_policy": vector_delta(
                left_prediction["rm_policy"],
                right_prediction["rm_policy"],
                top_k=top_k,
            ),
        },
        "bottleneck": {
            "q_response_ratio": ratio(prediction_q_delta, target_q_delta),
            "regret_response_ratio": ratio(prediction_regret_delta, target_regret_delta),
            "legal_q_response_ratio": ratio(prediction_legal_q_delta, target_legal_q_delta),
            "legal_regret_response_ratio": ratio(prediction_legal_regret_delta, target_legal_regret_delta),
            "target_q_l2_minus_prediction_q_l2": float(target_q_delta["l2"] - prediction_q_delta["l2"]),
            "target_regret_l2_minus_prediction_regret_l2": float(
                target_regret_delta["l2"] - prediction_regret_delta["l2"]
            ),
            "target_legal_q_l2_minus_prediction_legal_q_l2": float(
                target_legal_q_delta["l2"] - prediction_legal_q_delta["l2"]
            ),
            "target_legal_regret_l2_minus_prediction_legal_regret_l2": float(
                target_legal_regret_delta["l2"] - prediction_legal_regret_delta["l2"]
            ),
        },
        "root_returns": {
            "left": float(left.root_return),
            "right": float(right.root_return),
            "delta": float(right.root_return - left.root_return),
        },
    }


def _network_device(network: Any) -> torch.device:
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _target_batch(targets: list[TraversalTarget], *, device: torch.device) -> dict[str, torch.Tensor]:
    if not targets:
        raise ValueError("focused overfit требует хотя бы один target")
    return {
        "states": torch.stack([_as_1d_float_tensor(target.encoded) for target in targets]).to(device),
        "q": torch.stack([_as_1d_float_tensor(target.action_values) for target in targets]).to(device),
        "v": torch.stack([_as_1d_float_tensor(target.state_value) for target in targets]).reshape(-1, 1).to(device),
        "regrets": torch.stack([_as_1d_float_tensor(target.regrets) for target in targets]).to(device),
        "mask": torch.stack([_as_1d_float_tensor(target.mask) for target in targets]).to(device),
    }


def _target_fit_loss(network: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    components = network.forward_components(batch["states"])
    predicted_q = components.action_values
    predicted_v = components.state_values.reshape(-1, 1)
    predicted_regrets = components.regrets
    mask = batch["mask"]
    legal_slots = torch.clamp(mask.sum(), min=1.0)
    q_loss = (((predicted_q - batch["q"]) * mask) ** 2).sum() / legal_slots
    regret_loss = (((predicted_regrets - batch["regrets"]) * mask) ** 2).sum() / legal_slots
    value_loss = torch.mean((predicted_v - batch["v"]) ** 2)
    total = q_loss + regret_loss + value_loss
    return total, {
        "total": float(total.detach().cpu().item()),
        "q": float(q_loss.detach().cpu().item()),
        "regrets": float(regret_loss.detach().cpu().item()),
        "v": float(value_loss.detach().cpu().item()),
    }


def run_focused_overfit_probe(
    network: Any,
    target_pairs: list[dict[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    top_k: int = 12,
    fit_ratio_threshold: float = 0.8,
) -> dict[str, Any]:
    """Проверяет, может ли сеть локально выучить фиксированные target-пары."""
    if steps < 0:
        raise ValueError("steps должен быть >= 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate должен быть > 0")
    if not target_pairs:
        raise ValueError("target_pairs пуст")

    targets: list[TraversalTarget] = []
    for pair in target_pairs:
        targets.extend([pair["left"], pair["right"]])
    device = _network_device(network)
    batch = _target_batch(targets, device=device)

    was_training = bool(getattr(network, "training", False))
    network.train()
    before_loss_tensor, before_loss = _target_fit_loss(network, batch)
    del before_loss_tensor
    before_comparisons = [
        compare_target_prediction_pair(
            network,
            pair["left"],
            pair["right"],
            label=str(pair["label"]),
            top_k=top_k,
        )
        for pair in target_pairs
    ]

    loss_curve = [before_loss["total"]]
    if steps > 0:
        parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("network не содержит обучаемых параметров")
        optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
        report_every = max(1, int(steps) // 10)
        for step in range(1, int(steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _target_fit_loss(network, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("focused overfit получил non-finite loss")
            loss.backward()
            optimizer.step()
            if step == int(steps) or step % report_every == 0:
                loss_curve.append(float(loss.detach().cpu().item()))

    after_loss_tensor, after_loss = _target_fit_loss(network, batch)
    del after_loss_tensor
    after_comparisons = [
        compare_target_prediction_pair(
            network,
            pair["left"],
            pair["right"],
            label=str(pair["label"]),
            top_k=top_k,
        )
        for pair in target_pairs
    ]
    if not was_training:
        network.eval()

    pairs = []
    for pair, before, after in zip(target_pairs, before_comparisons, after_comparisons, strict=True):
        target_l2 = float(after["target"]["legal_regrets"]["l2"])
        ratio = float(after["bottleneck"]["legal_regret_response_ratio"])
        if target_l2 <= 1e-12:
            status = "no_target_signal"
        elif ratio >= float(fit_ratio_threshold):
            status = "fits_fixed_targets"
        else:
            status = "still_underfits_fixed_targets"
        pairs.append(
            {
                "label": str(pair["label"]),
                "status": status,
                "before": {
                    "loss": before_loss,
                    "comparison": before,
                },
                "after": {
                    "loss": after_loss,
                    "comparison": after,
                },
            }
        )

    return {
        "mode": "focused_overfit_probe",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "fit_ratio_threshold": float(fit_ratio_threshold),
        "loss_curve": loss_curve,
        "pairs": pairs,
    }


def _safe_apply(state: Any, action: pkrs.Action) -> Any:
    next_state = state.apply_action(action)
    if next_state.status != pkrs.StateStatus.Ok:
        raise ValueError(f"apply_action вернул status={next_state.status}")
    return next_state


def _flop_after_call_check(seed: int, *, button: int = 0) -> Any:
    state = pkrs.State.from_seed(n_players=2, button=button, sb=1, bb=2, stake=200.0, seed=int(seed))
    state = _safe_apply(state, cast(pkrs.Action, pkrs.Action(pkrs.ActionEnum.Call)))
    state = _safe_apply(state, cast(pkrs.Action, pkrs.Action(pkrs.ActionEnum.Check)))
    if state.stage != pkrs.Stage.Flop:
        raise ValueError(f"Ожидался flop, получен stage={state.stage}")
    return state


def _raise_response_state(seed: int, *, button: int = 0) -> Any:
    state = pkrs.State.from_seed(n_players=2, button=button, sb=1, bb=2, stake=200.0, seed=int(seed))
    return _safe_apply(state, cast(pkrs.Action, pkrs.Action(pkrs.ActionEnum.Raise, 2.0)))


def _state_meta(state: Any, player_id: int) -> dict[str, Any]:
    return {
        "stage": str(state.stage),
        "current_player": int(state.current_player),
        "player_id": int(player_id),
        "pot": float(state.pot),
        "legal_mask": legal_action_mask(state).astype(np.float32).tolist(),
        "public_cards": [repr(card) for card in getattr(state, "public_cards", [])],
    }


def build_live_state_pairs(agent: Any, *, seed: int) -> list[dict[str, Any]]:
    """Строит реальные encodable пары из State.from_seed/apply_action."""
    root_a = pkrs.State.from_seed(n_players=2, button=0, sb=1, bb=2, stake=200.0, seed=int(seed))
    root_b = pkrs.State.from_seed(n_players=2, button=0, sb=1, bb=2, stake=200.0, seed=int(seed) + 1)
    button_b = pkrs.State.from_seed(n_players=2, button=1, sb=1, bb=2, stake=200.0, seed=int(seed))
    raised = _raise_response_state(seed, button=0)
    flop_a = _flop_after_call_check(seed, button=0)
    flop_b = _flop_after_call_check(seed + 1, button=0)
    pairs = [
        ("deal_seed_change_preflop", root_a, root_b, 0),
        ("position_button_change_preflop", root_a, button_b, 0),
        ("betting_history_root_vs_raise_response", root_a, raised, 0),
        ("street_progression_preflop_vs_flop", root_a, flop_a, 0),
        ("flop_deal_change", flop_a, flop_b, int(flop_a.current_player)),
    ]
    result = []
    for label, left_state, right_state, player_id in pairs:
        left_encoded = agent._encode_state(left_state, int(player_id))
        right_encoded = agent._encode_state(right_state, int(player_id))
        result.append(
            {
                "label": label,
                "player_id": int(player_id),
                "left_state": _state_meta(left_state, int(player_id)),
                "right_state": _state_meta(right_state, int(player_id)),
                "left_state_obj": left_state,
                "right_state_obj": right_state,
                "legal_mask": legal_action_mask(left_state).astype(np.float32),
                "left_encoded": left_encoded,
                "right_encoded": right_encoded,
            }
        )
    return result


def _latest_d2cfr_target(agent: Any, *, root_return: float) -> TraversalTarget:
    buffer = agent.d2cfr_buffer
    if buffer is None or len(buffer) <= 0:
        raise ValueError("D2CFR buffer пуст после traversal")
    index = len(buffer) - 1
    return TraversalTarget(
        encoded=buffer._states[index].copy(),
        action_values=buffer._action_values[index].copy(),
        state_value=np.float32(buffer._state_values[index]),
        regrets=buffer._regrets[index].copy(),
        mask=buffer._masks[index].copy(),
        root_return=float(root_return),
    )


def capture_root_target(
    agent: Any,
    state: Any,
    *,
    iteration: int,
    traversing_player: int,
    seed: int,
) -> TraversalTarget:
    """Берёт target, который production traversal записал для root traverser node."""
    from tools.benchmark_traversal import seed_everything

    if int(state.current_player) != int(traversing_player):
        raise ValueError(
            f"root target требует current_player == traversing_player, "
            f"получено current={state.current_player}, traversing={traversing_player}"
        )
    before = len(agent.d2cfr_buffer)
    seed_everything(int(seed))
    root_return = float(agent.cfr_traverse_multi(state, int(iteration), int(traversing_player)))
    if len(agent.d2cfr_buffer) <= before:
        raise ValueError("traversal не записал D2CFR sample")
    return _latest_d2cfr_target(agent, root_return=root_return)


def _collect_and_train(agent: Any, *, traversals: int, seed: int, batch_size: int | None, epochs: int | None) -> dict[str, Any]:
    if traversals <= 0:
        return {"traversals": 0, "buffer_size": len(agent.d2cfr_buffer) if agent.d2cfr_buffer else 0}
    from tools.benchmark_traversal import generate_root_bank, make_state, seed_everything

    seed_everything(int(seed))
    agent.prepare_iteration(1, 0)
    agent.reset_traversal_stats()
    for root in generate_root_bank(int(traversals), int(seed), int(agent.num_players)):
        agent.cfr_traverse_multi(make_state(root, int(agent.num_players)), 1, 0)
    loss = float(agent.train_advantage_network_multi(batch_size=batch_size, epochs=epochs, player_id=0))
    return {
        "traversals": int(traversals),
        "buffer_size": int(len(agent.d2cfr_buffer)),
        "train_loss": loss,
        "train_steps": int(getattr(agent, "last_advantage_train_steps", 0) or 0),
    }


def run_live_representation_probe(
    config_path: str | Path | None,
    *,
    seed: int,
    collect_traversals: int,
    batch_size: int | None,
    epochs: int | None,
    device: str | torch.device,
    top_k: int,
    focused_overfit_steps: int = 0,
    focused_overfit_learning_rate: float = 0.01,
    focused_overfit_labels: list[str] | None = None,
) -> dict[str, Any]:
    from src.core.deep_cfr import DeepCFRAgent
    from src.utils import config as config_mod

    if config_path is not None:
        config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=str(device))
    training = _collect_and_train(
        agent,
        traversals=int(collect_traversals),
        seed=int(seed),
        batch_size=batch_size,
        epochs=epochs,
    )
    pairs = []
    target_pairs = []
    focused_targets = []
    focused_label_set = set(
        focused_overfit_labels
        or [
            "betting_history_root_vs_raise_response",
            "flop_deal_change",
        ]
    )
    for index, item in enumerate(build_live_state_pairs(agent, seed=int(seed))):
        comparison = compare_encoded_pair(
            agent.advantage_net,
            item["left_encoded"],
            item["right_encoded"],
            label=item["label"],
            legal_mask=item["legal_mask"],
            top_k=top_k,
        )
        comparison["player_id"] = item["player_id"]
        comparison["left_state"] = item["left_state"]
        comparison["right_state"] = item["right_state"]
        pairs.append(comparison)
        try:
            left_target = capture_root_target(
                agent,
                item["left_state_obj"],
                iteration=100 + index * 2,
                traversing_player=int(item["left_state_obj"].current_player),
                seed=int(seed) + index * 1009,
            )
            right_target = capture_root_target(
                agent,
                item["right_state_obj"],
                iteration=101 + index * 2,
                traversing_player=int(item["right_state_obj"].current_player),
                seed=int(seed) + index * 1009 + 1,
            )
            target_comparison = compare_target_prediction_pair(
                agent.advantage_net,
                left_target,
                right_target,
                label=item["label"],
                top_k=top_k,
            )
            target_comparison["left_state"] = item["left_state"]
            target_comparison["right_state"] = item["right_state"]
            target_pairs.append(target_comparison)
            if item["label"] in focused_label_set:
                focused_targets.append(
                    {
                        "label": item["label"],
                        "left": left_target,
                        "right": right_target,
                    }
                )
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            target_pairs.append(
                {
                    "label": item["label"],
                    "error": str(error),
                    "left_state": item["left_state"],
                    "right_state": item["right_state"],
                }
            )
    report = {
        "mode": "live_representation_probe",
        "config": str(config_path) if config_path is not None else None,
        "seed": int(seed),
        "training": training,
        "pairs": pairs,
        "target_pairs": target_pairs,
    }
    if int(focused_overfit_steps) > 0:
        if focused_targets:
            report["focused_overfit"] = run_focused_overfit_probe(
                agent.advantage_net,
                focused_targets,
                steps=int(focused_overfit_steps),
                learning_rate=float(focused_overfit_learning_rate),
                top_k=top_k,
            )
        else:
            report["focused_overfit"] = {
                "mode": "focused_overfit_probe",
                "error": "нет target-пар для focused overfit",
                "requested_labels": sorted(focused_label_set),
            }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--collect-traversals", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--focused-overfit-steps", type=int, default=0)
    parser.add_argument("--focused-overfit-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--focused-overfit-label",
        action="append",
        dest="focused_overfit_labels",
        default=None,
        help="label пары для focused overfit; можно указать несколько раз",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_live_representation_probe(
        args.config,
        seed=args.seed,
        collect_traversals=args.collect_traversals,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        top_k=args.top_k,
        focused_overfit_steps=args.focused_overfit_steps,
        focused_overfit_learning_rate=args.focused_overfit_learning_rate,
        focused_overfit_labels=args.focused_overfit_labels,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
