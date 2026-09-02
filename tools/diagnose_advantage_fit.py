from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import pokers as pkrs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.deep_cfr import DeepCFRAgent


def _new_hand(num_players: int, seed: int) -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=num_players,
        button=seed % num_players,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=seed,
    )


def _build_targets(agent: DeepCFRAgent, states, regrets, iterations):
    state_t = torch.from_numpy(states.copy()).to(agent.device)
    regret_t = torch.from_numpy(regrets.copy()).to(agent.device)
    iteration_t = torch.from_numpy(iterations.copy()).to(agent.device)
    iteration_now = max(int(agent.iteration_count), 1)
    with torch.no_grad():
        previous = agent.advantage_target_net(state_t)
        previous_positive = torch.clamp(previous, min=0.0)
        is_fresh = (iteration_t == iteration_now).unsqueeze(1).float()
        if iteration_now <= 1:
            targets = is_fresh * regret_t + (1.0 - is_fresh) * previous_positive
        elif agent.advantage_accumulation == "plain":
            targets = is_fresh * (previous + regret_t) + (1.0 - is_fresh) * previous
        else:
            discount = (iteration_now - 1) ** agent.discount_alpha
            discount /= discount + 1.0
            targets = is_fresh * (previous_positive * discount + regret_t) + (1.0 - is_fresh) * previous_positive
    return targets.cpu()


def _fit_metrics(agent: DeepCFRAgent, states, targets, masks, indices):
    if len(indices) == 0:
        return None
    with torch.no_grad():
        state_t = torch.from_numpy(states[indices].copy()).to(agent.device)
        mask_t = torch.from_numpy(masks[indices].copy()).to(agent.device)
        target_t = targets[indices].to(agent.device)
        pred_t = agent.advantage_net(state_t)
        legal = mask_t > 0.0
        errors = torch.abs(pred_t - target_t)[legal]
        target_abs = torch.abs(target_t)[legal]
        if errors.numel() == 0:
            return None
        denom = torch.quantile(target_abs.float(), 0.90).clamp_min(1e-8)
        return {
            "fit_error": float(torch.median(errors.float()).item() / denom.item()),
            "median_abs": float(torch.median(errors.float()).item()),
            "p90_target_abs": float(denom.item()),
        }


def _print_metrics(step: int, agent, states, targets, masks, train_idx, holdout_idx):
    train = _fit_metrics(agent, states, targets, masks, train_idx)
    holdout = _fit_metrics(agent, states, targets, masks, holdout_idx)
    if train is None or holdout is None:
        print(f"{step:4d}  нет legal samples для метрик")
        return
    print(
        f"{step:4d}  "
        f"train={train['fit_error']:.6f} "
        f"holdout={holdout['fit_error']:.6f} "
        f"train_med_abs={train['median_abs']:.6f} "
        f"holdout_med_abs={holdout['median_abs']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/test113/multi_checkpoint_iter_2000.pt")
    parser.add_argument("--traversals", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.load_model(checkpoint)
    iteration = int(agent.iteration_count) + 1
    agent.iteration_count = iteration
    agent.prepare_iteration(iteration, traversing_player=0)
    agent.reset_traversal_stats()

    started = time.perf_counter()
    for traversal in range(args.traversals):
        state_seed = args.seed + iteration * max(1, args.traversals) + traversal
        agent.cfr_traverse_multi(_new_hand(agent.num_players, state_seed), iteration, traversing_player=0)
    traversal_seconds = time.perf_counter() - started

    count = len(agent.advantage_buffer)
    samples = agent.advantage_buffer.sample(count)
    if samples is None:
        raise RuntimeError("Advantage buffer пуст после traversal")
    states, regrets, masks, iterations = samples
    targets = _build_targets(agent, states, regrets, iterations)

    legal_targets = targets[torch.from_numpy(masks.copy()).bool()]
    print(f"checkpoint={checkpoint}")
    print(f"iteration={iteration} traversals={args.traversals} samples={count} traversal_seconds={traversal_seconds:.3f}")
    print(f"target_min={float(legal_targets.min().item()):.6f} target_max={float(legal_targets.max().item()):.6f}")
    print(f"batch_size={args.batch_size} baseline_full_epoch_steps={int(np.ceil(count / min(args.batch_size, count)))}")

    order = np.random.permutation(count)
    holdout_count = max(1, int(round(count * 0.10)))
    holdout_idx = order[:holdout_count]
    train_idx = order[holdout_count:]
    checkpoints = {0, 1, 2, 4, 8, 16, 32, 50, 100, 150, args.steps}
    checkpoints = {step for step in checkpoints if 0 <= step <= args.steps}

    print("step  fit_error_train fit_error_holdout median_abs_train median_abs_holdout")
    _print_metrics(0, agent, states, targets, masks, train_idx, holdout_idx)

    agent.advantage_net.train()
    target_device = targets.to(agent.device)
    for step in range(1, args.steps + 1):
        batch_idx = np.random.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=True)
        state_t = torch.from_numpy(states[batch_idx].copy()).to(agent.device)
        mask_t = torch.from_numpy(masks[batch_idx].copy()).to(agent.device)
        target_t = target_device[batch_idx]
        predictions = agent.advantage_net(state_t)
        loss = F.mse_loss(predictions * mask_t, target_t * mask_t)
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.advantage_net.parameters(), max_norm=1.0)
        agent.optimizer.step()
        if step in checkpoints:
            _print_metrics(step, agent, states, targets, masks, train_idx, holdout_idx)


if __name__ == "__main__":
    main()
