import sys
import os
import time
import random
import numpy as np
import torch
import pokers as pkrs
from policy_runtime.core import PolicyRuntimeAgent
from policy_runtime.adapters.pokers import PokersGameState, action_to_pokers
from src.agents.random_agent import RandomAgent

CHECKPOINT = r"C:\Users\Cassmall\Desktop\deepcfr-test\models\multi\multi_checkpoint_iter_1500_light.pt"
NUM_GAMES = 3000
N_PLAYERS = 6
INITIAL_STAKE = 200.0
SB = 1.0
BB = 2.0
MODEL_PLAYER_ID = 0


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    agent = PolicyRuntimeAgent(CHECKPOINT, device=device)
    errors = agent.validate()
    for e in errors:
        print(f"  Валидация: {e}")

    random_agents = {i: RandomAgent(i) for i in range(N_PLAYERS) if i != MODEL_PLAYER_ID}

    model_rewards = []
    model_wins = 0
    model_losses = 0
    model_ties = 0
    error_games = 0
    start_time = time.time()

    for game_idx in range(NUM_GAMES):
        if (game_idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (game_idx + 1) / elapsed
            avg_r = np.mean(model_rewards) if model_rewards else 0
            wr = model_wins / max(1, game_idx + 1 - error_games) * 100
            print(f"  {game_idx+1}/{NUM_GAMES} | "
                  f"{rate:.0f} games/s | "
                  f"avg_reward={avg_r:.2f} | "
                  f"winrate={wr:.1f}%")

        button = game_idx % N_PLAYERS
        state = pkrs.State.from_seed(
            n_players=N_PLAYERS,
            button=button,
            sb=SB,
            bb=BB,
            stake=INITIAL_STAKE,
            seed=game_idx,
        )

        try:
            while not state.final_state:
                cp = state.current_player
                if cp == MODEL_PLAYER_ID:
                    gs = PokersGameState(state, player_id=MODEL_PLAYER_ID, bb=BB)
                    action_type, mul = agent.choose_action(
                        gs, player_id=MODEL_PLAYER_ID, deterministic=False)
                    action = action_to_pokers(action_type, mul, state)
                else:
                    action = random_agents[cp].choose_action(state)
                state = state.apply_action(action)

            reward = state.players_state[MODEL_PLAYER_ID].reward
            model_rewards.append(reward)
            if reward > 0:
                model_wins += 1
            elif reward < 0:
                model_losses += 1
            else:
                model_ties += 1
        except Exception as ex:
            error_games += 1

    elapsed = time.time() - start_time
    valid_games = len(model_rewards)
    rewards = np.array(model_rewards)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("=" * 60)
    print(f"Чекпоинт:       {os.path.basename(CHECKPOINT)}")
    print(f"Всего игр:      {NUM_GAMES}")
    print(f"Корректных:     {valid_games}")
    print(f"Ошибок:         {error_games}")
    print(f"Время:          {elapsed:.1f}s ({valid_games/max(elapsed,0.01):.0f} games/s)")
    print("-" * 60)
    print(f"Побед:          {model_wins} ({model_wins/max(valid_games,1)*100:.1f}%)")
    print(f"Поражений:      {model_losses} ({model_losses/max(valid_games,1)*100:.1f}%)")
    print(f"Ничьих:         {model_ties} ({model_ties/max(valid_games,1)*100:.1f}%)")
    print("-" * 60)
    print(f"Средний reward: {rewards.mean():.3f}")
    print(f"Медиана reward: {np.median(rewards):.3f}")
    print(f"Std reward:     {rewards.std():.3f}")
    print(f"Min reward:     {rewards.min():.3f}")
    print(f"Max reward:     {rewards.max():.3f}")
    print(f"Суммарный reward: {rewards.sum():.1f}")
    print("-" * 60)

    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        print(f"P{p}:            {np.percentile(rewards, p):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
