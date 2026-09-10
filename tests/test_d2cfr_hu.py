import numpy as np
import pytest
import torch

from src.core.buffers import DuelingAdvantageBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.core.hu_self_play import HuCurrentPolicySelfPlayCoordinator, HuStrategyBuffer, HuTraversalAdapter
from src.training import train as train_mod
from src.utils import config as config_mod


class _DuelingLeg(torch.nn.Module):
    def __init__(self, regrets):
        super().__init__()
        self.regrets = torch.nn.Parameter(torch.tensor(regrets, dtype=torch.float32))

    def forward(self, states):
        return self.regrets.unsqueeze(0).expand(states.shape[0], -1)


class _HuTree:
    def current_player(self, state):
        return state[0]

    def is_terminal(self, state):
        return state[0] is None

    def legal_mask(self, _state):
        return np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)

    def encode(self, state, actor_id):
        return np.array([float(actor_id), float(state[1])], dtype=np.float32)

    def apply(self, state, action):
        actor, branch = state
        if actor == 0:
            return (1, action)
        return (None, branch * 10 + action)

    def terminal_value(self, state, traverser):
        return float(state[1] if traverser == 0 else -state[1])


def _d2_coordinator():
    tree = _HuTree()
    networks = [_DuelingLeg([1, 3, 0, 0, 0, 0]), _DuelingLeg([4, 2, 0, 0, 0, 0])]
    strategy_net = torch.nn.Linear(4, 6)
    return HuCurrentPolicySelfPlayCoordinator(
        advantage_nets=networks,
        advantage_target_nets=None,
        advantage_optimizers=[
            torch.optim.SGD(network.parameters(), lr=0.1) for network in networks
        ],
        advantage_buffers=[DuelingAdvantageBuffer(8, 2), DuelingAdvantageBuffer(8, 2)],
        strategy_net=strategy_net,
        strategy_optimizer=torch.optim.SGD(strategy_net.parameters(), lr=0.1),
        strategy_buffer=HuStrategyBuffer(8, 2),
        adapter=HuTraversalAdapter(
            current_player=tree.current_player,
            is_terminal=tree.is_terminal,
            legal_mask=tree.legal_mask,
            encode=tree.encode,
            apply=tree.apply,
            terminal_value=tree.terminal_value,
            normalise_d2cfr_targets=lambda _state, q, v, mask: (
                q.copy(),
                np.float32(v),
                ((q - v) * mask).astype(np.float32),
            ),
        ),
        d2cfr_enabled=True,
    )


def test_hu_d2_records_q_v_r_only_in_traversing_player_buffer():
    coordinator = _d2_coordinator()
    coordinator.begin_iteration()

    coordinator.traverse((0, 0), traversing_player=0, iteration=3)

    assert len(coordinator.advantage_buffers[0]) == 1
    assert len(coordinator.advantage_buffers[1]) == 0
    _, action_values, state_values, regrets, masks, iterations = coordinator.advantage_buffers[0].sample()
    legal = masks[0] == 1.0
    assert np.allclose(action_values[0, legal] - state_values[0], regrets[0, legal])
    assert iterations.tolist() == [3.0]


@pytest.fixture
def d2_hu_agent(tmp_path):
    config_path = tmp_path / "hu-d2cfr.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "num_trainable_players: 2",
                "hu_current_policy_self_play: true",
                "hidden_size: 8",
                "d2cfr_enabled: true",
                "d2cfr_mc_correction_enabled: false",
                "advantage_memory_size: 16",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        yield DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    finally:
        config_mod.load_config("config.yaml")


def test_hu_d2_factory_uses_two_dueling_legs_without_target_networks(d2_hu_agent):
    coordinator = train_mod._create_hu_current_policy_coordinator(d2_hu_agent)

    assert coordinator.advantage_target_nets is None
    assert all(isinstance(buffer, DuelingAdvantageBuffer) for buffer in coordinator.advantage_buffers)
    assert all(network is not d2_hu_agent.advantage_net for network in coordinator.advantage_nets[1:])
