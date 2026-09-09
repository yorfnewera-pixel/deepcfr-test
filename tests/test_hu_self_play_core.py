import numpy as np
import pytest
import torch

from src.core.buffers import AdvantageBuffer
from src.core.hu_self_play import (
    HuCurrentPolicySelfPlayCoordinator,
    HuStrategyBuffer,
    HuTraversalAdapter,
)
from src.core.traversal_errors import TraversalFailure


class _ОднослойнаяСеть(torch.nn.Module):
    def __init__(self, advantages):
        super().__init__()
        self.advantages = torch.nn.Parameter(torch.tensor(advantages, dtype=torch.float32))

    def forward(self, states):
        return self.advantages.unsqueeze(0).expand(states.shape[0], -1)


class _СетьСРазделяемымПараметром(torch.nn.Module):
    """Отдельный Module, который намеренно может владеть общим Parameter."""

    def __init__(self, parameter):
        super().__init__()
        self.advantages = parameter

    def forward(self, states):
        return self.advantages.unsqueeze(0).expand(states.shape[0], -1)


class _Дерево:
    """Минимальное дерево: P0 выбирает ветку, затем P1 завершает раздачу."""

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


def _координатор(sampler=None, train_advantage=None, normalize_regrets=None):
    advantages = [_ОднослойнаяСеть([1, 3, 0, 0, 0, 0]), _ОднослойнаяСеть([4, 2, 0, 0, 0, 0])]
    targets = [_ОднослойнаяСеть([0] * 6), _ОднослойнаяСеть([0] * 6)]
    optimizers = [torch.optim.SGD(network.parameters(), lr=0.1) for network in advantages]
    return HuCurrentPolicySelfPlayCoordinator(
        advantage_nets=advantages,
        advantage_target_nets=targets,
        advantage_optimizers=optimizers,
        advantage_buffers=[AdvantageBuffer(8, 2), AdvantageBuffer(8, 2)],
        strategy_net=torch.nn.Linear(4, 6),
        strategy_optimizer=torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1),
        strategy_buffer=HuStrategyBuffer(8, 2),
        adapter=HuTraversalAdapter(
            current_player=_Дерево().current_player,
            is_terminal=_Дерево().is_terminal,
            legal_mask=_Дерево().legal_mask,
            encode=_Дерево().encode,
            apply=_Дерево().apply,
            terminal_value=_Дерево().terminal_value,
            normalize_regrets=normalize_regrets,
        ),
        sampler=sampler,
        train_advantage=train_advantage,
    )


def _аргументы_координатора():
    advantages = [_ОднослойнаяСеть([1, 0, 0, 0, 0, 0]), _ОднослойнаяСеть([0, 1, 0, 0, 0, 0])]
    targets = [_ОднослойнаяСеть([0] * 6), _ОднослойнаяСеть([0] * 6)]
    strategy_net = torch.nn.Linear(4, 6)
    tree = _Дерево()
    return {
        "advantage_nets": advantages,
        "advantage_target_nets": targets,
        "advantage_optimizers": [torch.optim.SGD(network.parameters(), lr=0.1) for network in advantages],
        "advantage_buffers": [AdvantageBuffer(8, 2), AdvantageBuffer(8, 2)],
        "strategy_net": strategy_net,
        "strategy_optimizer": torch.optim.SGD(strategy_net.parameters(), lr=0.1),
        "strategy_buffer": HuStrategyBuffer(8, 2),
        "adapter": HuTraversalAdapter(
            current_player=tree.current_player,
            is_terminal=tree.is_terminal,
            legal_mask=tree.legal_mask,
            encode=tree.encode,
            apply=tree.apply,
            terminal_value=tree.terminal_value,
        ),
    }


@pytest.mark.parametrize(
    "resource_name",
    ["advantage_nets", "advantage_target_nets", "advantage_optimizers", "advantage_buffers"],
)
def test_hu_otklonyaet_obshchiy_resurs_mezhdu_p0_i_p1(resource_name):
    arguments = _аргументы_координатора()
    arguments[resource_name][1] = arguments[resource_name][0]

    with pytest.raises(ValueError, match="независим"):
        HuCurrentPolicySelfPlayCoordinator(**arguments)


def test_hu_otklonyaet_optimizer_ne_svyazannyy_s_setyu_igroka():
    arguments = _аргументы_координатора()
    unrelated_network = _ОднослойнаяСеть([0] * 6)
    arguments["advantage_optimizers"][1] = torch.optim.SGD(unrelated_network.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="optimizer P1"):
        HuCurrentPolicySelfPlayCoordinator(**arguments)


def test_hu_otklonyaet_target_kak_ssylku_na_advantage_set():
    arguments = _аргументы_координатора()
    arguments["advantage_target_nets"][1] = arguments["advantage_nets"][1]

    with pytest.raises(ValueError, match="независим"):
        HuCurrentPolicySelfPlayCoordinator(**arguments)


@pytest.mark.parametrize("resource_name", ["advantage_nets", "advantage_target_nets"])
def test_hu_otklonyaet_raznye_seti_p0_i_p1_s_obshchim_parameterom(resource_name):
    arguments = _аргументы_координатора()
    shared_parameter = torch.nn.Parameter(torch.zeros(6))
    first_network = _СетьСРазделяемымПараметром(shared_parameter)
    second_network = _СетьСРазделяемымПараметром(shared_parameter)
    arguments[resource_name] = [first_network, second_network]
    if resource_name == "advantage_nets":
        arguments["advantage_optimizers"] = [
            torch.optim.SGD(first_network.parameters(), lr=0.1),
            torch.optim.SGD(second_network.parameters(), lr=0.1),
        ]

    with pytest.raises(ValueError, match="P0/P1"):
        HuCurrentPolicySelfPlayCoordinator(**arguments)


def test_hu_otklonyaet_advantage_i_target_s_obshchim_parameterom():
    arguments = _аргументы_координатора()
    shared_parameter = next(arguments["advantage_nets"][0].parameters())
    arguments["advantage_target_nets"][1] = _СетьСРазделяемымПараметром(shared_parameter)

    with pytest.raises(ValueError, match="advantage-сети и target-сети"):
        HuCurrentPolicySelfPlayCoordinator(**arguments)


def test_hu_traversals_marshrutiziruyut_regrety_i_strategy_mezhdu_igrokami():
    coordinator = _координатор()
    coordinator.begin_iteration()

    coordinator.traverse((0, 0), traversing_player=0, iteration=7)
    assert len(coordinator.advantage_buffers[0]) == 1
    assert len(coordinator.advantage_buffers[1]) == 0
    assert coordinator.strategy_buffer.actor_ids().tolist() == [1, 1]

    coordinator.traverse((0, 0), traversing_player=1, iteration=7)
    assert len(coordinator.advantage_buffers[0]) == 1
    assert len(coordinator.advantage_buffers[1]) == 1
    assert coordinator.strategy_buffer.actor_ids().tolist() == [1, 1, 0]


def test_hu_normaliziruet_regrety_dlya_p0_i_p1_pered_zapisyu_v_advantage_buffer():
    raw_regrets_by_actor = {}

    def normalize_regrets(state, regrets, _mask):
        actor_id = int(state[0])
        raw_regrets_by_actor[actor_id] = regrets.copy()
        return regrets / 200.0

    coordinator = _координатор(normalize_regrets=normalize_regrets)
    coordinator.begin_iteration()

    coordinator.traverse((0, 0), traversing_player=0, iteration=7)
    coordinator.traverse((0, 0), traversing_player=1, iteration=7)

    for actor_id in (0, 1):
        _, written_regrets, _, _ = coordinator.advantage_buffers[actor_id].sample()
        assert actor_id in raw_regrets_by_actor
        assert np.allclose(written_regrets[0], raw_regrets_by_actor[actor_id] / 200.0)


def test_obshchiy_strategy_buffer_hranitt_actor_i_actor_conditioned_input():
    buffer = HuStrategyBuffer(capacity=2, state_dim=2)
    buffer.add(
        actor_id=1,
        state=np.array([5, 6], dtype=np.float32),
        policy=np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
        mask=np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
        iteration=3,
    )

    states, actor_ids, policies, masks, iterations = buffer.sample()

    assert actor_ids.tolist() == [1]
    assert np.array_equal(HuStrategyBuffer.condition_state(states[0], actor_ids[0]), [5, 6, 0, 1])
    assert policies[0, 1] == 1
    assert masks[0, 1] == 1
    assert iterations.tolist() == [3.0]


def test_strategy_buffer_otdaet_actor_conditioned_batch_dlya_obshchey_seti():
    buffer = HuStrategyBuffer(capacity=2, state_dim=2)
    buffer.add(
        actor_id=0,
        state=np.array([7, 8], dtype=np.float32),
        policy=np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
        mask=np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
        iteration=1,
    )

    states, policies, masks, iterations = buffer.sample_conditioned()

    assert states.tolist() == [[7, 8, 1, 0]]
    assert policies[0, 0] == 1
    assert masks[0, 0] == 1
    assert iterations.tolist() == [1.0]


def test_target_policy_tozhdestvenna_policy_sampling_do_vybora_deystviya():
    observed = []

    def sampler(legal_slots, policy):
        observed.append((legal_slots.copy(), policy.copy()))
        return int(legal_slots[np.argmax(policy[legal_slots])])

    coordinator = _координатор(sampler=sampler)
    coordinator.begin_iteration()
    coordinator.traverse((0, 0), traversing_player=0, iteration=1)

    _, _, policies, _, _ = coordinator.strategy_buffer.sample()
    assert np.array_equal(policies[0], observed[0][1])
    assert observed[0][0].tolist() == [0, 1]
    assert policies[0].tolist() == pytest.approx([2 / 3, 1 / 3, 0, 0, 0, 0])


def test_snapshots_neizmenny_i_odny_dlya_obeih_faz_iteratsii():
    snapshot_values = []

    def train_advantage(player_id, _network, _target, _optimizer, _buffer):
        with torch.no_grad():
            coordinator.advantage_nets[player_id].advantages.add_(100)

    coordinator = _координатор(train_advantage=train_advantage)

    coordinator.run_iteration(
        iteration=1,
        traversals_per_player=1,
        new_initial_state=lambda _player, _index: (0, 0),
        after_phase=lambda: snapshot_values.append(
            [snapshot.advantages.detach().clone() for snapshot in coordinator.snapshots]
        ),
    )

    assert len(snapshot_values) == 2
    assert torch.equal(snapshot_values[0][0], snapshot_values[1][0])
    assert torch.equal(snapshot_values[0][1], snapshot_values[1][1])
    assert torch.equal(coordinator.snapshots[0].advantages, torch.tensor([1, 3, 0, 0, 0, 0]))
    assert coordinator.advantage_nets[0].advantages[0].item() == 101


def test_hu_zavershaet_obe_fazy_do_obucheniya_advantage_nog():
    events = []

    def train_advantage(player_id, *_args):
        events.append(f"train_p{player_id}")

    coordinator = _координатор(train_advantage=train_advantage)
    coordinator.traverse = lambda *_args, traversing_player, **_kwargs: events.append(
        f"traverse_p{traversing_player}"
    )

    coordinator.run_iteration(
        iteration=1,
        traversals_per_player=1,
        new_initial_state=lambda _player, _index: (0, 0),
    )

    assert events == ["traverse_p0", "traverse_p1", "train_p0", "train_p1"]


def test_hu_otkatyvaet_chastichnye_samples_pri_neustranimoy_oshibke_obhoda():
    coordinator = _координатор(sampler=lambda _slots, _policy: 99)

    with pytest.raises(TraversalFailure) as caught:
        coordinator.run_iteration(
            iteration=4,
            traversals_per_player=1,
            new_initial_state=lambda _player, _index: (0, 0),
        )

    assert caught.value.context.iteration == 4
    assert caught.value.context.traversing_player == 0
    assert caught.value.context.traversal_index == 0
    assert caught.value.context.acting_player == 1
    assert caught.value.context.depth == 1
    assert caught.value.context.action_trace == ("P0:slot 0",)
    assert len(coordinator.advantage_buffers[0]) == 0
    assert len(coordinator.advantage_buffers[1]) == 0
    assert len(coordinator.strategy_buffer) == 0


def test_hu_udalyaet_samples_upavshego_traversal_no_prodolzhaet_iteratsiyu():
    samples = iter([99, 0, 0])
    failures = []
    coordinator = _координатор(sampler=lambda _slots, _policy: next(samples))

    coordinator.run_iteration(
        iteration=5,
        traversals_per_player=1,
        new_initial_state=lambda _player, _index: (0, 0),
        handle_traversal_failure=lambda error: failures.append(error) or True,
    )

    assert len(failures) == 1
    assert failures[0].context.traversal_index == 0
    assert len(coordinator.advantage_buffers[0]) == 0
    assert len(coordinator.advantage_buffers[1]) == 1
    assert coordinator.strategy_buffer.actor_ids().tolist() == [0]


def test_hu_ne_ispolzuet_staryy_context_pri_oshibke_sozdaniya_kornya():
    coordinator = _координатор(sampler=lambda _slots, _policy: 99)
    with pytest.raises(TraversalFailure):
        coordinator.run_iteration(
            iteration=6,
            traversals_per_player=1,
            new_initial_state=lambda _player, _index: (0, 0),
        )

    def broken_initial_state(_player, _index):
        raise ValueError("не удалось создать root state")

    with pytest.raises(TraversalFailure) as caught:
        coordinator.run_iteration(
            iteration=7,
            traversals_per_player=1,
            new_initial_state=broken_initial_state,
        )

    assert caught.value.context.traversing_player == 0
    assert caught.value.context.acting_player is None
    assert caught.value.context.depth == 0
    assert caught.value.context.action_trace == ()
