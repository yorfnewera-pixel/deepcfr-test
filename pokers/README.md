[![CI](https://github.com/Reinforcement-Poker/pokers/actions/workflows/CI.yml/badge.svg)](https://github.com/Reinforcement-Poker/pokers/actions/workflows/CI.yml)
[![PyPI version](https://badge.fury.io/py/pokers.svg)](https://badge.fury.io/py/pokers)

# Pokers — локальный MTT-контракт v1

Локальный движок одной 2--8-max NLHE MTT-раздачи для RL. Контракт `mtt_per_player_ante_v1` несовместим со старыми HU checkpoint: обучение, replay и оценку необходимо начать заново.

## Правила MTT v1

- Внутри движка суммы -- целые единицы `chip_unit`; Python-величина обязана быть кратна `chip_unit`.
- `stakes` задаёт отдельный положительный стек каждого раздаваемого игрока.
- Перед раздачей собираются `ante -> SB -> BB`. Ante участвует в банках, но не является ставкой улицы и не уменьшает колл.
- Поддерживаются короткие обязательные взносы, all-in, side pots, возврат непокрытого избытка и odd chips.
- `min_raise` -- единственный публичный источник минимального полного рейза для action-space.
- `action_history` содержит только успешные публичные действия; карт, колоды и seed в ней нет.

## Why another poker environment?

Poker is an incredibly deep game with very simple rules, so why are all the environments so overly complex? Heck, someone could say that you need to publish a paper before building one (looking at you RLCard 👀). Pokers way is to discard the agent environment cycle and all that stuff, just the good old new_state = state + action model. Through its simplicity pokers tries to be flexible and easily integrable into any framework.

### Why not to use pokers

Pokers is a side project inside another side project. This means that it is guaranteed to have bugs, which is not very nice for a RL environment. We have done our best to minimize the errors, testing it against the 10k hands pluribus logs. However, this doesn't cover some areas of the state space, so if you need a more reliable environment RLCard is a better option.

## Installation

Pokers can be installed directly from pypi.

```bash
pip install pokers
```

## Usage

Just create the initial state and act over it. Easy peasy.
```python
import pokers as pkrs

agents = [agent0, agent1, agent2, agent3, agent4, agent5] # Build the agents however you want
initial_state = pkrs.State.from_seed(
    n_players=len(agents), button=0, sb=1.0, bb=2.0, stake=0.0,
    stakes=[31.0, 28.0, 24.0, 19.0, 13.0, 8.0], ante=0.25,
    chip_unit=0.25, seed=1234,
)
trace = [initial_state]

while not trace[-1].final_state:
    state = trace[-1]
    action = agents[state.current_player].choose_action(trace)
    new_state = state.apply_action(action)
    trace.append(new_state)
```

Начальное состояние также создаётся через `State.from_deck()`. Для следующей раздачи внешний MTT-контур передаёт только ненулевые конечные `player.stake` как новые `stakes`.

Curious about what info a state contains? Just go to [pokers.pyi](pokers.pyi) and see it yourself, I bet there's all you need.

As a bonus you can print the entire hand as text. Who wants GUIs anyway?
```python
print(pkrs.visualize_trace(trace))
```

### Error handling

Ошибочные состояния представлены `StateStatus`: `IllegalAction`, `LowBet`, `HighBet` и `InvalidAmount`; корректное состояние -- `Ok`. Некратные `chip_unit`, отрицательные и не конечные суммы отклоняются.

Every erroneous state is also final. So applying an action over it will return the same exact state.


### Parallel actions

If you have a bunch of independent states and want to perform multiple actions in parallel you can easily trick the GIL with `parallel_apply_action()`.

```python
import pokers as pkrs

agents = [agent0, agent1, agent2, agent3, agent4, agent5]
states = [pkrs.State.from_seed(n_players=len(agents), button=0, sb=0.5, bb=1.0, stake=100.0, seed=seed) for seed in range(10)]

while not all([s.final_state for s in states]):
    actions = [agents[s.current_player].choose_action(s) for s in states]
    states = pkrs.parallel_apply_action(states, actions)
```

Since final states do not change when an action is performed, you can safely wait for all hands in the batch to end.

## Alternatives

To our knowledge these are some other poker environments that you would want to consider.

- [RLCard](https://github.com/datamllab/rlcard): Great RL environment for multiple card games.
- [neuron_poker](https://github.com/dickreuter/neuron_poker): OpenAI gym for texas holdem.
- [pgx](https://github.com/sotetsuk/pgx): Pretty cool project with jax-native game simulators. Sadly (at the moment) it doesn't implement NLTH.
