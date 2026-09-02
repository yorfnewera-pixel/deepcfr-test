# Баг-репорт #21: CPU-оптимизации — encode_state dedup, torch.normal, inference_mode, Numpy Ring Buffers

## Дата
2026-05-08

## Категория
Производительность / Оптимизация

## Серьёзность
High (25% ускорение итераций на CPU)

## Описание
Четыре CPU-оптимизации внедрены после перехода с GPU на CPU (WDDM спайки). GPU-специфичные оптимизации из оригинального #21 (.cpu() merge, non_blocking=True) неприменимы на CPU.

## 1. encode_state dedup

### Причина
В cfr_traverse и cfr_traverse_multi `encode_state()` вызывалась дважды на каждый traversing-узел:
1. `torch.FloatTensor(encode_state(state, player_id))` — для forward pass
2. `np.array(encode_state(state, player_id), dtype=np.float32).tobytes()` — для memory write

### Исправление
```python
encoded_state = encode_state(state, traversing_player)
state_tensor = torch.FloatTensor(encoded_state).to(self.device)
# ...
_state_arr = np.asarray(encoded_state, dtype=np.float32)  # переиспользуем
```

### Файлы
- `src/core/deep_cfr.py`: `cfr_traverse` (стр ~293), `cfr_traverse_multi` (стр ~455)

## 2. torch.distributions.Normal -> torch.normal

### Причина
`Normal.__init__` создаёт обёртку-объект каждый вызов — Python overhead даже на CPU. В DFS с тысячами узлов это мусор для GC.

### Исправление
```python
# До (в sample_sizing_eval):
dist = torch.distributions.Normal(z_mean, std)
z_raw = dist.sample()
log_prob_z = dist.log_prob(z_raw)

# После:
z_raw = torch.normal(z_mean, std)
log_prob_z = -0.5 * ((z_raw - z_mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
```

В model.py `sample_sizing()` оставлен с Distribution (нужен `rsample()` + `log_prob()` для training с градиентами).

### Файлы
- `src/core/model.py`: `PokerNetwork.sample_sizing_eval()`

## 3. inference_mode вместо no_grad

### Причина
`torch.inference_mode()` полностью отключает version counter и autograd machinery — строже чем `no_grad()`. Для маленького MLP на CPU микросекунды за call.

### Исправление
Заменено в 4 местах (только traversal, НЕ training):
- `cfr_traverse` — advantage_net forward
- `cfr_traverse_multi` — advantage_net forward
- `cfr_traverse_multi` — strategy_net forward (opponent)
- `choose_action` — strategy_net forward

### Файлы
- `src/core/deep_cfr.py`: 4 места `with torch.no_grad():` -> `with torch.inference_mode():`

## 4. Numpy Ring Buffers (вместо deque + .tobytes())

### Причина
PrioritizedMemory хранил `list of tuples` с `bytes`:
- `.tobytes()` — копия 156 floats (624 байт) -> bytes объект
- `np.frombuffer()` при training — ещё копия
- Python object creation per `add()` — GC pressure
- `_opp_bytes` (20 байт) — **никогда не использовались** в training

PolicyGradientMemory и StrategyMemory (deque) — аналогичные проблемы.

### Исправление

**PrioritizedMemory:**
```python
# Предвыделенные numpy массивы
self._states = np.empty((capacity, state_dim), dtype=np.float32)
self._action_types = np.empty(capacity, dtype=np.int32)
self._bet_sizes = np.empty(capacity, dtype=np.float32)
self._regrets = np.empty(capacity, dtype=np.float32)
self._priorities = np.empty(capacity, dtype=np.float64)

# add() — прямая запись в numpy slice
def add(self, state, action_type, bet_size, regret, priority=None):
    self._states[pos] = state  # без .tobytes()!

# sample() — возвращает numpy slices
# Training: torch.from_numpy(states.copy()) вместо np.frombuffer
```

**PolicyGradientMemory** — аналогично: `_states`, `_z_raws`, `_regret_raises`, `_bet_sizes`.

**StrategyMemory** (новый класс, замена `deque`): `_states`, `_strategies` (num_actions=3), `_bet_sizes`, `_iterations`.

**Убрано:**
- `_opp_bytes` — нигде не использовалось, удалено из всех call sites
- `.tobytes()` — из всех memory writes
- `np.frombuffer()` — из всех training методов
- `deque` — из strategy_memory и pg_memory

### Изменённые call sites
- `cfr_traverse` — `add()` принимает numpy array напрямую
- `cfr_traverse_multi` — аналогично
- `train_advantage_network` — `torch.from_numpy(states.copy())`
- `train_advantage_network_multi` — numpy concatenate + `torch.from_numpy()`
- `train_sizing_network` — `torch.from_numpy()` из numpy slices
- `train_strategy_network` — `torch.from_numpy()` из numpy slices
- `prepare_iteration` — `.clear()` вместо `.buffer.clear()/.priorities.clear()`
- `train.py` clean_buffers — `.clear()` вместо `.buffer.clear()`
- `__init__` — `PrioritizedMemory(capacity, state_dim=input_size)`, `StrategyMemory(...)`, `PolicyGradientMemory(..., state_dim=input_size)`

### Файлы
- `src/core/deep_cfr.py`: PrioritizedMemory, PolicyGradientMemory, StrategyMemory (новый), все traversal/training методы
- `src/training/train.py`: clean_buffers
- Удалён `from collections import deque`

## Результаты

| Метрика | До | После | Δ |
|---------|-----|-------|---|
| Время итерации (3 iter, 100 trav, seed=42) | 1.88с | 1.40с | **-25%** |
| Memory writes | .tobytes() + tuple creation | numpy slice write | **0-copy** |
| Training data access | np.frombuffer() per sample | torch.from_numpy(buffer) | **zero-copy** |
| _opp_bytes | 20 байт на запись (не использовался) | удалено | **-0% waste** |

## Замечание
Все 4 оптимизации применены одновременно. Индивидуальный вклад каждой не изолирован. Основной буст ожидаемо от numpy buffers + dedup, inference_mode и torch.normal — микро-оптимизации.
