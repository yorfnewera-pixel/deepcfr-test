# #96 — E1: OS Force Full Traversal (action-OS off, full traversal on postflop multiway hero-узлах)

**Дата:** 2026-06-18
**Статус:** CLOSED DISPROVED — гипотеза OS опровергнута, relapse идентичен
**Связанные:** #81 (raise reward-grounding asymmetry), #84b (relapse mechanism — advantage flip fold/raise)
**Лейбл:** test94seed1E1

## Проблема

В R2 (test94seed1R2) метрики:
- `next_strategy_raise_mass_by_action["raise"]` = 0.0 / 0.10 (opponent mass vs hero mass)
- `relapse_diff`: advantage raise flip — от +0.73 к −4.65 за 100 итераций
- Raise структурно обездолен — return ceiling через bootstrap-only сэмплы

Гипотеза: OS-сэмплирование на action-ноге hero-узлов (_cfr_traverse_multi_outcome_node с IS-weight) вызывает коллапс ре-рейза — через targeted exploration вместо полного обхода raise теряет сигнал, advantage сдвигается в fold.

## Решение

Одна переменная: `os_force_full_traversal` — принудительный полный обход postflop multiway hero-узлов, минуя OS-узел.

### Правка 1 — `src/utils/config.py` (строка 48)

```python
'os_force_full_traversal': False,
```

### Правка 2 — `src/core/deep_cfr.py` `__init__` (после строки 1095)

```python
self.os_force_full_traversal = cfg_get('os_force_full_traversal', False)
```

### Правка 3 — `src/core/deep_cfr.py` `_should_use_outcome_sampling` (строка 1578, первой строкой тела)

```python
if self.os_force_full_traversal:
    return False
```

### Правка 4 — `config.yaml`

```yaml
os_force_full_traversal: true
```

Точная копия R2 + один флаг.

## Механика

- Логика `cfr_traverse_multi:2548`: `if self._should_use_outcome_sampling(...)` → при `os_force_full_traversal=True` метод возвращает `False` → исполнение идёт в полный обход (строки 2551–2719)
- OS-нода (`_cfr_traverse_multi_outcome_node`) с IS-weight не вызывается
- Sizing внутри raise — один анкер через `_hierarchical_sizing` + Q-baseline (строка 2569, 2644–2667) — **без изменений**
- Sizing-путь, per-player Q, Q-режим — не тронуты

## Результат: CLOSED DISPROVED (гипотеза OS опровергнута)

**Прогон:** test95seed1, iter 100–700, длинный прогон E1.

### Верификация OS-off

`raise_funnel_diag.hero_os.raise_sampled = 0` на всех чекпоинтах. Весь hero-raise идёт через `hero_full` (3241 raise-узлов на iter 700). OS реально выключен, регреты точные (полный обход, без IS-веса).

### Траектория eval (против рандома, 1000 игр)

| iter | raise% | win% | reward | verdict |
|------|--------|------|--------|---------|
| 100 | 22.0 | 39.8 | 15.2 | WARN (low_raise) |
| 200 | 10.7 | 39.7 | 12.0 | WARN (low_raise + concentration) |
| 300 | 24.9 | 38.6 | 15.6 | WARN (low_raise) |
| 400 | 10.1 | 29.9 | 10.6 | WARN (low_raise) |
| 500 | 7.4 | 35.3 | 13.0 | WARN (low_raise) |
| 600 | 5.3 | 28.6 | −0.9 | WARN (low_raise) |
| 700 | 4.3 | 21.3 | 9.2 | WARN (raise <5%) |

### relapse_diff (greedy-политика, OS off)

Advantage-флип идентичен прогонам с OS: raise advantage −0.01 → 0.03 → −0.15 → −1.84 → … → −2.45, fold свингует −8.7/−8.0 → +1.5. Текущая политика коллапсит в фолд: fold_freq → 81.7 → 89.7 → 87.7% к iter 500–700. `_first_to_break` = raise_freq shift (Q и advantage малы).

### Head-to-head 700_vs_100

iter700 в матче против iter100: win **12.4%**, reward **−7.8**, raise 6.8%. Пассивная iter700 проигрывает агрессивной iter100.

### Ключевые наблюдения

1. **Sizing-Q — единственный явный успех E1.** `max.mean`: −2.88 (iter 100) → −0.16 (iter 400) → **+0.37** (iter 700). Впервые положительный mean. Sizing-Q converged.

2. **Модель ВЫИГРЫВАЕТ у рандома даже при raise 4%.** win_rate 21–40% ≫ 16.7% (random baseline), reward в плюсе (+9…+15). Пассивная игра против каллинг-стейшнов реально +EV.

3. **Q-сеть плоская, не драйвер.** q_raise_mean 0.027 → −0.007, q_fold ~0 на всём прогоне. Q ничего не решает (согласуется с тем, что C низкорычажный).

4. **next_strategy_raise_mass пульсирует:** 0.0 (100) → 0.261 (200) → 0.033 (300) → 0.0 (400+) — стратегия временно оживает, но regret-накопление снова давит.

## Решающий вывод

**Гипотеза про OS-сэмплирование опровергнута.** OS выключен, регреты точные — relapse остался идентичным. Причина — не алгоритм и не сэмплирование, а оппонент:

- Против **рандома** фолд-пассив реально +EV → CFR **корректно** туда сходится (пассив выигрывает у рандома)
- Против **настоящей стратегии** (iter100) этот же пассив сливает −7.8 → модель переобучилась на эксплуатацию рандома и сама эксплуатируема
- Мультивей-агрессия против рандома наиболее −EV (раздуваешь банк с рандом-коллерами) → CFR-гарантия сходимости даёт best-response под рандом, а не под сильного соперника

**Следствие:** пивот на self-play (оппонент = mirroring стратегии героя, а не рандом).

## Откат

```yaml
os_force_full_traversal: false
```

## Откат

```yaml
os_force_full_traversal: false
```

# #96 — Advantage-regret scale: normalization + clip + Huber loss

**Дата:** 2026-06-18
**Статус:** IMPLEMENTED (ждёт прогонов)
**База:** R2 (A-core) + OS off (E1, `os_force_full_traversal: true`) — baseline #95
**Связано:** #84b/#88 (relapse H3), #87/#87b (нормировка Q — НЕ той ноги)

## Мотивация

OS выключен (полный обход, точные регреты) — relapse в фолд ОСТАЛСЯ.
Значит relapse интринсик advantage/regret-ноги self-play, не сэмплирование и не оппонент.

Корень: advantage-регреты пишутся в **сырых фишках**.
`cfr_traverse_multi` возвращает `state.players_state[traversing_player].reward` (сырьё),
`cf_regrets[a] = action_values[a] − ev` → `_adv_buffer.add` без нормировки.
Нормировка `pot_stack_relative` применяется ТОЛЬКО к Q-таргету,
а политикой рулит advantage (regret-matching). Чинили не ту ногу.

Два независимых дефекта шкалы:
1. **Межсостоянческий:** одна сеть фитит все состояния; крупнобанковые (±200) давят MSE, мелкобанковые недофитятся → fold-equity теряется. → лечит `advantage_regret_norm: pot_stack`.
2. **Внутрисостоянческий:** редкий стек-офф (−200) vs частый стил (+3) при одном pot — нормировка НЕ помогает (один denom). Всплески тянут рейз вниз. → лечит **Huber loss** (robust к выбросам, несмещённый).

Теоретическое обоснование выбора `pot_stack` над `per_node_max`:
- `pot_stack` = деление на константу инфосета (несмещённое: E[r/const] = E[r]/const)
- `per_node_max` = деление на величину сэмпла (смещённое: E[r/m_i] ≠ E[r])
- `pot_stack` + Huber закрывает ОБА дефекта без искажения таргета

## Новые конфиг-ключи

| Ключ | Значение по умолчанию | Описание |
|---|---|---|
| `advantage_regret_norm` | `'none'` | `'none'` / `'pot_stack'` / `'per_node_max'` — нормировка регретов перед `_adv_buffer.add` |
| `advantage_regret_clip` | `None` | `None` / float — клип регрета на пост-норм шкале |
| `advantage_loss` | `'mse'` | `'mse'` / `'huber'` — тип лосса в `train_advantage_network_multi` |
| `advantage_huber_delta` | `1.0` | float — delta для `smooth_l1_loss` (на нормированной шкале) |

## Изменения в коде

### config.py
Добавлены 4 ключа в `_DEFAULTS` (рядом с `q_terminal_loss_alpha`).

### deep_cfr.py `__init__`
Чтение 4 ключей через `cfg_get` (рядом с `q_target_norm` / `q_bootstrap_per_player_enabled`).

```python
self.advantage_regret_norm = cfg_get('advantage_regret_norm', 'none')
_arc = cfg_get('advantage_regret_clip', None)
self.advantage_regret_clip = float(_arc) if _arc is not None else None
self.advantage_loss = cfg_get('advantage_loss', 'mse')
self.advantage_huber_delta = float(cfg_get('advantage_huber_delta', 1.0))
```

### deep_cfr.py `cfr_traverse_multi` — full-traversal сайт
Между циклом `for a in legal_action_types: cf_regrets[a] = ...` и `self._adv_buffer(...).add(...)`:

```python
if self.advantage_regret_norm == 'pot_stack':
    denom = max(float(state.pot) + float(state.players_state[traversing_player].stake), 1.0)
    cf_regrets = cf_regrets / denom
elif self.advantage_regret_norm == 'per_node_max':
    max_abs_val = max(abs(action_values[a]) for a in legal_action_types)
    denom = max(max_abs_val, 1.0)
    cf_regrets = cf_regrets / denom
if self.advantage_regret_clip is not None:
    cf_regrets = np.clip(cf_regrets, -self.advantage_regret_clip, self.advantage_regret_clip)
cf_regrets = cf_regrets.astype(np.float32)
```

OS-сайт (`_cfr_traverse_multi_outcome_node`) **НЕ тронут** — при `os_force_full_traversal: true` OS дормант.

### deep_cfr.py `train_advantage_network_multi`
MSE заменён на Huber при `advantage_loss == 'huber'`:

```python
if self.advantage_loss == 'huber':
    loss = F.smooth_l1_loss(predicted_masked, target_masked.detach(), beta=self.advantage_huber_delta)
else:
    loss = F.mse_loss(predicted_masked, target_masked.detach())
```

### deep_cfr.py checkpoint save/load
4 ключа добавлены в `config_dict` при сохранении и в whitelist при загрузке — для resume-train консистентности.

## Pre-registered прогоны

| Run | advantage_regret_norm | advantage_loss | advantage_regret_clip | lr | Горизонт | Label | Проверяет |
|---|---|---|---|---|---|---|---|
| **N1** | `pot_stack` | `huber` | None | 1e-4 | 400 iter | `test96seed1N1` | Межсостоянческая ёмкость + внутрисостоянческая дисперсия (несмещённо) |
| **L1** | `none` | `mse` | None | **1e-6** | 800-1000 iter | `test96seed1L1` | Гипотеза шага/lr (правота GPT) |
| **N1'** (резерв) | `per_node_max` | `huber` | None | 1e-4 | 400 iter | — | Авторский подход (per-sample bias, но единственный рабочий репо) |

## Метрики и гейты

- **relapse_diff:** advantage raise flip — перестаёт ли уходить в минус к iter 300–400
- **raise_freq (eval):** не схлопывается 100→400
- **next_strategy_raise_mass by action["raise"]:** держится ли ре-рейз
- **Head-to-head поздний-vs-ранний:** late reward ≥ 0

**PASS:** raise не коллапсит И advantage raise flip не уходит в сильный минус И late-vs-early reward ≥ 0.
**FAIL:** raise < 10% к iter400 или late проигрывает early.

## Чтение исхода

- N1 чинит, L1 нет → корень в масштабе (несмещённо)
- L1 чинит, N1 нет → шаг/lr (правота GPT)
- Оба чинят → один корень Kuhn→NLHE, две дороги; выбираем чище/устойчивее
- Ни один → резерв N4 (снять бутстрап/clamp)

## Откат

```yaml
advantage_regret_norm: none
advantage_regret_clip: null
advantage_loss: mse
advantage_huber_delta: 1.0
```
→ исходное поведение, DCFR+ нетронут.

