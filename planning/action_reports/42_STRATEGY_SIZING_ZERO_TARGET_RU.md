# Баг-репорт #42: strategy_net sizing обучается на target=0 — финальный сайзинг всегда минимальный

## Серьёзность: CRITICAL

## Категория: Баг обучения / Sizing policy

## Дата обнаружения: 2026-05-16

---

## Описание

В `train_strategy_network` sizing loss обучает `strategy_net.mean_sizing` на **невозможный target = 0**:

```python
bet_targets = torch.zeros(len(raise_indices), device=self.device)
sizing_loss = F.mse_loss(raise_mean_preds, bet_targets)
```

`mean_sizing` выдаёт pot-multiplier ∈ [0.1, 3.0] (tanh-squash). Target = 0 — **недостижим**. Сеть тянет предсказание к нижней границе диапазона (0.1), что означает:

- **Финальный strategy_net всегда ставит минимальный сайзинг** независимо от state
- **PG sizing из advantage_net не переносится** в strategy_net
- **Inference (choose_action) использует strategy_net.mean_sizing** — агент всегда ставит минимальную ставку

Это критический баг: sizing policy в игре сломан.

---

## Затронутые файлы

- `src/core/deep_cfr.py` — `StrategyBuffer`, `cfr_traverse`, `_cfr_traverse_multi_outcome_node`, `cfr_traverse_multi`, `train_strategy_network`

---

## Фикс

### 1. StrategyBuffer: добавлено _bet_sizes

```python
# ДО: 4 поля
_states, _policies, _masks, _iterations

# ПОСЛЕ: 5 полей
_states, _policies, _masks, _iterations, _bet_sizes
```

`add()` теперь принимает `bet_size=0.0` (default). `sample()` возвращает 5-элементный tuple.

### 2. Все вызовы strategy_buffer.add обновлены

| Контекст | bet_size |
|----------|----------|
| Legacy `cfr_traverse` | `sampled_bet_size or bet_size_multiplier` (если Raise legal), иначе `0.0` |
| OS node | `sampled_bet_size` (если sampled_action == Raise), иначе `0.0` |
| Traversing non-OS | `sampled_bet_size or bet_size_multiplier` (если Raise legal), иначе `0.0` |

Значение — **pot-multiplier** ∈ [0.1, 3.0], то же что предсказывает `mean_sizing`.

### 3. train_strategy_network: реальный target

```python
# ДО (баг):
bet_targets = torch.zeros(len(raise_indices), device=self.device)

# ПОСЛЕ:
valid_bet_mask = (policy_tensors[:, 3] > 0) & (bet_size_tensors > 0)
bet_targets = bet_size_tensors[raise_indices]  # реальный sampled sizing
```

Фильтрация: только записи где Raise в стратегии **и** есть реальный sizing sample (bet_size > 0).

---

## Совместимость

- StrategyBuffer **не сохраняется** в checkpoint — backward compatibility не нужна
- Старые вызовы без `bet_size` используют default `0.0` — корректно

---

## Ограничения фикса (не решается в этом баг-репорте)

- Хранится **один scalar** bet_size (sampled/mean multiplier), не распределение
- `strategy_net.mean_sizing` выучит **условное среднее** sizing, не мультимодальный микс
- Mixture-of-Gaussians для sizing отложен до когда single Gaussian покажет ограничение
- Preflop sizing scale (pot-relative vs BB-relative) отложен

---

## Критерии успеха

1. `train_strategy_network` обучает sizing на реальный bet_size target
2. `strategy_net.mean_sizing` предсказывает значения > min_bet_size
3. Sizing в inference (choose_action) зависит от state, а не всегда min
4. Тесты проходят
