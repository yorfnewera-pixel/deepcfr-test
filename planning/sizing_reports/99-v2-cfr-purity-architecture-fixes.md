# #99v2 — CFR Purity: Strategy-Advantage Decoupling + Per-Player Sizing + Opponent Sizing Fix + Adaptive Batch

**Дата:** 2026-06-20
**Статус:** IMPLEMENTED
**База:** #99 v1 (random agent + per-iter traversing + optional Q)

---

## Диагноз

После v1 обнаружены архитектурные нарушения чистоты CFR:

1. **Strategy-сеть влияла на CFR-дерево** через opp-ноду — оппоненты использовали `strategy_sizing_net` для бет-сайзинга. Strategy — это distillation output, она не должна быть входом в обучение advantage.
2. **Strategy-буферы были per-player (5 шт)** — избыточно. Раз сеть одна общая, буфер должен быть один.
3. **Advantage sizing — одна shared сеть** вместо per-player. Каждый игрок должен иметь свою sizing-ногу с независимыми CFR-регретами.
4. **Sizing Q влиял на оппонентов** через `_hierarchical_sizing` без различения hero/opp.
5. **Strategy training имел жёсткий порог batch_size** — не тренировался пока буфер не наберёт 1024 сэмпла.

---

## Изменения

### 1. Откат per-player → сингл стратеджи-буферы

**Причина:** strategy-сеть одна общая — достаточно одного буфера. Очищается каждую итерацию, содержит данные только текущего traversing.

| Что | Было (v1) | Стало (v2) |
|---|---|---|
| `strategy_buffer` | `strategy_buffers[0..4]` (5 шт) | `strategy_buffer` (1 общий) |
| `sizing_strategy_buffer` | `sizing_strategy_buffers[0..4]` (5 шт) | `sizing_strategy_buffer` (1 общий) |
| `prepare_iteration` | Чистит `buffers[tp]` | Чистит сингл-буферы целиком |
| `train_strategy_network` | `player_id` параметр | Без `player_id` |
| `train_strategy_sizing_anchor_network` | `player_id` параметр | Без `player_id` |
| Логи | `sum(len(b) ...)` | `len(agent.strategy_buffer)` |
| `_cfr_traverse_with_opponents` | `strategy_buffers[0]` | `strategy_buffer` |

### 2. Оппонент sizing: `strategy_sizing_net` → `_hierarchical_sizing`

**Причина:** strategy-сеть не должна влиять на CFR-дерево. Оппонент должен использовать advantage-сеть для sizing.

```diff
# opp-нода cfr_traverse_multi
- with torch.inference_mode():
-     _, slot_logits, scalar_bet = self.strategy_sizing_net(opp_state_tensor)
- probs = F.softmax(slot_logits, dim=1)[0].cpu().numpy()
- opp_bet_size = float(np.random.choice(self.anchors, p=probs))
+ opp_bet_size, _, opp_anchor_idx = self._hierarchical_sizing(
+     opp_state_tensor, iteration, state=state, use_q=False)
```

### 3. Sizing Q только для traversing (`use_q` в `_hierarchical_sizing`)

**Причина:** sizing Q — инструмент traversing player'а. Оппоненты не должны использовать Q-guided sizing.

```python
def _hierarchical_sizing(self, state_tensor, iteration, state=None, use_q=True, player_id=None):
    ...
    elif use_q and self.sizing_q_enabled and self._sizing_q_is_ready():
        # Q-guided (только hero)
    else:
        # advantage_sizing_net + regret matching (всегда opp, fallback для hero)
```

| Узел | Action | Sizing |
|---|---|---|
| Hero | `advantage_nets[trav]` + regret matching | `_hierarchical_sizing(use_q=True)` |
| Opponent | `advantage_nets[cp]` + regret matching | `_hierarchical_sizing(use_q=False)` |

### 4. Per-player advantage sizing (5 сетей + 5 буферов + 5 оптимизаторов)

**Причина:** каждый игрок должен иметь свою sizing-ногу с независимыми CFR-регретами — симметрично с action-advantage.

| Компонент | Было (v1) | Стало (v2) |
|---|---|---|
| `advantage_sizing_net` | 1 общая | `advantage_sizing_nets[0..4]` (5 шт) |
| `sizing_optimizer` | 1 общий | `sizing_optimizers[0..4]` (5 шт) |
| `sizing_advantage_buffer` | 1 общий (или None) | `sizing_advantage_buffers[0..4]` (5 шт) |
| `_hierarchical_sizing` | `self.advantage_sizing_net(...)` | `self.advantage_sizing_nets[player_id](...)` |
| `train_sizing_anchor_network` | без `player_id` | с `player_id`, per-player nets/buffers |
| `_train_pg_and_log` | без `player_id` | с `player_id` |
| Чекпоинт save/load | Одиночная сеть | Список сетей/оптимизаторов |

### 5. Адаптивный batch size

**Причина:** strategy не тренировалась до 42-й итерации — `len(buffer) < 1024 → return 0`. Advantage использовал `min(batch_size, n)`, strategy — жёсткий порог. Приведено к единообразию.

```diff
# train_strategy_network
- if len(self.strategy_buffer) < batch_size:
-     return 0
+ n = len(self.strategy_buffer)
+ if n == 0:
+     return 0
+ effective_batch_size = min(batch_size, n)
```

Такой же фикс применён к `train_strategy_sizing_anchor_network` и `train_sizing_anchor_network` (оба пути: CFR и Q-derived).

---

## Итоговая архитектура

| Компонент | Сеть | Буфер | Per-player? |
|---|---|---|---|
| **Advantage action** | `advantage_nets[0..4]` (5) | `advantage_buffers[0..4]` (5) | Да |
| **Advantage sizing** | `advantage_sizing_nets[0..4]` (5) | `sizing_advantage_buffers[0..4]` (5) | Да |
| **Strategy** | `strategy_net` (1) | `strategy_buffer` (1) | Нет |
| **Strategy sizing** | `strategy_sizing_net` (1) | `sizing_strategy_buffer` (1) | Нет |
| **Sizing Q** | `sizing_q_net` (1) | `sizing_q_buffer` (1) | Нет |

### Принцип чистоты CFR

```
Advantage (per-player) → CFR regret matching → стратегия в буфер
Strategy (shared) → distillation буфера → быстрый inference
                                ↑
                     НЕ участвует в CFR-дереве!
```

Strategy-сеть — чистый distillation output. Ни hero, ни opp не используют её для принятия решений в CFR-обходе. Sizing Q влияет только на traversing player.

### Дополнительные правки (code hygiene)

1. **Коммент в `prepare_iteration`**: документирована асимметрия буферов — advantage/strategy очищаются, sizing_advantage — reservoir (НЕ чистить).
2. **`_hierarchical_sizing`**: попытка убрать re-encode при `state is not None` **сломала обучение** (max_depth упал до 10). **Откачено** — re-encode восстановлен, он необходим для правильной размерности тензора. `player_id` оставлен keyword/опциональным (не обязательный позиционный).
3. **Явный `player_id` во всех вызовах**: hero full (`traversing_player`), hero OS (`traversing_player`), opponent (`int(state.current_player)`).
4. **Конфиг**: `sizing_advantage_buffer_size: 16384 → 100000`. Больше места для reservoir-накопления CFR-регретов sizing'а.

### #100 — Config ablations поверх N5_4 (A1–A4, стабилизация деплой-политики)

**A1** `sizing_advantage_buffer_size: 16384 → 100000` (выше — reservoir плотнее)  
**A2** `discount_gamma: 2.0 → 1.0` (линейный strategy-вес, нет recency-перекоса)  
**A3** `advantage_epochs: 50 → 20` (меньше переобучения, быстрее)  
**A4** `strategy_buffer_reservoir: true` (НЕ чистить strategy_buffer — деплой-среднее = CFR-среднее по всей истории)

**A4 детали:** Новый флаг `strategy_buffer_reservoir` (default `false`). При `true` `prepare_iteration` не чистит `strategy_buffer` — он накапливается как reservoir (capacity = `strategy_memory_size` = 1M). Деплой-политика = честное CFR-среднее по всей истории → демпфирует дрейф raise, меньше train↔eval gap.

---

---

## Результаты (N5_4 — прогон 99v2)

**Стабильно и чуть лучше N5_3.**

| iter | raise% | win% | reward |
|------|--------|------|--------|
| 100 | 22.5 | 11.3 | 9.1 |
| 300 | 25.0 | 12.3 | 9.1 |
| 500 | 25.5 | 15.0 | 16.0 |
| 800 | 24.0 | 15.1 | 14.2 |
| 1100 | 19.5 | 12.3 | 12.0 |
| 1400 | 22.2 | 13.2 | 15.5 |
| 1600 | 23.0 | 12.2 | 12.2 |

**Ключевые наблюдения:**
- **raise держится ~20-25% все 1600 итераций — дрейфа нет** (как N5_3, но уровень повыше: ~23% vs ~19%)
- **reward стабильно выше N5_3**: пики 15-16 vs 12-15, среднее ~12.6 vs ~9.6
- **H2H 1600_vs_100**: reward +1.1, win 18.1% (выше baseline 16.7%), raise 28.5 → поздний уверенно бьёт ранний ✓
- **N5_4 ≥ N5_3 по всем метрикам**: raise агрессивнее, reward выше, H2H win выше baseline

### Что именно дало улучшение

#99v2 — правильные правки (strategy убрана из CFR-дерева, оппонент сайзит через advantage, per-player sizing). Они не сломали стабильность — даже чуть улучшили. Чистота CFR теперь корректна:

```
Advantage (per-player) → CFR regret matching → буфер
Strategy (shared) → distillation буфера → быстрый inference (НЕ вход в дерево)
```

### Оговорка (та же дисциплина)

В #99/#99v2 поменялось много всего сразу: per-iter traversing, random-агент, advantage_epochs 50, Q off, EMA off, per-player sizing, источник opp-sizing, adaptive batch. **Что именно убило дрейф — не атрибутировано.** Главные кандидаты: per-iteration traversing + EMA off. Для чистоты потом стоит ablation.

### Вердикт

Самый чистый и стабильный результат за всю серию:
- Катастрофический relapse — **снят** (N5 масштаб)
- Остаточный дрейф — **устранён** (#99: per-iter traversing + EMA off)
- Архитектура — **приведена к CFR-чистоте** (#99v2)
- Результат: стабильно ~23% raise, профит +12-16 vs рандом, late > early

**Дальше:** self-play против пула чекпоинтов (league) для роста win_rate; длинный прогон 10k+; ablation #99-переменных.

---

## См. также

- `sizing_reports/99-random-agent-per-iteration-strategy.md` — v1 отчёт
- `sizing_reports/SIZING.md` — полная хронология
