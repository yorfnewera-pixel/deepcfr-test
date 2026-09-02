# Баг #13: Взрыв лосса advantage-net (64 млрд+)

**Серьёзность:** Критическая
**Статус:** Исправлено
**Дата:** 2026-05-06

## Суть проблемы

Advantage loss взрывался до 64+ миллиардов, делая обучение невозможным. Выяснилось, что это **4 бага одновременно**, каждый из которых усиливал остальные.

## Баг 13-A: initial_stake < 1 → деление на ≈0 в encode_state

**Корневая причина**: `encode_state` нормализует pot/bet/stake/min_bet на `initial_stake = state.players_state[0].stake`. Когда P0 шёл all-in, его stake мог стать < 1 (но > 0), fallback на 1.0 не срабатывал, и деление на 0.000001 давало **секстиллионы** (10^16).

**Проявление**: Данные P1-P5 в буфере содержали нормы до 10^16, предсказания сети — до 10^13. P0 был чище (максимум 248), потому что в `cfr_traverse` (оригинальный) P0 traversed раньше, до all-in.

**Фикс**:
```python
# БЫЛО:
if initial_stake <= 0:
    initial_stake = 1.0

# СТАЛО:
if initial_stake < 1.0:
    initial_stake = 1.0
```

## Баг 13-B: sqrt(iteration) при записи регретов → растущие таргеты

**Корневая причина**: `scale_factor = sqrt(iteration)` применялся **при записи** в буфер, а не при обучении strategy. Это значит что таргеты advantage-net росли с каждой итерацией (×1, ×10, ×31, ×100...), а сеть с lr=1e-6 не могла их догнать.

**Оригинал Deep CFR** (Brown et al. 2018): линейные веса `⌊(t+1)/2⌋` применяются при **сэмплировании из буфера для strategy-net**, не при хранении регретов.

**Фикс**: Убрать `scale_factor` из хранения, добавить Linear CFR веса в `train_strategy_network`:
```python
# Стратегия: Linear CFR веса ⌊(t+1)/2⌋
linear_weights = torch.floor((iteration_tensors + 1) / 2)
linear_weights = linear_weights / linear_weights.sum()
```

## Баг 13-C: lr=1e-6 — сеть не обучается

**Корневая причина**: Оригинальный Deep CFR использует lr=0.001. Наш lr=1e-6 — в **1000 раз** меньше. При clip ±2 таргеты в [-2, +2], а сеть едва сдвигается от случайной инициализации.

**Фикс**: lr=1e-6 → 1e-4 (компромисс: оригинал 0.001, но у нас incremental обучение + 6 игроков → нужен консервативный lr).

## Баг 13-D: Приоритетная выборка (PER) дестабилизирует

**Корневая причина**: PER выбирает записи с наибольшей ошибкой → даёт наибольшие веса → наибольшие обновления → взрыв. Combined с багом 13-A (триллионы в данных), PER концентрировался на мусорных записях.

**Оригинал Deep CFR**: Использует **reservoir sampling** (uniform), не PER.

**Фикс**: Заменить PER на uniform sampling в `train_advantage_network`. PER оставлен в классе PrioritizedMemory для потенциального будущего использования.

## Итоговые изменения

| Файл | Изменение |
|------|-----------|
| `model.py:86-88` | `initial_stake <= 0` → `initial_stake < 1.0` |
| `deep_cfr.py:118-119` | Убран `scale_factor = sqrt(iteration)` из cfr_traverse |
| `deep_cfr.py:506-510` | Убран `scale_factor` из cfr_traverse_multi |
| `train.py:191-193` | Убран `scale_factor` из _cfr_traverse_with_opponents |
| `deep_cfr.py:lr` | lr=1e-6 → 1e-4 |
| `deep_cfr.py:clip` | clip ±10 → ±2 (ограничение масштаба таргетов) |
| `deep_cfr.py:train_adv` | PER → uniform sampling |
| `deep_cfr.py:batch/epochs` | batch=128/epochs=3 → batch=256/epochs=3 |
| `deep_cfr.py:strategy` | Linear CFR веса `⌊(t+1)/2⌋` вместо `t/sum(t)` |
| `deep_cfr.py:grad_clip` | 0.5 → 0.5 (оставлен) |
| `deep_cfr.py:train_adv_multi` | Новый метод — обучает на данных ВСЕХ 6 игроков одновременно |

## Результат

До: loss = 64,864,387,072
После: loss = 0.24–0.35 (стабильно для всех 6 игроков)
