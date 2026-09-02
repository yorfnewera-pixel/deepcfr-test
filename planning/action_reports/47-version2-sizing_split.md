# Bug #47-v2 — Изоляция SizingNet и удаление log1p

## Статус: APPLIED (Фазы 0–3.4)

## Контекст

Первая версия репорта #47 предлагала косметические правки (EMA running std, entropy floor, sigmoid weighting, diagnostic var(size|cluster)). Эмпирически прогон sizing47 показал: эти правки не решают корень — sizing head всё равно застрял (z_mean ≈ 0, log_std ≈ -1.0, PGMeanAdvantage шумит около нуля, cfv_running_std гонится за растущим CFV).

Повторный анализ выявил три патологии, из которых репорт-v1 атаковал только одну:

1. **Патология 1 — Single-sample variance leakage.** Один (state, sampled_size, CFV) на Raise-узел не даёт градиент ∂CFV/∂size. Без variance в size внутри одного state PG-сигнал структурно нулевой.
2. **Патология 2 — Strategy_net.sizing_head учится по копии.** MSE на sampled_bet_size из strategy_buffer.
3. **Патология 3 — Shared base конфликт.** PG-loss и advantage-MSE-loss оба бэкпропятся в advantage_net.base.

Полная история фаз 0–3.3 — в предыдущих версиях этого файла; ниже актуальный статус и новый раздел Фазы 3.4.

---

## Диагностика чекпоинта sizing4-5 / iter_1900 (Фаза 3.4 — подготовка)

### Утилиты

Удалены устаревшие: `diagnose_checkpoint.py`, `inspect_checkpoint.py`, `check_ckpt_format.py`, `_tmp_smoke_actions.py`. Заменены одной утилитой `diagnose_sizing.py` (376 строк): сравнение `advantage_sizing_net` vs `strategy_sizing_net` на одних state, инспекция весов голов (mean_head / log_std_head / value_head), dead-ReLU по слоям base, JSON-отчёт, поддержка нескольких чекпоинтов сразу для построения траектории.

### Результаты прогона на iter_1900 (100 state, seed=42, CPU)

| Метрика | advantage_sizing | strategy_sizing |
|---|---|---|
| z_mean mean | +0.2271 | +0.1675 |
| z_mean std | **0.0320** | **0.0095** |
| log_std mean | -1.2950 | -0.6500 |
| log_std std | 0.0023 | **0.0000** |
| bet_size mean | 1.8735 | 1.7906 |
| bet_size std | 0.0441 | 0.0134 |

Веса голов:

| Сеть | mean_head w_norm | log_std_head w_norm | value_head w_norm |
|---|---|---|---|
| advantage_sizing | 0.1011 | 0.5406 | 6.3890 |
| strategy_sizing | 0.0849 | **0.0000** | **0.0000** |

### Выводы

1. **advantage_sizing_net жива**: z_mean std=0.032, контекст различается, головы сдвинулись от zero-init.
2. **strategy_sizing_net мертва на log_std и value**: w_norm = 0.0000 ровно — ни одного шага градиента за 1900 итераций. Только mean_head обучалась.
3. **Причина**: distillation в `train_strategy_network` (до Фазы 3.4) копировала только `mean_sizing(z_mean)` через MSE. `log_std_head` и `value_head` strategy-сети не входили в граф loss → zero-init навсегда.
4. **log_std=-0.65 у strategy** — это zero-init артефакт (sigmoid(0) как midpoint диапазона [-1.3, 0]), а не выученное значение.
5. **Light-чекпоинты в текущем коде не битые** — все 6 мест сохранения в `src/training/train.py` пишут обе сети. Старые light без `strategy_sizing_net` — наследие предыдущих версий, не активный баг.

---

## Фаза 3.4 — distillation полного распределения (APPLIED)

### Правки

1. **`src/core/deep_cfr.py:1424-1453`** — в `train_strategy_network` distillation расширена:
   - target теперь `(z_mean, log_std)` оба из `advantage_sizing_net`;
   - loss = `mean_loss + log_std_weight * log_std_loss`;
   - оба члена взвешиваются тем же DCFR·raise_mask весом.

```python
with torch.no_grad():
    adv_z_mean, adv_log_std, _ = self.advantage_sizing_net(state_tensors)
    target_sizing = self.advantage_sizing_net.mean_sizing(adv_z_mean).detach()
    target_log_std = adv_log_std.detach()

strat_z_mean, strat_log_std, _ = self.strategy_sizing_net(state_tensors)
predicted_sizing = self.strategy_sizing_net.mean_sizing(strat_z_mean)

raise_mask = (policy_tensors[:, 3] > 0).float().unsqueeze(1)
sizing_weight = weights_expanded * raise_mask
denom = raise_mask.sum() + 1e-6

mean_loss = ((predicted_sizing - target_sizing) ** 2 * sizing_weight).sum() / denom
log_std_weight = cfg_get('strategy_log_std_distill_weight', 0.3)
log_std_loss = ((strat_log_std - target_log_std) ** 2 * sizing_weight).sum() / denom
sizing_loss = mean_loss + log_std_weight * log_std_loss
```

2. **`src/utils/config.py`** — `_DEFAULTS`: `strategy_log_std_distill_weight: 0.3`.
3. **`config.yaml`** — `strategy_log_std_distill_weight: 0.3`.

### Обоснование веса 0.3

- **1.0 риск обратного коллапса**: strategy_sizing_net начнёт копировать также шум log_std из advantage_sizing_net, а log_std сам ходит от PG-сигнала.
- **0.3** даёт log_std_head заметный градиент, чтобы выйти из zero-init, но mean остаётся доминирующим (~3× по весу).
- Параметр в конфиге — можно поднять до 0.5/1.0 или понизить до 0.1 без правки кода.

### Совместимость чекпоинтов

Архитектура не меняется — старые full-чекпоинты технически совместимы. Но запускать лучше с нуля: у sizing4-5 `log_std_head` strategy-сети ровно zero-init, и новая distillation начнёт его тянуть к `adv_log_std ≈ -1.3` (на хвосте). Чистый эксперимент = старт с нуля.

### Критерии успеха для следующего прогона (~500–800 ит., с нуля)

1. **`strategy_sizing.log_std_head.w_norm > 0.1`** после 500+ ит. — голова реально обучается.
2. **`gap (adv-strat) z_mean std < 0.02`** — strategy догоняет advantage по контексту.
3. **`strategy_sizing.bet_size_std ≥ 0.03`** на eval-state (против 0.013 сейчас) — sizing на инференсе различает state.
4. **Критерии Фазы 3.3 не регрессируют**: `ZMean ∈ ±0.3`, `RaiseFreq ≥ 0.20`, `Predicted_Size ∈ 1.5–1.9`, `LossAdvantage` стабильно в тысячах.

### Сигналы провала

- `log_std_loss` доминирует над `mean_loss` (даже при 0.3 веса) → strategy log_std дёргается шумом advantage. Понизить вес до 0.1.
- `strategy_sizing.log_std_head.w_norm` всё ещё ≈ 0 после 500 ит. → distillation log_std не доходит, проверить `log_std_weight × raise_mask.sum()` (мало raise-узлов в batch?).

### Чем Фаза 3.4 отличается от вариантов 2/3

- **Вариант 2 (прямое копирование state_dict / EMA)** — теряет DCFR-усреднение по итерациям.
- **Вариант 3 (использовать advantage_sizing на инференсе)** — нарушает архитектурную идею Deep CFR, где политика для inference усредняется через strategy-сеть.
- **Фаза 3.4 (Вариант 1)** — минимальная правка, сохраняет архитектуру, добавляет ровно те градиенты, которых не хватало двум головам strategy-сети.

Если после прогона критерии не выполнятся — fallback: Вариант 2 в форме EMA с малым τ или временный переход на Вариант 3 для проверки гипотезы паритета.
