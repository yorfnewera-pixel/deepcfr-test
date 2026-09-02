# Bug #71: отключение target-only z-score, чистый контроль src2 anneal

## Статус: IMPLEMENTED (требуется validation run)

## Контекст

#70 включил `sizing_q_train_zscore=true`: z-score применялся **только к `targets_t`**
в `train_sizing_q_network`, `q_pred` оставался raw. Это сознательно меняло шкалу выхода
`sizing_q_net`.

Validation iter200 показал FAIL:
- `verdict=WARN`, `reason=low_raise 17.8%`
- `win_rate=22.1`
- `best_sizing_anchor_dist`: `3.00 = 99.6%`
- policy raw sizing ушёл в small/0.10, q_best почти всегда 3.00 — полный рассинхрон

## Гипотеза

Target-only z-score изменил scale/семантику выхода `sizing_q_net`, а downstream читает raw Q:

1. `_hierarchical_sizing` использует `sizing_q_net` через `compute_sizing_heat_weights` с
   **абсолютным** `min_advantage=0.005`
2. `train_sizing_anchor_network` учит `target_adv = q_vals - mean(q_vals)` на raw Q
3. Раздутый spread Q сатурирует heat → argmax-коллапс `best_sizing_anchor_dist`

Первичная причина — scale shift, не src2 (src2 плоский, сокращается в `q - mean`).

## Изменения

Одно изменение в `config.yaml`:

```yaml
sizing_q_train_zscore: false   # было true — ВЫКЛЮЧЕНО
```

Оставлено без изменений:
```yaml
sizing_q_src2_loss_weight_enabled: true
sizing_q_src2_weight_start: 1.0
sizing_q_src2_weight_end: 0.25
sizing_q_src2_weight_decay_iterations: 400
```

## Что НЕ трогалось

- inference (`_hierarchical_sizing`, `choose_action`)
- `_normalize_sizing_q_target`, clip ±5.0
- `_bootstrap_sizing_value`, `_add_sizing_q_sample`
- существующая target diagnostics
- код z-score в `train_sizing_q_network` — оставлен за флагом (не удалён)
- src2 anneal strength/speed — не менялся (это отдельный #72)
- новая диагностика — не добавлялась (чистый контрольный эксперимент)

## PASS-критерии (по диагностике прогона до iter400)

- `raise_freq` возвращается к #69/#70 iter100 диапазону (~28-35%)
- `win_rate` не продолжает деградировать
- `best_sizing_anchor_dist` не доминируется одним anchor >90%
- policy raw sizing и q_best не расходятся как small-vs-3.00
- `source2_intra_anchor_range` не растёт монотонно без z-score

## Валидация

1. Baseline: #69 (`sizing_q_bootstrap_target_net=true`).
2. Прогон #71 до iter400 с `sizing_q_train_zscore=false`.
3. Сравнить метрики из `full_report` по PASS-критериям.
4. Если PASS на iter200 — прогон до iter400 для стабильности.

## Если FAIL

- **#72**: усилить/ускорить src2 down-weight anneal (например `1.0 -> 0.1 / 200`)
- **#73**: loss-normalization `(q-μ)/σ` vs `(t-μ)/σ` с detached EMA/fixed σ — если MSE-scale imbalance подтверждён диагностикой

## Дата: 2026-06-07
