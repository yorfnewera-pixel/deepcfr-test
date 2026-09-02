# Bug #70: z-score targets + src2 loss-weight anneal для SizingQ MSE

## Статус: IMPLEMENTED (требуется validation run)

## Описание

После #69 `sizing_q_bootstrap_target_net=true` частично решил deadly-triad drift:
barbell-collapse ломается впервые на iter200, raise_freq восстановился с 2% до 28%.
Но мультимодальность **не держится** — к iter300/400 best_anchor снова в экстремуме.

Корневая проблема #69: source2 (lookahead bootstrap) остаётся **плоским по anchors**
(`source2_cross_anchor.mean_std≈0.02`, `mean_range≈0.05`) и **дрейфует по состояниям**
(`source2_intra_anchor_range` растёт 0.00→2.10→2.81→4.00).

Причина: `_bootstrap_sizing_value` вычисляет V(s) через Q-сети с масштабом,
отличным от src0/1 (src0 = raw chips, src2 = BB-скейлинг). 
Mixed-scale targets в буфере дают MSE, доминируемый крупными по модулю записями,
и source2 создаёт плоский шум, тянущий Q-surface к ~+2.1 независимо от anchor.

## Изменения

### 1. train-time z-score targets перед MSE

**Файл:** `src/core/deep_cfr.py` — метод `train_sizing_q_network`

При флаге `sizing_q_train_zscore=True` перед вычислением MSE targets нормализуются
per-batch: `targets_z = (targets - μ) / (σ + eps)`.

- z-score применяется **только к targets_t**, `q_pred` не нормализуется.
- Scale shift `sizing_q_net` ожидаем и принимаем как часть эксперимента:
  downstream использует относительные `q_vals - mean(q_vals)`.
- batch-статистики (μ/σ) считаются по всем строкам батча (включая source2).

### 2. src2 per-sample loss weight с линейным anneal

**Файл:** `src/core/deep_cfr.py` — методы `train_sizing_q_network` + `_current_sizing_q_src2_weight`

При флаге `sizing_q_src2_loss_weight_enabled=True`:
- per-sample MSE: `loss_i = (q_pred_i - target_i)²`
- вес: `w_i = 1.0` для source 0/1, `w_i = annealed_weight` для source 2
- итоговый loss: `sum(w_i * loss_i) / sum(w_i)`
- anneal: линейно от `sizing_q_src2_weight_start` (1.0) к `sizing_q_src2_weight_end` (0.25)
  за `sizing_q_src2_weight_decay_iterations` (400) итераций по `self.iteration_count`

### 3. Buffer возвращает sources

**Файл:** `src/core/deep_cfr.py` — `SizingQBuffer.sample` / `sample_stratified`

Добавлен параметр `return_sources=False` в оба метода. При `True` возвращается 4-tuple
`(states, norm_sizes, targets, sources)`. По умолчанию `False` — backward-compat.

### 4. Config

**Файл:** `config.yaml`

```yaml
sizing_q_train_zscore: true
sizing_q_zscore_eps: 1.0e-6
sizing_q_src2_loss_weight_enabled: true
sizing_q_src2_weight_start: 1.0
sizing_q_src2_weight_end: 0.25
sizing_q_src2_weight_decay_iterations: 400
```

### 5. Тесты

**Файл:** `tests/test_hybrid_outcome_sampling.py`

Добавлены 8 тестов:
- `test_sizing_q_buffer_sample_return_sources_false/true` — buffer sources
- `test_sizing_q_buffer_sample_stratified_return_sources` — stratified sources
- `test_sizing_q_current_src2_weight_anneal` — anneal formula iter0/200/400/800
- `test_sizing_q_train_zscore_targets` — zscore ON → zero-mean/unit-var targets
- `test_sizing_q_train_zscore_off_old_behavior` — OFF → raw targets unchanged
- `test_sizing_q_src2_loss_weight_formula` — weighted loss конечный
- `test_sizing_q_src2_loss_weight_off_normal_mse` — OFF → обычный reduction='mean'
- `test_checkpoint_saves_zscore_and_src2_flags` — 6 флагов в чекпоинте

Старый тест `test_train_sizing_q_network_uses_stratified_when_enabled` обновлён:
tracking-функция принимает `**kwargs` (т.к. `return_sources=True` теперь передаётся).

## Что НЕ трогалось

- inference (`_hierarchical_sizing`, `choose_action`) — без изменений
- `target_adv = q_vals - baseline` в `train_sizing_anchor_network`
- `_normalize_sizing_q_target`, clip ±5.0 — без изменений
- `_bootstrap_sizing_value`, `_add_sizing_q_sample` — без изменений
- diagnostics (`_record_sizing_q_target_diag`, `run_sizing_q_target_diag_stats`)
- monitoring-флаг не добавлялся

## Риски

1. **Scale shift `sizing_q_net`**: z-score targets меняет масштаб выхода Q-сети,
   что через `_hierarchical_sizing` и `_bootstrap_sizing_value` влияет на downstream.
   Это ожидаемый и принятый caveat эксперимента: проверяем normalized Q-surface.

2. **Per-batch статистика**: μ/σ батча могут быть шумными при малых batch_size.
   Если понадобится стабилизация — #71 (EMA или running stats).

3. **Взаимодействие с source2 bootstrap**: `_bootstrap_sizing_value` использует `sizing_q_net`
   для построения source2 targets: z-score Q-сети → изменит масштаб source2 таргетов.
   Именно поэтому source2 дополнительно down-weight через anneal.

## PASS-критерии (по диагностике прогона)

- `source2_intra_anchor_range` не растёт монотонно
- `source2_cross_anchor.mean_std` выше плоского `~0.02`
- `best_sizing_anchor_dist` не возвращается в dominance 0.10/3.00
- `raise_freq` желательно около/выше 25%
- `win_rate` не продолжает деградировать

## Валидация

1. Baseline: текущий ON (#69, commit с `sizing_q_bootstrap_target_net=true`).
2. Прогон #70 до iter400 с новыми флагами в `config.yaml`.
3. Сравнить метрики из `full_report` по PASS-критериям.
4. Если PASS — прогон до iter600.

## Дата: 2026-06-06
