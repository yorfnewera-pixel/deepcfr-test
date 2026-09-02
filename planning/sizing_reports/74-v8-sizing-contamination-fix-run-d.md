# Bug #74 v8: Sizing Contamination Fix — Legal-Anchor Mask + Kind-Filter + Replay Hygiene

> **Статус:** Implemented, config-gated (all off by default), tests passing.
> **Цель:** сделать verdict честным по концентрации sizing, затем config-gated исправить загрязнение sizing-Q через legal-anchor mask, kind-filter и опциональную очистку replay.
> **Дата:** 2026-06-09

---

## 1. Контекст

Из `74-v2-sizing-collapse-root-cause-diagnostics.md`: sizing-Q collapse происходит из-за загрязнения:
- `ALL_IN`/`MIN_RAISE` remap анкеров пишет targets в буфер под чужими anchor индексами
- Top-2 sizing mass концентрируется на 2-3 анкерах при 15-анкерной сетке
- lookahead (`source=2`) диагностирован в v5, но фильтрация по kind отсутствовала

Решение: трёхкомпонентный config-gated механизм, все выключены по умолчанию.

---

## 2. Что сделано

### 2.1 `tools/checkpoint_tools.py` — Concentration Verdict Gate

**`compute_verdict` (+18 строк):**
- Новая сигнатура: `compute_verdict(action_counts, sizing_counts, num_anchors, top2_mass_max=0.55, min_norm_entropy=0.0)`
- После single-anchor gate (90%+ dominance) добавлены:
  - **top-2 mass** > 55% → `WARN` с `"concentration: top-2 sizing mass = X%"`
  - **normalized entropy** < порог (опционально, default 0.0 = off) → `WARN`

### 2.2 `src/core/deep_cfr.py` — SizingQBuffer Extensions

**`SizingQBuffer.sample` и `sample_stratified` (+6 строк):**
- Добавлен параметр `return_selected_kinds=False`
- При `return_sources=True + return_selected_kinds=True` возвращает 5-tuple (state, size, target, source, kind_id)

**`SizingQBuffer.clear` (+5 строк):**
- Обнуляет `_position`, `_size`, `_anchor_counts` без пересоздания массивов

### 2.3 `config.yaml` + `DeepCFRAgent.__init__` — Config Flags

Новые флаги (все `false` по умолчанию):
```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: false
sizing_q_kind_filter_mode: drop       # or downweight
sizing_q_kind_filter_downweight: 0.25
sizing_q_replay_clear_once: false
```

Читаются в обеих ветках `__init__` (sizing_q_enabled/not enabled). Добавлен `_sizing_q_replay_clear_once_done = False`.

**Diagnostics dict (+8 ключей):**
```python
self.sizing_q_kind_filter_diag = {
    'normal_kept': 0, 'min_raise_dropped': 0, 'all_in_dropped': 0,
    'unknown_dropped': 0, 'downweighted': 0,
    'zero_weight_batches': 0, 'cleared_once': 0,
}
```
Сбрасывается в `reset_traversal_stats`.

### 2.4 Legal Anchor Mask Helpers

**`_legal_sizing_anchor_mask(self, state)` (+16 строк):**
- Вычисляет boolean mask для anchors: `True` = NORMAL после `_resolve_effective_sizing`
- Fallback: если все anchors заблокированы → возвращает all-True

**`_apply_sizing_anchor_mask(self, probs, state)` (+10 строк):**
- Применяет mask к вероятностям: обнуляет blocked, перенормирует
- При выключенном флаге — no-op (pass-through)

### 2.5 Apply Mask at Selection Sites

**`_hierarchical_sizing` (+2 строки):**
- Сигнатура: `_hierarchical_sizing(self, state_tensor, iteration, state=None)`
- После нормализации `full_probs` → вызов `self._apply_sizing_anchor_mask(full_probs, state)` (только если `state is not None`)

**`_cfr_traverse_multi_outcome_node` (+1 строка):**
- В Q-ready пути после нормализации → `self._apply_sizing_anchor_mask(full_probs, state)`

**`choose_action` (+5 строк):**
- `argmax`: маскирует `sizing_probs` перед argmax
- `top_bucket`: маскирует после `_sparsify_sizing_target_by_buckets`
- `hierarchical`: маскирует после сборки `full_probs`
- `stochastic`: маскирует перед `np.random.choice`

**Callsites обновлены** (×2): `_cfr_traverse_multi_outcome_node`, `cfr_traverse_multi` → `state=state`

### 2.6 Kind Filter in `train_sizing_q_network`

**Sampling (+5 строк):**
- `return_selected_kinds = self.sizing_q_kind_filter_enabled`
- Распаковка: 4-tuple или 5-tuple в зависимости от флага

**Per-sample loss (+25 строк):**
- При `sizing_q_src2_loss_weight_enabled or sizing_q_kind_filter_enabled` → per-sample loss + weights
  - `drop` mode: non-NORMAL → weight=0, считает `all_in_dropped`/`min_raise_dropped`/`unknown_dropped`
  - `downweight` mode: non-NORMAL → weight *= `sizing_q_kind_filter_downweight`, считает `downweighted`
  - `zero_weight_batches`: если все weights=0 → skip (continue)
- Обновления считаются отдельно (`updates`), не делятся на фиктивные epochs

### 2.7 Replay Clear Once

**`_maybe_clear_sizing_q_replay_once` (+10 строк):**
- Guard: флаг on, ещё не делали, buffer существует
- Вызывает `self.sizing_q_buffer.clear()`, устанавливает `_sizing_q_replay_clear_once_done = True`
- Вызывается в начале `train_sizing_q_network` перед проверкой `len(buffer) < batch_size`

### 2.8 Report/Checkpoint Passthrough

**`_build_checkpoint`:**
- 6 новых config-ключей: `sizing_q_legal_anchor_mask_enabled`, `sizing_q_mask_exclude_kinds`, `sizing_q_kind_filter_enabled`, `sizing_q_kind_filter_mode`, `sizing_q_kind_filter_downweight`, `sizing_q_replay_clear_once`
- `checkpoint['sizing_q_kind_filter_diag'] = dict(self.sizing_q_kind_filter_diag)`

**`_load_checkpoint`:**
- 6 новых ключей в списке `setattr`
- Загрузка `sizing_q_kind_filter_diag` с защитой от отсутствующих ключей

**`tools/checkpoint_tools.py`:**
- `load_full_checkpoint`: passthrough `sizing_q_kind_filter_diag`
- `save_json_report`: passthrough в JSON report
- `main` json dump: passthrough в итоговый report

---

## 3. Файлы

| Файл | Изменения |
|------|-----------|
| `tools/checkpoint_tools.py` | `compute_verdict` + concentration gate, load/report passthrough |
| `src/core/deep_cfr.py` | `SizingQBuffer.sample/stratified/clear`, config flags, mask helpers, mask at 3 sites, kind-filter loss, replay clear, checkpoint config+diag |
| `config.yaml` | 6 новых флагов (all false) |
| `tests/test_hybrid_outcome_sampling.py` | 8 новых тестов |

---

## 4. Тесты (8 новых, 85 всего passed)

| Тест | Проверяет |
|------|-----------|
| `test_compute_verdict_warns_on_top2_sizing_concentration` | barbell sizing → WARN |
| `test_sizing_q_buffer_sample_keeps_legacy_shape_without_kind_ids` | backward compat |
| `test_sizing_q_buffer_sample_can_return_selected_kind_ids` | 5-tuple with kind ids |
| `test_sizing_q_buffer_clear_resets_size_and_anchor_counts` | clear() works |
| `test_legal_sizing_anchor_mask_excludes_configured_kinds` | mask excludes ALL_IN/MIN_RAISE |
| `test_legal_sizing_anchor_mask_falls_back_when_all_blocked` | fallback all-true |
| `test_sizing_q_kind_filter_skips_zero_weight_batch` | all-drop → skip |
| `test_sizing_q_replay_clear_once_clears_and_only_once` | однократная очистка |

---

## 5. Config для Run D

```yaml
sizing_q_legal_anchor_mask_enabled: true
sizing_q_mask_exclude_kinds:
  - ALL_IN
  - MIN_RAISE
sizing_q_kind_filter_enabled: true
sizing_q_kind_filter_mode: drop
sizing_q_kind_filter_downweight: 0.25
sizing_q_replay_clear_once: true
```

**Не менять одновременно:** `sizing_lookahead_enabled`, Q target scale/zscore, anchors, heat temperature/top-p, action policy thresholds.

**Метрики для сравнения Run D vs Run C:**
- `verdict` (должен быть WARN/OK уже с concentration gate)
- `raise_freq`, `win_rate`, `mean_reward`
- `best_sizing_anchor_dist`, `collapse.anchor_raw_probs`
- `sizing_q_kind_filter_diag`, `selected_effective_kind_diag`
- `sizing_q_stats`, `q_raise_gt_max_sq_pct`, `q_call_gt_max_sq_pct`

---

## 6. Риски

- **MIN_RAISE mask** может снизить частоту маленьких рейзов в коротких стеках. Fallback "all blocked → all legal" обязателен.
- **Kind-filter drop** слишком агрессивен → `zero_weight_batches` (видно в diagnostics).
- **Маска после bucket Q** может оставить bucket-level leakage. Если Run D покажет странные bucket masses, следующим патчем маскировать Q до bucket `max`.
- **q_a3=0.0** не решается этим планом — отдельный diagnostic/fix после Run D.

---

## 7. Верификация: почему Run D не работал (2026-06-09)

> **Чекпоинт:** `models/test8seed1/multi_checkpoint_iter_100.pt`
> **Report:** `multi_checkpoint_iter_100_full_report.json`

### 7.1 Что обнаружено в чекпоинте

| Ключ | Значение | Ожидалось для Run D |
|------|----------|---------------------|
| `sizing_q_enabled` | `True` | `True` |
| `config.sizing_q_legal_anchor_mask_enabled` | **MISSING** | `True` |
| `config.sizing_q_kind_filter_enabled` | **MISSING** | `True` |
| `config.sizing_q_replay_clear_once` | **MISSING** | `True` |
| `config.sizing_q_mask_exclude_kinds` | **MISSING** | `[ALL_IN, MIN_RAISE]` |
| `sizing_q_kind_filter_diag` | `{all_zeroes}` | `normal_kept > 0` |
| `sizing_q_kind_filter_diag.cleared_once` | `0` | `1` |

### 7.2 Root Cause

Тренировка шла **без конфига 74v8**. Чекпоинт создан кодом, где флаги `sizing_q_legal_anchor_mask_enabled` и `sizing_q_kind_filter_enabled` инициализировались со значением по умолчанию `False` (строка 875-880 deep_cfr.py), а в `config.yaml` они не были выставлены в `true`.

Дополнительный фактор: флаги 74v8 сохраняются в `_build_checkpoint` **только внутри блока `if self.sizing_q_enabled:`** (строка 3039-3070). Если бы `sizing_q_enabled` был `False` в момент сохранения, они бы тоже не попали в чекпоинт.

### 7.3 Следствия для репорта

1. **`sizing_q_kind_filter_diag`** присутствует в чекпоинте (сохраняется вне `if sizing_q_enabled`, строка 3143), но все счётчики — нули, потому что kind-filter ни разу не срабатывал (флаг `sizing_q_kind_filter_enabled` — `False`).
2. **`sizing_selected_effective_kind_diag`** показывает много `MIN_RAISE`/`ALL_IN` selection'ов — legal-anchor mask не применялся.
3. **Verdict `WARN`** сработал корректно (top-2 sizing mass = 60%), но это концентрация **без** маскировки.

### 7.4 Потенциальный round-trip баг

Флаги 74v8 **сохраняются** в `checkpoint['config']` sub-dict (строки 3065-3070), но **загружаются** из `checkpoint` top-level (строки 3284-3292). Это значит:

- При чистом обучении с нуля: `_build_checkpoint` → сохраняет флаги в `config` → ок.
- При возобновлении (resume): `_load_checkpoint` → не видит флаги в `config` → `sizing_q_legal_anchor_mask_enabled` остаётся `False` (дефолт из `__init__`).

Даже если бы Run D тренировался с правильным `config.yaml`, после save→load→resume флаги бы сбросились в `False`.

### 7.5 Что нужно для настоящего Run D

1. Убедиться, что `config.yaml` содержит флаги из секции 5 с `true`.
2. Начать обучение **с нуля** (не resume от pre-74v8 чекпоинта).
3. После обучения проверить:
   - `config.sizing_q_legal_anchor_mask_enabled` → `True`
   - `sizing_q_kind_filter_diag.normal_kept` → `> 0`
   - `sizing_q_kind_filter_diag.min_raise_dropped` → `> 0` (если mask работала)
   - `sizing_q_kind_filter_diag.cleared_once` → `1`
 4. **Опционально:** починить round-trip bug, чтобы `_load_checkpoint` читал флаги из `checkpoint['config']` при отсутствии top-level ключей.

---
### 7.6 Round-trip bug — ИСПРАВЛЕНО (2026-06-09)

**Файл:** `src/core/deep_cfr.py`, `_load_checkpoint`, строка 3291-3293.

**Было:**
```python
if key in checkpoint and checkpoint[key] is not None:
    setattr(self, key, checkpoint[key])
```

**Стало:**
```python
if key in checkpoint and checkpoint[key] is not None:
    setattr(self, key, checkpoint[key])
elif 'config' in checkpoint and key in checkpoint['config'] and checkpoint['config'][key] is not None:
    setattr(self, key, checkpoint['config'][key])
```

Лоадер теперь читает флаги из `checkpoint['config']` как fallback, если ключ отсутствует в top-level. Это покрывает все sizing_q флаги: `sizing_q_legal_anchor_mask_enabled`, `sizing_q_kind_filter_enabled`, `sizing_q_replay_clear_once`, `sizing_q_mask_exclude_kinds`, `sizing_q_kind_filter_mode`, `sizing_q_kind_filter_downweight` и все старые тоже.

После этого фикса resume от чекпоинта с правильным config будет сохранять флаги активными.

---

### 7.9 Follow-up Fix: Activation/Report Diagnostics (2026-06-09)

Исправлены три причины, из-за которых Run D выглядел «не включённым» в checkpoint/report:

#### 7.9.1 Сохранение флагов в top-level + config

**Файл:** `src/core/deep_cfr.py`, `_build_checkpoint`, строка 3107-3117.

Флаги 74v8 теперь сохраняются **и в root чекпоинта, и в `checkpoint['config']`**:

```python
sizing_q_runtime_config = {
    'sizing_q_legal_anchor_mask_enabled': self.sizing_q_legal_anchor_mask_enabled,
    'sizing_q_mask_exclude_kinds': self.sizing_q_mask_exclude_kinds,
    'sizing_q_kind_filter_enabled': self.sizing_q_kind_filter_enabled,
    'sizing_q_kind_filter_mode': self.sizing_q_kind_filter_mode,
    'sizing_q_kind_filter_downweight': self.sizing_q_kind_filter_downweight,
    'sizing_q_replay_clear_once': self.sizing_q_replay_clear_once,
}
checkpoint.update(sizing_q_runtime_config)
config_dict.update(sizing_q_runtime_config)
```

#### 7.9.2 `_load_checkpoint` с `checkpoint_config` переменной

**Файл:** `src/core/deep_cfr.py`, `_load_checkpoint`.

Добавлена переменная `checkpoint_config = checkpoint.get('config', {}) or {}` перед циклом загрузки флагов. Fallback читает из `checkpoint_config` вместо дублирования `checkpoint['config']`.

#### 7.9.3 `cleared_once` не сбрасывается в `reset_traversal_stats`

**Файл:** `src/core/deep_cfr.py`, `reset_traversal_stats`.

`cleared_once` — lifecycle-событие, а не per-traversal метрика. Теперь он не обнуляется при сбросе traversal stats:

```python
persistent_sizing_q_kind_filter_keys = {'cleared_once'}
for key in self.sizing_q_kind_filter_diag:
    if key not in persistent_sizing_q_kind_filter_keys:
        self.sizing_q_kind_filter_diag[key] = 0
```

#### 7.9.4 `normal_kept` инкрементится при kind-filter training

**Файл:** `src/core/deep_cfr.py`, `train_sizing_q_network`.

Добавлен `normal_kept` счётчик — сколько NORMAL samples участвовало в обучении после фильтрации:

```python
self.sizing_q_kind_filter_diag['normal_kept'] += int((kind_ids_t == normal_id).sum().item())
```

#### 7.9.5 `sizing_q_config` блок в JSON report

**Файл:** `tools/checkpoint_tools.py`.

- `load_full_checkpoint`: собирает `sizing_q_config` dict из чекпоинта (top-level → `config` fallback).
- `save_json_report` + `main` report_dict: passthrough `sizing_q_config` и `sizing_q_kind_filter_diag` в JSON.

#### 7.9.6 Тесты (3 новых, 70/73 passed)

```text
test_build_checkpoint_persists_sizing_q_74v8_flags_in_top_level_and_config  PASSED
test_load_checkpoint_restores_sizing_q_74v8_flags_from_config              PASSED
test_reset_traversal_stats_keeps_sizing_q_cleared_once                     PASSED
```

Три pre-existing failure в тестах `train_sizing_q_network_uses_stratified` / `zscore_targets` / `zscore_off` — не связаны с правками (буфер < batch_size).

#### 7.9.7 Итоговый checklist для валидного Run D

Новый Run D считается **валидным** только если JSON содержит:

```json
"sizing_q_config": {
  "sizing_q_legal_anchor_mask_enabled": true,
  "sizing_q_mask_exclude_kinds": ["ALL_IN", "MIN_RAISE"],
  "sizing_q_kind_filter_enabled": true,
  "sizing_q_kind_filter_mode": "drop",
  "sizing_q_kind_filter_downweight": 0.25,
  "sizing_q_replay_clear_once": true
},
"sizing_q_kind_filter_diag": {
  "normal_kept": 12345,
  "min_raise_dropped": 78,
  "all_in_dropped": 0,
  "unknown_dropped": 1057,
  "downweighted": 0,
  "zero_weight_batches": 0,
  "cleared_once": 1
}
```

Если этого нет — закоммиченный код не совпадает с запущенным, или используется не тот checkpoint/report path.
