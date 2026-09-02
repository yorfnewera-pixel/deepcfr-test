# Bug #74 v5: Source=2 Lookahead Diagnostics Instrumentation

> **Статус:** Diagnostics deployed, обучение НЕ изменено.
> **Цель:** выяснить, почему `source=2` (lookahead) пишет `target=0.0` по всем 15 anchors.
> **Следующий шаг:** diagnostic run (1-2 итерации) + новый report с `sizing_lookahead_diag`.
> **Дата:** 2026-06-09

---

## 1. Контекст

Из `models/test3seed1/multi_checkpoint_iter_100_full_report.json`:

- `sizing_q_buffer.mean_target_by_source_anchor`: `source=2` = `0.0` по всем 15 anchors.
- `source=2` = 14 565 samples (29% буфера, 50k total).
- `sizing_q_target_diag`: у всех `source=2` бакетов `clip_high=0, clip_low=0` (в отличие от `source=0/1`, где клипы массовые).

Это либо bootstrap-проблема, либо баг в записи. Данные не позволяют выбрать между гипотезами без прямой диагностики post-raise состояний.

Обсуждение OPUS+GPT сошлось: сначала root-cause `source=2`, потом legal-mask + kind-filter + replay hygiene + verdict-gate.

---

## 2. Что сделано

### 2.1 `src/core/deep_cfr.py`

**`__init__` (+16 строк):**
```python
self.sizing_lookahead_diag = {
    'counts': {
        'attempts': 0, 'applied_ok': 0, 'added': 0,
        'final_state': 0, 'nonterminal': 0,
        'bootstrap_debug_missing': 0,
        'vk_near_zero': 0, 'target_near_zero': 0,
    },
    'stats': {},
}
self._last_bootstrap_debug = None
```

**`_bootstrap_sizing_value` (+7 строк):**
В конце метода перед `return v` захватывается:
```python
self._last_bootstrap_debug = {
    'strategy_sum': float(strategy.sum().item()),
    'legal_sum': float(mask_t.sum().item()),
    'q_vals': q_vals.detach().reshape(-1).cpu().numpy().copy(),
}
```

**`_perform_sizing_lookahead` (+40 строк):**
- `diag_on = self.sizing_q_target_diagnostics_enabled`
- Счётчик `attempts` при входе в цикл по anchors
- Счётчик `applied_ok` + сброс `_last_bootstrap_debug` после `s_next.status == Ok`
- После `_compute_sizing_q_target_values`: обновляются `added`, `final_state`, `nonterminal`, `vk_near_zero`, `target_near_zero` (eps=1e-8)
- Running stats через `_upd_la_stat`: `v_k`, `pre_clip`, `target`, `pot`, `strategy_sum`, `legal_sum`, `q_a0..q_a3`
- Если `nonterminal` но `_last_bootstrap_debug is None` → `bootstrap_debug_missing += 1`

**`_upd_la_stat` (+13 строк):**
```python
def _upd_la_stat(self, key, val):
    """обновляет бегущие [n, sum, sumsq, min, max] для метрики."""
    val = float(val)
    s = self.sizing_lookahead_diag['stats'].get(key)
    if s is None:
        self.sizing_lookahead_diag['stats'][key] = [1, val, val * val, val, val]
    else:
        s[0] += 1; s[1] += val; s[2] += val * val; s[3] = min(s[3], val); s[4] = max(s[4], val)
```

**Checkpoint save (+2 строк):**
```python
if self.sizing_q_target_diagnostics_enabled:
    checkpoint['sizing_lookahead_diag'] = self.sizing_lookahead_diag
```

### 2.2 `tools/checkpoint_tools.py`

**Loader (+3 строк):**
```python
if 'sizing_lookahead_diag' in ckpt:
    nets['sizing_lookahead_diag'] = ckpt['sizing_lookahead_diag']
```

**Formatter (+30 строк):**
`run_sizing_lookahead_diag_stats(nets)` — превращает `[n, sum, sumsq, min, max]` в `{count, mean, std, min, max}`.

**Report JSON (full mode, +4 строки):**
Добавлен `report['sizing_lookahead_diag']`.

**Result assembly (+3 строки):**
Вызов `run_sizing_lookahead_diag_stats(nets)` → `result['sizing_lookahead_diag']`.

**Diagnose JSON (+2 строки):**
Добавлен `report_dict['sizing_lookahead_diag']`.

### 2.3 Гейт

Вся диагностика включается/выключается через имеющийся флаг `sizing_q_target_diagnostics_enabled` (уже включён в test3seed1). При выключенном флаге overhead = 0.

---

## 3. Что НЕ менялось

- Обучение, inference, loss — без изменений.
- `_add_sizing_q_sample` для source=2: metadata не прокидывается (в этом плане не требуется).
- `sizing_lookahead_enabled` не менялся.
- `SizingQBuffer.sample()` / `sample_stratified()` не менялись.

---

## 4. Тесты

```text
py_compile src/core/deep_cfr.py → OK
py_compile tools/checkpoint_tools.py → OK
70/70 tests pass (test_hybrid_outcome_sampling.py + test_sizing_q_regret.py)
```

---

## 5. Как читать новый `sizing_lookahead_diag` в report

| Условие | Интерпретация |
|---------|---------------|
| `final_state ≈ added` | Проблема в terminal reward ветке: reward в точке apply_action = 0 |
| `nonterminal большой + vk_near_zero ≈ nonterminal` | Проблема в bootstrap path |
| `bootstrap_debug_missing > 0` | `_bootstrap_sizing_value` не оставляет debug для части nonterminal states (пропуск) |
| `strategy_sum ≈ 0 или legal_sum ≈ 0` | Проблема в маске/перспективе |
| `q_a0..q_a3 ≈ 0` на post-raise states | Distribution mismatch: action q_net ненулевой на diagnostic states, но нулевой на post-raise |
| `q_a* ненулевые, но v_k ≈ 0` | Проблема в суммировании `strategy · Q` |

---

## 6. Следующий шаг

Выполнить diagnostic run (1-2 итерации с `sizing_q_target_diagnostics_enabled=True`, `sizing_lookahead_enabled=True`), затем:

```powershell
py -3 tools/checkpoint_tools.py full models/test3seed1/<checkpoint>.pt --games 1000 --num-states 500 --seed 1
```

По `sizing_lookahead_diag` в JSON выбрать конкретную ветку фикса.

После root-cause `source=2` — следующий этап (отдельный план):
- legal-anchor mask на выборе
- kind-aware loss filter (NORMAL vs MIN_RAISE vs ALL_IN)
- replay hygiene
- verdict-gate по концентрации
