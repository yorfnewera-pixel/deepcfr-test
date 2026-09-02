# Bug #65: checkpoint_tools — диагностика sizing_q_buffer из full checkpoint

**Severity:** LOW (инструментальная, не ломает обучение)

**Дата:** 2026-06-04

**Связан с:** Bug #61 (sizing collapse), Bug #62 (q_network_gap), Bug #63 (probe decay), Bug #64 (one-step lookahead)

---

## Симптомы

При отладке sizing collapse (#61) и проверке эффективности probe (#63) / lookahead (#64) невозможно понять из отчёта `checkpoint_tools.py`:

1. **Какие source'ы доминируют в буфере** (regular=0, probe=1, lookahead=2)?
2. **Какие анкоры перепредставлены / недопредставлены** в буфере?
3. **Какие mean_target'ы по анкерам** — реальные ли значения или нули/константы?

В `deep_cfr.py:2483-2488` эти поля уже сохраняются в full checkpoint:
- `sizing_q_buffer_sources` (uint8)
- `sizing_q_buffer_anchor_indices` (int64)
- `sizing_q_buffer_targets` (float32)

Но `checkpoint_tools.py` их не показывает.

---

## Root Cause

`load_full_checkpoint()` грузила только веса сетей, но не извлекала буферные массивы из чекпоинта. Не было функции для подсчёта статистики по ним.

---

## Fix

**Файл:** `tools/checkpoint_tools.py`

### 1. `load_full_checkpoint()` (стр. 245-248)
Добавлено извлечение трёх массивов (опционально — только для full чекпоинтов):

```python
if 'sizing_q_buffer_sources' in ckpt:
    nets['sizing_q_buffer_sources'] = ckpt['sizing_q_buffer_sources']
    nets['sizing_q_buffer_anchor_indices'] = ckpt['sizing_q_buffer_anchor_indices']
    nets['sizing_q_buffer_targets'] = ckpt['sizing_q_buffer_targets']
```

### 2. `run_sizing_q_buffer_stats(nets)` — новая функция (стр. 458-518)
Считает:
- `count_by_source` — распределение {0: N, 1: N, 2: N}
- `count_by_anchor` — {anchor_label: count}
- `count_by_source_anchor` — {(source, anchor_label): count}
- `mean_target_by_anchor` — {anchor_label: mean}
- `mean_target_by_source_anchor` — {(source, anchor_label): mean}

Возвращает `None` если буферные массивы отсутствуют (light чекпоинт).

### 3. `print_sizing_q_buffer_stats(stats)` — новая функция (стр. 521-549)
Форматированный вывод в консоль: source-распределение, таблица анкер × count × mean_target, top-20 пар (source, anchor).

### 4. Интеграция в `main()` (стр. 932-935, 947-948)
Вызов после `run_q_comparison` в diagnose/full режимах. Результат сохраняется в `result['sizing_q_buffer']` и попадает в JSON-отчёты (`_diagnose_report.json`, `_full_report.json`).

### 5. `save_json_report()` (стр. 847-849)
Добавлен экспорт `sizing_q_buffer` в full JSON.

---

## Использование

```bash
# Diagnose — покажет sizing_q_buffer статистику если чекпоинт full
python tools/checkpoint_tools.py diagnose checkpoint_800.pt

# Full — то же + eval + JSON с sizing_q_buffer
python tools/checkpoint_tools.py full checkpoint_800.pt
```

Для light чекпоинтов — silently skipped (нет буферных массивов).

---

## Ожидаемый эффект

- Видна доля probe/lookahead/regular сэмплов в буфере → оценка эффективности #63 и #64
- Видно покрытие анкоров → диагностика причины collapse (#61)
- Видны mean_target'ы → проверка, не застыли ли Q-значения на нулях
- Данные в JSON → можно строить графики динамики по итерациям
