# Bug #74 — Nearest-Anchor Snapping: Bucket Anchor Может Быть Нелегален

**Дата:** 2026-06-08
**Статус:** Open (диагностика добавлена, фикс — отдельно)
**Severity:** Medium (data quality: Q-targets assigned to illegal bucket anchors)

---

## Краткое описание

После `_resolve_effective_sizing` эффективный multiplier `eff_mult` snap'ится к ближайшему фиксированному anchor из `anchors_arr` через `argmin(abs(diff))`. Этот ближайший anchor может быть **нелегален** в текущем game state — например, ниже `min_raise`.

### Конкретный пример

```
state.pot = 100
state.last_raise_increment = 12  (min_raise = 12 чипов)

selected anchor  = 0.10  (10 чипов)
_resolve_effective_sizing:
    raw = 10 < min_raise = 12 → effective = 12, kind = MIN_RAISE
    eff_mult = 12/100 = 0.12

Ближайший anchor в сетке [0.10, 0.25, 0.33, ..., 3.00]:
    |0.12 - 0.10| = 0.02  ← ближе
    |0.12 - 0.25| = 0.13
    eff_idx = индекс 0.10
```

**Результат:** в `sizing_q_buffer` записывается `anchor_idx` для 0.10 — который в этом стейте как самостоятельный sizing **нелегален** (ниже min_raise=12).

---

## Где происходит

Все три места, где эффективный sizing маппится к bucket anchor:

| Файл | Строка | Путь |
|------|--------|------|
| `src/core/deep_cfr.py` | ~1816 | Hero OS sizing-Q sample |
| `src/core/deep_cfr.py` | ~1983 | Hero full traversal sizing-Q sample |
| `src/training/train.py` | 317 | External training OS |

Во всех трёх случаях:
```python
eff_mult, _, _ = self._resolve_effective_sizing(state, sampled_bet_size)
eff_idx = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
```

Нет проверки, является ли `self.anchors_arr[eff_idx]` **легальным** sizing для этого `state`.

---

## Последствия

1. **Q-target noise:** Q-сеть учится предсказывать target для anchor 0.10 из стейта, где 0.10 никогда не мог быть применён средой. Это зашумляет обучение.

2. **Стратифицированный sampling:** `sizing_q_buffer.sample_stratified()` полагается на `anchor_idx` для балансировки по anchors. Нелегальный bucket получает "фантомные" сэмплы, искажая распределение.

3. **Скрытая проблема:** В простой матрице `selected_idx → eff_idx` кейс `0.10 → 0.10` выглядит как диагональ (корректный mapping), но на самом деле `eff_mult = 0.12 ≠ 0.10`.

---

## Добавленная диагностика (Bug #74)

Добавлена read-only диагностика, которая **не фиксит проблему**, но делает её видимой:

### 1. `sizing_selected_effective_diag`
Матрица `num_anchors × num_anchors` для трёх путей (`hero_os_raise`, `hero_full_raise`, `opponent_raise`). Считает пары `(selected_anchor_idx, bucket_anchor_idx)`.

### 2. `sizing_selected_effective_kind_diag`
Для каждого пути:
- `selected_kind`: что вернул `_resolve_effective_sizing(state, selected_anchor)` — NORMAL/MIN_RAISE/ALL_IN
- `bucket_kind`: что вернул `_resolve_effective_sizing(state, nearest_bucket_anchor)` — **этот счётчик покажет MIN_RAISE > 0 для кейса выше**
- `bucket_relation`: ниже/равен/выше ли bucket anchor относительно реального `eff_mult`

### Как интерпретировать

```json
"hero_full_raise": {
  "selected_kind": {"NORMAL": 950, "MIN_RAISE": 50, "ALL_IN": 0},
  "bucket_kind":   {"NORMAL": 950, "MIN_RAISE": 50, "ALL_IN": 0},
  "bucket_relation": {"below_effective": 25, "equal_effective": 975, "above_effective": 0}
}
```

- `selected_kind.MIN_RAISE = 50` — 50 раз выбранный sizing был форсирован до min_raise
- `bucket_kind.MIN_RAISE = 50` — все 50 раз ближайший bucket anchor **тоже был бы** форсирован до min_raise → **bucket нелегален как самостоятельный sizing**
- `bucket_relation.below_effective = 25` — 25 раз ближайший anchor был **ниже** реального effective — это подтверждение проблемы

---

## Предлагаемый фикс (отдельно от диагностики)

Заменить `argmin(abs(diff))` на поиск **ближайшего legal anchor**:

```python
def _nearest_legal_anchor_idx(self, eff_mult, state):
    """Возвращает индекс ближайшего anchor, который легален в state."""
    eff_mult = float(eff_mult)
    legal_mask = np.ones(self.num_anchors, dtype=bool)
    for i, anchor in enumerate(self.anchors_arr):
        _, _, kind = self._resolve_effective_sizing(state, float(anchor))
        if kind == 'NORMAL':
            legal_mask[i] = True
        else:
            legal_mask[i] = False  # MIN_RAISE или ALL_IN → нелегален
    
    if not legal_mask.any():
        # fallback: если ни один anchor не легален, вернуть nearest
        return int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
    
    legal_indices = np.where(legal_mask)[0]
    legal_anchors = self.anchors_arr[legal_mask]
    nearest_legal_idx_in_subset = int(np.argmin(np.abs(legal_anchors - eff_mult)))
    return int(legal_indices[nearest_legal_idx_in_subset])
```

**Trade-off:** 
- Минус: дополнительный вызов `_resolve_effective_sizing` для каждого anchor (15 вызовов × 3 пути ≈ 45 вызовов per sizing sample) — но функция чистая, без side effects
- Плюс: устраняет data quality issue
- Можно оптимизировать: кешировать legal_mask по `state` в рамках одного sizing-сэмпла

---

## Вердикт

**Диагностика готова.** Нужен ли фикс nearest-legal-anchor — решать после анализа реальных цифр из боевого чекпоинта. Если `bucket_relation.below_effective > 5%` от общего числа — фикс рекомендуется.

---

## Изменённые файлы

| Файл | Что изменено |
|------|-------------|
| `src/core/deep_cfr.py` | +`sizing_selected_effective_diag`, +`sizing_selected_effective_kind_diag`, +`_record_selected_effective_sizing()`, wiring в 3 пути, checkpoint save |
| `tools/checkpoint_tools.py` | +`format_selected_effective_diag()`, +`format_selected_effective_kind_diag()`, загрузка из checkpoint, добавление в full/diagnose JSON |
| `tests/test_hybrid_outcome_sampling.py` | 5 новых тестов: init/reset, helper valid/invalid, illegal bucket visibility, checkpoint serialization |
