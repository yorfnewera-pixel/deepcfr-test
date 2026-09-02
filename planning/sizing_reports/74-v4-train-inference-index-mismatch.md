# Bug #74 v4: Train/Inference Sizing Index Mismatch + Metadata Plumbing

> **Статус:** Diagnostics deployed (read-only), fix NOT applied.
> **План:** Шаг 1 — измерить; Шаг 2 — выбрать B1/B2.
> **Дата:** 2026-06-08
> **Затронутые коммиты:** changeset `selected_effective_metadata_plumbing`

---

## 1. Корень проблемы: double indexing

| Слой | Пространство анкоров | Что происходит |
|------|---------------------|----------------|
| **Inference** (argmax Q) | `selected` (15 anchors) | Модель выбирает любой анкор, в том числе нелегальный |
| **Training** (Q-target) | `effective` (после _resolve_effective_sizing) | Target пишется в `effective_idx`, а не в `selected_idx` |
| **Remap** (среда) | MIN_RAISE cap, ALL_IN floor | Крупный выбор (3.0) режется до all-in (~0.10) при коротком стеке |

**Train/inference index mismatch**: Q[selected=3.0] никогда не получает кредит за выбор "3.0" — target ушёл в Q[effective=0]. Q[effective=0] загрязнён исходами швов при коротком стеке и нерелевантен при инференс-argmax.

Из `multi_checkpoint_iter_100_full_report.json`:
- **equal effective** (точное попадание): hero_os 28%, hero_full 40%, opponent 34%
- **MIN_RAISE selected**: os 17%, full 10%, opp 23%
- **ALL_IN selected**: os 49%, full 41%, opp 36%
- **88% всех raise-записей** = opponent_path → напрямую связано с #75 (player-aware)

---

## 2. Что сделано (Tasks 1-4)

### 2.1 SizingQBuffer — новые поля

Файл: `src/core/deep_cfr.py:481-573`

```python
class SizingQBuffer:
    SELECTED_KIND_TO_ID = {'UNKNOWN': 0, 'NORMAL': 1, 'MIN_RAISE': 2, 'ALL_IN': 3}
    SELECTED_ID_TO_KIND = {v: k for k, v in SELECTED_KIND_TO_ID.items()}

    # Новые массивы (не влияют на sampling):
    self._selected_anchor_indices = np.full(capacity, -1, dtype=np.int32)
    self._effective_anchor_indices = np.full(capacity, -1, dtype=np.int32)
    self._selected_kind_ids = np.full(capacity, SELECTED_KIND_TO_ID['UNKNOWN'], dtype=np.int32)

    # add() расширен keyword-only параметрами:
    #   selected_anchor_idx=-1, effective_anchor_idx=None, selected_kind='UNKNOWN'
    # effective_anchor_idx fallback → anchor_idx если None
```

### 2.2 selected_effective_target_summary()

Метод `selected_effective_target_summary(num_anchors)` возвращает dict:

```json
{
  "source=0|selected=14|effective=0|kind=ALL_IN": {
    "count": 42,
    "target_mean": 0.123,
    "target_min": -1.5,
    "target_max": 3.2
  }
}
```

Ключ кодирует: `source`, `selected_anchor_idx`, `effective_anchor_idx`, `selected_kind`.

### 2.3 _add_sizing_q_sample — прокидывание metadata

Расширена сигнатура keyword-only параметрами. Source-0 call sites:

| Путь | Файл:строка | Передаёт |
|------|------------|----------|
| hero_os_raise | `deep_cfr.py:1893` | `sizing_anchor_idx`, `selected_kind` |
| hero_full_raise | `deep_cfr.py:2060` | `sizing_anchor_idx`, `selected_kind` |
| train.py raise | `train.py:318` | `sizing_anchor_idx`, `selected_kind` |

Probe/lookahead (source=1/2) остаются с `selected_kind='UNKNOWN'`.

### 2.4 Checkpoint + Report

- **Save** (`deep_cfr.py:2907`): `checkpoint['sizing_q_selected_effective_target_summary']`
- **Load** (`checkpoint_tools.py:268`): passthrough из ckpt
- **Report JSON** (`checkpoint_tools.py`): и в `diagnose`, и в `full` режиме

---

## 3. Что НЕ сделано (ждёт данных)

- `SizingQBuffer.sample() / sample_stratified()` return shape **не менялся** (обратная совместимость)
- Training loss **не менялся**
- Inference legal-anchor mask **не добавлена**
- eps/exploration **не трогался**
- Вердикт-гейт **не менялся** (всё ещё может false-pass на сколлапсированной метрике)

---

## 4. Тесты

- `tests/test_hybrid_outcome_sampling.py` — 62/62 pass
- `tests/test_sizing_q_regret.py` — 8/8 pass
- Full suite: 96/99 pass (3 pre-existing failures: `advantage_memory`, `PrioritizedMemory`, mixed training smoke)

---

## 5. Следующий шаг: Decision Gate

### Шаг 0 (готово)
Планируемый репорт: `sizing_q_selected_effective_target_summary` в JSON.

### Шаг 1 — измерить (нужен checkpoint)

После тренировки с новым кодом выполнить:

```powershell
py -3 tools/checkpoint_tools.py full models/test2seed1/<checkpoint>.pt --games 1000 --num-states 500 --seed 1
```

Проверить в JSON:
- `count` по `kind=NORMAL` vs `kind=ALL_IN`
- `target_mean` по `selected == effective` vs `selected > effective`
- `source=0` отдельно от `source=1/2`

### Правило выбора B1 vs B2

| Условие | Решение |
|---------|---------|
| `ALL_IN` samples имеют отличающийся `target_mean` от `NORMAL` при том же `effective_idx` | **B1 + kind filtering**, не чистый B1 |
| `selected -> effective` remap имеет устойчивый target, который должен кредитоваться selected действию | **B1** (учить под selected_anchor_idx) |
| `MIN_RAISE/ALL_IN` занимают большую долю и target-профиль отличается | **B2** (сегментация/исключение forced-remap из normal sizing-Q) |

### B1 (preferred)
Писать Q под `selected_anchor_idx` (target = реальный исход ремапа). Чинит train/inference index mismatch.

### B2
Оставить `effective_idx`, но сегментировать по kind: исключить/учить отдельно MIN_RAISE/ALL_IN.

### Важно
B1 и B2 НЕ взаимоисключающие. Возможен B1 + маскирование форсов из кредита.

---

## 6. Ортогональные баги (не трогать в этом PR)

- **#75** (player-aware backup): 88% загрязнения в opponent-ветке
- **#58** (_resolve_effective_sizing): только пост-фактум атрибуция, не маскирует выбор
- **Verdict false-pass**: `sizes=15/15 OK` считается по поллюцированной метрике

---

## 7. Что не делать без отдельного плана

- Не добавлять legal-anchor маску в этом же PR
- Не менять `anchor_idx` meaning без миграции тестов
- Не менять `best_sizing_anchor_dist` verdict logic
- Не трогать #75
- Не менять eps/exploration до восстановления разнообразия
