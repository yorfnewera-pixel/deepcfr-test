# Bug #90b — Discount Alignment (DCFR+ Reference Recipe)

**Дата:** 2026-06-14
**Статус:** CLOSED → #91 (test90seed1 выполнен. Результат: CLOSED PARTIAL. M0 FAIL — discount не первопричина. См. `sizing_reports/91-discount-alignment-result.md`)
**Severity:** High (прямой рычаг по H3 — накопление отрицательного raise-regret)
**База:** test88seed1 (#88, MC propagation baseline)
**Предшествует:** Probe 1 (#89) — коллапс реален (head-to-head 6-max: iter400 raise 1%, win 12.3% vs iter100)
**Конфиг:** `discount_alpha: 2.0 → 1.5`, знаменатель `+ 1 → + 1.5` (hardcode в `deep_cfr.py:2617`)

---

## Мотивация

Подтверждённый корень — **H3 (накопление отрицательного raise-regret)**. Точка смерти 200→300: `adv_raise: +0.43 → −0.003` (пересекает 0), `adv_fold` уходит в +. У проекта **форгеттинг слабее референса**:

| | Формула d(t) | α | Знаменатель | Forgetting при t=11 |
|---|---|---|---|---|
| **Проект** | `(t−1)^α / ((t−1)^α + 1)` | 2.0 | `+ 1` | d=0.990 (~1%/iter) |
| **Референс DCFR+** | `(T−1)^α / ((T−1)^α + 1.5)` | 1.5 | `+ 1.5` | d=0.955 (~4.5%/iter) |

Референс забывает старые regret'ы в ~4.5× быстрее. Сильнее форгеттинг → отрицательный regret за raise не успевает накопиться до flip → advantage не пересекает 0.

**Почему одна переменная:** α и знаменатель со-определяют DCFR+-рецепт. Это валидированная точка референса (`DeepPDCFR-master/deeppdcfr/os_deep_cumu_adv_variants.py:231-247`), не произвольная пара.

---

## Дизайн

| Файл | Строка | Было | Стало |
|------|--------|------|-------|
| `config.yaml` | 11 | `discount_alpha: 2.0` | `discount_alpha: 1.5` |
| `src/core/deep_cfr.py` | 2617 | `+ 1)` | `+ 1.5)` |

**Формула (deep_cfr.py:2617):**
```python
# было
discount = (t - 1) ** self.discount_alpha / ((t - 1) ** self.discount_alpha + 1)
# стало
discount = (t - 1) ** self.discount_alpha / ((t - 1) ** self.discount_alpha + 1.5)
```

**НЕ трогаем:**
- Q-сеть / sizing / max_norm — чистая advantage-сторона
- `discount_gamma` (recency-вес стратегии — #88c)
- `q_target_update_interval` (возвращён к baseline 100)

**Прогон:** test90seed1, сид как у #88, iter 400, чекпоинты 100/200/300/400.

---

## Критерии (baseline test88 в скобках)

| Gate | Условие | Baseline #88 |
|------|---------|-------------|
| **M0** (обрыв) | `raise_freq(200) ≥ 30%` | 16.0% |
| **M1** (механизм) | `adv_raise` не пересекает 0 к iter 300; `next_strategy_raise_mass` не падает <0.1 | +0.43→−0.003; 0.37→0.04 |
| **M2** (durability) | verdict=OK на ≥3/4 чекпоинтов; raise≥30%, unique≥12, top-2≤60% на всех | OK только iter 100 |

### Исходы

| Исход | Условие | Действие |
|--------|---------|----------|
| **PASS** | M0+M1+M2 | discount = первопричина durability, DCFR+-рецепт принять дефолтом → консолидация seed2 |
| **PARTIAL** | M0+M1, M2 проседает позже 200 | Форгеттинг помогает, но не до конца → Probe 3 (refit Q baseline) поверх discount |
| **FAIL** | M0 не выполнен / хуже baseline / больше осцилляции | Накопление не главный канал, ИЛИ сильный discount дестабилизирует → #88c recency-вес / advantage-регуляризация |

---

## Дерево решений

```
Probe 2 (#90b discount α=1.5,+1.5), iter 400
├── M0+M1+M2 PASS → discount = fix, новый baseline → консолидация seed2
├── M0+M1 PASS, M2 FAIL → Probe 3 (refit Q) поверх discount
└── M0 FAIL / хуже baseline → #88c recency-вес ИЛИ advantage-регуляризация
```

---

## Риски / guardrail

- **Сильный discount может увеличить осцилляцию** (меньше сглаживания). Guardrail: если raise_freq скачет резче, чем у #88, или M0 хуже baseline — FAIL-сигнал, не продолжать крутить α.
- **Discount затрагивает только advantage-bootstrap** — Q/sizing не меняются, чистая одна переменная по смыслу.
- **При FAIL не пытаться подбирать α вручную** — переходить к #88c (recency-вес).

---

## Что запустить

1. Правки из §1 (config + строка 2617)
2. Тренинг до iter 400 (тот же запуск, что давал test88seed1), сохранить чекпоинты 100/200/300/400 (full + light)
3. `tools/checkpoint_tools.py eval` (1000 игр) на каждом чекпоинте → `multi_checkpoint_iter_*_full_report.json`
4. `tools/relapse_diff.py 100/200/300/400` (seed=42, 500 состояний) → `relapse_diff_all.json`
5. Заполнить таблицу критериев и вынести PASS/PARTIAL/FAIL

---

## Изменения

| Файл | Что |
|------|-----|
| `config.yaml:11` | `discount_alpha: 2.0 → 1.5` |
| `src/core/deep_cfr.py:2617` | `+ 1` → `+ 1.5` |

## Референс-источники

- `DeepPDCFR-master/deeppdcfr/os_deep_cumu_adv_variants.py:231-247` — VRDCFRPlus discount-формула (α=1.5, знаменатель +1.5)
- `DeepPDCFR-master/configs/VRDeepDCFRPlus.yaml` — config α=2 (note: config override vs class default 1.5; reference code uses class default 1.5)
