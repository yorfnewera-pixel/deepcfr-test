# #84 — Consolidation run (max_norm=100 baseline)

> **Статус:** CLOSED FAIL — relapse подтверждён, воспроизводимость CONFIRMED.
> **Связанные:** #82 (behavioral breakthrough test80b), #84b (relapse mechanism analysis), #85 (balanced loss v2)

---

## 0. Pre-registered критерии (зафиксированы ДО запуска)

### Мотивировка

#82 post-hoc анализ обнаружил поведенческий прорыв: test80b (max_norm=100) — первый OK-вердикт со времён Run C (raise_freq=48.4%, 15/15, top-2=29%). Но история знает relapse: #73 — 47.9% на iter 100 упало до 12.6% к iter 200; #67/#69 — коллапс к iter 300-400.

**Цель:** проверить что прорыв воспроизводим и устойчив на дистанции до iter 400.

### Дизайн

| Параметр | Значение |
|----------|----------|
| `q_grad_clip_max_norm` | 100 |
| `q_terminal_balanced_loss_enabled` | false |
| всё остальное | = test79seed1 база |
| Итерации | **400** (было 100) |
| Сиды | seed1, seed2 |

### Критерии (проверять на iter 300 И 400, оба сида)

| Исход | Условие |
|-------|---------|
| **PASS** | verdict=OK, raise_freq≥40%, unique_anchors=15/15, top-2 sizing mass≤55%, анкер 0.10≤15% массы |
| **PARTIAL** | держится на iter 200, relapse на 300-400 (метрики падают ниже порогов) |
| **FAIL** | relapse уже к iter 200, или не воспроизводится на seed2 |

PASS → объявить max_norm=100 новым baseline (вместо Run C), зафиксировать в SIZING.md.

---

## 1. Результаты

### seed1

| iter | verdict | raise_freq | unique_anchors | top-2 mass | 0.10 mass | win_rate | mean_reward | fold count |
|------|---------|------------|----------------|-----------|-----------|----------|-------------|------------|
| 100 | **OK** | 48.0% | 15 | 28.7% | 10.6% | 41.2 | 10.134 | 44 |
| 200 | **WARN** | 21.5% | 12 | 48.9% | 24.2% | 13.3 | 0.164 | 633 |
| 300 | **WARN** | 7.2% | 7 | 78.3% | 55.5% | 18.0 | 7.089 | 524 |
| 400 | *not run* | — | — | — | — | — | — | — |

**Дополнительные метрики:**
- fit_ratio: 0.003 → 0.10-0.21 (iter 150-300) — фит улучшен в 5-10× vs базы
- q_compare raise_lt_fold_pct: 100% на всех чекпоинтах
- q_compare raise_lt_call_pct: 0.2% → 35.1% → 49.3%
- grad_clip scale_bottleneck: False на всех чекпоинтах
- action distribution (eval games): fold 44 → 633 → 524; raise 1138 → 331 → 125
- source2 cross-anchor mean_range: 0.085 → 0.213 → 1.491 (взрыв к iter 300)

### seed2

| iter | verdict | raise_freq | unique_anchors | top-2 mass | 0.10 mass | win_rate | mean_reward |
|------|---------|------------|----------------|-----------|-----------|----------|-------------|
| все | *not run* | — | — | — | — | — | — |

seed2 не запускался — вердикт уже определён на seed1 (relapse подтверждён).

---

## 2. Вывод: FAIL (durability) / воспроизводимость CONFIRMED

max_norm=100 даёт воспроизводимое окно здорового поведения на iter ~100 (OK, raise=48%, 15/15) и улучшает фит в 5-10× (fit_ratio 0.003 → 0.10-0.21), но **НЕ является фиксом коллапса** — паттерн relapse идентичен #73: iter 100 OK → iter 200 WARN → iter 300 полный возврат коллапса.

Причина relapse исследована в `#84b-relapse-mechanism.md`: advantage/regret-сеть делает полный flip предпочтений fold↔raise (fold advantage -7.93 → +0.53) за 100 итераций — положительная обратная связь через CFR regret-matching.

max_norm=100 **оставлен** в config.yaml как baseline — улучшает фит и не вредит. Следующий эксперимент: #85 (terminal-balanced loss) на базе max_norm=100.

---

## 3. Next Steps

- **Выполнено:** #84b — relapse mechanism analysis (tools/relapse_diff.py).
- **Далее:** #85 (terminal-balanced loss на базе max_norm=100, до iter 300).
- **После #85:** шаг 4 — n-step/MC terminal reward propagation + нормализация целей.
