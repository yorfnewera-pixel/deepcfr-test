# Bug #77: Isolation Result — GREY/PARTIAL, Mask Catastrophic Trigger Confirmed, Residual Open

> **Статус:** Isolation (Full-D minus mask) выполнен. Result: GREY/PARTIAL.
> **Дата:** 2026-06-10

---

## 1. Контекст и конфиг

Прогон test76seed1 — исполнение isolation-плана из #76 §6: Full-D с отключённой legal-anchor mask, но включёнными kind-filter (drop) и replay_clear_once. Цель: доказать каузально — mask единственный виновник D-коллапса, отделив его от kind-filter/replay_clear.

### 1.1 Конфиг (из test76seed1 `sizing_q_config`)

```yaml
sizing_q_legal_anchor_mask_enabled: false   # ← отличие от Full-D
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: true          # как в Full-D
sizing_q_kind_filter_mode: drop
sizing_q_replay_clear_once: true            # как в Full-D
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

### 1.2 Pre-registered критерии (из #76 §6.2)

| PASS | FAIL | Grey |
|------|------|------|
| `raise_freq >= 33%` | `< 5%` | `15-30%` |
| `next_strategy_raise_mass >= 0.38` | `≈ 0` | промежуточно |
| `action_q_buffer.raise >= 230k` | `≈ 27k` | |
| `unique_anchors = 15` | `≤ 10` | |
| `opponent funnel ≠ 0` | `= 0` | |

---

## 2. Результат: Сверка с pre-registered критериями

| Критерий | Threshold | Факт (test76seed1) | Статус |
|---|---|---|---|
| `raise_freq` | ≥33% | **27.7%** | **FAIL** (Grey 15–30 zone) |
| `next_strategy_raise_mass` (raise→raise) | ≥0.38 | **0.603** | **PASS** |
| `action_q_buffer.raise` | ≥230k | **233 746** | **PASS** |
| `unique_anchors` | =15 | **14** | **NEAR** |
| `opponent funnel` | ≠0 | **7 951** | **PASS** |

**Verdict: 3 PASS, 1 NEAR, 1 FAIL → GREY.**

---

## 3. Сравнение C / Full-D / Isolation

| Метрика | Run C (test7) | Full-D (test8) | Isolation (test76seed1) |
|---|---|---|---|
| `verdict` | OK | WARN | WARN (conc 83%) |
| `raise_freq` | 37.6% | 2.7% | 27.7% |
| `win_rate` | 42.1 | 38.5 | 37.7 |
| `mean_reward` | ~13-15 | 11.7 | 9.41 |
| `unique_anchors` | 15/15 | 10/15 | 14/15 |
| `best_sizing_anchor_dist` | 0.10=424/499 | 0.10=499/499 | 0.10=499/499 |
| `top-2 sizing mass` | ~61% | ~83% | 83% |
| `action_q_buffer.raise` | 262 597 | 27 556 | 233 746 |
| `next_strategy_raise_mass` | 0.406 | 0.0 | 0.603 |
| `opponent raise funnel` | 5 169 | 0 | 7 951 |

---

## 4. Что подтверждено (Mask = Catastrophic-Collapse Trigger)

Mask OFF восстановил катастрофическое плечо D-коллапса:

| Метрика | Full-D → Isolation | Интерпретация |
|---|---|---|
| `raise_freq` | 2.7 → **27.7%** | катастрофический collapse устранён |
| `action_q_buffer.raise` | 27 556 → **233 746** | raise training квота восстановлена |
| `next_strategy_raise_mass` | 0.0 → **0.603** | deadly triad разорван |
| `opponent raise funnel` | 0 → **7 951** | оппонентский raise path жив |

**Hard legal-anchor mask — catastrophic-collapse trigger. Это доказано.**

---

## 5. Что НЕ подтверждено (Mask ≠ Sole Cause Полной C-Регрессии)

Остаточный зазор до C/A не закрылся:

| Метрика | Isolation | Run C | Зазор |
|---|---|---|---|
| `raise_freq` | 27.7% | 37.6% | **−9.9pp** |
| `win_rate` | 37.7 | 42.1 | **−4.4pp** |
| `mean_reward` | 9.41 | ~13 | **−3.6** |
| `best_sizing_anchor_dist` | 0.10=499/499 | 0.10=424/499 | **75 more states** |
| `top-2 sizing mass` | 83% | ~61% | **+22pp** |
| `unique_anchors` | 14/15 | 15/15 | **−1** |

**Mask убрал катастрофическое плечо, но не восстановил C/A полностью. Sole-cause гипотеза опровергнута.**

---

## 6. Ревизия вывода #75 v5 о kind-filter

### 6.1 Исходный вывод #75 v5 §1

> «UNKNOWN = source2 lookahead (не передаёт selected_kind) → kind-filter drop редундантен (src2 уже weight=0).»

### 6.2 Code-evidence: kind-filter=drop режет ВСЕ non-NORMAL

Из `src/core/deep_cfr.py:3033–3057`:

```python
if self.sizing_q_kind_filter_enabled:
    non_normal = kind_ids_t != normal_id   # MIN_RAISE + ALL_IN + UNKNOWN — ВСЁ
    if mode == 'drop':
        # счётчики:
        self.sizing_q_kind_filter_diag['normal_kept'] += ...
        self.sizing_q_kind_filter_diag['all_in_dropped'] += ...
        self.sizing_q_kind_filter_diag['min_raise_dropped'] += ...
        self.sizing_q_kind_filter_diag['unknown_dropped'] += ...
        weights[non_normal] = 0.0   # ← РЕАЛЬНЫЙ ДРОП В LOSS
```

**kind-filter в режиме `drop` зануляет вес для всех non-NORMAL samples** — включая known MIN_RAISE/ALL_IN из source=0/1, не только UNKNOWN из source=2.

### 6.3 Эмпирическое подтверждение из test76seed1

`kind_filter_diag`:
```
normal_kept:      772
min_raise_dropped: 267
all_in_dropped:    293
unknown_dropped:  1228
```

`insert_kind_diag.by_source`:
```
source=0: {ALL_IN: 134, MIN_RAISE: 240, NORMAL: 237}
source=1: {ALL_IN: 97,  MIN_RAISE: 15,  NORMAL: 134}
source=2: {UNKNOWN: 425}
```

За training шаги дропнулось **560 known non-NORMAL** (267 MIN_RAISE + 293 ALL_IN) из source=0/1 против 772 normal_kept → **≈42% live known mass отбрасывается из loss.**

### 6.4 Уточнённый вывод

Вывод #75 v5 **узко верен** для UNKNOWN-канала: source2 и так weight=0 через `sizing_q_src2_weight_*`, так что kind-filter-drop UNKNOWN из source2 действительно редундантен.

**Но** вывод некорректен для общего эффекта: kind-filter режет **и MIN_RAISE/ALL_IN из source=0/1**, где они являются живыми, обучаемыми сэмплами. В Full-D этот эффект прятался upstream под маской (mask обнуляла эти sizing-выборы на selection-time, не допуская до buffer/loss). В isolation mask OFF открывает поток MIN_RAISE/ALL_IN → они попадают в buffer → kind-filter дропает их в loss.

### 6.5 Заключение

**kind-filter=drop — активный loss-side источник residual regression**, не сводящийся к UNKNOWN-редундантности. Вывод #75 v5 требует уточнения.

---

## 7. Residual-каналы (что осталось объяснить)

| # | Канал | Механизм | Evidence в test76seed1 |
|---|---|---|---|
| 1 | **Loss-side — kind-filter=drop** | дропает 42% known non-NORMAL (MIN_RAISE+ALL_IN) из source=0/1 в loss | `kind_filter_diag`: 267+293 dropped vs 772 kept |
| 2 | **Loss-side — replay_clear interactor** | clear стирает 3453 разнообразных сэмпла (all 15 anchors, all 3 sources), refill идёт под уже искажённой kind-filter dynamic | `replay_clear_diag`: before_clear=all15, cleared=true |
| 3 | **Inference-side — sizing-Q argmax concentration** | `argmax sizing_q` даёт 0.10 на всех 499 eval-states при healthy buffer (14/15 anchors) | `best_sizing_anchor_dist.0.10=499/499`, `source2_cross_anchor.mean_std=0.008` (source2 не различает anchors) |

**Канал 3 не объясняется ни kind-filter, ни replay_clear** — это самостоятельный inference-side баг. Source2 равномерно ≈0.88-0.93 по всем анкерам (mean_std=0.008) → sizing_q_net не получает от source2 дифференцирующего сигнала по размерам → argmax коллапсирует в 0.10 за счёт source0/source1 systematic bias.

---

## 8. Заключение

По pre-registered правилу #76 §6.2: **Grey → не PASS.** Mask — catastrophic-collapse trigger, но не sole cause полной C-регрессии.

Residual может объясняться kind-filter=drop (loss-side, канал 1), replay_clear interactor (канал 2), inference sizing-Q концентрацией (канал 3), или их комбинацией. Из одного isolation-прогона эти гипотезы неразличимы.

---

## 9. Файлы

| Файл | Изменения |
|------|-----------|
| `sizing_reports/77-...md` | Этот отчёт |
| `sizing_reports/SIZING.md` | Обновлён (isolation result) |
| `models/test76seed1/` | Прогон isolation (Full-D minus mask) — исходные данные этого отчёта |
| `models/test77seed1/` | **Следующий прогон — second control: test76 minus kind_filter** (конфиг в config.yaml) |

---

## 9. Комментарий: консенсус GPT + OPUS и ожидания от test77seed1

### 9.1 Общее мнение о результате #77

**GPT и OPUS — полный консенсус.** Mask = catastrophic-collapse trigger доказан однозначно (4 ключевых метрики восстановлены: opponent 0→7951, actionQ.raise 27.5k→233.7k, next_strategy_raise 0→0.603, raise_freq 2.7→27.7). Но mask ≠ sole cause; residual зазор 27.7→37.6 указывает на второй активный канал.

kind-filter=drop — **strongest loss-side suspect** с подтверждённым code-mechanism (`deep_cfr.py:3040–3052`: weights[non_normal]=0.0 для ВСЕХ non-NORMAL, не только UNKNOWN) и численным evidence (42% live known mass dropped из source=0/1). Вывод #75 v5 «kind-filter не culprit» был узко верен только для UNKNOWN-канала — kind-filter активен и для known MIN_RAISE/ALL_IN.

Inference concentration (`best_sizing 0.10=499/499`) — третий residual канал. В текущем конфиге `src2_weight=0`, поэтому source2 не обучает sizing_q_net напрямую; источник концентрации скорее в systematic bias source0/source1 targets (`mean_target_by_anchor.0.10=-0.81` — наименее отрицательная метка) + downstream amplification через kind_filter/replay_clear. Канал 3 может быть побочным эффектом канала 1 (kind_filter=drop удаляет обучающий сигнал для не-0.10 анкеров → sizing_q_net не учится различать размеры → argmax коллапсирует в 0.10).

### 9.2 Следующий прогон: test77seed1 (second control)

**Решение:** один config-only прогон, без изменений кода. Единственная дельта от test76seed1:

```yaml
sizing_q_kind_filter_enabled: true → false
```

Всё остальное идентично: mask OFF, replay_clear ON, diagnostics ON, dry_run mask diag ON, тот же seed, iter_100.

**Конфиг уже в config.yaml** (секция «Bug #77 second control»).

### 9.3 Decision rules и ожидания

| Исход | Критерии | Интерпретация | Следующий шаг |
|---|---|---|---|
| **C-like** | raise_freq ≥ 33%, actionQ.raise ≥ 250k, unique=15, top2 ≤ 65%, best_sizing 0.10 < 499/499 | kind_filter=drop = root cause residual regression | Шаг 2a: **longer-run stability check** (iter200, iter300) с тем же конфигом — перед любым production fix'ом, т.к. в истории проекта iter100 красивые показатели релапсировали к iter200–400 |
| **GREY** | raise 15–30%, top2 ≥ 80%, best_sizing 0.10=499/499 | kind_filter не виновен; residual = replay_clear interactor | Шаг 2b: **config-ablation replay_clear OFF** → mask OFF + kind_filter OFF + replay_clear OFF |
| **FAIL** | ~Full-D collapse | конфиг/loader bug (не ожидается при mask OFF) | Перепроверить конфиг |

**Общее ожидание (GPT + OPUS):** оба исхода информативны. C-like — kind_filter root cause доказан, fix через config (a) `kind_filter_enabled: false` permanent или (b) `mode: downweight`. GREY — kind_filter elimination closes loss-side, передаёт подозрение replay_clear.

### 9.4 Что НЕ реализуем сейчас (консенсус)

- ❌ `drop_unknown_only` (новый mode c.1) — преждевременно; нужен только если kind_filter подтверждён как частичный виновник, но C-like не достижим через (a)/(b)
- ❌ Soft mask (#77 original plan) — mask уже OFF, лечить нечего
- ❌ Inference/B1-B2 fix (#74-v4) — отдельный канал, не смешивать с текущим ablation; открывается только если оба loss-side канала исключены
- ❌ Warmup/probe/target-net interval changes — safety-gates, не residual виновники
