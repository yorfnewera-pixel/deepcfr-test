# Bug #78: Second Control Result — UNEXPECTED, kind_filter=drop Hypothesis Refuted, Action-Q Regression

> **Статус:** test77seed1 (test76 minus kind_filter) выполнен. Result: UNEXPECTED regression — не C-like, не Grey.
> **Дата:** 2026-06-10

---

## 1. Контекст

Прогон test77seed1 — второй pre-registered контроль из #77 §9: test76seed1 с отключённым kind_filter. Гипотеза GPT+OPUS: kind_filter=drop — главный residual виновник зазора 27.7→37.6. Ожидание: `kind_filter_enabled: false` → C-like восстановление.

### 1.1 Конфиг (из test77seed1 `sizing_q_config`)

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_kind_filter_enabled: false        # ← ЕДИНСТВЕННАЯ ДЕЛЬТА от test76seed1
sizing_q_replay_clear_once: true
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

**Seed:** тот же что test76seed1. **Длина:** iter_100. **Папка:** `models/test77seed1/`.

---

## 2. Результат

### 2.1 Сводка

| Метрика | test76seed1 (kind ON) | test77seed1 (kind OFF) | Δ | Ожидалось |
|---|---|---|---|---|
| `verdict` | WARN (conc 83%) | WARN (conc 84%, **low_raise 13%**) | хуже | лучше |
| `raise_freq` | 27.7% | **13.0%** | **−14.7pp** | ↑ |
| `win_rate` | 37.7 | **28.1** | **−9.6pp** | ↑ |
| `mean_reward` | 9.41 | **2.06** | **−7.35** | ↑ |
| `unique_anchors` | 14/15 | **15/15** | +1 | ↔ |
| `best_sizing_anchor_dist` | 0.10=499/499 | 0.10=**499/499** | ≡ | должен был ↓ |
| `top-2 sizing mass` | 83% | **84%** | +1pp | ↓ |
| `action_q_buffer.raise` | 233 746 | **147 878** | **−37%** | ↑ |
| `next_strategy_raise_mass` (raise→raise) | 0.603 | **0.702** | +0.099 | ↔ |
| `opponent raise funnel` | 7 951 | 7 262 | −9% | ↔ |

**Verdict: НЕ C-like, НЕ Grey в прежнем понимании, а НОВЫЙ ПАТТЕРН regression.** kind_filter=drop гипотеза **опровергнута** — выключение kind_filter ухудшило метрики почти по всем осям.

### 2.2 Action-Q Regression — ключевой smoking gun

| Метрика | test76seed1 | test77seed1 | Изменение |
|---|---|---|---|
| `q_action.raise.mean` | +0.384 | **+0.205** | −47% |
| `q_action.fold.mean` | −0.294 | **+0.690** | знак сменился |
| `q_action.call.mean` | +0.215 | +0.167 | −22% |
| `raise_lt_fold_pct` | 0.0% | **100.0%** | fold > raise всегда |
| `raise_lt_call_pct` | 40.1% | 0.0% | raise > call всегда |

**Модель перестала считать raise валюабельным.** `q_action.fold.mean = +0.690` положительный — модель думает что fold лучше raise в 100% тестовых состояний. Это **тот же failure mode что в #71**: action-level fold-preference.

### 2.3 Sizing-Q targets — что изменилось в обучении

`mean_target_by_source_anchor` для source=0:

| Anchor | test76seed1 | test77seed1 | Δ |
|---|---|---|---|
| 0.10 | −0.93 | **−0.49** | +0.44 |
| 0.25 | −1.41 | **−0.87** | +0.54 |
| 2.00 | −2.50 | **−2.10** | +0.40 |
| 3.00 | — | −0.43 | новый |

Targets **в среднем менее отрицательные** в test77 — kind_filter OFF пропускает MIN_RAISE/ALL_IN targets в loss. Но **относительная картина не изменилась**: 0.10 остаётся наименее отрицательным anchor.

### 2.4 Inference concentration не сдвинулась

`best_sizing_anchor_dist`: 0.10=499/499 в обоих прогонах. `source2_cross_anchor.mean_std`: 0.008 в обоих. **Inference concentration — самостоятельный канал, не зависящий от kind_filter.**

### 2.5 kind_filter_diag — ожидаемые нули

```
kind_filter_diag: { normal_kept: 0, min_raise_dropped: 0, all_in_dropped: 0,
                    unknown_dropped: 0, downweighted: 0, zero_weight_batches: 0, cleared_once: 1 }
```

Конфирмация что `kind_filter_enabled: false` работает корректно — счётчики на нуле, никаких дропов.

### 2.6 Raise funnel — opponent path сохранился

| Path | test76seed1 | test77seed1 |
|---|---|---|
| hero_os | 0 | 0 |
| hero_full | 857 | 698 |
| opponent | 7 951 | 7 262 |

Mask OFF → opponent path жив. Небольшой спад (951→262) — следствие падения raise_freq.

---

## 3. Интерпретация: kind_filter=drop был полезным фильтром, а не виновником

### 3.1 Гипотеза опровергнута

Предсказание: «kind_filter=drop — strongest residual suspect. Выключить → C-like».
Факт: **все метрики ухудшились или не изменились.** Гипотеза опровергнута.

### 3.2 Почему kind_filter=drop был полезен

```python
# deep_cfr.py:3040-3052
non_normal = kind_ids_t != normal_id   # MIN_RAISE + ALL_IN + UNKNOWN
weights[non_normal] = 0.0              # не участвуют в sizing-Q loss
```

kind_filter=drop **фильтровал систематически отрицательные targets** от forced MIN_RAISE/ALL_IN sizing choices. Без фильтрации эти targets учатся → sizing-q в среднем ниже → action-Q видит raise как убыточный → self-suppressing loop.

### 3.3 Почему MIN_RAISE targets систематически отрицательные

Из `sizing_q_selected_effective_target_summary` test77seed1 (source=0, выборка):

| Путь | Count | target_mean |
|---|---|---|
| selected=0.10 → effective=0.10 (NORMAL) | 6673 | −0.524 |
| selected=0.10 → effective=0.25 (MIN_RAISE, force up) | 2431 | **−0.988** |
| selected=0.10 → effective=0.66 (MIN_RAISE, force up) | 1247 | **−1.513** |
| selected=0.10 → effective=0.50 (MIN_RAISE, force up) | 930 | **−2.011** |
| selected=0.25 → effective=0.50 (MIN_RAISE, force up) | 409 | **−2.041** |
| selected=0.10 → effective=0.10 (ALL_IN) | 1091 | −0.002 |

**Паттерн чёткий:** чем дальше effective-анкер от selected-анкера (MIN_RAISE force вверх), тем отрицательнее target. Когда модель хотела 0.10, но min_raise был 0.50 — она «наказывается» target'ом −2.01.

Это прямое следствие **train/inference index mismatch** (#74-v4 OPEN):

- **Inference:** argmax Q по `selected_idx` (модель хочет анкер 0.10).
- **Training:** target пишется в `effective_idx` (Q[effective=0.66] получает метку −1.51, или Q[effective=0.50] получает −2.01).

Q[selected=0.10] не получает кредит за то что модель выбрала 0.10 — target уходит в другой индекс. При этом effective-idx получает большую отрицательную метку, которая при kind_filter OFF распространяется через loss на весь sizing-Q → action-Q переучивается на fold-preference.

### 3.4 Вывод

**kind_filter=drop маскировал последствия train/inference index mismatch (#74-v4).** Он не был residual виновником — он был **safety filter**, удалявшим токсичный сигнал от forced sizings. Его отключение не починило деградацию, а раскрыло более глубокую проблему.

---

## 4. Что это значит для трёх residual каналов (#77 §7)

| # | Канал | До test77 | После test77 |
|---|---|---|---|
| 1 | Loss-side — kind_filter=drop | strongest suspect | **опровергнут** как viable fix через OFF |
| 2 | Loss-side — replay_clear interactor | вторичный | **открыт** (не тестировался отдельно) |
| 3 | Inference-side — sizing-Q argmax concentration | самостоятельный | **подтверждён** (не сдвинулся, независим от kind_filter) |

И **появился канал 4**: train/inference selected↔effective index mismatch (#74-v4). kind_filter случайно маскировал его последствия, отбрасывая отрицательные MIN_RAISE targets.

**Приоритет на следующем шаге изменился:** не replay_clear ablation, а **#74-v4 B1/B2 fix.**

---

## 5. Сравнительная таблица (полная цепочка)

| Метрика | Run C (test7) | Full-D (test8) | Isolation (test76) | Second control (test77) |
|---|---|---|---|---|
| `mask_enabled` | ❌ | ✅ | ❌ | ❌ |
| `kind_filter_enabled` | ❌ | ✅ | ✅ | **❌** |
| `replay_clear_once` | ❌ | ✅ | ✅ | ✅ |
| `raise_freq` | 37.6% | 2.7% | 27.7% | **13.0%** |
| `win_rate` | 42.1 | 38.5 | 37.7 | **28.1** |
| `mean_reward` | ~13 | 11.7 | 9.41 | **2.06** |
| `action_q.raise count` | 262 597 | 27 556 | 233 746 | **147 878** |
| `unique_anchors` | 15/15 | 10/15 | 14/15 | 15/15 |
| `best_sizing 0.10` | 424/499 | 499/499 | 499/499 | **499/499** |
| `top-2 sizing mass` | ~61% | ~83% | 83% | **84%** |
| `next_strategy_raise` | 0.406 | 0.0 | 0.603 | 0.702 |
| `opponent funnel` | 5169 | 0 | 7951 | 7262 |
| `q_action.raise.mean` | 0.406* | — | 0.384 | **0.205** |
| `q_action.fold.mean` | — | — | −0.294 | **+0.690** |
| `raise_lt_fold_pct` | — | — | 0% | **100%** |

---

## 6. Что НЕ делаем сейчас (консенсус отменён)

- ❌ `kind_filter_enabled: false` permanent — **опровергнуто**, ухудшает
- ❌ `kind_filter_mode: downweight` — не тестирован, но механизм тот же (non-NORMAL targets тянут sizing-Q вниз)
- ❌ `drop_unknown_only` (новый mode) — UNKNOWN только в source2 (weight=0), редундантен
- ❌ `replay_clear OFF` ablation — сейчас не приоритетно; после test77 ясно что bigger issue в index mismatch, а не в clear
- ❌ Soft mask — mask уже OFF

---

## 7. Следующий шаг: #74-v4 B1/B2 — fix train/inference index mismatch

**Приоритет изменился.** Test77 показал что root cause residual проблемы — **не kind_filter, не replay_clear, а фундаментальный train/inference index mismatch** (#74-v4):

1. Инференс: `argmax sizing_q` по `selected_idx` (15 anchors)
2. Training: target в `effective_idx` после `_resolve_effective_sizing` (MIN_RAISE/ALL_IN ремап)
3. MIN_RAISE/ALL_IN targets **систематически отрицательные** (−0.99...−2.04), потому что forced anchor получает плохой reward
4. **kind_filter=drop маскировал это**, удаляя эти targets из loss. Без него они тянут sizing-Q вниз → action-Q переучивается на fold-preference

### 7.5 Code-evidence: точная сигнатура _add_sizing_q_sample

Из `src/core/deep_cfr.py` (проверено GPT по исходному коду):

```python
_add_sizing_q_sample(..., eff_mult, ...,
                     anchor_idx=eff_idx,                    # target пишется по effective
                     selected_anchor_idx=sizing_anchor_idx, # selected сохраняется как metadata
                     selected_kind=selected_kind)
```

**Подтверждение mismatch:** сеть учится на `effective` sizing (target в `anchor_idx=eff_idx`), но policy/inference выбирает `selected` anchor. Инференс argmax по `selected_idx`, но Q[selected=0.10] не получает кредит за выбор — target уходит в Q[effective].

---

## 8. Консенсус GPT + OPUS после результата #78

### 8.1 Общее согласие

GPT и OPUS — **полный консенсус** по следующим пунктам:

1. **kind_filter=drop гипотеза опровергнута.** Выключение ухудшило все метрики (raise 27.7→13.0%, win 37.7→28.1, actionQ.raise 233.7k→147.9k). kind_filter не был residual виновником.

2. **kind_filter=drop оказался safety filter**, маскировавшим токсичные MIN_RAISE/ALL_IN targets от forced sizings. Без фильтрации targets от `selected=0→eff=1 (−0.99)`, `selected=0→eff=4 (−2.01)` учатся напрямую → sizing-Q падает → action-Q переучивается на fold-preference.

3. **#74-v4 selected↔effective index mismatch — root cause** токсичности. Инференс argmax по selected, training target по effective. Для MIN_RAISE forced-remap это создаёт систематически отрицательные targets.

4. **B1 — правильное направление fix'а**: писать Q-target под `selected_idx` (что модель реально выбрала). Меньше кода, чистый fix mismatch. B2 (сегментировать по kind) — избыточная сложность.

5. **B1 только как config-gated experimental flag** (`sizing_q_selected_credit_enabled`), не unconditional production.

### 8.2 4-уровневая иерархия каналов (точная формулировка GPT, принята OPUS)

| Уровень | Компонент | Роль |
|---|---|---|
| Фундамент | selected/effective mismatch (#74-v4) | **фундаментальная токсичность** forced-remap targets |
| Слой 2 | kind_filter=drop | **safety filter** — частично скрывает токсичность, удаляя MIN_RAISE/ALL_IN targets из loss |
| Слой 3 | replay_clear | **exposer / amplifier** — убирает historical diversity, позволяет токсичному refill доминировать на iter 100 |
| Отдельно | hard legal-anchor mask (#76) | catastrophic trigger из Full-D (доказан, не участвует в residual) |

### 8.3 Replay_clear как exposer mismatch — ключевое уточнение GPT

GPT показал сравнительные данные, которые я в первоначальном отчёте не подсветил:

| Прогон | Конфиг | raise_freq | Статус |
|---|---|---|---|
| Run A (dry-run) | mask OFF / kind OFF / **clear OFF** | **36.2%** | healthy |
| test77seed1 | mask OFF / kind OFF / **clear ON** | **13.0%** | catastrophe |

**Дельта 23pp по raise_freq от единственной разницы (clear ON vs OFF).** Это сильный сигнал: replay_clear не просто interactor, а **необходимое условие проявления** mismatch на iter_100.

Механизм: без clear буфер сохраняет historical diversity, которая буферит токсичный сигнал от forced-remap targets. C clear: буфер чистится → refill под altered policy → токсичный поток доминирует.

**Вывод: replay_clear не «деприоритизирован», не «виновен per se», но exposer/amplifier фундаментального mismatch.** Это меняет приоритизацию — перед B1 кодом нужно закрыть config-контроль для измерения роли clear.

### 8.4 Полная 2x2 матрица — недостающая клетка

| | clear OFF | clear ON |
|---|---|---|
| kind OFF | Run A dry-run: **36.2%** | test77seed1: **13.0%** |
| kind ON | **? (test78seed1)** ← нужно закрыть | test76seed1: **27.7%** |

4-я клетка (`kind ON / clear OFF`) — единственная неизмеренная комбинация. Ответит на вопрос: насколько test76 residual (27.7 vs 37.6) был от replay_clear, и насколько — от mismatch.

---

## 9. Следующий шаг: test78seed1 — config control (mask OFF / kind ON / clear OFF)

### 9.1 Конфиг (уже в config.yaml)

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: true        # safety filter активен
sizing_q_kind_filter_mode: drop
sizing_q_kind_filter_downweight: 0.25
sizing_q_replay_clear_once: false         # ← ЕДИНСТВЕННАЯ ДЕЛЬТА от test76seed1
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

**Папка:** `models/test78seed1/`. **Seed:** тот же что test76/test77. **Длина:** iter_100.

### 9.2 Pre-registered критерии (GPT-калиброванные, менее строгие по top2)

| Исход | Критерии | Интерпретация → Следующий шаг |
|---|---|---|
| **C-like** | raise ≥ 33%, win ~40+, actionQ.raise ≥ 230k, top2 ≤ 65-70% | replay_clear — **главная причина** test76 residual (27.7→37.6 зазор). B1 — отдельный mismatch fix, аккуратно, смотреть longer iter |
| **test76-like** | raise 24-30%, actionQ.raise ~200-240k, best_sizing 0.10=499/499 | clear **не важна** при kind ON. Mismatch/concentration активны даже под safety filter. B1 priority высокий |
| **bad/unexpected** | raise < 20% **или** fold > raise failure mode | clear OFF не спасает. Роль kind/mismatch сложнее, нужен пере-анализ |

### 9.3 Decision tree

- **C-like** → residual test76 в основном от replay_clear при kind ON; B1 всё равно нужен, но как отдельный mismatch fix
- **test76-like** → clear почти не влияет при kind ON; mismatch остаётся даже под safety filter
- **bad** → значит clear OFF не спасает, и роль kind/mismatch ещё сложнее

### 9.4 Что дальше после test78seed1

1. **Если C-like или test76-like:** B1 design как config-gated experimental flag `sizing_q_selected_credit_enabled`.
2. **B1 stress test:** конфиг `mask OFF / kind OFF / clear ON / B1 ON`. Pre-registered: raise_freq ≥ 27% (≥ test76 уровень) → mismatch fix успешен.
3. **Если bad:** пере-анализ роли kind/mismatch, возможно нужен другой experimental подход.

---

## 10. Файлы

| Файл | Изменения |
|------|-----------|
| `config.yaml` | Переключён на test78seed1 (kind ON, clear OFF) |
| `sizing_reports/78-...md` | Этот отчёт (+ консенсус GPT+OPUS §8, next step §9) |
| `sizing_reports/SIZING.md` | Обновлён (секция #78 с консенсусом + test78seed1 план) |
| `sizing_reports/77-...md` | Гипотеза kind_filter из §9 опровергнута результатом #78 |
| `models/test77seed1/` | Прогон second control — исходные данные этого отчёта |
| `models/test78seed1/` | **Следующий прогон** — config control (конфиг готов) |
