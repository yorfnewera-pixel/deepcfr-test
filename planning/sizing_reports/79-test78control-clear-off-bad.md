# Bug #79: Config Control Result — BAD, replay_clear-as-exposer гипотеза не подтверждена, config-fix исчерпан

> **Статус:** test78seed1 (mask OFF / kind ON / clear OFF) выполнен. Result: BAD — 4-я клетка 2x2 матрицы закрыта, config-only fix исчерпан.
> **Дата:** 2026-06-10

---

## 1. Контекст

Прогон test78seed1 — последняя клетка 2x2 матрицы из #78 §9: закрыть конфиг-контроль `mask OFF / kind ON / clear OFF`. Гипотеза (GPT+OPUS): replay_clear — exposer/amplifier mismatch; выключение clear должно улучшить метрики до C-like или test76-like зоны.

### 1.1 Конфиг

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_kind_filter_enabled: true         # safety filter активен
sizing_q_kind_filter_mode: drop
sizing_q_replay_clear_once: false          # ← ЕДИНСТВЕННАЯ ДЕЛЬТА от test76seed1
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

**Seed:** тот же. **Длина:** iter_100. **Папка:** `models/test78seed1/`.

### 1.2 Pre-registered критерии (из #78 §9.2)

| Исход | Критерии |
|---|---|
| C-like | raise ≥ 33%, win ~40+, actionQ.raise ≥ 230k |
| test76-like | raise 24-30%, actionQ.raise ~200-240k |
| bad | raise < 20% **или** fold > raise failure mode |

---

## 2. Результат

### 2.1 Сводка — BAD, close-to-Full-D collapse

| Метрика | test76seed1 (kind ON/clear ON) | test78seed1 (kind ON/clear OFF) | Δ |
|---|---|---|---|
| `verdict` | WARN (conc 83%) | WARN (**raise 4.6%**, conc 89%) | хуже |
| `raise_freq` | 27.7% | **4.6%** | **−23.1pp** |
| `win_rate` | 37.7 | 38.1 | +0.4 |
| `mean_reward` | 9.41 | 10.28 | +0.87 |
| `unique_anchors` | 14 | **9** | **−5** |
| `best_sizing_anchor_dist` | 0.10=499/499 | 0.10=**499/499** | ≡ |
| `top-2 sizing mass` | 83% | **89%** | +6pp |
| `action_q_buffer.raise` | 233 746 | **27 474** | **−88%** |
| `next_strategy_raise_mass` (raise→raise) | 0.603 | **0.000** | collapse |
| `opponent raise funnel` | 7 951 | **0** | vanished |

**Verdict: BAD.** `raise_freq=4.6%` — практически Full-D зона (2.7%). `actionQ.raise=27 474` — Full-D уровень (27 556). `opponent funnel=0`, `next_strategy_raise=0.0`. Это не просто не C-like/test76-like — это **регрессия даже относительно test76**.

### 2.2 Action-Q — fold-preference доминирует

| Метрика | test76seed1 | test78seed1 | Δ |
|---|---|---|---|
| `q_action.fold.mean` | −0.29 | **+1.75** | flip |
| `q_action.raise.mean` | +0.38 | **−0.20** | flip |
| `q_action.call.mean` | +0.22 | **−0.17** | flip |
| `raise_lt_fold_pct` | 0% | **100%** | fold > raise всегда |
| `raise_lt_call_pct` | 40% | **79%** | raise хуже call |
| `q_raise_gt_max_sq_pct` | 100% | **35%** | raise Q редко > best sizing Q |

**Модель категорически не хочет рейзить.** `q_action.fold.mean=+1.75` — самый сильный fold-preference из всех прогонов (Full-D: не было замерено, test77: +0.69, test76: −0.29).

### 2.3 Raise funnel — opponent path исчез, hero_full упал

| Path | test76seed1 | test78seed1 | Δ |
|---|---|---|---|
| hero_os | 0 | 691 | появился |
| hero_full | 857 | **400** | −53% |
| opponent | 7 951 | **0** | vanished |

Интересно: hero_os **появился** (0→691), но opponent исчез (7951→0), hero_full сильно упал (857→400). OS-node стал выбирать raise чаще (2398 legal→691 sampled, 29%), но opponent-raise из strategy_sizing_net полностью исчез — модель оппонента не считает raise валюабельным.

### 2.4 kind_filter_diag — работает штатно

```
normal_kept:      1206
min_raise_dropped: 334
all_in_dropped:     43
unknown_dropped:   977
cleared_once:        0
```

`cleared_once=0` — подтверждение что `replay_clear_once: false` работает корректно.

kind_filter режет 334+43=377 known non-NORMAL из source=0/1 против 1206 NORMAL kept → **24% live mass dropped** (vs 42% в test76). Меньше live mass дропается — но это не спасло от коллапса.

### 2.5 Replay_clear — не чистился

```
replay_clear_diag: { before_clear: null, cleared: false, refill_snapshots: [] }
```

Конфирмация: replay_clear_once не сработал. Буфер/история сохранена.

---

## 3. Интерпретация: replay_clear-as-exposer гипотеза не подтвердилась

### 3.1 Ожидание было

Если replay_clear — exposer mismatch: выключить clear → historical diversity в буфере сохранится → raise_freq улучшится до test76-like или C-like.

### 3.2 Факт

Выключение clear **ухудшило** метрики в 6 раз: 27.7% → 4.6%. Это bad ветка — худший исход из pre-registered.

### 3.3 Механизм — гипотеза, не доказательство (уточнено GPT+OPUS)

**Факт: BAD стальной** (raise 4.6%, actionQ 27.5k, opponent 0, fold.mean +1.75). 2x2 матрица закрыта.

**Конкретный механизм почему kind ON/clear OFF — худшая клетка — открытая гипотеза.** Кодовая проверка (`deep_cfr.py:3040–3064`) показывает: в drop-mode non-NORMAL получают `weight=0` и loss нормируется на `weight_sum`, поэтому они **не вносят прямого градиента** в sizing_q_net. Однако могут вредить косвенно:

1. **Anchor coverage через stratified sampling:** non-NORMAL сэмплы занимают слоты в `count_by_anchor`, выталкивая NORMAL coverage. Стратифицированный sampling распределяет batch по anchor counts → искажение распределения.
2. **Effective-index target misassignment (#74-v4 mismatch):** `selected=0.10→effective=0.25 (MIN_RAISE)` создаёт запись с `anchor_idx=eff_idx=0.25`, которая обучает Q[0.25] под контекст «модель хотела 0.10». Это toxic-from-target-misassignment — kind_filter этого не трогает (anchor_idx уже effective, kind_id не смотрит на это). Самый прямой кандидат на root cause.
3. **Stale state distribution / replay composition:** старые samples от ранних policy iterations не совпадают с current policy distribution. Без clear эта история доминирует replay количественно.

Из трёх кандидатов effective-index target misassignment — самый прямой: он напрямую связан с #74-v4 mismatch и не фильтруется kind_filter ни в каком режиме. Без clear записи с misassigned effective-index накапливаются. Это согласуется с тем, что B1 (target под selected_idx) должен устранять источник, не симптом.

### 3.4 Что это доказывает

**Replay_clear в связке с kind_filter не просто "exposer" — он выполняет полезную функцию очистки** от forced-remap samples при старте. Без него буфер сохраняет накопленную токсичную историю, и модель не может выбраться из ямы.

**Root cause — не clear, не kind_filter, а forced-remap targets.** Clear и kind_filter — два разных паллиативных механизма, которые взаимодействуют сложнее, чем мы предполагали.

---

## 4. Полная 2x2 матрица (завершена)

| | clear OFF | clear ON |
|---|---:|---:|
| kind OFF | Run A dry-run: **36.2%** | test77: **13.0%** |
| kind ON | test78: **4.6%** | test76: **27.7%** |

**Interpretation:**

- **Лучшая клетка** (36.2%): kind OFF / clear OFF — никакой фильтрации, исторический буфер без forced-remap dominance на iter 100. Базовый C-подобный результат, накопленный до того как mismatch стал доминировать.
- **2-я** (27.7%): kind ON / clear ON — kind_filter фильтрует forced-remap из loss, clear очищает старый буфер → mixed grey.
- **3-я** (13.0%): kind OFF / clear ON — kind_filter выключен, forced-remap targets учатся; clear очищает diversity, refill под altered policy → action-Q fold-preference.
- **Худшая** (4.6%): kind ON / clear OFF — kind_filter фильтрует только из loss, но не из буфера; clear не чистит старую токсичную историю → доминирование forced-remap history двойного эффекта.

**Conclusion:** ни одна single-variable гипотеза не работает. Система сложная interaction-system между kind_filter, replay_clear, forced-remap targets, и buffer distribution dynamics.

---

## 5. Что это меняет

### 5.1 Config-only fix исчерпан

2x2 матрица закрыта. Ни один конфиг не дал стабильно C-like при mask OFF. Config-only путь больше ничего не даст — все 4 клетки измерены.

### 5.2 Root cause — #74-v4 selected/effective mismatch

Все плохие конфиги объединяет одно: токсичные forced-remap targets (MIN_RAISE selected→effective mismatch) **либо учатся в loss, либо лежат в буфере, либо оба варианта**. Ни kind_filter (filter from loss only), ни replay_clear (clear from buffer only) не закрывают оба пути одновременно.

Единственный fix который может работать **фундаментально**: **B1 — писать sizing-Q target под selected_idx**, а не под effective_idx. Тогда:
- forced-remap targets не будут токсичными ни в буфере, ни в loss
- kind_filter становится ненужным для non-NORMAL (потому что больше нет токсичного mismatch)
- replay_clear может остаться или уйти — неважно, потому что mismatch устранён

### 5.3 B1 — теперь единственный viable next step

**Config-gated experimental flag:**
```yaml
sizing_q_selected_credit_enabled: true  # B1: target пишется под selected_idx, не effective_idx
```

**Stress-test config:**
```yaml
mask OFF
kind_filter OFF          # stress: никакого фильтра
replay_clear ON           # stress: clear раскрывает mismatch максимально
sizing_q_selected_credit_enabled: true
```

Pre-registered PASS:
- `raise_freq ≥ 27%` (≥ test76 уровень) — mismatch fix частично работает
- `raise_freq ≥ 33%` (C-like) — mismatch fix успешен полностью
- `raise_lt_fold_pct < 50%` — fold-preference должен уйти
- `best_sizing_anchor_dist.0.10 < 499` — sizing diversity должен появиться

---

## 6. Сравнительная таблица (все 4 клетки)

| Метрика | kind OFF/clear OFF (Run A) | kind OFF/clear ON (test77) | kind ON/clear ON (test76) | kind ON/clear OFF (test78) |
|---|---|---|---|---|
| `raise_freq` | 36.2% | 13.0% | 27.7% | **4.6%** |
| `win_rate` | 40.5 | 28.1 | 37.7 | 38.1 |
| `mean_reward` | — | 2.06 | 9.41 | 10.28 |
| `unique_anchors` | 15/15 | 15/15 | 14/15 | **9/15** |
| `best_sizing 0.10` | 424/499 | 499/499 | 499/499 | 499/499 |
| `top-2 sizing mass` | ~61% | 84% | 83% | **89%** |
| `action_q.raise count` | 262 597 | 147 878 | 233 746 | **27 474** |
| `opponent raise funnel` | 5 169 | 7 262 | 7 951 | **0** |
| `next_strategy_raise_mass` | 0.406 | 0.702 | 0.603 | **0.000** |
| `q_action.fold.mean` | — | +0.69 | −0.29 | **+1.75** |
| `q_action.raise.mean` | — | +0.21 | +0.38 | **−0.20** |
| `raise_lt_fold_pct` | — | 100% | 0% | **100%** |

---

## 7. Что НЕ делаем

- ❌ Больше config-ablation. 2x2 закрыта, все 4 клетки — либо grey, либо bad (кроме Run A dry-run который не full run).
- ❌ downweight/drop_unknown_only — механизм тот же: forced-remap targets живут в буфере или loss, фильтр не помогает.
- ❌ soft mask — mask уже OFF, лечить нечего.
- ❌ warmup/clear/probe tuning — не закроет фундаментальный mismatch.

---

## 8. Следующий шаг: B1 selected-credit experimental fix

### 8.1 Code change (уточнён GPT+OPUS — меняется и credit_size, и credit_idx)

**GPT поправка:** менять только `anchor_idx` недостаточно. `_add_sizing_q_sample` считает `norm_size` из переданного `bet_size` (строка 1729), и `sizing_q_net` обучается именно на continuous `norm_size`. Если оставить `bet_size=eff_mult`, сеть продолжит получать effective-sizing continuous input — это полумера, даст ложный FAIL.

**Полная B1-спецификация (верифицирована по коду, commit 26):**

В 3 sizing-Q source=0/1 callsites (2 hero-raise пути `deep_cfr.py:2203, 2375` + `train.py:318` sync-now) при выбранном рейзе:

```python
if self.sizing_q_selected_credit_enabled and sizing_anchor_idx >= 0:
    credit_size = sampled_bet_size          # selected-size для norm_size
    credit_idx  = int(sizing_anchor_idx)    # selected anchor index
else:
    credit_size = eff_mult                  # effective-size (поведение по умолчанию)
    credit_idx  = eff_idx                   # effective anchor index

self._add_sizing_q_sample(
    _state_arr, credit_size, sizing_target, iteration,
    is_probe=...,                           # 2203: False; 2375: is_probe
    anchor_idx=credit_idx,
    selected_anchor_idx=sizing_anchor_idx,
    effective_anchor_idx=eff_idx,           # ЯВНО сохранить effective для диагностики
    selected_kind=selected_kind,
)
```

**Ключевые детали:**
- Параметр `effective_anchor_idx` уже есть в сигнатуре `_add_sizing_q_sample` и в `SizingQBuffer.add`. Сейчас callsites его не передают → buffer fallback: `eff_idx = anchor_idx`. Под B1 **надо явно передать `effective_anchor_idx=eff_idx`**, иначе потеряем effective metadata.
- `norm_size = (bet_size - min_bet_size) / (max_bet_size - min_bet_size)` — замена `eff_mult→sampled_bet_size` меняет continuous-Q credit.
- `target_value` остаётся target фактически сыгранного effective raise (не counterfactual «что было бы при selected»). При forced-remap это сознательный credit-assignment: selected anchor получает credit за effective outcome. Потенциальный источник шума и причина возможного PARTIAL.
- `anchor_counts` и stratified sampling после B1 перейдут на selected-index distribution. Это ожидаемо и правильно.
- `source=2 lookahead` (строка 1951) **не трогаем** — там перебор candidate anchors без selected/effective remap.
- `opponent_raise` (строка 2447) **не трогаем** — sizing-Q sample не пишется.

**Scope callsites (3, sync-now):**
- `deep_cfr.py:2203` hero_os_raise — на self-play test79 path.
- `deep_cfr.py:2375` hero_full_raise — на self-play test79 path.
- `train.py:318` `_cfr_traverse_with_opponents` — не на self-play path, но sync-now для единой семантики флага.

**Только под config flag.** Всегда `false` по умолчанию → поведение bit-for-bit как сейчас.

### 8.2 Stress-test прогон (после кода)

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_kind_filter_enabled: false        # stress: никакого фильтра
sizing_q_replay_clear_once: true           # stress: clear раскрывает mismatch
sizing_q_selected_credit_enabled: true     # ← B1 fix
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

**Seed:** тот же. **Длина:** iter_100. **Папка:** `models/test79seed1/`.

### 8.3 Pre-registered критерии

| Исход | Критерии | Интерпретация |
|---|---|---|
| **PASS** | raise ≥ 33%, actionQ.raise ≥ 250k, best_sizing 0.10 < 499/499, raise_lt_fold < 50% | B1 fix mismatch полностью успешен; forced-remap selected↔effective — единственный root cause. Production-ready B1 (но всё равно под flag, не unconditional). |
| **PARTIAL** | raise ≥ 27% (≥ test76), actionQ.raise ≥ 200k, raise_lt_fold < 90% | B1 fix частично работает. Возможные причины: (a) target_value остаётся effective-outcome — шум при forced-remap; (b) longer iter нужен для стабилизации; (c) anchor coverage через stratified sampling меняет dynamics. Смотреть longer run + ablation (downweight, anchor coverage тюнинг). |
| **FAIL** | raise < 20%, fold > raise режим сохранён | B1 недостаточен ИЛИ implementation bug. Проверить: правильность credit_size/credit_idx routing; что `effective_anchor_idx` не потерян; что `norm_size` действительно считается от `selected` а не `effective`. Если routing корректен: mismatch — не единственная причина коллапса; источник #74-v4 устранён, но нужен дополнительный ablation по action-Q deadly triad или replay composition.

---

## 9. Файлы

| Файл | Изменения |
|------|-----------|
| `config.yaml` | В состоянии test78seed1 (kind ON, clear OFF) |
| `sizing_reports/79-...md` | Этот отчёт — уточнён mechanism (§3.3), B1-спецификация (§8.1) GPT+OPUS consensus |
| `sizing_reports/SIZING.md` | Обновлён (test78seed1 результат + B1 план) |
| `sizing_reports/78-...md` | Гипотеза replay_clear-as-exposer опровергнута результатом #79 |
| `models/test78seed1/` | Прогон config control — исходные данные этого отчёта |
| `models/test79seed1/` | **Следующий прогон** — B1 stress-test (после кода) |
| `src/core/deep_cfr.py` | B1 flag + credit-routing в hero_os_raise (2203) и hero_full_raise (2375) + checkpoint save/load |
| `src/training/train.py` | B1 sync-now credit-routing (318) |
| `tests/test_sizing_q_regret.py` | Тесты B1: default false, flag true credit routing, checkpoint roundtrip |

**B1 prerequisites (read-only верификация, June 10):**
- `_add_sizing_q_sample` сигнатура уже имеет `effective_anchor_idx=None` ✓
- `SizingQBuffer.add` хранит `_selected_anchor_indices`, `_effective_anchor_indices`, `_selected_kind_ids` ✓
- `sampled_bet_size`, `sizing_anchor_idx`, `eff_idx` доступны во всех 3 callsites ✓
- `norm_size` semantics: `(bet_size - min_bet_size) / (max_bet_size - min_bet_size)`, замена `eff_mult→sampled_bet_size` корректна ✓
- `source=2 lookahead` и `opponent_raise` не требуют B1 ✓
