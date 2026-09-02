# SIZING.md — Сводка по сайзингу (bugs #61–#99v2)

> **Последнее обновление:** 2026-08-18 (**#100 и #101 добавлены как пропущенные ветки. #104–#113 добавлены. #113 §18: измерение динамического диапазона (план 179) — приза при 1000 traversals нет. Дальше 10k traversals.**)
> **Общая тема:** sizing collapse — модель коллапсирует в один анкер (обычно 0.10 или 3.00), raise_freq падает до единиц процентов, win_rate рушится. Корень: sizing feedback loop (#61), action-Q deadly triad (#73), train/inference index mismatch (#74-v4), contamination через нелегальные анкеры (#74-v3, #74-v8), **action-Q reward-grounding asymmetry — raise структурно обездолен delayed-reward ceiling (#80, #81).**

---

## Результаты каждого репорта

### #61 — Sizing collapse в анкер 0.10 (feedback loop через sparsify)

**Результат: FIX CONFIRMED.** Коллапс полностью устранён.

- `top_per_bucket: 1→2` — теперь training targets получают 4 анкера вместо 2
- До фикса (iter_700): 0.10 = 84.5%, unique_anchors = 2, raise_freq = 20.3%, win_rate = 25.1%
- После фикса (iter_100, с нуля): 0.10 = 5.5%, unique_anchors = 15/15, raise_freq = 46.9%, win_rate = 42.3%, mean_reward = 23.20
- Риск: 1.50 занял 47.3% — потенциальный новый коллапс, нужен мониторинг
- Large bucket (2.00-3.00) всё ещё обнуляется sparsify — `top_buckets=2` а large стабильно третий

---

### #62 — Разрыв между q_net[Raise] и sizing_q_net (корень call-heavy стратегии)

**Результат: OPEN.** Диагностирован, не исправлен.

- После #61 модель всё равно коллапсирует к iter_400: raise_freq 46.9% → 8.4%, call% = 64%
- `q_net[Raise] < q_net[Call]` в 64% состояний, `q_net[Raise] < q_net[Fold]` в 96%
- `max_sizing_q.mean = +0.57`, но `q_net[Raise] − max_sizing_q = −6.39` — разрыв >6 единиц
- Две Q-сети живут в разных мирах: sizing_q_net видит «ставка 0.10 = хорошо», а q_net видит «Raise → бет мелкий → мало EV → Raise = плохо»
- Feedback loop: модель рейзит мелко → Q(Raise) учится что рейз = мало EV → модель ещё реже рейзит → (повтор)
- Создан инструмент `checkpoint_tools.py` с режимами `diagnose` / `eval` / `full`

---

### #63 — Probe затухает к iter 1000 (крупные размеры недосэмплированы)

**Результат: IMPLEMENTED.** Конфиг изменён, ждёт валидации с чекпоинта 600.

- Проблема: probe падает с 30% до 5% к iter 1000 — exploration floor исчезает
- На iter_600: 0.10 = best_sizing в 92.8% состояний, sizing_q монотонно убывает по размеру
- Фикс: `probe_prob_end: 0.05→0.15`, `decay_iterations: 1000→2000`
- Новая формула: `probe_prob(iter) = 0.30 − 0.15 × clamp(iter/2000, 0, 1)`
- На iter 600 probe 15% → 25.5% (+70%), на iter 1000: 5% → 22.5% (+350%)
- Probe влияет только на Q-буфер, не на inference-политику
- Ограничение: probe отсутствует в OS-узле (`_cfr_traverse_multi_outcome_node`)

---

### #64 — One-step lookahead (контрфактические sizing Q-target'ы)

**Результат: IMPLEMENTED.** Механизм добавлен, введён source=2.

- Каждый raise-узел с вероятностью 10% пишет Q-target'ы для 5 случайных несыгранных анкеров
- Стоимость: 5 × (один шаг apply_action + bootstrap через q_net), а не 5 × (дерево до showdown)
- Новые поля в буфере: source=2 для lookahead (source=0 — rollout, source=1 — probe)
- `_bootstrap_sizing_value`: V(s_next) = Σ strategy[a] × q_net(s_next)[a] — TD-bootstrap через regret-matching стратегию
- Ожидаемый эффект: sizing_q_net получает плотное покрытие по размерной сетке без взрыва стоимости

---

### #65 — checkpoint_tools: диагностика sizing_q_buffer

**Результат: FIXED.** Инструментарий добавлен.

- `load_full_checkpoint` теперь грузит `sizing_q_buffer_sources`, `_anchor_indices`, `_targets`
- Новая функция `run_sizing_q_buffer_stats`: count_by_source, count_by_anchor, mean_target_by_source_anchor
- Данные попадают в `_diagnose_report.json` и `_full_report.json`
- Для light чекпоинтов — silently skipped

---

### #66 — Стабилизация q_net через reward scaling + target network

**Результат: FIXED.** q_net больше не главный источник шума.

- До фикса: loss = 1000-3000 (MSE на масштабе 200²), без снижения
- `q_reward_scale: bb` — reward делится на big_blind при обучении q_net
- Добавлен `q_target_net` (замороженная копия), синхронизация каждые 100 итераций
- CFR/regret/sizing target не затронуты — только q_net учится в нормализованных единицах
- 6 новых тестов, 26/26 passed

---

### #67 — Унификация шкал sizing_q + стратифицированный sampling

**Результат: FIXED.** Разнообразие временно появилось, но relapse к iter 400/500.

- Дефект 1: source2 был в bb (через q_net после #66), source0/1 в raw chips — рассинхрон ~30-100×
- Фикс: `sizing_q_bootstrap_scale_fix=true` — домножает bootstrap-значение на big_blind → raw chips
- Дефект 2: равномерный sampling → source0 (67%) и анкоры 0.10/0.25 доминируют в MSE
- Фикс: `sizing_q_stratified_sampling=true` — равная квота на каждый непустой анкер, малые группы с заменой
- **Iter 300 — лучший чекпоинт:** best_sizing показал разнообразие (не all-0.10)
- **Iter 400/500 — relapse:** коллапс вернулся. Причина relapse на тот момент неизвестна.
- 4 новых теста, 31/31 passed

---

### #68 — Диагностика Target Formation + Phase 0 hotfixes

**Результат: FIXED (instrumentation).** Инструментарий для поиска причины relapse после #67.

- **САЙЗИНГ-КОЛЛАПС ПОБЕЖДЁН ЧАСТИЧНО:** после #67 diversity держался до iter 300 (лучший чекпоинт) но relapse к 400/500
- Причина relapse на тот момент неизвестна — гадание без диагностики
- Добавлена read-only instrumentation: pre-clip/post-clip target values при записи в буфер
- Агрегация по `(source, anchor_idx)`: count, clip_high, clip_low, raw/pre/post min/max
- **Hotfix Phase 0:** `load_full_checkpoint` не читал `sizing_q_target_diag` → исправлено
- Regression guard: `compute_verdict` теперь WARN при single anchor >60%
- Новые метрики: `low_count_cells` (count < 5), `single_anchor_dominance_pct`, `amplification_risk`
- Future phases (не реализованы): src0/source1 root cause, плоский source2, durable fix, honest eval
- 7 новых тестов, 38/38 passed

---

### #69 — Target-net для bootstrap sizing_q + коррекция диагностики source2

**Результат: FIXED.** Barbell-collapse сломан впервые на iter 200, raise_freq восстановлен с 2% до 28%. Но мультимодальность не держится к iter 300/400.

- **САЙЗИНГ-КОЛЛАПС ПОБЕЖДЁН ЧАСТИЧНО (продолжение #68):** на iter 200 впервые нет barbell — anchor diversity работает
- source2 дрейфовал монотонно вверх (`source2_intra_anchor_range` 1.03→7.39) — deadly triad: bootstrap через live-сети
- Фикс: `sizing_q_bootstrap_target_net=true` — bootstrap использует `q_target_net` и `target_net` (замороженные)
- Метрика `source2_flatness` переименована в `source2_intra_anchor_range` — мерила разброс по состояниям, а не различимость между анкорами
- Добавлена `source2_cross_anchor`: mean_range, mean_std, argmax_anchor, n_anchors_represented
- Verdict gates ужесточены: CRITICAL при одном anchor >90%, WARN при raise_freq <25%
- **raise_freq восстановился с 2% до 28%**
- **НО к iter 300/400 best_anchor снова в экстремуме** — мультимодальность не держится
- 3 новых теста, 41/41 passed

---

### #70 — z-score targets + src2 loss-weight anneal

**Результат: FAIL.** z-score вызвал полный рассинхрон между policy и Q.

- source2 после #69 оставался плоским по анкорам (`cross_anchor.mean_std≈0.02`) и дрейфовал по состояниям
- Фикс: train-time z-score targets (только к targets_t, не к q_pred) + src2 per-sample loss weight с anneal 1.0→0.25 за 400 итераций
- `SizingQBuffer.sample` / `sample_stratified` теперь возвращают sources через `return_sources=True`
- **Validation iter 200:** FAIL — `verdict=WARN (low_raise 17.8%)`, win_rate=22.1, `best_sizing_anchor_dist: 3.00=99.6%`
- Policy raw sizing ушёл в small/0.10, q_best почти всегда 3.00 — полный рассинхрон
- Первопричина: target-only z-score изменил scale/семантику `sizing_q_net`, а downstream читает raw Q
- 8 новых тестов

---

### #71 — Отключение z-score, чистый контроль src2 anneal

**Результат: FAIL.** Проблема сместилась с sizing на action-level.

- Одно изменение: `sizing_q_train_zscore: true→false`. src2 anneal оставлен.
- **Validation iter 100:** FAIL нового типа — `raise_freq=1.6%`, `q_action.fold.mean=3.06`, `q_action.raise.mean=0.63`, `raise_lt_fold_pct=100%`
- `best_sizing_anchor_dist: 0.10=2993/2993=100%`
- Первичная проблема теперь action-level: модель считает fold лучше raise почти всегда
- Sizing-collapse в 0.10 вторичен — raise почти не выбирается
- Гипотезы: H1 action-level deadly triad, H2 starvation raise transitions, H3 fold bootstrap неоднозначен

---

### #72 — action-Q low-raise collapse diagnostics

**Результат: IMPLEMENTED (diagnostic).** Подтверждён self-suppressing loop через `next_strategy`.

- Diagnostic-only: никаких правок в динамику обучения
- Новый флаг `save_q_buffer_in_checkpoint: true` — сохраняет q_buffer states/actions/rewards/next_states/terminals
- `run_action_q_buffer_diagnostics` — count, terminal ratio, reward stats by action
- `run_action_q_model_diagnostics` — q_net vs q_target diff, bootstrap_value, target, next_strategy_raise_mass
- **Diagnostic run iter 900:** `raise_freq=23.1%`, `win_rate=17.3`, `raise_lt_call_pct=99.36%`
- `q_action.raise.mean=-1.867`, `q_action.call.mean=-0.033`
- `q_buffer.raise.count=374185` — raise НЕ starving, сэмплов много
- `q_buffer.raise.terminal_pct=0.0%`, `reward=0.0` — все raise-сэмплы bootstrap-only
- `bootstrap_value.raise.mean=-6.465` — экстремально отрицательный
- `next_strategy_raise_mass.raise.mean=0.117` — политика подавляет raise в next-состояниях
- **Self-suppressing loop подтверждён:** next_strategy ≈ 0.12 raise → bootstrap value отрицательный → Q[raise] падает → policy ещё сильнее подавляет raise

---

### #73 — action-Q bootstrap policy mix для raise suppression

**Результат: FAIL (не держится).** Deadly triad разорван на iter 100, но коллапс возвращается к iter 200.

- Фикс: в `train_q_network` TD-target backup: `next_v = (1−mix)×next_strategy + mix×uniform × q_target`
- `q_bootstrap_policy_mix_enabled: true`, `uniform_mix: 0.10`
- Mix применяется только в TD-target, не в игре/inference/sizing
- **Iter 100: ОГРОМНЫЙ УСПЕХ** — `verdict=OK`, `raise_freq=47.9%`, `win_rate=41.8`
- `raise_bootstrap_value.mean=+0.175` (было −6.465 в #72)
- `next_strategy_raise_mass.raise.mean=0.591` (было 0.117)
- Deadly triad разорван на iter 100 — модель агрессивно рейзит и выигрывает
- **Iter 200: КОЛЛАПС** — `verdict=WARN (low_raise)`, `raise_freq=12.6%`, `win_rate=19.6`
- `raise_bootstrap_value.mean=−0.885`, `raise_mass=0.098` — self-suppressing loop вернулся
- Mix=0.10 разрывает цикл на iter 100, но статичный floor недостаточен против самоусиления
- 4 новых теста

---

### #74 — dose-response: mix 0.10→0.15

**Результат: FAIL (отсрочка, не фикс).** Держится до iter 200, коллапсирует к iter 300-400.

- Одна правка: `q_bootstrap_policy_uniform_mix: 0.10→0.15`
- **Iter 100:** `verdict=OK`, `raise_freq=54.8%`, `win_rate=44.0`
- **Iter 200:** `verdict=OK`, `raise_freq=34.8%`, но `best_sizing_anchor_dist` уже `0.10: 2918/2993`
- **Iter 300:** `verdict=WARN`, `raise_freq=16.9%`, `best_sizing: 0.10=2993/2993`
- **Iter 400:** `verdict=WARN`, `raise_freq=11.2%`, sizing коллапс в small bucket
- Вывод: mix — это отсрочка self-suppressing collapse, а не фикс

---

### #74-v2 — Sizing collapse root-cause diagnostics (next_is_hero / sizing_path_diag)

**Результат: DIAGNOSTICS.** Инструментарий для определения, где замыкается collapse.

- Вместо слепого #75 нужен диагностический прогон для доказательного определения источника collapse
- `QValueBuffer.next_is_hero`: новый массив — True если next_state.current_player == traversing_player
- `bootstrap_by_next_role` в report: разрезы hero/opponent/terminal по каждому action
- `sizing_path_diag`: три счётчика — hero_os_raise, hero_full_raise, opponent_raise
- Правила интерпретации: Case A (opponent портит raise) → #75, Case B (action норм, sizing схлопнулся) → копать OS path, Case C (оба слоя) → #75 + OS fix
- 57/57 тестов в test_hybrid_outcome_sampling.py

---

### #74-v3 — Nearest-anchor snapping: bucket anchor может быть нелегален

**Результат: OPEN (диагностика готова, фикс отдельно).**

- `eff_mult` snap'ится к ближайшему anchor через `argmin(|diff|)` — этот anchor может быть нелегален (например ниже min_raise)
- Конкретный пример: selected=0.10 (10 чипов), min_raise=12 → effective=12, eff_mult=0.12 → ближайший anchor=0.10 (который 10 чипов, НИЖЕ min_raise=12)
- Три пути: hero_os_raise, hero_full_raise, opponent_raise — все без проверки легальности bucket
- Последствия: Q-target noise, фантомные сэмплы в стратифицированном sampling, скрытая проблема
- Диагностика: `sizing_selected_effective_diag` (матрица selected_idx→bucket_idx) + `_kind_diag` (NORMAL/MIN_RAISE/ALL_IN + below_effective/equal/above)
- Предложен фикс `_nearest_legal_anchor_idx` — поиск ближайшего legal anchor, с fallback all-true если все заблокированы
- 5 новых тестов

---

### #74-v4 — Train/inference index mismatch + metadata plumbing

**Результат: OPEN (диагностика готова).** Подтверждено double indexing.

- Инференс: argmax Q по `selected_idx` (15 anchors, любой может быть нелегален)
- Training: target пишется в `effective_idx` после `_resolve_effective_sizing` (MIN_RAISE/ALL_IN ремап)
- Q[selected=3.0] никогда не получает кредит за выбор «3.0» — target ушёл в Q[effective=0]
- **Из iter 100 чекпоинта:** equal_effective (точное попадание) — hero_os 28%, hero_full 40%, opponent 34%
- MIN_RAISE selected: os 17%, full 10%, opp 23%
- ALL_IN selected: os 49%, full 41%, opp 36% — огромная доля форсов!
- **88% всех raise-записей = opponent_path** → напрямую связано с #75 (player-aware)
- SizingQBuffer расширен: `_selected_anchor_indices`, `_effective_anchor_indices`, `_selected_kind_ids`
- `selected_effective_target_summary()` — группировка по (source, selected, effective, kind)
- Decision gate: B1 (писать Q под selected_idx) или B2 (сегментировать по kind) — ждёт данных
- 62/62 тестов

---

### #74-v5 — Source=2 lookahead diagnostics v3

**Результат: DIAGNOSTICS.** source2 = 0.0 по всем анкорам, причина пока неизвестна.

- Из iter 100: source=2 = 14 565 samples (29% буфера), `mean_target_by_source_anchor` = 0.0 по всем 15 anchors
- `clip_high=0, clip_low=0` у всех source=2 (в отличие от source=0/1 где клипы массовые)
- Либо bootstrap-проблема, либо баг в записи
- Новый `sizing_lookahead_diag`: counts (attempts, applied_ok, added, final_state, nonterminal, bootstrap_debug_missing, vk_near_zero, target_near_zero)
- `_bootstrap_sizing_value` захватывает debug: strategy_sum, legal_sum, q_vals
- Running stats per anchor: v_k, pre_clip, target, pot, strategy_sum, legal_sum, q_a0..q_a3

---

### #74-v6 — Source=2 root-cause: stale q_target_net + Q-сеть ОЖИВЛЕНА на source2

**Результат: CONFIRMED. Q-сеть на source2 ОЖИВЛЕНА через отключение target-net в bootstrap.**

- **V3 диагностика показала:** `q_a0..q_a3 = 0.0, std=0` на всех post-raise состояниях (pot 3→1191)
- V4 добавила: `base_out_abs_sum`, `q_canary_abs_sum` (ones_like), `q_input_abs_sum/min/max`, `using_target_net`
- **Прогон A (`bootstrap_target_net=true`):**
  - attempts=33665, `final_state=0` — все через bootstrap
  - `q_input_has_nan=0`, `q_input_abs_sum=23.14` — вход нормальный
  - `base_out_abs_sum=6.10` — base живой (ReLU выдаёт ненулевые features)
  - **`q_canary_abs_sum=0.0`** — сеть на ones_like даёт ровно 0 → q_head мёртвый
  - `q_a0..q_a3=0.0 (std=0)`, `v_k=0.0`, `target=0.0` — все 33 665 сэмплов нулевые
  - `using_target_net=33665` — все 100% через q_target_net
  - **Root cause:** `q_target_net.q_head = zero-init`. Sync каждые 100 итераций. Iter 1-99: q_net обучается (ненулевой), но q_target_net ни разу не синхронизирован → q_head = 0. Checkpoint на iter 100 сохраняется ПОСЛЕ sync.
- **Прогон B (`bootstrap_target_net=false`):** ← Q-СЕТЬ ОЖИВЛЕНА
  - `using_target_net=0`, `q_canary_abs_sum=6.25` — **сеть ЖИВАЯ**
  - `q_a0=1.17, q_a1=-0.09, q_a2=-0.02` — **Q на post-raise НЕНУЛЕВОЙ**
  - `v_k mean=-0.074, std=0.99`, `target mean=0.002, std=0.052`
  - `source2_means: 0.003..0.008` по анкорам — source2 перестал быть константным нулём!
  - `vk_near_zero`: 33665 → 1113 (было все, стало 3%)
  - **НО стратегия регрессировала:** `raise_freq 28%→19.7%`, `win_rate 42%→37.2%`, `best_sizing=0.10=100%`
  - Причина: source2 хоть и не ноль, но слабый/почти плоский (~0.003..0.008 std 0.05), с весом ~0.81 в loss размывает sizing-Q

---

### #74-v7 — Run C: source2 исключён из sizing-Q loss

**Результат: IMPLEMENTED (ждёт прогона).** Ожидается восстановление raise_freq.

- `sizing_q_src2_weight_start: 1.0→0.0`, `sizing_q_src2_weight_end: 0.25→0.0`
- `sizing_q_bootstrap_target_net: false` (как в прогоне B — q_net ЖИВОЙ)
- Механизм: существующий per-sample вес (#70) обнуляет source2 в loss, но source2 продолжает писаться в буфер
- **Ожидаемый результат:** raise_freq >25% (восстановление с 19.7%), best_sizing больше 1 anchor, win_rate >40%
- **После прогона C (roadmap):** v5-диагностика q_a3, legal-anchor mask, kind-filter, replay hygiene, verdict-gate

---

### #74-v8 — Run D: Legal-anchor mask + kind-filter + replay hygiene + verdict concentration gate

**Результат: IMPLEMENTED (config-gated, ждёт валидного прогона).**

- Трёхкомпонентный механизм, все выключены по умолчанию. Новые флаги:
  - `sizing_q_legal_anchor_mask_enabled: true` — исключает ALL_IN/MIN_RAISE из выбора sizing
  - `sizing_q_kind_filter_enabled: true`, `mode: drop` — дропает не-NORMAL сэмплы из sizing-Q loss
  - `sizing_q_replay_clear_once: true` — однократная очистка буфера при старте
- `compute_verdict`: WARN при top-2 sizing mass > 55%
- Legal-anchor mask: `_legal_sizing_anchor_mask` вычисляет boolean mask (NORMAL после resolve), fallback all-true если все заблокированы
- Маска применяется в: `_hierarchical_sizing`, `choose_action` (argmax/top_bucket/hierarchical/stochastic), OS node
- Kind-filter: per-sample loss с weight=0 для MIN_RAISE/ALL_IN при mode=drop
- Replay clear: `_maybe_clear_sizing_q_replay_once` — однократный вызов `buffer.clear()`
- **Run D не работал:** тренировка шла без конфига 74v8 (флаги missing в чекпоинте), kind_filter_diag = all zeros
- **Round-trip bug исправлен:** `_build_checkpoint` сохраняет флаги в top-level + config; `_load_checkpoint` читает из config как fallback
- **Follow-up fix:** `cleared_once` теперь persistent (не сбрасывается в reset_traversal_stats), `normal_kept` инкрементится
- 8 новых тестов, 70/73 passed (3 pre-existing failure)
- Критерии валидного Run D: `sizing_q_config.sizing_q_legal_anchor_mask_enabled=true`, `kind_filter_diag.normal_kept>0`, `cleared_once=1`

---

### #75 v5 — Mask vs Replay-Clear Diagnostics (C→D Root-Cause Diff)

**Результат: IMPLEMENTED (instrumentation-only, config-gated).** Все 12 новых тестов passed (83/86 total).

- Run C/test7 = profitable baseline; Run D/test8 = v8-induced regression.
- q_a3=0, raise reward=0, terminal=0% — не smoking gun (идентичны в C и D).
- UNKNOWN = source2 lookahead (не передаёт selected_kind) → kind-filter drop редундантен (src2 уже weight=0).
- Подозреваемые: (1) legal-anchor mask, (2) replay_clear_once, вероятно в связке.

**Компоненты v5:**
- 3 config flags: `sizing_q_mask_diagnostics_enabled`, `sizing_q_mask_dry_run`, `sizing_q_replay_clear_diagnostics_enabled` (all false).
- `_compute_sizing_anchor_kind_mask` — counterfactual mask helper без зависимости от `mask_enabled`.
- `_apply_sizing_anchor_mask(probs, state, callsite)` — diagnostics + dry-run + callsite labels.
- 6 callsite labels: `hierarchical`, `os_q_ready`, `choose_argmax`, `choose_top_bucket`, `choose_hierarchical`, `choose_stochastic`.
- Raise selection funnel: 3 пути (hero_os/hero_full/opponent) × 4 counters (legal/sampled/apply/buffer_add).
- Insert-time UNKNOWN audit: by_source + by_callsite.
- Replay clear audit: before_clear + cleared + refill_snapshots.

**Предрегистрированные прогоны:**
- Run A: C + dry-run + diagnostics (same seed as test7). Guardrail: должен воспроизвести test7.
  - Если counterfactual post-mask концентрируется даже на здоровом C → legal-anchor mask culprit.
- Run B: full D + diagnostics (подтверждает applied-mask и replay refill).
- Ablation (C + mask/clear/оба) — только если A/B неоднозначны.

**Decision rules:**
- Dry-run benign → replay_clear/interactor.
- UNKNOWN только source2 → kind-filter не culprit.
- raise_legal stable, sampled падает → policy/selection issue.

---

### #76 — Legal-Anchor Mask Culprit Confirmed (Run A dry-run + Full D diagnostics)

**Результат: CONFIRMED.** Mask — primary culprit (20-53% removed mass). Replay clear — interactor (не primary).

**Run A (dry-run):**
- C-поведение: raise_freq=36.2%, win_rate=40.5, unique_anchors=15/15.
- Counterfactual mask removed: hierarchical 42%, os_q_ready 29%, choose_top_bucket 31%.
- UNKNOWN только source2 (315). Replay не чистился.
- Raise funnel: hero_full 100% legal→buffer, hero_os 56%, opponent 5169.

**Full D + diagnostics:**
- D-regression: raise_freq 2.7%, unique_anchors 10/15, action_q_buffer.raise 27556, next_strategy_raise_mass 0.0.
- Applied mask removed: hierarchical 53%, os_q_ready 21%, choose_top_bucket 30%.
- Removed mass: MIN_RAISE (львиная доля), ALL_IN (вторично), NORMAL почти 0.
- Replay before_clear: total 3816 (all 15 anchors, all 3 sources). Clear стёр diversity, но refill дал полный buffer, значит clear не primary.
- Opponent raise path полностью исчез (5169 → 0).
- UNKNOWN только source2.

**Root cause:** mask обнуляет MIN_RAISE/ALL_IN → срезает 20-53% sizing-policy массы → ранний D collapse. Replay clear усиливает через удаление historical diversity.

**Isolation run (Full-D minus mask, конфиг в config.yaml):**
mask OFF / kind ON / clear ON / diagnostics ON. Единственное отличие от Full-D: `mask_enabled: true→false`.
Pre-registered PASS: `raise_freq≥33%, next_strategy_raise_mass≥0.38, action_q_buffer.raise≥230k, opponent funnel≠0`.

**Следующий шаг:** см. #77 для результата isolation и #78 для second control.

---

### #77 — Isolation Result (GREY/PARTIAL, test76seed1)

**Результат: GREY/PARTIAL.** Mask catastrophic-collapse trigger подтверждён; sole-cause гипотеза опровергнута. kind-filter=drop режет 42% live mass.

**Pre-registered сверка (#76 §6.2):**
- `raise_freq=27.7%` (FAIL, Grey zone 15–30), `next_strategy_raise_mass=0.603` (PASS), `action_q_buffer.raise=233 746` (PASS), `unique_anchors=14` (NEAR), `opponent funnel=7951` (PASS).
- **Verdict: 3 PASS, 1 NEAR, 1 FAIL → GREY.**

**Сравнение C / Full-D / Isolation:**
- C: raise_freq=37.6%, win=42.1, unique=15, best_sizing 0.10=424/499, top2≈61%.
- Full-D: raise_freq=2.7%, win=38.5, opponent=0, actionQ.raise=27 556.
- Isolation: raise_freq=27.7%, win=37.7, opponent=7951, actionQ.raise=233 746, but best_sizing 0.10=499/499, top2=83%.

**Mask = catastrophic-collapse trigger (доказано):**
- Mask OFF → opponent funnel 0→7951, actionQ.raise 27.5k→233.7k, next_strategy_raise 0→0.603, raise_freq 2.7→27.7.
- 4 катастрофические метрики восстановлены полностью.

**Mask ≠ sole cause (не подтверждено):**
- Residual зазор: raise 27.7 vs 37.6, win 37.7 vs 42.1, top2 83% vs 61%, best_sizing 0.10=499/499 vs 424/499.

**Ревизия #75 v5 о kind-filter:**
- Code-evidence из `deep_cfr.py:3040–3052`: kind_filter=drop зануляет вес для ВСЕХ non-NORMAL (MIN_RAISE+ALL_IN+UNKNOWN), не только UNKNOWN.
- test76seed1 `kind_filter_diag`: 267 MIN_RAISE + 293 ALL_IN dropped из source=0/1 vs 772 NORMAL kept → **42% live known mass отбрасывается из loss.**
- Вывод #75 v5 «kind-filter не culprit» узко верен для UNKNOWN-канала, но ошибочен для общего эффекта.

**Три residual канала:**
1. Loss-side — kind_filter=drop (42% known live mass dropped).
2. Loss-side — replay_clear interactor (clear 3453 samples → refill под altered policy).
3. Inference-side — sizing-Q argmax concentration (`0.10=499/499` при healthy buffer 14/15 anchors).

**Следующий шаг:** pre-registered GREY → second control `C + replay_clear only` (#78, kind_filter OFF). See: `77-isolation-result-grey.md`.

---

### #78 — Second Control Result (UNEXPECTED, kind_filter=drop гипотеза опровергнута, replay_clear = exposer mismatch)

**Результат: UNEXPECTED.** Выключение kind_filter=drop ухудшило метрики — гипотеза «kind_filter = residual виновник» опровергнута. Root cause: #74-v4 train/inference index mismatch.

**Сводка test76 → test77:**
- `raise_freq`: 27.7 → **13.0%** (−14.7pp), `win_rate`: 37.7 → **28.1** (−9.6pp), `mean_reward`: 9.41 → **2.06** (−7.35).
- `actionQ.raise`: 233 746 → **147 878** (−37%), `q_action.raise.mean`: 0.384 → **0.205** (−47%).
- `q_action.fold.mean`: −0.294 → **+0.690**, `raise_lt_fold_pct`: 0 → **100%** — fold > raise всегда (тот же failure mode что #71).
- `best_sizing_anchor_dist`: 0.10=499/499 в обоих (inference concentration не сдвинулась).

**Action-Q regression:**
- kind_filter OFF → токсичные MIN_RAISE targets учатся в loss (selected=0→eff=1: −0.99, eff=4: −2.01). **Паттерн: чем дальше effective от selected, тем отрицательнее target.**
- Code-evidence: `_add_sizing_q_sample(..., anchor_idx=eff_idx, selected_anchor_idx=sizing_anchor_idx)` — target пишется по effective, inference argmax по selected. **Фундаментальный mismatch.**

**Консенсус GPT + OPUS — 4-уровневая иерархия:**
1. **selected/effective mismatch (#74-v4)** — фундаментальная токсичность forced-remap targets.
2. **kind_filter=drop** — safety filter, частично скрывает токсичность (удалял 42% MIN_RAISE/ALL_IN targets из loss).
3. **replay_clear** — **exposer/amplifier** mismatch. Убирает historical diversity, позволяет токсичному refill доминировать. Run A (kind OFF/clear OFF) 36.2% vs test77 (kind OFF/clear ON) 13.0% — дельта **23pp** от единственной разницы.
4. **hard mask** — отдельный catastrophic trigger из Full-D (доказан в #76).

**Следующий шаг:** test78seed1 → результат #79 (см. ниже).

---

### #79 — Config Control Result (BAD, replay_clear-as-exposer гипотеза не подтверждена, config-fix исчерпан)

**Результат: BAD.** `raise_freq=4.6%` — close to Full-D collapse. replay_clear-as-exposer гипотеза не подтвердилась. 2x2 матрица закрыта, config-only fix исчерпан.

**Сводка test76 → test78:**
- `raise_freq`: 27.7 → **4.6%** (−23pp), `actionQ.raise`: 233 746 → **27 474** (−88%), `opponent funnel`: 7951 → **0**.
- `next_strategy_raise_mass`: 0.603 → **0.0**, `unique_anchors`: 14 → **9**, `top2`: 83% → **89%**.
- `q_action.fold.mean`: −0.29 → **+1.75** (сильнейший fold-preference всех прогонов), `raise_lt_fold=100%`.
- `hero_os` появился (0→691), но `opponent` исчез, `hero_full` упал (−53%).

**Механизм почему kind ON/clear OFF — худшая клетка — открытая гипотеза (уточнено GPT+OPUS):**

Кодовая проверка (`deep_cfr.py:3040–3064`): в drop-mode non-NORMAL получают `weight=0`, loss нормируется на `weight_sum` — **прямого градиента нет**. Возможные косвенные каналы: (1) anchor coverage через stratified sampling; (2) effective-index target misassignment (#74-v4 mismatch) — записи обучают Q[effective_idx] под контекст другого selected; (3) stale replay composition. Самый прямой кандидат — effective-index misassignment: не фильтруется kind_filter, без clear накапливается.

**Завершённая 2x2 матрица:**

| | clear OFF | clear ON |
|---|---:|---:|
| kind OFF | Run A: **36.2%** | test77: 13.0% |
| kind ON | test78: **4.6%** | test76: 27.7% |

**Ни одна single-variable гипотеза не работает** — система сложная interaction dynamics. Все конфиги кроме Run A — grey или bad.

**Вывод:** config-only fix исчерпан. Root cause — #74-v4 forced-remap selected↔effective mismatch. kind_filter и replay_clear — паллиативные механизмы, взаимодействующие через buffer distribution. Единственный фундаментальный fix — **B1**: писать sizing-Q target под `selected_idx` (одновременно `credit_size=sampled_bet_size` и `credit_idx=sizing_anchor_idx`), не под `effective_idx`.

**Следующий шаг:** B1 experimental flag (`sizing_q_selected_credit_enabled: true`). Stress-test: mask OFF/kind OFF/clear ON/B1 ON. Pre-registered: raise≥33% (PASS), raise≥27% (PARTIAL), raise<20% (FAIL). Полная спецификация в `79-test78control-clear-off-bad.md` §8.1.

---

### #80 — B1 Stress-Test Result (PRE-REGISTERED FAIL, raise_freq=2.1%, B1 input-side fix недостаточен)

**Результат: PRE-REGISTERED FAIL.** `raise_freq=2.1%` — хуже всех предыдущих конфигов. **B1 был ПОЛНОСТЬЮ активен** (`sizing_q_selected_credit_enabled: true` в config.yaml, credit-routing во всех 3 callsites). Предыдущая версия отчёта ошибочно интерпретировала прогон как "B1 NOT PRESENT" из-за reporting bug: `tools/checkpoint_tools.py` не включает флаг в whitelist `sizing_q_config_keys`. `sizing_q_mask_dry_run=true` относится только к legal-anchor mask, не к B1.

**Сводка:**
- `raise_freq=2.1% < 20%`, `raise_lt_fold_pct=100%` → **FAIL по pre-registration**
- `win_rate=35.1%` — высокий для почти нулевого рейза; модель как calling station
- `mean_reward=-7.33` — отрицательный
- `best_sizing_anchor_dist: 0.10=499/499` — **B1 не сдвинул ни на одно состояние**
- `unique_anchors=5/15`, `top-2 mass=89%`
- `actions`: fold=30, check=548, call=2484 (79.4%), raise=65

**Self-suppressing loop НЕ активен:**
- `bootstrap_value.raise.mean=+0.266` (положительный!)
- `next_strategy_raise_mass.raise.mean=0.667` (next-политика хочет рейзить)

**Sizing funnel здоров:**
- `opponent_raise=7225`, буфер 15/15 анкеров, source2 amplification_risk=false

**Action-Q fold-preference — primary blocker:**
- `q_action.fold.mean=+0.51 > raise.mean=+0.33` на 100% eval-состояний
- raise_freq=2.1% → sizing-Q фикс не проявляется

**Почему B1 не сработал — два независимых блокера:**
1. **Target-side caveat:** B1 починил input (credit_size/credit_idx), но `sizing_target` всё ещё effective outcome. `best_sizing=0.10=499/499` не сдвинулся → источник на target-стороне.
2. **Action-Q fold>raise:** `fold.mean=+0.51 > raise.mean=+0.33`. На 499 eval-состояниях модель предпочитает fold. Пока action-голова глушит raise, sizing фикс не виден.

**Следующие шаги (по приоритету):**
1. P0: Action-Q fold>raise investigation — почему `q_net[fold]=+0.51` на eval при `target=-2.70` на replay
2. P1: B1 target-side завершение — counterfactual `sizing_target` под selected
3. P2: Серия чекпоинтов iter 100/200/400 — исключить переходный артефакт после replay clear
4. P3: Tooling fix — `checkpoint_tools.py` whitelist
5. P4: Sizing-Q negative shift investigation (`max.mean=-0.78`)
6. См. `80-b1-stress-test-catastrophic-fail.md` для полного отчёта.

---

### #81 — Action-Q Role & Legal Diagnostics (H1 CONFIRMED, H2 REJECTED, raise reward-grounding asymmetry)

**Результат:** Read-only диагностика `tools/action_q_role_legal_diag.py` на чекпоинте test79seed1 iter 100.

**H1 vs H2:**
- **H1 (raise bootstrap-only / ceiling): CONFIRMED.** Raise — 0 terminal samples. TD-target потолок +0.418 (p99=+0.38). Fold/call имеют terminal targets до +988/+1000. Проблема — reward-grounding асимметрия: raise никогда не получает terminal reward напрямую.
- **H2 (q_compare misleading): REJECTED.** Fold легален в 97.4% q_compare состояний. Fold>raise confirmed on legal states (100% when fold legal). `best_legal_action_dist = {fold: 486, raise: 13}`.

**q_compare Legal Mask:**
- `fold_legal_pct=97.4%`, `raise_lt_fold_when_fold_legal_pct=100.0%`
- fold — реально best action, не diagnostic artifact

**Raise Bootstrap Ceiling (TD-targets):**
- raise nonterm max: **+0.418**, p99: +0.38
- fold nonterm max: +0.295 (НИЖЕ raise!)
- fold terminal max: **+988.04**, call terminal max: **+1000.0**
- Raise nonterminal target — наивысший среди действий (+0.42)
- Но fold/call terminal tails orders of magnitude больше → тянут action-Q вверх

**Terminal Positive Tail:**
- hero_fold_positive_tail_suspected: **false** (positive fold = opponent fold = normal poker outcome)
- Нет reward sign/role bug

**Self-suppressing loop уточнение:**
- #73 deadly triad: bootstrap_value.raise=+0.266 (положительный, loop не активен)
- Новая проблема: delayed-reward **ceiling** + terminal-reward **asymmetry**
- Raise nonterminal target (+0.42) > fold nonterminal max (+0.295), НО fold имеет terminal +988

**Следующие шаги:**
1. P0: Action-Q n-step/MC delayed reward propagation — донести terminal value через raise chains
2. P1: Sizing-Q negative shift investigation (max.mean=-0.78)
3. P2: После action-Q fix — повтор B1 test79
4. Не включать mask/kind-filter — H2 rejected, проблема реальная
5. См. `80-b1-stress-test-catastrophic-fail.md` §9 и `81-action-q-role-legal-diagnostics.md`.

**#81 Grad-Clip & Scale Audit (reward_unit=2.0):**

| Scenario | Target abs(mean) | Raw grad | Post grad | Throttle |
|---|---|---|---|---|
| mixed_natural | 2.94 | 120.04 | 1.00 | 120.04× |
| terminal_only | 37.12 | 734.09 | 1.00 | 734.09× |
| bootstrap_only | 0.73 | 7.36 | 1.00 | 7.36× |
| raise_only | 0.29 | 0.69 | 0.67 | 1.03× |

- Mixed decomposition: terminal grad 671.39 vs bootstrap grad 3.45 — ratio **194.72×**
- `scale_bottleneck_confirmed` = **False**: grad-clip не причина Q-сжатия
- `bootstrap_grad_barely_clipped` = **False** (throttle 7.36× > 1.5× порог): bootstrap тоже клипается
- Terminal в mixed batch: 3-8% сэмплов, их сигнал задавлен не clipping'ом, а loss weighting / batch composition

**#81 Grad-Clip & Scale Audit (reward_unit=500.0 / bb500):**

| Scenario | Target abs(mean) | Raw grad | Post grad | Throttle |
|---|---|---|---|---|
| mixed_natural | 0.65 | 28.04 | 1.00 | 28.04× |
| terminal_only | 0.15 | 554.01 | 1.00 | 554.01× |
| bootstrap_only | 0.73 | 7.36 | 1.00 | 7.36× |
| raise_only | 0.29 | 0.69 | 0.67 | 1.03× |

- Mixed decomposition: terminal grad 232.94 vs bootstrap 3.45 — ratio **67.56×**
- `scale_bottleneck_confirmed` = **False** (same conclusion)
- Terminal targets сжаты reward_unit (0.15 vs 37.12 при 2.0), но terminal градиенты всё равно 554×

**Interpretation:** Q-сжатие — не grad-clip баг. Причина: (а) bootstrap численно доминирует в loss; (б) mean reduction размазывает terminal градиенты; (в) выходной диапазон сети ограничен. Next: loss reduction diagnosis, terminal upweighting, `max_norm` ablation.

---

### #82 — action-Q max_norm ablation (изоляция grad-clip)

**Результат: B CONFIRMED (fit_ratio) + BEHAVIORAL BREAKTHROUGH (post-hoc).**

- **Pre-registered (fit_ratio):** B подтверждена — \|q_pred\|_term 1.30/1.75 <2, fit_ratio стабилен ~0.02. Clip не первопричина Q-сжатия.
- **Post-hoc (поведение):** Монотонная дозозависимость: 1.0→raise 1.5% / 10→52.8% WARN / 100→48.4% **OK** — первый OK со времён Run C.
- **max_norm=100:** raise_freq=48.4%, 15/15 unique, win_rate=42.2, mean_reward=20.7, анкер 0.10=10.6% (было 56-100%).
- **Механизм:** fit_ratio не улучшился (~0.02), но Q-дифференциация между действиями выросла → политика начала рейзить.
- См. `sizing_reports/82-max-norm-ablation.md` §4.


### #83 — terminal-balanced Q-loss (ветка B после #82)

**Результат: CLOSED (WARN).** test81seed1: raise_freq=3.0%, unique_anchors=5/15 — коллапс не сломан.

- Balanced loss α=0.5 при max_norm=1.0 не работает: Q слишком сжат (fit_ratio ~0.02), дифференциация между действиями отсутствует.
- **Перезапущен как #85 на базе max_norm=100 → CLOSED FAIL.** Balanced loss закрыт навсегда (#83 + #85).
- См. `sizing_reports/83-terminal-balanced-loss.md`, `sizing_reports/85-terminal-balanced-loss-v2.md`.


### #84 / #84b — Consolidation + Relapse Mechanism (max_norm=100 baseline)

**Результат: CLOSED FAIL (durability) / воспроизводимость CONFIRMED / механизм relapse НАЙДЕН.**

#### #84 — Консолидационный прогон

Pre-registered: max_norm=100, seed1 до iter 400.

| iter | verdict | raise_freq | unique_anchors | top-2 mass | 0.10 mass | win_rate | fold count |
|------|---------|------------|----------------|-----------|-----------|----------|------------|
| 100 | **OK** | 48.0% | 15 | 28.7% | 10.6% | 41.2 | 44 |
| 200 | **WARN** | 21.5% | 12 | 48.9% | 24.2% | 13.3 | 633 |
| 300 | **WARN** | 7.2% | 7 | 78.3% | 55.5% | 18.0 | 524 |

**Вывод:** max_norm=100 даёт воспроизводимое окно здорового поведения на iter ~100 (первый OK со времён Run C) и улучшает фит в 5-10× (fit_ratio 0.003 → 0.10-0.21). Но паттерн relapse идентичен #73: iter 100 OK → iter 200 WARN → iter 300 полный коллапс. **FAIL по durability.**

#### #84b — Анализ механизма relapse

Инструмент: `tools/relapse_diff.py` — сравнение чекпоинтов 100/200/300 на фиксированном наборе состояний.

**Что ломается ПЕРВЫМ (срез 100→200):**

| Метрика | iter 100 | iter 200 | Δ |
|---------|----------|----------|---|
| advantage fold mean | -7.93 | +0.53 | **+8.47** |
| advantage raise mean | +0.73 | -1.76 | **-2.48** |
| policy raise % | 88.4% | 1.0% | -87.4 |
| q_raise mean | 0.11 | 0.29 | +0.18 |
| q_fold mean | 1.48 | 1.15 | -0.33 |

Advantage/regret-сеть делает ПОЛНЫЙ FLIP fold↔raise за 100 итераций — изменение advantage в 10× больше любых изменений Q. Q-сеть стабильна (raise_lt_fold = 100% на всех чекпоинтах — она ВСЕГДА считала fold > raise, но advantage-сеть компенсировала это на iter 100).

**Smoking gun (из buffer-диагностики):** у ВСЕХ raise-сэмплов в q_buffer `reward = 0.0`, `terminal% = 0`. Raise никогда не видит исхода руки. Без сигнала о выигрыше advantage-сеть физически не может удержать raise > fold.

**Цепная реакция relapse (подтверждена данными):**
1. Advantage-сеть теряет предпочтение raise (H3 — накопление regret) → политика сдвигается в fold
2. Fold-сэмплы в буфере взрываются ×57 (3209 → 184893) → буфер отражает сдвиг стратегии
3. next_strategy.raise_mass падает 0.58 → 0.006 → bootstrap V для raise обнуляется (H2 — bootstrap усиливает)
4. target для raise-сэмплов = 0 → петля замыкается

**Гипотезы:**
- **H3 (regret-накопление): ✓ ПОДТВЕРЖДЕНА** — advantage-сеть ломается первой.
- **H2 (Q-деградация → bootstrap): ⚠ ЧАСТИЧНО** — Q-сеть стабильна, но bootstrap feedback loop УСИЛИВАЕТ коллапс.
- **H1 (буфер fold-terminal): ✗ ОПРОВЕРГНУТА как причина** — буфер реагирует, но не инициирует.

#### Общий итог #84

max_norm=100 **оставлен как baseline** (улучшает фит, не вредит). Механизм relapse доказан — первопричина в advantage/regret-сети, smoking gun — reward=0 у raise. Это закрывает эпоху конфиг-гипотез: дальше только код (→ #86 MC reward propagation).

- `sizing_reports/84-consolidation-run.md` — полный pre-registration отчёт
- `sizing_reports/84b-relapse-mechanism.md` — анализ механизма relapse
- `tools/relapse_diff.py` — инструмент сравнения чекпоинтов

---


### #85 — Terminal-balanced loss v2 (max_norm=100 + balanced α=0.5)

**Результат: CLOSED FAIL.** Хуже #84, даже PARTIAL недостижим.

- **seed1:** iter 100 raise 4.4% (было 48%) — окно уничтожено. Коллапс сменил форму: call-station на 100-200 (call 75-79%), min-raise-spam на 300 (raise 21%, но 0.10 = 67.8% raises).
- **Q дестабилизирована:** взрывы градиентов до ~9500 (×18 vs #84), дрейф q-значений до -4.5.
- **Вместе с #83 (FAIL на max_norm=1.0) — ветка balanced loss закрыта навсегда.**
- **Win_rate стабилен (~36-37), но это дегенеративная call-station стратегия, не успех.**
- См. `sizing_reports/85-terminal-balanced-loss-v2.md`.

### #86 — MC Reward Propagation (одна переменная vs #84)

**Результат: CLOSED PARTIAL.** M1 PASS (smoking gun устранён), M2 PASS (mini-flip без relapse), B FAIL (плато ~21%).

- **seed1:** остановлен на iter ~250. Raise_freq: 22.4% (iter 100) → 14.2% (iter 200) → 21.2% (iter 250 eval) — плато, не relapse.
- **⚠ Кавеат:** 1000 traversals vs 400 у #84 — сравнения с #84 конфаундятся. Замедление итераций в основном отсюда.
- **Главный научный результат:** relapse-распад ОТСУТСТВУЕТ. #84: 48→21.5→7.2 (распад), #86: 22→14→21 (плато). Первый прогон без коллапса.
- **Причина плато:** сырые чипсовые цели (target_term 36-39, std 152) — fit_ratio застревает на 0.14. Нужна нормализация.
- **M2 (relapse_diff 100↔200):** adv_fold mini-flip +0.60 (в 14× меньше #84: +8.47), |δ adv_raise|=0.27<1.0. Буфер fold уменьшился (17.8%→12.3%).
- См. `sizing_reports/86-mc-propagation.md`, `sizing_reports/86-reward-propagation-rfc.md`.

### #87 — Pot-relative Normalization (одна переменная vs #86)

**Результат: CLOSED FAIL-M0.** Масштаб улучшен (target_term 0.7-1.4, q_loss 28), но хвосты не убраны.

- **seed1:** остановлен на ~iter 140. fit_ratio 0.04-0.07 вместо >0.5 — M0 провален.
- **Причина:** деление на preflop pot (2-3) при v_sampled ±hundreds chips → reward сотни «потов» (raise std=28.5, max=+787).
- **Advantage хуже #86:** raise mean=-0.25, max=-0.15 (отрицательный на ВСЕХ состояниях). next_raise_mass=0.
- **Урок:** early-stop по M0 на iter 50 применять БЕЗ исключений.
- См. `sizing_reports/87-pot-relative.md`.

### #87b — Pot+Stack Normalization (одна переменная vs #87, PRE-REGISTERED)

**Дизайн:** `denom = max(pot + stake, 1.0)` — reward гарантированно ∈ [-1, +1]. `q_target_norm: pot_stack_relative`. M0 жёсткий: fit_ratio>0.5 к iter 50, иначе СТОП.

- См. `sizing_reports/87b-pot-stack-relative.md`.

### #88 — Relapse Under MC Propagation (диагностика iter 100→400, test88seed1)

**Результат: Arm B' PRE-REGISTERED.** Диагностика test88seed1 завершена — relapse подтверждён. Гипотеза: обрыв на iter 100 — от `q_target_update_interval=100` (референс синкает каждые 50 train-шагов внутри итерации, проект — раз в 100 CFR-итераций). Arm B': `q_target_update_interval: 100 → 10`.

- **test88seed1 (диагностика):** multi-checkpoint 100/200/300/400 на фиксированных 500 состояниях. Конфиг: kind_filter=false, mask dry_run, max_norm=100, MC propagation baseline.
- **Trajectory (fixed-state raise_freq):** 58.5% → 32.7% → 0.2% → 0.8%. Стратегия умирает за 200 итераций.
- **Trajectory (eval):** 40.8% → 16.0% → 9.2% → 8.1%. Win-rate: 42.2% → 17.5% → 17.2% → 20.6%.
- **Advantage-flip (100→200):** adv_fold Δ=+3.69, adv_raise Δ=−0.22 — в 2.3× и 11× мягче #84. H2+H3 подтверждены.
- **Arm B' (PRE-REGISTERED):** `q_target_update_interval: 100 → 10` — одна переменная. Меняет **только** config.yaml. Основание: референс (`DeepPDCFR-master`, `QValueTrainer.train_model`) синкает target каждые 50 train-шагов внутри итерации на свеже-инициализированной сети. Проектная заморозка на 100 итераций создаёт ступеньку на iter 100. sync=10 — компромисс (не 1 чтобы избежать осцилляции, не 100 чтобы убрать обрыв).
- **Критерии:** M0 (нет ступеньки на iter 100, raise≥30% на 200), M1 (q_raise≥q_fold держится), M2 (OK на ≥3/4 чекпоинтов).
- **Следующие шаги:** M0/M1 PASS + M2 FAIL → Arm C (refit — истинный референс). И C FAIL → #88b (discount α=1.5,+1.5), #88c (recency). M0 FAIL → обрыв не от sync.
- **НЕ трогаем:** `discount_alpha` (отдельный #88b), recency-вес стратегии (отдельный #88c).
- См. `sizing_reports/88-relapse-mc-propagation.md`.

### #89 — Arm B': q_target_update_interval 100→10 (CLOSED FAIL, test89seed1)

**Результат: CLOSED FAIL.** Поведение полностью идентично baseline (test88seed1, sync=100). Sync-интервал не первопричина relapse.

- **test89seed1:** `q_target_update_interval: 100 → 10` (одна переменная, config.yaml). Прогон до iter 400, чекпоинты 100/200/300/400.
- **Eval raise_freq:** 40.7% → 17.8% → 10.4% → 7.8% — идентично test88 (40.8 → 16.0 → 9.2 → 8.1).
- **Fixed-state:** все метрики (raise_freq, q_raise, advantage, anchor dist) **полностью идентичны** test88.
- **M0: FAIL** — raise_freq(200)=17.8% <30%. Ступенька на iter 100 не от sync-режима.
- **M1: FAIL** — raise_lt_fold = 67-88% на всех iter, q_raise никогда не превосходит q_fold.
- **M2: FAIL** — OK только на iter 100, WARN на 200/300/400.
- **Причина:** Q-сеть структурно не может обучиться prefer raise > fold из-за reward-grounding asymmetry (#81). Свежесть target не меняет этот дисбаланс — advantage-сеть на iter 100 компенсирует, но к 200-300 ломается (H3 regret-накопление).
- **Cross-play:** iter 400 vs iter 100: win **12.3%** (raise 1%, call 80%). iter 100 vs iter 400: win **40.5%** (raise 71.5%, 15/15 anchors). Iter 100 exploits iter 400 — деградация доказана.
- **Следствие:** M0 FAIL по дереву #88 → обрыв не от value-target → копать advantage-сторону. Следующий шаг: **Arm C — refit Q с нуля каждую итерацию + best-model** (истинный референс).
- См. `sizing_reports/89-arm-b-q-target-sync.md`.

### #90b — Discount Alignment (DCFR+ Reference Recipe, PRE-REGISTERED)

**Результат: PRE-REGISTERED.** Одна переменная: discount-рецепт advantage-сети → референс DCFR+. Прямой рычаг по H3 (накопление raise-regret).

- **Мотивация:** проект `α=2.0, +1` → forgetting ~1%/iter. Референс DCFR+ `α=1.5, +1.5` → ~4.5%/iter. Сильнее форгеттинг → отрицательный regret не накапливается до flip.
- **Правки:** `config.yaml:11`: `discount_alpha: 2.0 → 1.5`. `deep_cfr.py:2617`: `+ 1` → `+ 1.5` (hardcode).
- **НЕ трогаем:** Q-сеть/sizing/max_norm/discount_gamma.
- **Критерии vs baseline test88:** M0: raise(200)≥30% (было 16%). M1: adv_raise не пересекает 0 к iter 300. M2: OK на ≥3/4.
- **Дерево:** PASS → discount = fix. PARTIAL → Probe 3 refit Q поверх. FAIL → #88c recency-вес.
- См. `sizing_reports/90b-discount-alignment.md`.

### #91 — Discount Alignment Result (CLOSED PARTIAL, test90seed1)

**Результат: CLOSED PARTIAL.** M0 FAIL, но механизм relapse изменился — discount решает H3 (raise-накопление), создаёт H3* (fold-инфляция). Симптомы частично смягчены.

- **test90seed1:** discount α=1.5, знаменатель +1.5. Прогон до iter 300, чекпоинты 100/200/300.
- **Eval:** raise 58.2% → 22.6% → 17.9%. Win 44.8% → 19.9% → 17.7%. Все выше baseline (#88).
- **Fixed-state:** raise 98.8% → 14.3% → 3.9%. Anchor 0.10: 0% → 0% → 100%.
- **Механизм (новый):** adv_raise **растёт** (+0.13→+0.38) — H3 не работает. Но adv_fold взлетает (−1.26→+1.89) — сильный форгеттинг обнуляет старый негативный regret за fold → fold «слишком хорош» → raise проигрывает конкуренцию, не потому что плох.
- **M0: FAIL** — raise(200)=14.3% fixed / 22.6% eval. **M1: PASS*** — adv_raise не пересекает 0. **M2: FAIL** — OK только iter 100.
- **Следствие:** discount — не silver bullet. Промежуточный α — тупик (trade-off H3↔H3*). Next: Arm C — refit Q с нуля каждую итерацию (истинный референс).
- См. `sizing_reports/91-discount-alignment-result.md`.

### #91b — Discount Corrected (α=2.0, +1.5, PRE-REGISTERED)

**Результат: PRE-REGISTERED.** Исправление #91: α возвращён к 2.0 (reference config), только знаменатель +1→+1.5. Одна переменная.

- **Ошибка #91:** использован α=1.5 (class-default референса), но конфиг `VRDeepDCFRPlus.yaml` переопределяет `alpha: 2`. Пере-форгеттинг (forgetting 4.5× быстрее) — вероятная причина fold-инфляции.
- **#91b:** `discount_alpha: 1.5 → 2.0` (config.yaml). `deep_cfr.py:2617` уже содержит `+1.5` — не трогаем.
- **Forgetting при t=11:** d=0.985 (~1.5%/iter) — умереннее #91 (4.5%/iter), сильнее baseline (1%/iter).
- **M0:** raise(200)≥30%. **M1 (ключевое):** adv_fold не раздувается как в #91 (+1.89 на iter 200). **M2:** OK ≥3/4.
- **Дерево:** fold-инфляция спала → discount жив → combine с #92. Или идентично #91 → discount не канал → #92 главный.
- См. `sizing_reports/91b-discount-corrected.md`.

### #92 — Refit Q Baseline / Arm C (PRE-REGISTERED)

**Результат: PRE-REGISTERED.** Одна переменная: процедура обучения q_net → reference QValueTrainer (re-init с нуля каждый iter). Истинный референсный режим.

- **Референс:** `QValueTrainer.train_model` — `self.model = init_model()` каждую итерацию, target-sync/50 шагов, best-model, restore в конце.
- **Проект:** персистентная Q-сеть, 2-5 шагов/iter. Накопление багов между итерациями.
- **Правки:** `config.yaml` +4 параметра (`q_refit_enabled`, `q_refit_steps=200`, `q_refit_batch_size=2048`, `q_refit_sync_every=50`). `deep_cfr.py` — сохранение `_q_input_size/_q_hidden/_q_lr` + новый метод `_train_q_network_refit` + ветвление в `train_q_network`.
- **При MC:** все сэмплы terminal → target=rewards, bootstrap=0 → target-sync инертен. Refit = чистый per-iter regression fit + best-model.
- **Compute:** ×40-100 рост. Рекомендация: сначала iter 200 как быстрый M0-чек.
- **M0:** raise(200)≥30%. **M1:** adv_fold<+1.0 И adv_raise>0. **M2:** OK ≥3/4.
- **Дерево:** PASS → baseline-bias = корень. PARTIAL → combine #91b+#92. FAIL → full reference-alignment.
- См. `sizing_reports/92-refit-q-baseline.md`.

### #93 — Sizing как CFR-подзадача (advantage-нога) + удаление bucket-loss

**Результат: IMPLEMENTED + RESULT (M1 PASS, M0/M2 PARTIAL).** Sizing-коллапс в 0.10 сломан. Regret-matching с min_prob-floor держит diversity на всём прогоне 100→800. Но action-уровень регрессирует (self-suppressing loop).

**Прогон 1 (sizing_cfr_mode + top_bucket inference, 300 trav):**
- **Iter 200 — исторический момент:** 0.10=4.4%, доминанты 0.33(27%), 0.75(21%), 2.50(13%), 2.00(9%), 3.00(9%) — 5 interior мод. Впервые не граничные анкеры.
- **Iter 300:** 0.33(28%) + 1.25(26%) — interior доминанты. Sizing diversity держится, но raise_freq упал с 52% до 10.6%.
- **M1 PASS, M0 FAIL, M2 FAIL.**

**Прогон 2 (sizing_cfr_mode + hierarchical inference, 300 trav, перезапуск):**
- **Iter 200:** модель самовосстановилась — `next_strategy_raise_mass` 0.0→0.497.
- **Iter 300 (пик):** raise_freq=50.9%, win=38.9%, mean_reward=+23.5, 0.10=1.6%. Лучший чекпоинт.
- **Iter 500-800:** рецидив action-уровня, но sizing держится: 0.66(31%), 1.00(23%), 2.25(29%) — 3 interior моды. 0.10 < 4%.
- **Iter 700:** sizing_q.max.mean = **+0.09** (впервые положительный sizing-Q!).
- **mean_reward +17.6 на iter 600** при win_rate=16.7% — модель выигрывает крупно на сильных руках.

**Сравнение с #92 baseline (iter 300):** 0.10 масса: 51-57% → **1.6%**. Доминант: 0.10 → **0.66**. Бакеты: small-biased → **сбалансированы (35/30/35)**.

**Вывод:** sizing-механизм работает. Action-регрессия — ортогональная проблема (нужен оппонент сильнее рандома). См. `sizing_reports/93-sizing-cfr-advantage-leg.md`.


### #94 — Sizing Legality, Boundary All-in, Preflop no-Buckets, Q-bootstrap per-Player

**Результат: IMPLEMENTED + ERRATA FIXED (ждёт прогона R1→R2).** Три фичи за флагами, откатываемо.

**Что сделано:**
- **A1:** `_anchor_availability(state)` — per-state доступность анкеров по стеку/мин-рейзу с `can_raise` guard
- **A2:** Гейтинг — недоступные анкеры зануляются во всех точках сэмплирования (OS-raise, full-raise, `_hierarchical_sizing`, `choose_action`)
- **A3:** `eff_idx_cfr`/`credit_idx` → `sizing_anchor_idx` (убрал nearest-snap argmin)
- **B:** Префлоп без бакетов — плоский regret-matching при `stage==0`
- **A4:** `_encode_state_for_sizing` — расширенный энкод (+30 availability-фич) для sizing-сетей
- **C:** `_next_player_ids` в `QValueBuffer` + per-player bootstrap в `train_q_network`

**Errata (2026-06-17):** `sizing_input_size` был безусловным → холодный старт sizing даже при выключенных флагах. Исправлен на условный: +30 только при `sizing_availability_on_input: true`.

**Новые флаги:** `sizing_anchor_availability_enabled`, `sizing_allin_boundary_enabled`, `sizing_preflop_disable_buckets`, `sizing_availability_on_input`, `q_bootstrap_per_player_enabled`

**План прогонов (OPUS):** R0 (патч размерности) ✓, R1 (baseline, все false) в прогоне, R2 (A-core: availability/boundary/preflop=true) — решающий, R3 (A4: +on_input), R4 (C: per-player Q, propagation=none). См. `sizing_reports/94-szing-legality-boundary-allin.md`.


### #95 — Результаты #94: A-core принят, A4 отклонён, C отложен

**Результат: RESULT.** R2 (A-core) — PASS, принять как baseline. R3 (A4) — FAIL. R4 (C) — неубедительно.

**Матрица прогонов (seed1, raise%/win%/reward, iter 100→400):**
- R1 (baseline): raise 35→49, win 37→24, reward 0→12 — PASS, коллапса нет
- **R2 (A-core): raise 38→52, win 44→26, reward 29→14 — PASS, лучший. Принят.**
- R3 (A4): raise 59→40, win 28→15, reward 6→−3 — FAIL, отклонить
- R4a (C контроль): raise 26→33, win 41→30, reward 10→18
- R4b (C фикс): raise 38→37, win 43→24, reward 13→11 — ≈R4a, неубедительно

**Ключевое открытие: relapse_diff vs eval.** `relapse_diff` меряет greedy advantage-политику — она осциллирует (H3-флип). Eval меряет усреднённую `strategy_net` — она стабильна. Осцилляция текущей политики ≠ несходимость. Здоровье R2 держится на усреднении.

**Принятые флаги (baseline):** `sizing_anchor_availability_enabled: true`, `sizing_allin_boundary_enabled: true`, `sizing_preflop_disable_buckets: true`, `sizing_availability_on_input: false`, `q_bootstrap_per_player_enabled: false`.

**Следующие шаги:** D1 (средняя политика в relapse_diff), D2 (EV raise vs fold vs random), D3 (self-play), D4 (seed2/seed3). Floor ТОЛЬКО если D1+D2 покажут реальную деградацию.

**UPD** К D1–D3 добавить D4 — разложить next_strategy_raise_mass (и raise-regret) по глубине/улице и по «после raise vs после call», чтобы прямо подтвердить: коллапс концентрируется в глубоких/редких узлах. Если да — фикс целится точечно в сэмплирование глубоких raise-линий (targeted exploration), а не глобальный floor.

См. `sizing_reports/95-szing-legality-results.md`.

### #96 — E1: OS Force Full Traversal (action-OS off)

**Гипотеза опровергнута.** OS выключен, регреты точные — relapse идентичен: raise 22→4.3% к iter700.
CFR корректно сходится к пассивному best-response против рандома. Пивот на self-play.
**См.** `sizing_reports/96-os-force-full-traversal-e1.md`.

### #96 — Advantage-regret scale fix (pot_stack + Huber)

**Корень:** advantage-регреты пишутся в сырых фишках (рулит политикой), нормировка была только на Q-ноге.
**Решение:** 4 новых ключа — `advantage_regret_norm` (`pot_stack`/`per_node_max`), `advantage_regret_clip`, `advantage_loss` (`huber`), `advantage_huber_delta`.
**Primary прогон (N1):** `pot_stack` + `huber`, lr=1e-4, 400 iter. **L1:** `advantage_lr=1e-6`, 800-1000 iter.
**См.** `sizing_reports/96-advantage-regret-scale.md`.

### #97 — Результаты N1 / L1 / N1' (все три опровергнуты) + N4 (IMPLEMENTED)

**N1 (pot_stack+Huber):** FAIL — регреты пересушены до O(0.01), коллапс быстрее baseline.
**L1 (lr=1e-6):** FAIL — осцилляции, iter300 OK (raise 26.1%), но дрейф тот же.
**N1' (per_node_max+Huber):** FAIL — per-sample нормировка ломает бутстрап.
**Вывод:** масштаб регрета — НЕ корень. Next: N4 (без clamp/discount).

### #97-N4 — Plain advantage accumulation (IMPLEMENTED, ждёт прогона)

**Результат: PENDING.** Код реализован. Прогон `test96seed1N4` не запущен.

- **Одна переменная vs R2+OS-off:** `advantage_accumulation: plain` — `fresh_target = prev_pred + regret_tensors` без clamp(min=0) и без discount.
- **Новый флаг:** `advantage_accumulation` (default `dcfr_plus`) в `config.py`, `__init__`, checkpoint save/load.
- **Багфикс:** `discount_alpha` добавлен в load whitelist (сохранялся, но не восстанавливался).
- **Мониторинг:** `max|bootstrap_target|` и `max|adv_net_out|` per-iteration (только plain-режим) — отслеживание безграничного роста кумулятива.
- **Метрики/гейты:** `relapse_diff` advantage raise flip (главное), `raise_freq` eval, h2h late-vs-early.
- **Чтение исхода:** raise держится → clamp+discount был причиной. Коллапс остался → корень в value/динамике. Advantage взрывается → нужен discount.
- **Откат:** `advantage_accumulation: dcfr_plus`.
- **См.** `sizing_reports/97-n1-l1-n1prime-results.md`.

### #97-диагностика: relapse — variance-driven, не EV

**Результат: DIAGNOSIS.** `reward_stats_by_action` test97seed1 + TB `PerformanceRaiseFreq` / `LossAdvantage`.

- **Reward-диагностика:** iter 300 raise mean reward = **+0.0074 (ВЫСШИЙ среди действий)**, но raise_freq = 5.4%. Raise безубыточен/прибылен, но имеет наибольшую дисперсию (std 1.27). Политика абандонит прибыльное высокодисперсное действие.
- **Вывод:** relapse — артефакт обучения регрета (variance), а НЕ −EV. Объясняет провал всех accumulation-вмешательств (N1/N1'/L1/N4): чинили правило накопления, корень — шум сигнала raise.
- **TB-анализ:** `LossAdvantage` MSE ~40k–55k (таргеты ~±220 фишек). Advantage-сеть фитит регреты масштаба сотен фишек → тонкая структура raise-vs-fold тонет → грубая политика бросает raise.
- **Корень:** СЫРОЙ масштаб регрета на уровне фита (loss ~50k). Per-iter нормировки (N1/N1') ломают бутстрап. Нужен ГЛОБАЛЬНЫЙ фиксированный масштаб `cf_regret` (одна константа), гомогенный с бутстрапом.
- **Направление:** variance reduction регрета raise (усиленный baseline / больше сэмплов на raise-узлах / control variates), не accumulation.


### #99 — Random Agent + Per-Iteration Traversing Player + Optional Q (v1)

**Результат: IMPLEMENTED + RESULT (N5_3 — лучший прогон по стабильности).** Добавлен random-агент на одном сиденье, traversing_player ротируется по итерациям (не травесам), стратеджи учится только на данных текущего наблюдаемого. Q сделан опциональным (action-level выкл, sizing вкл).

**Архитектура (5+1):** 5 advantage-сетей на игроков 0-4, RandomAgent на seat 5. `traversing_player = iteration % 5`. Button случайный каждый травес. 1 общий strategy-буфер — чистится каждую итерацию, содержит данные только текущего traversing. Strategy-сеть общая — учится каждую итерацию.

**N5_3 результат (98seed1N5_3):** raise держится ~16-22% на всех 1600 итерациях — **дрейфа нет**. Ключевое отличие от N5/N5_2: raise не сползает. H2H: late > early (+1.3 reward). Стабильность достигнута. **`sizing_target_ema_enabled: false`** — сырые per-state sizing-таргеты без EMA-бакетизации (симметрично action-сети).

**Что изменилось:** `self.num_trainable_players = 5`, random-агент ветка в `cfr_traverse_multi`, `random_agent` через instance-переменную, Q guard'ы в `__init__`/training/checkpoint/tools. Багфикс: `random_agent` после `depth` в сигнатуре. См. `sizing_reports/99-random-agent-per-iteration-strategy.md`.

### #99v2 — CFR Purity: Strategy-Advantage Decoupling + Per-Player Sizing + Opp Sizing Fix + Adaptive Batch

**Результат: IMPLEMENTED.** Архитектурная чистка после v1. Исправлены нарушения CFR-чистоты и training efficiency.

**Ключевые изменения:**

1. **Strategy-буферы откачены к синглу** (1 общий вместо 5 per-player) — сеть одна, буфер один.
2. **Opponent sizing через `_hierarchical_sizing(use_q=False)`** — оппоненты больше не используют `strategy_sizing_net`. Strategy-сеть — чистый distillation output, не влияет на CFR-дерево.
3. **Sizing Q только для traversing** — `_hierarchical_sizing(use_q=True)` для hero, `use_q=False` для opp.
4. **Per-player advantage sizing** — `advantage_sizing_nets[0..4]` (5 шт), `sizing_advantage_buffers[0..4]` (5 шт), `sizing_optimizers[0..4]` (5 шт). Каждый игрок имеет независимую sizing-ногу с CFR-регретами.
5. **Адаптивный batch size** — `min(batch_size, n)` вместо `len < batch_size → return 0`. Strategy тренируется с первой итерации.

**Итоговая архитектура:**

| Компонент | Сеть | Буфер |
|---|---|---|
| Advantage action | 5 per-player | 5 per-player |
| Advantage sizing | 5 per-player | 5 per-player |
| Strategy | 1 общая | 1 общий |
| Strategy sizing | 1 общая | 1 общий |
| Sizing Q | 1 общая | 1 общий |

**Code hygiene (#99v2 дополнение):** коммент buffer asymmetry в `prepare_iteration`, явный `player_id` во всех вызовах `_hierarchical_sizing`. **Попытка убрать re-encode откачена** — сломала обучение (max_depth упал до 10), re-encode необходим для правильной размерности тензора. `player_id` keyword/опциональный.

**#100 — Config ablations поверх N5_4:** A1 `sizing_advantage_buffer_size: 100000`, A2 `discount_gamma: 1.0`, A3 `advantage_epochs: 20`, A4 `strategy_buffer_reservoir: true` (НЕ чистить strategy_buffer — деплой-среднее = CFR-среднее по истории). См. `sizing_reports/99-v2-cfr-purity-architecture-fixes.md`.

**N5_4 результат:** raise ~20-25% на всех 1600 итерациях — **дрейфа нет, чуть лучше N5_3**. Reward стабильно выше (пики 15-16 vs 12-15). H2H late > early (+1.1 reward, win 18.1% > 16.7% baseline). **Самый чистый и стабильный результат серии.** Per-player sizing + strategy=distillation → CFR-чистота без потери стабильности.

### N5_3 vs N5_4 Timeline

| iter | N5_3 raise% | N5_3 reward | N5_4 raise% | N5_4 reward |
|------|------------|------------|------------|------------|
| 100 | 19.1 | 6.4 | **22.5** | **9.1** |
| 300 | 19.3 | 6.8 | **25.0** | **9.1** |
| 500 | 21.6 | 12.0 | **25.5** | **16.0** |
| 800 | 19.1 | 9.2 | **24.0** | **14.2** |
| 1100 | 18.8 | 14.7 | **19.5** | 12.0 |
| 1400 | 22.2 | 10.2 | **22.2** | **15.5** |
| 1600 | 16.5 | 9.6 | **23.0** | **12.2** |

> **N5_4 ≥ N5_3 по всем метрикам. #99v2 правки не сломали стабильность — улучшили.**

---

### #100 — Sizing Target Sparsify: коллапс деплой-сайзинга через top_per_bucket=1

**Результат: FIXED (revert).** `top_per_bucket: 5→1` заострял таргеты до усреднения в резервуаре → деплой-стратегия схлопывалась до 1-2 анкеров.

- **Баг:** коммит 34b362d («69») сменил `sizing_target_top_per_bucket: 5→1` — sparsify оставлял внутри каждого бакета только top-1 анкер. Резервуар усреднял argmax'ы вместо миксов → concentration (top-2 mass 61% / 56%).
- **Механизм:** Deep CFR average-strategy: резервуар обязан усреднять истинную смешанную per-state стратегию. При `top_per_bucket=1` таргет схлопывается до ≤3 точечных масс ещё до усреднения. Усреднение argmax'ов ≠ усреднение микса.
- **Фикс:** коммит 1401075 — revert `top_per_bucket: 5` (no-op: все 5 анкеров в бакете), `hidden_size: 128` (proven baseline).
- **Подтверждение:** прогон `test98seed1N5_5_3`: concentration-варн исчез, sizes=15/15 на всех чекпоинтах, raise_freq 20→30%, win_rate 11.5→17.3. LossStrategySizingAnchor упал вдвое (0.91→0.51).
- **WP-4 (test102):** `bucket_loss_weight: 0→0.5` — эксперимент, не улучшение над _3. Конфликт градиентов bucket vs anchor в shared base. Оставлено как эксперимент.
- См. `sizing_reports/100-sizing-target-sparsify-collapse-fix.md`.

---

### #101 — Inference Threshold Masking (action + sizing)

**Результат: IMPLEMENTED.** Пороговая маска на инференсе: отсечение опций с вероятностью <5%, ренормализация.

- **Проблема:** softmax даёт всем 15 анкерам и 4 действиям ненулевые вероятности. Микро-вероятности (1-5%) размывают стратегию и добавляют дисперсию в eval.
- **Решение:** `inference_min_action_prob: 0.05`, `sizing_inference_min_prob: 0.05`. После softmax, до сэмплинга: `keep = probs >= threshold`, ренормализация с сохранением пропорций.
- **Только инференс:** тренировка (CFR-траверс, буферы, обучение сетей) не затронута. Откат: оба параметра в 0.0 → мгновенный возврат.
- **Eval fix:** `run_eval_games` использовал свой инференс-путь (regret-matching через `advantage_sizing_net`), а не `choose_action`. Eval переведён на те же сети/механизм что и деплой (strategy_net → softmax → порог 5%). Расхождение устранено полностью.
- См. `sizing_reports/101-inference-threshold-masking.md`.

---

### #104 — Пакет накопления sizing-регретов (RESTORED)

**Результат: RECONSTRUCTED.** Оригинальный #104 утрачен; восстановлен из истории переписки 2026-08-12/13.

- **Архитектурный контекст:** проект — VR-DeepDCFR+, где fresh_target = target_net(s)·discount + cf_regret. Накопление регрета живёт в весах сети через бутстрап, а не в буфере.
- **Пакет накопления (§2.6):** ключевой механизм записи sizing-регретов в буфер с clamping, маскингом кредитуемых анкеров и центрированием previous. Восстановлен как источник решений для пункта 7 плана.
- **Что утрачено:** точные номера строк, таблицы исходных измерений, §6.1 замеры мультиборда.
- См. `sizing_reports/104-RESTORED-accumulation-package.md`.

---

### #106 — Профиль итерации и шаги обучения sizing-ноги

**Результат: DIAGNOSIS.** Три исходных вывода #106 опровергнуты замерами.

- **Профиль итерации:** обучение — 6% времени (64 с из 1013 с). Гипотеза «дорого обучение на растущем буфере» опровергнута.
- **Дерево:** 281 узел на траверс, max_depth=32, depth_hits=0 — здоровое дерево. Гипотеза об экспоненте от ветвления по бакетам неверна.
- **Диспетчеризация PyTorch:** не доминирует (0.051 мс на eager-вызов). Разрыв 2-7× объясняется конкуренцией BLAS-потоков.
- **Sizing-обучение:** 10 шагов/итерацию — практически бесплатно (~0% времени). Шаги можно поднимать без значительного замедления.
- См. `sizing_reports/106 — Профиль итерации и шаги обучения sizing-ноги.md`.

---

### #107 — Профилирование и корневая причина плоскости sizing-регретов

**Результат: DIAGNOSIS + FIXES.** Три вывода #106 опровергнуты. Корневая причина плоскости найдена.

- **BLAS-потоки:** главный источник накладных расходов. `set_num_threads(1)` на обход даёт −19.2% мкс/узел изолированно.
- **Второй `_anchor_availability`:** удалён дублирующий вызов (2.832 → 1.832 на sizing-узел).
- **§2.8 — гейты отменены:** `ptp` — regret_matching инвариантен к равномерному масштабу. `sign_agreement < 0.5` — мерило сеть против таргетов, а не шум таргетов. `TV/RATIO` против одного визита — leave-one-out потолок 1.275 > 1.0, метрика не разрешает разницу.
- **ICC = 0.4512** — на тот момент казалась достаточной для гейта. **Позже опровергнуто (#112-Итог, #113 §15-16):** ICC по дубликатам (0.5% буфера) даёт потолок R² = 0.0004 при фактическом holdout R² = 0.041 — на два порядка мимо. ICC как потолок мёртв, в списке мёртвых метрик #112-Итог.
- **Sizing-регреты плоские по построению:** 48% строк single-credit (r_i = 0 тождественно). std(R_i) = 0.104 в группе 1 — бутстрап-член, не сигнал.
- См. `sizing_reports/107-profiling-and-sizing-regret-flatness.md`.

---

### #107Result — Исполнение #107 §6.2 и трек 1

**Результат: EXECUTED.** Оптимизации применены, трек 1 зафиксирован.

- **Совокупный результат правок 1+2:** −23.7% мкс/узел на iter 5-10.
- **Трек 1 (steps 10→300):** гейт зафиксирован до запуска. Позже отменён в #107 §2.8 п. 10-17.
- **Пункты §6.2:** 1-3 сделаны, 4-9 не начаты или отложены.
- См. `sizing_reports/107Result-execution-and-track1.md`.

---

### #108 — СПРАВОЧНИК: две статьи о параллелизации CFR

**Результат: REFERENCE ONLY.** Не план работ, нечего выполнять.

- **Статья A (Tsinghua):** Real-Time Parallel CFR — конвейер из 7 стадий, CPU+GPU. Ускорение 3.3-3.4×.
- **Статья B (CMU, Sandholm):** Parallelizing CFR — CFR как линейная алгебра на GPU. Ускорение до 4×10³ против C++ OpenSpiel.
- **Вывод:** A неприменима к нашему тренеру (batch=1 на узле). B требует табличного представления игры. Обе — для справки, не для исполнения.
- См. `sizing_reports/108-REFERENCE-parallel-cfr-literature.md`.

---

### #109 — Update-Equivalence Framework против PBS-солвера

**Результат: ANALYZED, РЕШЕНИЕ НЕ ПРИНЯТО.** Решать после S3b.

- Статья утверждает, что PBS-планирование плохо масштабируется в multiway: носитель PBS растёт с объёмом непубличной информации.
- Альтернатива: MDS/MMDS — планирование в пространстве диапазонов, а не решающих точек.
- **Вывод:** кандидат в замену рантайм-архитектуры солвера целиком. Не переписывать сейчас.
- См. `sizing_reports/109-update-equivalence-mmds-vs-pbs-solver.md`.

---

### #110 — «Деградация ноги» = артефакт; структурный дефект в сборке таргета

**Результат: DIAGNOSIS.** «Деградация» — артефакт чтения незначимых разниц как тренда.

- `blueprint − uniform` по итерациям v8: первые три точки — шум, растёт не деградация, а значимость (stderr падает).
- Статика: RATIO ≈ 1.0-1.2 на диапазоне 33k-1M кумулятивных узлов, без тренда.
- **Следствие:** накопительные гипотезы (бутстрап ×19, петля argmax, Q-смещение) отменены по построению — траектории «из плюса в минус» не существует.
- См. `sizing_reports/110-leg-degradation-artifact-and-ev-desync.md`.

---

### #111 — Баг харнесса: четыре руки мерили частоту рейза, а не сайзинг

**Результат: BUG FIXED.** Все замеры `sizing_oracle_ab.py` (v8→v13) признаны невалидными.

- **Баг 1:** oracle/uniform всегда рейзили на легальных узлах (841/813/827 raises против 156 у blueprint). `blueprint − uniform` мерило разницу в частоте рейза, а не качество выбора размера.
- **Баг 2:** ev считался по-разному в оракуле и конвейере — разные величины.
- **Починка:** `play_hand` теперь использует `choose_action` для решения рейзить/нет, sizing_source влияет только на размер рейза. Добавлен `ev_mode=weighted`.
- См. `sizing_reports/111-oracle-harness-raise-frequency-bug.md`.

---

### #112 — Мёртвый параметр steps; 4000 шагов не сработали; план стратификации (v16)

**Результат: CLOSED.** Стратификация не сработала.

- **Баг:** `sizing_anchor_train_steps_per_iteration` не читался кодом — `getattr`-цепочка молча падала на `pg_train_steps_per_iteration = 10`. Параметр был мёртв всю серию.
- **v15 (4000 шагов):** RATIO стабилизировался на ~1.11, не падает к потолку. `blueprint − uniform` = шум.
- **v16 (стратификация 50/50):** `fresh_share = 0.5`, но `fresh_loss` застрял на ~0.6-0.7. RATIO 1.155-1.240, не ниже 1.0.
- **LOO-потолок 1.275:** прежний «потолок 0.63» был с утечкой. Сеть v16 стоит на 1.15 — внутри скобки [0.72, 1.275].
- **Вывод:** TV/RATIO — третья мёртвая метрика серии. Единственный живой гейт — EV через `oracle_ab`.
- См. `sizing_reports/112-dead-steps-param-and-v16-stratification.md`.

---

### #112-Итог — Что доказано, что мертво

**Результат: SUMMARY.** Сводка по v11-v16 и офлайн-замерам.

- **Единственный честный результат:** `blueprint − uniform` = ноль на всех версиях и итерациях. Sizing-нога не отличается от равномерного выбора анкеров.
- **Кладбище метрик (пять):** ptp, sign_agreement, TV/RATIO, EV оракула, ICC-потолок. Все отменены — мерили не то.
- **Оракул завышен:** hindsight в `_rollout` — видел продолжение np.random и будущий борд.
- **Что про обучение выяснено:** сеть способна выразить таргет (in-sample RATIO 0.13). Обобщение не улучшается. Объём сигнала не узкое место. fresh_loss у шумового пола.
- **Вырождение таргетов:** 48% строк single-credit — r_i = 0 тождественно.
- **Вывод:** измеримого запаса у S6 нет. Приоритет вниз. Дальше 10k traversals и солвер.
- См. `sizing_reports/112-Итог по sizing-ноге что доказано, что мертво.txt`.

---

### #113 — Оракул мёртв, гейт откалиброван, IW-фикс, K-диагностика, накопление, представление, таргет v19

**Результат: MULTI-TOPIC.** Ключевой документ фазы S6 (v17→v19).

- **§1:** Оракул мёртв как сертификатор — утечка борда починена (независимый шафл остаточной колоды). После починки oracle_mc−uniform не реплицируется.
- **§2:** Гейт откалиброван. always_min−always_max: 6 замеров, значим 1. Динамический диапазон НЕ реплицирован. План 179: честный замер через min_available/max_available + фильтр K_avail≥2.
- **§4:** IW-фикс (v17) — exploration получал N вместо N/ε. Починено, клип впервые активен на 7.9% строк.
- **§5:** K-диагностика — K_avail=1 на 36.7% узлов, K_credited=1 на 47.4%.
- **§6:** Маргинальное разложение — голова = положительно-смещённая константа со слабой дискриминацией.
- **§7-8:** Причина — ratio_L 1.28→2.9 при ratio_R≈1.0. Общая компонента накапливается когерентно. Приз ≤ +0.65 фишки.
- **§9-10:** Вариант A (центрирование previous по кредитуемым). Pre-reg v18.
- **§11:** Результат A — полулечение (ratio_L ~1.4 вместо 2.9). net−marginal вырос вдвое. Линейка сжалась +0.65→+0.37.
- **§12:** Дискриминация входа — в 157 входах НЕТ информации, различающей размеры. Дело в представлении, не в CFR-механике.
- **§13:** B реализован (центрирование по доступным). НЕ ЗАПУСКАТЬ.
- **§14:** Путь представления — sizing_availability_on_input → 157→187 входов. Приоритет снижен §16.
- **§15:** Потолок R² = ICC таргета. corr(визит1,визит2) ≈ 0.001-0.077, R²-потолок ≤ 0.006. ICC по дубликатам ЗАНИЖАЕТ потолок (0.5% буфера).
- **§16:** Learnability чистого MC — offdiag перевернулся (121/124→18/124). HOLD R²=+0.034. Основной шум от Q, не от chance.
- **§17:** Правка таргета v19 (sizing_cfr_regret_grounded_ev). Within кредитуемых падает ~30x. Diag отделяется. Pre-reg + результат.
- **§18 (план 179):** Измерение динамического диапазона. Контрольные руки починены (min_available/max_available). Фильтр K_avail≥2 дал 9140/32000 раздач. Диапазон +0.27±0.25 (t≈1.1) — не реплицируется. Приза при 1000 traversals нет. Дальше 10k traversals.
- См. `sizing_reports/113.md`.

---

## Глоссарий ключевых концепций

| Термин | Значение |
|--------|----------|
| **source=0** | On-policy real rollout — Q-target от реального исхода раздачи |
| **source=1** | Probe — random exploration sizing |
| **source=2** | One-step lookahead — bootstrap через `q_net` на post-raise состоянии |
| **top_per_bucket** | Сколько анкеров выживает после `_sparsify_sizing_target_by_buckets` внутри каждого bucket (small/medium/large по 5 анкеров) |
| **kind=NORMAL** | Выбранный sizing после `_resolve_effective_sizing` остался без изменений |
| **kind=MIN_RAISE** | Выбранный sizing был меньше min_raise и форсирован вверх |
| **kind=ALL_IN** | Выбранный sizing превысил стек и форсирован в all-in |
| **deadly triad** | Self-reinforcing loop: policy подавляет raise → bootstrap undervalues raise → Q падает → policy ещё сильнее подавляет |
| **train/inference index mismatch** | Инференс делает argmax по `selected_anchor_idx`, но target пишется в `effective_anchor_idx` (после ремапа). Q[selected=3.0] не получает кредит за выбор. |
| **sizing collapse** | Модель использует 1-2 анкера из 15, остальные получают 0 градиентов → feedback loop |
| **barbell collapse** | Два анкера (обычно 0.10 и крупный) делят всю массу, остальные — 0 |
| **reward-grounding asymmetry** | Raise никогда не получает terminal reward напрямую (все 86k+ сэмплов bootstrap-only). Fold/call изредка получают terminal до +988/+1000. Raise TD-target ограничен bootstrap-ceiling ~+0.42. Это создаёт асимметрию в action-Q credit assignment. |

---

## Текущее состояние прогонов

| Прогон | bootstrap_target_net | src2_weight | source2 target | raise_freq | best_sizes | verdict |
|--------|---------------------|-------------|----------------|------------|------------|---------|
| A (test3-4) | true | 1.0→0.25 | exact 0 (q_head dead) | 27-29% | 0.10=65%, 1.75=23% | OK |
| B (test6) | false | 1.0→0.25 | ~0.002 ±0.05 (Q ОЖИВЛЕНА) | 19.7% | 0.10=100% | WARN |
| C (test7) | false | 0.0 | пишется, не в loss | 37.6% | 0.10=424/499, 15/15 unique | OK |
| D (test8) | false | 0.0 | пишется, не в loss + mask + kind-filter + replay clear | 2.6% | 0.10=499/499, 8/15 unique | WARN |
| A (dry-run) | false | 0.0 | C-like + counterfactual diagnostics | 36.2% | 0.10=424/499, 15/15 unique | OK* |
| Full D + diag | false | 0.0 | mask ON + kind ON + clear ON + all diag ON | 2.7% | 0.10=499/499, 10/15 unique | WARN |
| Isolation (test76seed1) | false | 0.0 | **mask OFF**, kind ON, clear ON, diag ON | 27.7% | 0.10=499/499, 14/15 unique | WARN (conc 83%) |
| test77seed1 (**#78**) | false | 0.0 | **mask OFF, kind OFF**, clear ON, diag ON | 13.0% | 0.10=499/499, 15/15 unique | WARN (conc 84%, low_raise 13%) |
| test78seed1 (**#79**) | false | 0.0 | **mask OFF, kind ON**, clear OFF, diag ON | 4.6% | 0.10=499/499, 9/15 unique | WARN (raise 4.6%<5%, conc 89%) |
| test79seed1 (**#80**) | false | 0.0 | **mask OFF, kind OFF**, clear ON, diag ON, **B1 ON** | 2.1% | 0.10=499/499, 5/15 unique | WARN (**FAIL** raise 2.1%<5%, conc 89%) |
| **test80aseed1 (#82)** | false | 0.0 | **≈test79seed1**, q_grad_clip_max_norm=**10** | ? | \|q_pred\|_term=1.30 | **B** (<2) |
| **test80bseed1 (#82)** | false | 0.0 | **≈test79seed1**, q_grad_clip_max_norm=**100** | ? | \|q_pred\|_term=1.75 | **B** (<2) |
| **test81seed1 (#83)** | false | 0.0 | **≈test79seed1**, balanced_loss α=0.5 | 3.0% | 5/15 unique | WARN (закрыт) |
| **test84seed1 (#84)** | false | 0.0 | **≈test80b**, max_norm=**100**, iter **300** | FAIL | WARN | **CLOSED FAIL** |
| **test84seed2 (#84)** | — | — | не запускался | — | — | **SKIPPED** |
| **test88seed1 (#88)** | false | 0.0 | **MC propagation baseline**, multi-checkpoint 100/200/300/400 | 40.8→16.0→9.2→8.1% | 0.10=56-65%, 6-15/15 unique | WARN (relapse iter 200+) |
| **test89seed1 (#89)** | false | 0.0 | **Arm B': q_target_update_interval=10**, iter 400 | 40.7→17.8→10.4→7.8% | 0.10=60-66%, 6-15/15 unique | **CLOSED FAIL** (идентичен baseline) |
| **#88 Arm B' (PRE-REGISTERED)** | false | 0.0 | **test88seed1 + q_target_update_interval=10**, iter 400 | TBD | TBD | TBD → **закрыт, см. #89** |
| **test90seed1 (#91)** | false | 0.0 | **discount α=1.5,+1.5, Q ON**, iter 300 | 58.2→22.6→17.9% | 0.10=64%, 0.66=25% (iter100), 10-15/15 | **CLOSED PARTIAL** (механизм сменился, коллапс остался) |
| **test90bseed1 (#91)** | false | 0.0 | **discount α=2.0,+1.5, Q OFF**, iter 100 | 32.2% | 0.66=dominant, 15/15 | OK (слабее test90) |
| **test90cseed1 (#91)** | false | 0.0 | **discount α=1.5,+1.5, Q OFF**, iter 100 | 24.9% | 0.66=dominant, 15/15 | WARN (худшая комбинация) |
| **#91b (PRE-REG)** | — | — | superseeded, см. #92 | — | — | **ЗАКРЫТ** (не запущен) |
| **test91seed1 (#92)** | true | 0.0 | **multi-agent, Q OFF, α=1.5**, 204 trav, iter 100 | 12.4% | 0.10=100%, 15/15 | **FAIL** (Q OFF + multi = нежизнеспособно) |
| **test92seed1 (#92)** | true | 0.0 | **multi-agent, Q ON, α=1.5**, 400 trav, iter 300 | 43→63→52% | 0.10=51-57%, 0.75=18-21%, 13-15/15 | **ACTIVE** (первый прогон без relapse!) |
| **test93seed1 (#93, прогон 1)** | true | 0.0 | **sizing_cfr_mode=true, top_bucket, Q ON, α=1.5,** 300 trav, iter 200 | 51.7% | 0.33=27%, 0.75=21%, 5 мод | **M1 PASS, M0 PARTIAL** |
| **test93seed1 (#93, прогон 2)** | true | 0.0 | **sizing_cfr_mode=true, hierarchical, Q ON, α=1.5,** 300 trav, iter 300 | **50.9%** | 0.66=38%, 1.00=16%, 2.25=20% | **M1 PASS, пик на 300** |
| **test96seed1N4 (#97-N4)** | false | 0.0 | **plain accumulation, R2+OS-off, lr=1e-4, MSE,** 400 trav | TBD | TBD | **PENDING** |
| **test98seed1N5_3 (#99)** | false | 0.0 | **v1: random agent + per-iter traversing + EMA off,** 400 trav, iter 1600 | 16-22% | 0.10 dom, 15/15 | **OK — ПЕРВЫЙ БЕЗ ДРЕЙФА** |
| **test98seed1N5_4 (#99v2)** | false | 0.0 | **v2: CFR purity + per-player sizing + opp fix + adaptive batch,** 400 trav, iter 1600 | **20-25%** | 0.10 dom, 15/15 | **OK — ЛУЧШИЙ ПРОГОН** |

> **Прогон B:** Q-сеть на source2 оживлена (q_canary_abs_sum 0→6.25, q_a* ненулевые), но слабый source2 с весом ~0.81 размывает sizing-Q.
> **Прогон C:** source2 исключён из loss → **восстановление подтверждено** (OK, raise_freq=37.6%, win_rate=42.1). Главный profitable baseline.
> **Прогон D:** legal-anchor mask, kind-filter, replay clear регрессировали: raise_freq 37.6%→2.6%, raise_mass 0.406→0.
> **Прогон A (dry-run):** C-like (raise_freq=36.2%, win=40.5%); counterfactual mask-diff доказал mask culprit (29-42% removed mass).
> **Прогон Full D + diag:** подтвердил applied mask 20-53% removed mass, opponent raise исчез, replay clear не primary. → #76.
> **Isolation (test76seed1):** GREY/PARTIAL. Mask OFF убрал catastrophic collapse (raise 2.7→27.7, opponent 0→7951, actionQ 27.5k→233.7k), но C-recovery не достигнут (27.7 vs 37.6, win 37.7 vs 42.1, top2 83%). kind-filter=drop режет 42% known live mass (267 MIN_RAISE + 293 ALL_IN dropped vs 772 NORMAL kept). See: `77-isolation-result-grey.md`.
> **test77seed1 (#78 second control, EXECUTED):** UNEXPECTED regression. kind_filter=drop гипотеза **опровергнута**. kind_filter — safety filter, replay_clear — exposer, root cause — #74-v4 index mismatch. GPT+OPUS: 4-уровневая иерархия каналов. See: `78-test77secondcontrol-unexpected-regression.md`.
> **test78seed1 (#79 config control, EXECUTED):** BAD — close to Full-D collapse (raise 4.6%, actionQ.raise 27.5k, opponent 0). replay_clear-as-exposer гипотеза не подтвердилась: kind ON/clear OFF — худшая клетка 2x2 матрицы. Config-only fix исчерпан. Next: B1 selected-credit experimental fix под флагом. See: `79-test78control-clear-off-bad.md`.
> **test79seed1 (#80 B1 stress-test, EXECUTED):** PRE-REGISTERED FAIL — raise_freq=2.1% (<20%), B1 input-side fix активен но не сдвинул поведение. #81 diagnostics: H1 CONFIRMED — raise 0 terminal samples, TD-target ceiling +0.418; fold/call terminal up to +988/+1000. H2 REJECTED — fold legal 97.4%. Action-Q fold>raise — реальная reward-grounding асимметрия, не diagnostic artifact. Next: action-Q n-step/MC delayed reward propagation. See: `80-...md` §9, `81-action-q-role-legal-diagnostics.md`.
> **test88seed1 (#88 MC propagation relapse, EXECUTED):** Multi-checkpoint 100/200/300/400 на фиксированных 500 состояниях. MC propagation задерживает но не устраняет relapse: iter 100 OK (raise 40.8%, 15/15), iter 200 WARN (raise 16%, 0.10=55%), iter 300-400 полный коллапс (raise <10%). Advantage-flip в 2-11× мягче #84, но политика гиперчувствительна. H2+H3 подтверждены. Smoking gun: reward=0 у raise-сэмплов. See: `sizing_reports/88-relapse-mc-propagation.md`.
> **test89seed1 (#89 Arm B' result, EXECUTED):** **CLOSED FAIL.** `q_target_update_interval=10` — поведение полностью идентично test88 (sync=100). Все fixed-state метрики совпадают. M0/M1/M2 провалены. Sync-интервал не первопричина relapse. Next: Arm C (refit Q с нуля каждую итерацию). → `sizing_reports/89-arm-b-q-target-sync.md`.
> **#88 Arm B' (PRE-REGISTERED):** `q_target_update_interval: 100 → 10` (config.yaml). → **CLOSED, результат в #89.**
> **#90b Discount Alignment (PRE-REGISTERED):** `discount_alpha: 2.0→1.5`, `+1→+1.5`. → **CLOSED, результат в #91.**
> **#91 Discount Result (EXECUTED):** test90seed1 iter 300: M0 FAIL. Discount решил H3 но создал H3* (fold-инфляция). Вероятная причина: α=1.5 вместо reference α=2.0 → пере-форгеттинг. Исправлено в #91b. → `sizing_reports/91-discount-alignment-result.md`.
> **#91b Discount Corrected (PRE-REGISTERED):** α=2.0, +1.5 — правильный reference recipe. Ключевое: adv_fold не должен раздуваться как в #91 (+1.89). → `sizing_reports/91b-discount-corrected.md`.
> **#92 Refit Q / Arm C (PRE-REGISTERED):** `q_refit_enabled: true` — re-init Q с нуля/iter + target-sync/50 + best-model. ×40-100 дороже. Сначала iter 200. → `sizing_reports/92-refit-q-baseline.md`.
> **#93 Sizing CFR Advantage Leg (IMPLEMENTED + RESULT):** `sizing_cfr_mode: true` + `hierarchical` inference. M1 PASS: 0.10 подавлен (1-4%), interior доминанты (0.66/1.00/2.25), sizing-Q впервые положительный (+0.09). M0/M2 PARTIAL: action-регрессия после iter 400. Два прогона на test92seed1 (top_bucket + hierarchical). → `sizing_reports/93-sizing-cfr-advantage-leg.md`.
> **#96 Advantage-regret scale (IMPLEMENTED):** 4 новых ключа — нормировка регретов (`pot_stack`/`per_node_max`), клип, Huber loss, huber_delta. Код: врезка перед `_adv_buffer.add` + Huber в `train_advantage_network_multi` + checkpoint save/load. N1 = `pot_stack`+`huber`, 400 iter. L1 = lr=1e-6, 800-1000 iter. → `sizing_reports/96-advantage-regret-scale.md`.
> **#97 Results N1/L1/N1' (CLOSED — все три опровергнуты):** N1 (pot_stack+Huber): регреты пересушены O(0.01) → голодный коллапс. L1 (lr=1e-6): осцилляции вместо коллапса, iter300 OK (26.1%), но итоговый FAIL — дрейф тот же, медленнее. N1' (per_node_max+Huber): масштаб O(1) удержан, но per-sample нормировка несовместима с бутстрапом → политика вырождена с iter100. **Масштаб регрета — НЕ корень relapse.** Next: N4 (без clamp/discount как в оригинале). → `sizing_reports/97-n1-l1-n1prime-results.md`.

---

## Ключевые вехи (что победили и что нет)

| Веха | Где | Статус |
|------|-----|--------|
| Коллапс в 0.10 через sparsify | #61 | **ПОБЕЖДЁН** (top_per_bucket 1→2) |
| q_net instability (loss 1000+) | #66 | **ПОБЕЖДЁН** (reward scaling + target net) |
| Sizing diversity временный (iter 300) | #67, #68 | **ЧАСТИЧНО** — держится до iter 300, relapse к 400/500 |
| Barbell-collapse сломан (iter 200) | #69 | **ЧАСТИЧНО** — raise_freq 2%→28%, но не держится к iter 300+ |
| Source2 = 0 (мёртвая Q-сеть) | #74-v6 | **ПОБЕЖДЁН** — q_canary 0→6.25, q_a* ненулевые через `bootstrap_target_net=false` |
| Deadly triad action-Q (iter 100) | #73 | **ЧАСТИЧНО** — разорван на iter 100 (raise_freq 47.9%), но возвращается к iter 200 |
| Action-Q self-suppressing loop | #72, #73, #74 | **ОТКРЫТ** — mix 0.15 даёт отсрочку до iter 300, не фикс |
| Train/inference index mismatch | #74-v4 | **ОТКРЫТ** — 88% opponent_path, ALL_IN/MIN_RAISE до 49% |
| Illegal bucket contamination | #74-v3, #74-v8 | **РЕГРЕССИЯ** — Run D fix регрессировал raise_freq 37.6%→2.6%, root cause — предмет #75 v5 diagnostics |
| Mask vs Replay-Clear Root-Cause | #75 v5 | **IMPLEMENTED** — instrumentation готова, 12 тестов |
| Dry-Run + Full D + Isolation Verdict | #76, #77 | **CONFIRMED catastrophic trigger; PARTIAL full regression** — mask culprit для catastrophic collapse доказан (raise 2.7→27.7); residual зазор до C (27.7→37.6) не изолирован; #77 isolation result: GREY; kind-filter=drop режет 42% live mass |
| Second Control Result | #78 | **kind_filter=drop гипотеза ОПРОВЕРГНУТА** — выключение ухудшило метрики (raise 27.7→13.0%). GPT+OPUS консенсус: 4-уровневая иерархия (mismatch→kind_filter safety→replay_clear exposer→hard mask). Root cause: #74-v4 index mismatch |
| Config Control Result | #79 | **BAD, replay_clear-as-exposer не подтверждён, config-fix исчерпан.** kind ON/clear OFF — худшая клетка 2x2 матрицы (raise 4.6%). kind_filter+clear — interaction dynamics через buffer distribution, не single-variable fix. 2x2 закрыта. Next: B1 selected-credit experimental fix |
| B1 Stress-Test Result | #80 | **PRE-REGISTERED FAIL (raise 2.1%), B1 input-side fix активен но не сдвинул поведение.** Self-suppressing loop не активен (bootstrap_value.raise=+0.266). Sizing funnel здоров (opponent=7225). Reporting bug: `checkpoint_tools.py` whitelist скрывал флаг. Next: action-Q delayed reward propagation. |
| Action-Q Role & Legal Diagnostics | #81 | **H1 CONFIRMED, H2 REJECTED.** Raise 0 terminal, TD-target ceiling +0.418. Fold/call terminal up to +988/+1000. Fold legal 97.4% on q_compare. Reward-grounding asymmetry confirmed. **Grad-clip audit:** scale bottleneck NOT confirmed — terminal grads 194× larger, clip equalizes post-norm; bootstrap also clipped 7.36×. Q-сжатие не от clip, от loss weighting/batch composition. Next: loss reduction diagnosis, terminal upweighting, max_norm ablation. |
| **Max-Norm Ablation (#82)** | #82 | **DONE (B + BREAKTHROUGH).** fit_ratio stable 0.02 (B confirmed). **Post-hoc:** test80b max_norm=100 → OK (raise 48.4%, 15/15, первый OK со времён Run C). max_norm=100 = новый baseline. |
| **Terminal-Balanced Q-Loss (#83)** | #83 | **CLOSED WARN.** test81seed1 raise 3.0%, 5/15. Balanced loss при max_norm=1.0 не работает. Будет перезапущен как #85 на базе max_norm=100. |
| **Consolidation + Relapse (#84/#84b)** | #84, #84b | **CLOSED FAIL (durability) / механизм НАЙДЕН.** max_norm=100: OK на 100, WARN на 200, коллапс на 300. Smoking gun: reward=0 у всех raise-сэмплов. Advantage/regret-сеть делает flip fold↔raise (H3 подтверждена). → #86 MC propagation. |
| **Balanced Loss v2 (#85)** | #85 | **CLOSED FAIL.** Balanced loss уничтожил окно iter 100 (raise 4.4%). Q дестабилизирована. Ветка закрыта навсегда (#83 + #85). |
| **MC Propagation (#86)** | #86 | **CLOSED PARTIAL.** M1+M2 PASS (smoking gun убран, relapse отсутствует), B FAIL (плато ~21%). Причина: сырые чипсовые цели → fit_ratio 0.14. Next: #87 нормализация. |
| **Pot-relative Norm (#87)** | #87 | **CLOSED FAIL-M0.** Масштаб улучшен, но хвосты остались (preflop pot=2-3, reward ±400-1000 «потов»). fit_ratio 0.04<0.5. M0 провален. |
| **Pot+Stack Norm (#87b)** | #87b | **PRE-REGISTERED.** denom = pot + stake → reward ∈ [-1,+1]. M0 жёсткий: fit_ratio>0.5 к iter 50, СТОП иначе. Next: запуск test87bseed1 до iter 300. |
| **MC Propagation Relapse (#88)** | #88 | **Arm B' PRE-REGISTERED.** Диагностика test88seed1: relapse подтверждён (adv-flip в 2-11× мягче #84, но raise всё равно коллапсирует). Гипотеза: обрыв от `q_target_update_interval=100`. Arm B': sync 100→10 — одна переменная (config.yaml). Если M0/M1 PASS но M2 FAIL → Arm C (refit с нуля — истинный референс). C FAIL → #88b discount, #88c recency. |
| **Arm B' Result (#89)** | #89 | **CLOSED FAIL.** test89seed1: поведение идентично baseline. Sync-интервал не первопричина. Q-сеть структурно не может prefer raise>fold (reward-grounding asymmetry #81). M0/M1/M2 — все FAIL. Cross-play: iter100 win 40.5% vs iter400 win 12.3%. Next: #90b discount. |
| **Discount Alignment (#90b)** | #90b | **PRE-REGISTERED → CLOSED.** → результат #91. |
| **Discount Result (#91)** | #91 | **CLOSED PARTIAL.** test90seed1 (α=1.5, Q ON): M0 FAIL, H3 сломан, H3* fold-инфляция. test90b (α=2.0, Q OFF): 32.2%. test90c (α=1.5, Q OFF): 24.9%. Вывод: α=1.5 требует Q ON, иначе хуже α=2.0. Best конфиг: α=1.5 + Q ON. |
| **Discount Corrected (#91b)** | #91b | **SUPERSEDED.** α=2.0, +1.5. Не запущен — test90b/c показали что α=2.0 слабее α=1.5 с Q ON. → `sizing_reports/91b-discount-corrected.md` (архив). |
| **Multi-Agent Result (#92)** | #92 | **ACTIVE — ПЕРВЫЙ ПРОГОН БЕЗ RELAPSE.** test91 (Q OFF): FAIL 12%. test92 (Q ON, α=1.5, 400 trav): iter 100 43%→iter 200 63%→iter 300 52%, win 39.9→45.0→47.1. Multi-agent advantage сломал H3 relapse. M0/M1/M2 все PASS. Sizing concentration остаётся (отдельная проблема). → `sizing_reports/92-multi-agent-result.md`. |
| **Sizing CFR Advantage Leg (#93)** | #93 | **IMPLEMENTED + RESULT (M1 PASS).** advantage_sizing_net → играющая нога через regret-matching. 0.10 подавлен (1-4%), interior моды (0.66/1.00/2.25), sizing-Q впервые положительный (+0.09). M0/M2 PARTIAL из-за action-регрессии. → `sizing_reports/93-sizing-cfr-advantage-leg.md`. |
| **N4 Plain Advantage Accumulation (#97-N4)** | #97-N4 | **IMPLEMENTED (ждёт прогона).** Флаг `advantage_accumulation: plain` — `target_net(s) + cf_regret` без clamp/discount. Мониторинг max\|advantage\|. whitelist fix (`discount_alpha`). Прогон `test96seed1N4`. → `sizing_reports/97-n1-l1-n1prime-results.md`. |
| **#97 Variance-Driven Relapse Diagnosis** | #97-диаг | **DIAGNOSIS.** Raise +EV но высокодисперсный → политика абандонит. Advantage-сеть тонет в шуме (MSE ~50k, таргеты ~±220). Корень — сырой масштаб регрета на уровне фита. → `sizing_reports/97-n1-l1-n1prime-results.md`. |
| **N5 Global Fixed Advantage-Reward Scale (#98)** | #98 | **IMPLEMENTED (ждёт прогона test97seed1N5).** Единственная переменная: `advantage_reward_scale: 200` — деление сырых cf_regrets (O(100)) на фиксированную константу перед записью в advantage-буфер (full-traversal сайт). Гомогенно, не ломает бутстрап (в отличие от N1/N1'). Цель: LossAdvantage с ~50k до O(1–10). OS-сайт НЕ масштабируется (Q-derived, уже O(1)). → `sizing_reports/98-n5-global-fixed-reward-scale.md`. |
| **Random Agent + Per-Iter Traversing + Per-Player Strategy + Optional Q (#99)** | #99 | **IMPLEMENTED + RESULT (N5_3). ЛУЧШИЙ ПРОГОН ПО СТАБИЛЬНОСТИ.** Raise ~16-22% на всех 1600 итерациях — дрейфа нет. Random агент на seat 5, traversing_player по итерациям, strategy смотрит по очереди, Q опционален. H2H late>early подтверждён. Стабильность достигнута. → `sizing_reports/99-random-agent-per-iteration-strategy.md`. |
| **CFR Purity: Strategy-Advantage Decoupling + Per-Player Sizing + Opp Fix + Adaptive Batch (#99v2)** | #99v2 | **IMPLEMENTED + RESULT (N5_4 — ЛУЧШИЙ ПРОГОН СЕРИИ).** N5_4 ≥ N5_3 по всем метрикам. Raise ~20-25%, reward 12-16, H2H 18.1%. Per-player sizing + strategy=distillation → CFR-чистота без потери стабильности. → `sizing_reports/99-v2-cfr-purity-architecture-fixes.md`. |
| **Config Ablations N5_4 (#100-A1..A4)** | #100 | **IMPLEMENTED.** A1: sizing buffer 100k. A2: discount_gamma 1.0. A3: advantage_epochs 20. A4: strategy_buffer_reservoir true (CFR-среднее по истории). → `sizing_reports/99-v2-cfr-purity-architecture-fixes.md`. |
| **Sizing Target Sparsify Collapse (#100)** | #100 | **FIXED (revert).** `top_per_bucket: 5→1` заострял таргеты до резервуара → concentration до 61%. Revert к 5 (no-op). Loss упал вдвое, sizes=15/15. WP-4 (bucket_loss_weight) — эксперимент, не улучшение. → `sizing_reports/100-sizing-target-sparsify-collapse-fix.md`. |
| **Inference Threshold Masking (#101)** | #101 | **IMPLEMENTED.** Порог 5% на инференсе для action и sizing. Только инференс, тренировка не затронута. Eval синхронизирован с деплоем. → `sizing_reports/101-inference-threshold-masking.md`. |
| **#97 Variance-Driven Relapse Diagnosis** | #97-диаг | **DIAGNOSIS.** Raise +EV но высокодисперсный → политика абандонит. Advantage-сеть тонет в шуме (MSE ~50k, таргеты ~±220). Корень — сырой масштаб регрета на уровне фита. → `sizing_reports/97-n1-l1-n1prime-results.md`. |
| **Sizing Accumulation Package (#104)** | #104 | **RECONSTRUCTED.** VR-DeepDCFR+ архитектура. Пакет накопления sizing-регретов восстановлен из истории. |
| **Iteration Profiling (#106)** | #106 | **DIAGNOSIS.** Обучение — 6% времени. Три вывода #106 опровергнуты замерами. |
| **Sizing Regret Flatness Root Cause (#107)** | #107 | **ROOT CAUSE FOUND.** BLAS overhead. Три гейта отменены (ptp, sign_agreement, TV/RATIO). 48% single-credit строк. |
| **#107 Execution (#107Result)** | #107Result | **EXECUTED.** −23.7% мкс/узел. Трек 1 отменён. |
| **Parallel CFR Literature (#108)** | #108 | **REFERENCE ONLY.** Две статьи не применимы. |
| **Update-Equivalence vs PBS (#109)** | #109 | **ANALYZED.** Кандидат в замену солвера. Отложено до S3b. |
| **Leg Degradation = Artifact (#110)** | #110 | **DIAGNOSIS.** «Деградация» — артефакт. Накопительные гипотезы отменены. |
| **Oracle Harness Bug (#111)** | #111 | **BUG FIXED.** Все замеры v8→v13 невалидны. Руки мерили частоту рейза. |
| **Dead Steps + Stratification (#112)** | #112 | **CLOSED.** steps-параметр был мёртв. TV/RATIO — третья мёртвая метрика. |
| **Sizing: Proven & Dead (#112-Итог)** | #112-Итог | **SUMMARY.** blueprint−uniform = ноль. Пять мёртвых метрик. S6 без запаса. |
| **Oracle Dead, v17→v19, Plan 179 (#113)** | #113 | **MULTI-TOPIC.** Оракул мёртв. v19 grounded_ev: diag отделился. §18: диапазон +0.27±0.25 (t≈1.1), приза нет. → 10k traversals. |

---

## Roadmap (остающееся)

1. ~~**Верификация Run C**~~ — подтверждено: OK, raise_freq=37.6%, win_rate=42.1.
2. ~~**Верификация Run D**~~ — подтверждено: WARN, raise_freq=2.6%, регрессия относительно C.
3. ~~**#75 v5 — Mask vs Replay-Clear Root-Cause Diff**~~ — Run A dry-run + Full D подтвердили legal-anchor mask culprit.
4. ~~**#76 — Full D + diagnostics**~~ — подтверждено: mask 20-53% removed mass, replay clear не primary.
5. **#76 isolation (Full-D minus mask)** — **DONE: GREY/PARTIAL** (test76seed1, iter 100). Mask catastrophic trigger confirmed. Residual 27.7→37.6 не закрыт. → See `77-isolation-result-grey.md`.
6. **#78 second control (test76 minus kind_filter)** — **DONE: UNEXPECTED** (test77seed1, iter 100). kind_filter=drop гипотеза опровергнута. kind_filter — safety filter, replay_clear — exposer mismatch, root cause — #74-v4 index mismatch. GPT+OPUS консенсус: 4-уровневая иерархия каналов. → See `78-test77secondcontrol-unexpected-regression.md`.
7. **test78seed1 — config control (mask OFF / kind ON / clear OFF)** — **DONE: BAD** (test78seed1, iter 100). raise 4.6% — close to Full-D collapse. replay_clear-as-exposer гипотеза не подтверждена. 2x2 матрица закрыта, config-only fix исчерпан. → See `79-test78control-clear-off-bad.md`.
7b. **test79seed1 — B1 stress-test (mask OFF / kind OFF / clear ON / B1 ON)** — **DONE: PRE-REGISTERED FAIL** (test79seed1, iter 100). B1 input-side fix полностью активен но не сдвинул поведение. → See `80-b1-stress-test-catastrophic-fail.md`.
8. ~~**#74-v4 B1 — code implementation**~~ — **DONE (commit 27, 29e9409).**
9. **#81 — Action-Q Role & Legal Diagnostics** — **DONE.** `tools/action_q_role_legal_diag.py` + `action_q_grad_clip_scale_audit.py` запущены на test79seed1. H1 CONFIRMED (raise reward-grounding asymmetry: 0 terminal, ceiling +0.418), H2 REJECTED (fold legal 97.4%). **Grad-clip audit:** scale bottleneck NOT confirmed — terminal grads 194× larger but clip equalizes; bootstrap also clipped 7.36×. Q-сжатие от loss weighting/batch composition, не от clip. → See `81-action-q-role-legal-diagnostics.md` §9.
10. ~~**Tooling fix — `checkpoint_tools.py` whitelist**~~ — **DONE (commit 28, 13d2a41).**
11. **P0: Action-Q n-step/MC delayed reward propagation** — **ПРИОРИТЕТ.** Raise никогда не получает terminal reward. TD-target потолок ~+0.42, fold/call имеют terminal up to +988/+1000. Нужен механизм credit propagation: n-step return, TD(λ), или MC target для raise chains.
12. **P1: Sizing-Q negative shift** — `sizing_q_stats.max.mean=-0.78`. Все sizing-Q отрицательные → argmax всегда 0.10. Возможно, sizing-Q тоже страдает от delayed-reward ceiling.
13. **P2: Повтор B1 test79 после action-Q fix** — Проверить, что sizing diversity появляется с работающим raise frequency.
14. ~~**q_a3=0.0, q_target_update_interval, B1 target-side**~~ (deprioritised: root cause — action-Q, не sizing).
15. **#82 max_norm ablation** — **DONE (B CONFIRMED).** test80a (`\|q_pred\|_term`=1.30) и test80b (1.75). Гипотеза A (clip=бутлнек) опровергнута: рост всего 2× при ослаблении клипа в 100×. Гипотеза B (loss weighting) подтверждена. → `82-max-norm-ablation.md`.
16. **Post-#82 (B): terminal upweighting** — → **#83 CLOSED WARN** (raise 3.0%). Будет перезапущен как #85.
17. **#83 terminal-balanced Q-loss** — **CLOSED WARN.** Balanced loss α=0.5 при max_norm=1.0. → `83-terminal-balanced-loss.md`.
18. ~~**#84 Consolidation + #84b Relapse mechanism**~~ — **CLOSED FAIL / механизм НАЙДЕН.** max_norm=100 OK на 100, relapse на 200-300. Smoking gun: reward=0 у raise. Advantage/regret flip (H3). max_norm=100 оставлен как baseline.
19. ~~**#85 Terminal-balanced loss v2**~~ — **CLOSED FAIL.** Balanced loss закрыт навсегда (#83 + #85).
20. ~~**#86 MC reward propagation**~~ — **CLOSED PARTIAL.** Smoking gun убран, relapse отсутствует (первый прогон без коллапса). Но raise плато ~21% — сырые чипсы шумят. Next: #87 pot-relative.
21. ~~**#87 Pot-relative normalization**~~ — **CLOSED FAIL-M0.** Масштаб ok, хвосты нет. fit_ratio 0.04<0.5. Урок: early-stop по M0 без исключений.
22. **#87b Pot+stack normalization** — **PRE-REGISTERED.** denom = pot+stake → reward ∈ [-1,+1]. M0 жёсткий: fit_ratio>0.5 к iter 50. См. `sizing_reports/87b-pot-stack-relative.md`.
23. **P0: n-step/MC terminal reward propagation** — propagation сделано (#86), нормализация идёт (#87b). Осталось: консолидация seed2/400.
24. **#88 — MC propagation durability до iter 400** — **DONE (диагностика).** test88seed1: advantage-flip в 2-11× мягче #84, но raise_freq 58.5%→0.2%. H2+H3 подтверждены. → `sizing_reports/88-relapse-mc-propagation.md`.
25. **#88 Arm B' — q_target_update_interval 100→10** — **CLOSED → #89.** → `sizing_reports/89-arm-b-q-target-sync.md`.
26. **#89 — Arm B' Result** — **CLOSED FAIL.** test89seed1: поведение идентично baseline. M0/M1/M2 все FAIL. Cross-play: iter100 40.5% vs iter400 12.3%. Next: #90b discount alignment. → `sizing_reports/89-arm-b-q-target-sync.md`.
27. **#90b — Discount Alignment (DCFR+ reference)** — **PRE-REGISTERED → CLOSED.** Результат в #91. → `sizing_reports/90b-discount-alignment.md`.
28. **#91 — Discount Alignment Result** — **CLOSED PARTIAL.** test90seed1 (α=1.5, Q ON): M0 FAIL, H3→H3*. test90b (α=2.0, Q OFF): 32.2%. test90c (α=1.5, Q OFF): 24.9%. Вывод: α=1.5 + Q ON = best single-net конфиг. → `sizing_reports/91-discount-alignment-result.md`.
29. **#91b — Discount Corrected** — **SUPERSEDED.** α=2.0, +1.5. Не запущен. test90b/c данные показали что α=2.0 (Q OFF) слабее α=1.5 (Q ON). → `sizing_reports/91b-discount-corrected.md` (архив).
30. **#92 — Multi-Agent Advantage Result** — **ACTIVE — ПЕРВЫЙ ПРОГОН БЕЗ RELAPSE.** test91 (Q OFF): FAIL 12%. test92 (Q ON, α=1.5): iter 100 43→200 63→300 52%, win 39.9→45.0→47.1. M0/M1/M2 все PASS. Первый прогон с монотонно растущим win rate. → `sizing_reports/92-multi-agent-result.md`.
31. **Следующий шаг** — iter 400 + seed2 для подтверждения воспроизводимости. Sizing concentration fix (отдельная проблема, не multi-agent).
32. **#93 — Sizing CFR Advantage Leg** — **IMPLEMENTED + RESULT (M1 PASS).** Два прогона на test92seed1. Sizing-коллапс сломан: 0.10 подавлен, interior anchors, sizing-Q положительный. Action-регрессия остаётся (нужен оппонент сильнее рандома). Ссылка: `sizing_reports/93-sizing-cfr-advantage-leg.md`.
33. **#94 — Sizing Legality, Boundary All-in, Preflop no-Buckets, Q-bootstrap per-Player** — **IMPLEMENTED + ERRATA FIXED → RESULT в #95.** Код реализован корректно, R0-патч применён. Результаты прогонов R1-R4 см. #95. Ссылка: `sizing_reports/94-szing-legality-boundary-allin.md`.
34. **#95 — Результаты #94: A-core PASS, A4 FAIL, C PASS на целевой метрике** — **RESULT.** R2 (A-core: availability/boundary/preflop=true) принят как новый baseline — raise 38→52%, reward 29→14, без relapse в eval. R3 (A4: +on_input, cold-start) отклонён — reward −4→−3. R4b (C: per-player Q ON) держит after-raise mass 0.104 vs R4a 0.000 — бутстрап-нога не обнуляется. Корректность-фикс, низкорычажный для relapse. Принят под TD (`q_reward_propagation: none`). Ключевое открытие: relapse_diff меряет greedy-политику (осциллирует), eval — усреднённую (стабильна). Ссылка: `sizing_reports/95-szing-legality-results.md`.
35. **#96 — E1: OS Force Full Traversal (action-OS off, CLOSED DISPROVED)** — Гипотеза опровергнута. OS выключен (hero_os=0, всё через hero_full), регреты точные — relapse идентичен: raise 22→10→25→10→7→5→4.3% к iter700, advantage-флип тот же. Модель выигрывает у рандома даже при raise 4% (win 21–40% > 16.7%), но проигрывает iter100 в h2h (12.4% win). Вывод: CFR корректно сходится к пассивному best-response против рандома — мультивей-агрессия против рандом-коллеров наиболее −EV. Пивот на self-play. См. `sizing_reports/96-os-force-full-traversal-e1.md`.
36. **#96 — Advantage-regret scale: normalization + Huber loss** — **IMPLEMENTED (ждёт прогонов N1/L1).** 4 новых ключа: `advantage_regret_norm` (`none`/`pot_stack`/`per_node_max`), `advantage_regret_clip` (None/float), `advantage_loss` (`mse`/`huber`), `advantage_huber_delta` (float, default 1.0). Код: нормировка регретов перед `_adv_buffer.add` в full-traversal сайте + Huber loss в `train_advantage_network_multi` + checkpoint save/load. N1 = `pot_stack` + `huber`, lr=1e-4, iter 400. L1 = `advantage_lr=1e-6`, iter 800-1000. Резерв N1' = `per_node_max` + `huber`. См. `sizing_reports/96-advantage-regret-scale.md`.
37. **#97 — Результаты N1 / L1 / N1'** — **CLOSED (все три опровергнуты).** N1 (pot_stack+Huber) FAIL — регреты пересушены до O(0.01). L1 (lr=1e-6) FAIL — дрейф тот же, медленнее; iter300 OK (26.1%) но к iter1000 raise=1.1%. N1' (per_node_max+Huber) FAIL — per-sample нормировка несовместима с бутстрапом. **Масштаб регрета — НЕ корень relapse.** Next: N4 (без clamp/discount как в оригинальном DeepPDCFR). См. `sizing_reports/97-n1-l1-n1prime-results.md`.
38. **#97-N4 — Plain advantage accumulation** — **IMPLEMENTED (ждёт прогона test96seed1N4).** Единственная переменная vs R2+OS-off: `advantage_accumulation: plain` (`fresh_target = prev_pred + regret_tensors`, без clamp, без discount). Новый флаг + мониторинг max|advantage| + whitelist fix (`discount_alpha`). Гейты: relapse_diff raise-advantage flip, raise_freq eval, h2h late-vs-early. См. `sizing_reports/97-n1-l1-n1prime-results.md`.
39. **#97-диагностика: relapse — variance-driven, не EV** — **DIAGNOSIS.** Raise безубыточен/прибылен (mean reward +0.0074 на iter 300), но имеет наибольшую дисперсию. Advantage-сеть фитит таргеты ~±220 фишек (MSE ~50k) → raise-vs-fold тонет в шуме. Корень — СЫРОЙ масштаб регрета на уровне фита. Нужен глобальный фиксированный масштаб `cf_regret`. Направление: variance reduction (baseline/sampling/control variates), не accumulation.
40. **#100 WP-4 — Bucket head training (bucket_loss_weight: 0→0.5)** — **EXPERIMENT (NOT better than _3).** test102: raise 22-25%, win 12-14%, 1/8 OK. Mean_reward выше _3 (фолды перетекли в коллы), но raise_freq/win_rate ниже и осциллируют. Конфликт градиентов в shared base. Оставлено как эксперимент. См. `sizing_reports/100-sizing-target-sparsify-collapse-fix.md` §WP-4.
41. **#101 — Inference Threshold Masking (action + sizing)** — **IMPLEMENTED.** `inference_min_action_prob: 0.05`, `sizing_inference_min_prob: 0.05`. Пороговая маска на инференсе: отсечение действий/анкеров с вероятностью <5%, ренормализация оставшихся. Только инференс, тренировка не затронута. См. `sizing_reports/101-inference-threshold-masking.md`.
42. **#98 — N5: Global Fixed Advantage-Reward Scale** — **IMPLEMENTED (ждёт прогона test97seed1N5).** Одна фиксированная константа `advantage_reward_scale: 200` — деление сырых cf_regrets перед записью в advantage-буфер (full-traversal сайт, `cfr_traverse_multi`). Гомогенно с бутстрапом (в отличие от per-state N1/N1'). OS-сайт не трогаем (Q-derived, уже O(1)). Стартовая магнитуда 200 = стек при 100bb. Первый гейт: LossAdvantage падает ~в 40000× (50k → O(1–10)). См. `sizing_reports/98-n5-global-fixed-reward-scale.md`.


