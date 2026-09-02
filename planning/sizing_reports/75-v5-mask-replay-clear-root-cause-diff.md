# Bug #75 v5: Mask vs Replay-Clear Diagnostics (C→D Root-Cause Diff)

> **Статус:** IMPLEMENTED (instrumentation-only, config-gated, all off by default).
> **Цель:** доказательно разделить `legal-anchor mask` vs `replay_clear_once` как причины ранней регрессии Run D (test8) относительно прибыльного Run C (test7).
> **Дата:** 2026-06-09

---

## 1. Контекст

Run C/test7: `verdict=OK`, `raise_freq=37.6%`, `win_rate=42.1`, `raise_mass=0.406`, `raise_samples=262597`.
Run D/test8: `verdict=WARN`, `raise_freq=2.6%`, `win_rate=36.2`, `raise_mass=0.0`, `raise_samples=27556`.

Оба прогона имеют идентичный базовый конфиг (bootstrap_target_net=false, src2_weight=0).
Разница только в 4 компонентах v8:
- `legal-anchor mask` (exclude ALL_IN/MIN_RAISE)
- `kind-filter` (drop non-NORMAL)
- `replay_clear_once`
- `verdict concentration gate`

Дополнительно из v7 известно:
- q_a3=0 — идентичен в Run C и Run D → **НЕ smoking gun.**
- raise reward=0, terminal=0% — идентичны в C и D → **структурная норма.**
- UNKNOWN = source2 lookahead (не передаёт selected_kind) → kind-filter drop редундантен (src2 уже weight=0).
- Регрессия на iter 100 слишком ранняя для старого action-Q self-suppression loop (#73/#74 проявлялся к iter 200-300).

Главные подозреваемые:
1. **legal-anchor mask** (selection-time, мгновенный эффект, iter-100 timing)
2. **replay_clear_once** — второй, вероятно в связке с маской: clear стирает разнообразный буфер → наполняется заново уже замаскированной/узкой policy.

---

## 2. Что сделано

### 2.1 Config Flags (3 новых, все `false` по умолчанию)

```yaml
sizing_q_mask_diagnostics_enabled: false
sizing_q_mask_dry_run: false
sizing_q_replay_clear_diagnostics_enabled: false
```

### 2.2 `_compute_sizing_anchor_kind_mask` — Counterfactual Mask Helper

Новый helper, **не зависящий** от `sizing_q_legal_anchor_mask_enabled`:
- Для каждого анкера вызывает `_resolve_effective_sizing(state, anchor)`.
- Строит mask по `sizing_q_mask_exclude_kinds`.
- Возвращает `(mask, kind_by_anchor, fallback_all_true)`.

`_legal_sizing_anchor_mask` отрефакторен: теперь это обёртка над helper с early-return по флагу.

### 2.3 `_apply_sizing_anchor_mask(probs, state, callsite)` — Diagnostics + Dry-Run

Приоритет:
`dry_run=true` > `mask_enabled=true` > no-op.

`dry_run=true`: считает counterfactual post-mask метрики, логирует их, но **возвращает исходные probs**.

Диагностика по callsite (6 меток):
- `calls`, `fallback_all_true`
- `pre_entropy_sum`, `post_entropy_sum`
- `pre_top1_idx_hist`, `post_top1_idx_hist`
- `pre_top1_mass_sum`, `post_top1_mass_sum`
- `pre_top2_mass_sum`, `post_top2_mass_sum`
- `removed_mass_total_sum`
- `removed_mass_by_kind`: NORMAL / MIN_RAISE / ALL_IN (UNKNOWN исключён — на selection-time не бывает)
- `kept_mass_total_sum`

### 2.4 Callsite Labels (6 точек)

| Callsite | Место |
|----------|-------|
| `hierarchical` | `_hierarchical_sizing` |
| `os_q_ready` | `_cfr_traverse_multi_outcome_node` (Q-ready sizing путь) |
| `choose_argmax` | `choose_action` mode=argmax |
| `choose_top_bucket` | `choose_action` mode=top_bucket |
| `choose_hierarchical` | `choose_action` mode=hierarchical |
| `choose_stochastic` | `choose_action` mode=stochastic |

### 2.5 Raise Selection Funnel

Счётчики по трём путям (hero_os / hero_full / opponent):
- `raise_legal` — узел с legal Raise
- `raise_sampled` — Raise выбран и sizing определён
- `raise_apply_ok` — apply_action успешен
- `raise_q_buffer_add` — сэмпл записан в action_q_buffer / sizing_q_buffer

### 2.6 Insert-Time UNKNOWN Audit

Метод `_update_insert_kind_diag(source, kind, callsite)`.
Считает `by_source` и `by_callsite` на точках вызова `_add_sizing_q_sample`:
- lookahead: source=2, kind=UNKNOWN
- hero_os_raise: source=0, kind из resolve
- hero_full_raise: source=0/1, kind из resolve

Ожидание: UNKNOWN почти весь в source=2.

### 2.7 Replay Clear Audit

При `sizing_q_replay_clear_diagnostics_enabled=true`:
- `before_clear`: total, by_source, by_anchor counts
- `cleared`: True после clear
- `refill_snapshots`: (зарезервировано для future use)

### 2.8 Report/Checkpoint Passthrough

- `_build_checkpoint`: 3 новых config-ключа + sizing_q_mask_diag, raise_funnel_diag, sizing_q_insert_kind_diag, sizing_q_replay_clear_diag.
- `_load_checkpoint`: 3 новых ключа в списке setattr (top-level → config fallback).
- `tools/checkpoint_tools.py`: passthrough новых diagnostics в full report и diagnose report.

---

## 3. Файлы

| Файл | Изменения |
|------|-----------|
| `config.yaml` | 3 новых флага (all false) |
| `src/core/deep_cfr.py` | Mask helper + refactor, mask diagnostics + dry-run, callsite labels, raise funnel, insert audit, replay clear audit, reset/checkpoint/loader passthrough |
| `tools/checkpoint_tools.py` | load_full_checkpoint + report passthrough для всех новых diagnostics |
| `tests/test_hybrid_outcome_sampling.py` | 12 новых тестов |

---

## 4. Тесты (12 новых, 83/86 passed)

| Тест | Проверяет |
|------|-----------|
| `test_compute_sizing_anchor_kind_mask_works_when_mask_disabled` | helper работает даже при mask_enabled=false |
| `test_legal_sizing_anchor_mask_returns_all_true_when_disabled` | старый метод сохраняет поведение |
| `test_apply_sizing_anchor_mask_dry_run_returns_original` | dry-run возвращает исходные probs |
| `test_apply_sizing_anchor_mask_dry_run_wins_over_mask_enabled` | dry-run побеждает mask_enabled |
| `test_apply_sizing_anchor_mask_no_op_when_all_disabled` | no-op когда всё выключено |
| `test_mask_diagnostics_records_per_callsite` | diagnostics пишется раздельно по callsite |
| `test_update_insert_kind_diag` | insert-kind audit работает |
| `test_raise_funnel_diag_initial_structure` | правильная структура raise_funnel |
| `test_reset_traversal_stats_clears_raise_funnel_and_mask_diag` | reset чистит новые diag |
| `test_build_checkpoint_saves_v5_diagnostics` | checkpoint содержит новые diag |
| `test_build_checkpoint_persists_v5_config_flags` | config-флаги в checkpoint/config |
| `test_load_checkpoint_restores_v5_config_flags_*` (×2) | loader читает флаги из top-level и config |

---

## 5. Предрегистрированные Прогоны

### Run A: C + Dry-Run Mask Diagnostics

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: false
sizing_q_replay_clear_once: false
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: false
```

**Seed:** тот же, что test7
**Ожидание:** поведение C-like: `raise_freq ~37%`, `win_rate ~42%`, `raise_mass ~0.40`.
Если не воспроизводит test7 → инструментация загрязнила прогон (guardrail).

**Decision rules:**
- Dry-run post-mask показывает резкую концентрацию / removed_mass → **legal-anchor mask culprit.**
- Dry-run mask benign → **переходим к Run B или replay-clear подозреваемому.**

### Run B: Full D + Diagnostics

```yaml
sizing_q_legal_anchor_mask_enabled: true
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: true
sizing_q_kind_filter_mode: drop
sizing_q_replay_clear_once: true
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: false
sizing_q_replay_clear_diagnostics_enabled: true
```

Подтверждает actual applied-mask behavior и replay refill после clear.

### Ablation (только если A/B неоднозначны)

- C + legal-anchor mask only
- C + replay_clear only
- C + legal-anchor mask + replay_clear

Не планируется сразу.

---

## 6. Decision Rules

| Сигнал | Вывод |
|--------|-------|
| Run A C-like, но counterfactual post-mask концентрируется | legal-anchor mask culprit |
| Run A mask benign, Run B refill узкий после clear | replay_clear culprit / interactor |
| UNKNOWN только source2 | kind-filter не culprit |
| UNKNOWN в source0/1 | metadata plumbing bug |
| `raise_legal` stable, `raise_sampled` падает | policy/selection issue |
| `raise_legal` тоже падает | earlier state-distribution shift |

---

## 7. Риски

- **Dry-run на C** может показать пустой отчёт, если diagnostics-ветка не применяется (проверено: helper не зависит от флага).
- **removed_mass_by_kind.UNKNOWN** на mask всегда 0 — это корректно, UNKNOWN живёт в insert-audit.
- **Replay clear audit** ловит before_clear только один раз (до первого clear); refill trajectory отслеживается через `refill_snapshots` (зарезервировано).
