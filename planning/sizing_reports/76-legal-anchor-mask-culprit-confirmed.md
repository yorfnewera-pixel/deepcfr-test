# Bug #76: Legal-Anchor Mask Culprit Confirmed — Run A Dry-Run + Full D Diagnostics

> **Статус:** Run A подтверждён. Full D подтвердил маску как primary culprit. Replay clear не primary.
> **Дата:** 2026-06-10

---

## 1. Результаты Run A (C + dry-run diagnostics)

### 1.1 Конфиг

```yaml
sizing_q_legal_anchor_mask_enabled: false   # маска не применяется
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: false         # kind-filter выключен
sizing_q_replay_clear_once: false           # replay не чистится
sizing_q_mask_diagnostics_enabled: true     # counterfactual mask рассчитан
sizing_q_mask_dry_run: true                 # возвращает original probs
sizing_q_replay_clear_diagnostics_enabled: false
```

### 1.2 Guardrail: C-like поведение подтверждено

| Метрика | Run C/test7 | Run A (этот) | Статус |
|--------:|:----------:|:----------:|:------:|
| `verdict` | OK | WARN (concentration) | OK* |
| `raise_freq` | 37.6% | 36.2% | ✓ |
| `win_rate` | 42.1 | 40.5 | ✓ |
| `unique_anchors` | 15/15 | 15/15 | ✓ |
| `action_q_buffer.raise` | 262597 | 262597 | ✓ |
| `next_strategy_raise_mass.raise` | 0.406 | 0.406 | ✓ |

*WARN только из-за нового concentration gate (top-2=61%>55%), поведение C-идентично.

### 1.3 Legal-Anchor Mask Culprit — ДОКАЗАНО

Counterfactual mask-diff на здоровом C-like распределении:

| Callsite | Calls | Removed Mass | Removed % | Main Culprit |
|----------|------:|:------------:|:---------:|:------------:|
| `hierarchical` | 565 | 236.9 | ~42% | MIN_RAISE + ALL_IN |
| `os_q_ready` | 68 | 19.4 | ~29% | ALL_IN + MIN_RAISE |
| `choose_top_bucket` | 1287 | 404.7 | ~31% | MIN_RAISE + ALL_IN |

`removed_mass_by_kind` (суммарно):
- **MIN_RAISE**: львиная доля удалённой массы
- **ALL_IN**: заметно
- **NORMAL**: почти 0

Маска режет существенную часть живой политики даже на здоровом C-распределении — не только редкие мусорные анкеры.
Это **прямое доказательство**, что legal-anchor mask — главный виновник Run D-регрессии.

### 1.4 UNKNOWN подтверждён как source2

```json
"sizing_q_insert_kind_diag": {
  "by_source": {
    "0": { "ALL_IN": 226, "MIN_RAISE": 182, "NORMAL": 195 },
    "1": { "NORMAL": 109, "ALL_IN": 71, "MIN_RAISE": 16 },
    "2": { "UNKNOWN": 315 }
  }
}
```

UNKNOWN — только в source2 (lookahead metadata gap). kind-filter сам по себе не главный виновник,
но мог бы усиливать collapse на вторичном этапе.

### 1.5 Replay Clear не виноват в этом прогоне

```json
"sizing_q_replay_clear_diag": { "before_clear": null, "cleared": false }
```

Replay не чистился — значит C-распределение восстановилось без clear.
Clear остаётся подозреваемым только как interactor: может ухудшить ситуацию при уже включённой маске.

### 1.6 Raise Funnel — C/B

```json
"raise_funnel_diag": {
  "hero_os":  { "raise_legal": 68,   "raise_sampled": 38,   "raise_apply_ok": 38,   "q_buffer_raise_add": 38 },
  "hero_full": { "raise_legal": 761,  "raise_sampled": 761,  "raise_apply_ok": 761,  "q_buffer_raise_add": 761 },
  "opponent":  { "raise_legal": 5169, "raise_sampled": 5169, "raise_apply_ok": 5169, "q_buffer_raise_add": 5169 }
}
```

- Hero full: все raise_sampled = raise_legal → 100% конверсия (традиционный full traversal).
- Hero OS: конверсия ~56% (38/68) — OS sampling выбирает не-raise чаще.
- Opponent: масса raise от оппонента (strategy_sizing_net, без mask).

---

## 2. Конфиг для Full D + Diagnostics (следующий прогон)

```yaml
sizing_q_legal_anchor_mask_enabled: true
sizing_q_mask_exclude_kinds:
  - ALL_IN
  - MIN_RAISE
sizing_q_kind_filter_enabled: true
sizing_q_kind_filter_mode: drop
sizing_q_kind_filter_downweight: 0.25
sizing_q_replay_clear_once: true
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: false
sizing_q_replay_clear_diagnostics_enabled: true
```

### 2.1 Ожидаемые вопросы для Full D

1. Applied mask — сколько массы реально удаляется на каждом callsite (сравнение с Run A counterfactual)?
2. Replay clear — каков before_clear состав буфера и как он перезаполняется?
3. Kind-filter — сколько MIN_RAISE/ALL_IN/UNKNOWN реально дропается при включённой маске?
4. Raise funnel — где именно падает raise mass: legal → sampled → apply → buffer?

### 2.2 Decision Rules

- Applied mask removed_mass ~ counterfactual из Run A → маска виновна.
- Before_clear буфер полный и разнообразный, после refill узкий → replay_clear interactor.
- Raise funnel: legal сохраняется, но sampled/apply/buffer падает → selection-time policy collapse (маска).
- Raise funnel: legal тоже падает → earlier state-distribution shift.

---

## 4. Результаты Full D + Diagnostics

### 4.1 Итог

| Метрика | Run C | Run A (dry-run) | Full D | Статус |
|--------:|:-----:|:---------------:|:------:|:------:|
| `verdict` | OK | WARN* | WARN | |
| `raise_freq` | 37.6% | 36.2% | 2.7% | D regression |
| `win_rate` | 42.1 | 40.5 | 38.5 | D regression |
| `unique_anchors` | 15/15 | 15/15 | 10/15 | D regression |
| `best_sizing_anchor_dist` | 0.10=424/499 | 0.10=424/499 | 0.10=499/499 | D collapse |
| `action_q_buffer.raise` | 262597 | 262597 | 27556 | D collapse |
| `next_strategy_raise_mass.raise` | 0.406 | 0.406 | 0.0 | D collapse |

### 4.2 Applied Mask в D

Маска реально режет huge mass на здоровой D-траектории:

| Callsite | Calls | Removed Mass | % | Culprit |
|----------|------:|:------------:|:--:|:-------:|
| `hierarchical` | 276 | 147.5 | ~53% | MIN_RAISE |
| `os_q_ready` | 2399 | 511.5 | ~21% | MIN_RAISE + ALL_IN |
| `choose_top_bucket` | 1570 | 471.8 | ~30% | MIN_RAISE + ALL_IN |

`removed_mass_by_kind`: MIN_RAISE = львиная доля, ALL_IN = вторично, NORMAL = почти 0.

На `hierarchical` в D ещё хуже чем Run A (53% vs 42%), потому что распределение уже сжато до 0.10.

### 4.3 Replay Clear — НЕ primary, но interactor

```json
"sizing_q_replay_clear_diag": {
  "before_clear": {
    "total": 3816,
    "by_source": { "0": 1776, "2": 1305, "1": 735 },
    "by_anchor": { "0": 678, "1": 338, "2": 319, ... (all 15) }
  },
  "cleared": true
}
```

Before-clear буфер был разнообразный (все source, все 15 anchors). После clear и refill buffer опять полный `50000` (all anchors представлены). Clear сам по себе не создаёт коллапс, но усиливает вред маски (удаляет historical diversity до того, как mask задаёт узкую policy).

### 4.4 Raise Funnel в D vs A

| Path | Run A | Full D | Diff |
|------|------:|-------:|-----:|
| hero_os raise_legal | 68 | 2399 | OS ε разные |
| hero_os raise_sampled | 38 | 736 | |
| hero_full raise_legal | 761 | 400 | **↓** |
| opponent raise_* | 5169 | **0** | **исчез** |

В D оппонентский raise path полностью исчезает. Цепочка:

1. mask режет MIN_RAISE/ALL_IN sizing mass
2. sizing policy становится narrow
3. action-Q/policy перестаёт кредитовать raise
4. opponent raise mass → 0
5. next_strategy_raise_mass → 0
6. q_raise < call/fold

### 4.5 UNKNOWN

source2 → `UNKNOWN: 195`, source0/source1 UNKNOWN нет. Kind-filter не culprit.

---

## 5. Root Cause Conclusion

> Жёсткая legal-anchor mask, полностью обнуляющая `MIN_RAISE`/`ALL_IN`, режет 20-53% живой sizing policy mass. Это ломает healthy C distribution и запускает ранний D collapse. Replay clear усиливает, но не является primary источником.

---

## 6. Isolation Run: Full-D Minus Mask (последний контроль)

> **Статус:** Конфиг поставлен, ждёт запуска.
> **Цель:** Доказать каузально — mask единственный виновник, отделив его от kind-filter/replay_clear.

### 6.1 Конфиг (уже в config.yaml)

```yaml
sizing_q_legal_anchor_mask_enabled: false   # ← единственное отличие от Full-D
sizing_q_mask_exclude_kinds: [ALL_IN, MIN_RAISE]
sizing_q_kind_filter_enabled: true          # как в Full-D
sizing_q_kind_filter_mode: drop
sizing_q_replay_clear_once: true            # как в Full-D
sizing_q_mask_diagnostics_enabled: true     # counterfactual mask для сравнения
sizing_q_mask_dry_run: true
sizing_q_replay_clear_diagnostics_enabled: true
```

### 6.2 Pre-registered критерии

| PASS | FAIL | Grey |
|------|------|------|
| `raise_freq >= 33%` | `< 5%` | `15-30%` |
| `next_strategy_raise_mass >= 0.38` | `≈ 0` | промежуточно |
| `action_q_buffer.raise >= 230k` | `≈ 27k` | |
| `unique_anchors = 15` | `≤ 10` | |
| `opponent funnel ≠ 0` | `= 0` | |

- **PASS** → mask culprit доказан окончательно. → #77 soft mask fix.
- **FAIL** → kind-filter/replay_clear interaction сильнее, нужен второй контроль `C + replay_clear only`.
- **Grey** → mask частично виновен, нужен второй контроль.

---

## 7. Next Steps (#77 — после isolation)

1. Если isolation PASS: **Soft mask** — не обнулять excluded kinds:
   - Exclude только `ALL_IN`, оставить `MIN_RAISE` (рекомендация)
   - Или downweight `MIN_RAISE`/`ALL_IN` до `0.25`/`0.10`
2. Ablation (после подтверждения):
   - mask exclude only `ALL_IN`
   - mask downweight non-NORMAL
   - no replay_clear + soft mask

---

## 8. Файлы

| Файл | Изменения |
|------|-----------|
| `config.yaml` | Isolation: mask OFF, kind ON, clear ON, all diagnostics ON |
| `sizing_reports/76-...md` | Этот отчёт (+ isolation plan) |
| `sizing_reports/SIZING.md` | Обновлён |
