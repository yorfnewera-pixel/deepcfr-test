# Bug #74 v7: Source2 Root-Cause Found + Source2-Excluded Run C Configuration

> **Статус:** Root-cause found (stale q_target_net). Config-only patch applied. Source2 excluded from sizing-Q loss.
> **Прогон A (true):** source2=exact-0, причина: stale q_target_net.
> **Прогон B (false):** source2≠0, но слабый/плоский, raise_freq регрессировал до 19.7% WARN.
> **Прогон C (false + src2_weight=0):** исключаем source2 из loss полностью. Ожидаем восстановление raise_freq и sizing-Q.
> **Дата:** 2026-06-09

---

## 1. Краткий итог всех прогонов

| Прогон | bootstrap_target_net | src2_weight | source2 target | raise_freq | best_sizes | verdict |
|--------|---------------------|-------------|----------------|------------|------------|---------|
| A (test3-4) | true | start=1.0, end=0.25 | exact 0 | 27-29% | 0.10=65%, 1.75=23% | OK |
| B (test6) | false | start=1.0, end=0.25 | ~0.002 ±0.05 | 19.7% | 0.10=100% | WARN |
| C (test7) | false | start=0.0, end=0.0 | запись в buffer, но не в loss | ? | ? | ? |

---

## 2. Root-Cause: stale `q_target_net` при `sizing_q_bootstrap_target_net=true`

```yaml
q_target_update_interval: 100
sizing_q_bootstrap_target_net: true
```

Таймлайн Прогона A на 100 итерациях:

```text
Init:    q_target_net = q_net, обе q_head = zero-init
Iter 1-99: q_net обучается → ненулевой
           q_target_net НИ РАЗУ не синхронизируется (iter % 100 != 0)
           → q_target_net.q_head = zero-init
           → lookahead bootstrap через q_target_net → все source2 = exact 0
Iter 100:  _sync_q_target_net() → q_target_net = q_net (ненулевой)
           Checkpoint сохраняется после sync
           → q_compare в report показывает ненулевые Q
```

Доказательства из sizing_lookahead_diag (прогон A):

```text
using_target_net=33665, q_canary_abs_sum=0.0, q_a0..q_a3=0.0, v_k=0.0, target=0.0
q_input_abs_sum=23.14, q_input_has_nan=0, base_out_abs_sum=6.10
```

---

## 3. Прогон B: `sizing_q_bootstrap_target_net=false`

```diff
- sizing_q_bootstrap_target_net: true
+ sizing_q_bootstrap_target_net: false
```

Результат (sizing_lookahead_diag, test6seed1):

```text
using_target_net=0
q_canary_abs_sum=6.25
q_a0=1.17, q_a1=-0.09, q_a2=-0.02
v_k mean=-0.074, std=0.99
target mean=0.002, std=0.052
source2_means: 0.003..0.008 by anchor
vk_near_zero: 33665 -> 1113
```

**Вывод:** source2 перестал быть константным нулём. Гипотеза stale-target подтверждена.

Но стратегия регрессировала:

```text
raise_freq: 28% -> 19.7% (WARN)
best_sizing_anchor_dist: 0.10=100% (499/499)
win_rate: 42% -> 37.2%
mean_reward: 18.6 -> 15.1
```

Причина: source2 хоть и не ноль, но слабый/почти плоский по anchors (~0.003..0.008 std 0.05) и с весом ~0.81 в loss размывает sizing-Q, вырождая argmax в 0.10.

---

## 4. Прогон C (текущий): исключить source2 из sizing-Q loss

### 4.1 Config changes

```diff
- sizing_q_src2_weight_start: 1.0
- sizing_q_src2_weight_end: 0.25
+ sizing_q_src2_weight_start: 0.0
+ sizing_q_src2_weight_end: 0.0
```

Оставлено:
```yaml
sizing_q_bootstrap_target_net: false
sizing_q_src2_loss_weight_enabled: true
```

### 4.2 Используемый механизм

Bug #70: `sizing_q_src2_weight_start/end/decay` — это существующий per-sample вес, применяемый к source=2 сэмплам в sizing-Q loss через `_current_sizing_q_src2_weight()`.

При `start=0.0, end=0.0` эффективный вес source=2 равен 0 на всех итерациях. Source=2 continues to be written to buffer but is ignored in loss.

### 4.3 Ожидаемый результат

Прямое сравнение с test6 (прогон B):

| Метрика | Прогон B (test6) | Прогон C (test7, ожидаем) |
|---------|-----------------|---------------------------|
| raise_freq | 19.7% WARN | >25% (восстановление) |
| best_sizing_anchor_dist | 0.10=100% | больше 1 anchor |
| sizing_q_stats.max.mean | -1.25 | > -1.0 |
| collapse.anchor_raw_probs | dominant=0.10 | менее доминантный |
| win_rate | 37.2% | >40% |

### 4.4 Что проверять в новом report

- `sizing_lookahead_diag`: source2 всё ещё пишется в buffer, counts/stats должны быть ненулевые
- `sizing_q_buffer.mean_target_by_source_anchor`: source=2 должен остаться как в test6 (~0.003..0.008)
- `collapse` + `best_sizing_anchor_dist`: anchor diversity должен восстановиться
- `verdict`: должен перестать быть WARN по low_raise
- `sizing_selected_effective_diag` + `_kind_diag`: less ALL_IN/MIN_RAISE remap?

---

## 5. Что дальше (не в этом прогоне)

После Прогона C:

1. **v5-диагностика q_a3** (raise-column обучение): отдельно от source2, поитерационные веса/градиенты.
2. **legal-anchor mask** на выборе sizing.
3. **kind-filter**: normal-Q только на `kind=NORMAL`.
4. **replay hygiene** после смены семантики.
5. **verdict-gate** по концентрации (энтропия / top-2 mass).
6. **При необходимости:** более частый `q_target_update_interval` или gate lookahead после первого sync, чтобы можно было вернуть `sizing_q_bootstrap_target_net=true` с осмысленным bootstrap.

---

## 6. Файлы

- `config.yaml:81`: `sizing_q_bootstrap_target_net: false` (Прогон B)
- `config.yaml:93`: `sizing_q_src2_weight_start: 0.0` (Прогон C)
- `config.yaml:94`: `sizing_q_src2_weight_end: 0.0` (Прогон C)

Код не менялся относительно коммита 18/19.
