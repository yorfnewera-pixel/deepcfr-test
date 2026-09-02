# Action-Q Role & Legal Diagnostics — iter 100

> **Checkpoint:** multi_checkpoint_iter_100.pt
> **B1 active:** True

---

## 1. Interpretation Flags

| Flag | Value |
|---|---|
| B1 selected-credit active | True |
| Raise — bootstrap-only (0 terminal samples) | True |
| Raise target ceiling < 1.0 confirmed | True |
| Fold has terminal positive targets >100 | True |
| HERO fold terminal positive suspected | False |
| q_compare fold>raise may be misleading (fold illegal) | False |
| Fold>raise confirmed on LEGAL states | True |
| Actor role not available (next_is_hero proxy) | True |

---

## 2. q_compare Legal Mask

- **n_states:** 499
- **fold_legal_pct:** 97.4%
- **raise_legal_pct:** 100.0%
- **raise_lt_best_legal_pct:** 97.4%
- **raise_lt_fold_when_fold_legal_pct:** 100.0%

**Best legal action distribution:**
```
  fold: 486
  raise: 13
```

---

## 3. Raise Bootstrap Ceiling

Target quantiles:

### fold
- **nonterminal** (n=3603): p50=0.05765800178050995, p90=0.1794548332691193, p99=0.26260085999965666, max=0.29496219754219055, mean=-2.913175344467163
- **terminal** (n=1007): p50=-14.5, p90=347.4900024414062, p99=814.2925793457026, max=988.0399780273438, mean=-3.843080997467041

### check
- **nonterminal** (n=482872): p50=0.2676517069339752, p90=0.30004891753196716, p99=0.3252301782369614, max=0.3784588575363159, mean=0.2668023705482483
- **terminal** (n=26724): p50=-2.0, p90=9.5, p99=140.0, max=958.6500244140625, mean=-0.07625750452280045

### call
- **nonterminal** (n=370833): p50=0.2755560278892517, p90=0.315596741437912, p99=0.3349231708049774, max=0.38336804509162903, mean=-0.9169718623161316
- **terminal** (n=28896): p50=-21.84000015258789, p90=-0.0, p99=1000.0, max=1000.0, mean=-15.258638381958008

### raise
- **nonterminal** (n=86065): p50=0.32151806354522705, p90=0.3618718087673187, p99=0.38294118642807007, max=0.4176066219806671, mean=0.26595625281333923
- **terminal:** no samples

---

## 4. Terminal Positive Tail

### fold
- **terminal_count:** 1007
- **next_opponent** (n=1007): gt_0=119, gt_50=118, gt_100=116, gt_250=108, gt_400=91, p99=814.2925793457026, max=988.0399780273438

### check
- **terminal_count:** 26724
- **next_opponent** (n=26724): gt_0=2882, gt_50=838, gt_100=445, gt_250=136, gt_400=50, p99=140.0, max=958.6500244140625

### call
- **terminal_count:** 28896
- **next_opponent** (n=28896): gt_0=2811, gt_50=2279, gt_100=2022, gt_250=1710, gt_400=1452, p99=1000.0, max=1000.0

### raise
- **terminal_count:** 0

---

## 5. Replay vs q_compare Distribution

- **q_compare pot:** mean=20.1, p50=9.2, p90=37.3, max=395.8, n=300
- **q_compare legal_actions:** mean=3.0, p50=3.0, p90=3.0, max=3.0, n=300
- **pot_distribution_match:** cannot_compare

---

## 6. Most Likely Root Cause

- **H1: Raise структурно обездолен.** Bootstrap-only + низкий потолок (~+0.4). Fold>raise подтверждён на legal states. Это reward-grounding асимметрия.

---

## 7. What NOT to do

- Не запускать новое обучение до разрешения H1/H2.
- Не включать legal-anchor mask / kind-filter.
- Не менять action-Q архитектуру без диагностики.

---

## 8. Next Experiment

1. **Action-Q delayed reward propagation** — n-step return или MC target для raise chains.
2. Отдельно: проверить reward sign для hero fold terminal samples.
3. После фикса action-Q — повторный прогон с B1 ON.

---

## 9. Grad-Clip & Scale Audit

> **Script:** `tools/action_q_grad_clip_scale_audit.py`
> **Checkpoints:** `multi_checkpoint_iter_100_q_grad_clip_audit.json` (+ `_bb500` variant)

### 9.1 reward_unit=2.0

| Scenario | Target abs(mean) | Target abs(max) | Raw grad | Post grad | Throttle | Eff step |
|---|---|---|---|---|---|---|
| mixed_natural | 2.94 | 244.67 | 120.04 | 1.00 | 120.04× | 0.001 |
| terminal_only | 37.12 | 500.00 | 734.09 | 1.00 | 734.09× | 0.001 |
| bootstrap_only | 0.73 | 24.49 | 7.36 | 1.00 | 7.36× | 0.001 |
| raise_only | 0.29 | 3.00 | 0.69 | 0.67 | 1.03× | 0.001 |

**Mixed decomposition:**
- terminal grad norm: 671.39, bootstrap grad norm: 3.45
- grad ratio terminal/bootstrap: **194.72×**
- terminal target abs(mean): 38.72, bootstrap target abs(mean): 0.68

**Verdict flags:**
| Flag | Value |
|---|---|
| terminal_raw_grad_much_larger (ratio > 5) | **True** (ratio=99.69) |
| post_clip_steps_comparable (0.5 < ratio < 2) | **True** (ratio=1.0) |
| terminal_grad_heavily_throttled (throttle > 5) | **True** (throttle=734.09) |
| bootstrap_grad_barely_clipped (throttle < 1.5) | **False** (throttle=7.36) |
| **scale_bottleneck_confirmed** | **False** |

### 9.2 reward_unit=500.0 (bb500)

| Scenario | Target abs(mean) | Target abs(max) | Raw grad | Post grad | Throttle | Eff step |
|---|---|---|---|---|---|---|
| mixed_natural | 0.65 | 21.83 | 28.04 | 1.00 | 28.04× | 0.001 |
| terminal_only | 0.15 | 2.00 | 554.01 | 1.00 | 554.01× | 0.001 |
| bootstrap_only | 0.73 | 24.49 | 7.36 | 1.00 | 7.36× | 0.001 |
| raise_only | 0.29 | 3.00 | 0.69 | 0.67 | 1.03× | 0.001 |

**Mixed decomposition:**
- terminal grad norm: 232.94, bootstrap grad norm: 3.45
- grad ratio terminal/bootstrap: **67.56×**
- terminal target abs(mean): 0.155, bootstrap target abs(mean): 0.684

**Verdict flags:**
| Flag | Value |
|---|---|
| terminal_raw_grad_much_larger (ratio > 5) | **True** (ratio=75.23) |
| post_clip_steps_comparable (0.5 < ratio < 2) | **True** (ratio=1.0) |
| terminal_grad_heavily_throttled (throttle > 5) | **True** (throttle=554.01) |
| bootstrap_grad_barely_clipped (throttle < 1.5) | **False** (throttle=7.36) |
| **scale_bottleneck_confirmed** | **False** |

### 9.3 Interpretation

- **Scale bottleneck NOT confirmed** в обоих вариантах. Grad-clip не первопричина Q-сжатия в ±1.
- При `reward_unit=2.0`: terminal abs(mean) = 37.12, bootstrap = 0.73. Terminal градиенты ~735× raw, но clip приводит оба к post-norm=1.0 — эффективный шаг одинаков.
- При `reward_unit=500.0`: terminal targets сжаты до 0.15 (деление на reward_unit), но terminal градиенты всё равно 554×.
- **Bootstrap тоже клипается** (7.36×, порог 1.5× не пройден → flag `bootstrap_grad_barely_clipped` = False). Это означает, что bootstrap-сигнал хоть и слабее terminal, но тоже испытывает clipping.
- Terminal сэмплов в mixed batch: ~8-21 из 256 (3-8%). Их сигнал задавлен не clipping'ом, а **loss weighting / batch composition**.
- **Причина Q-сжатия:** скорее всего (а) bootstrap численно доминирует в loss; (б) mean reduction размазывает terminal градиенты; (в) выходной диапазон сети ограничен архитектурой/инициализацией.

**Next checks:**
1. Проверить `loss reduction` (mean vs sum) и долю terminal в total loss.
2. Попробовать terminal loss upweighting (`weight = batch_size / n_terminal`).
3. Отдельный прогон с `max_norm = 10/100` для изоляции clipping vs weighting.