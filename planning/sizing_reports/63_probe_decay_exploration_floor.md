# Bug #63: Probe затухает к iter 1000 — крупные размеры недосэмплированы, sizing_q монотонно падает

**Severity:** HIGH

**Дата:** 2026-06-04

**Связан с:** Bug #61 (sizing collapse в 0.10), Bug #62 (q_network_gap)

---

## Симптомы

Диагностика `q_compare` на checkpoint 600 (`multi_checkpoint_iter_600_full_report.json`) показала:

```json
"best_sizing_anchor_dist": {
  "0.10": 2778,  ← 92.8% всех raise-legal состояний
  "0.25": 18,
  "0.75": 24,
  "3.00": 167,
  "остальные 11 анкеров": <10
}
"sizing_q_stats": {
  "max": { "mean": -1.43 }
}
"collapse": {
  "anchor_raw_probs": { "0.10": 0.595, "0.25": 0.267 },
  "small_bucket_mass": 0.865,
  "large_bucket_mass": 0.035
}
```

Q-поверхность `sizing_q_net` монотонно убывает по размеру: чем крупнее ставка, тем ниже Q. Крупные размеры (`1.00+`) почти не сэмплируются → их Q застывает заниженным.

---

## Root Cause: Probe затухает слишком быстро

Текущий probe-конфиг:

```
probe_prob(iter) = 0.30 - 0.25 * clamp(iter / 1000, 0, 1)
```

| iter | probe_prob |
|------|-----------|
| 0    | 0.300     |
| 200  | 0.250     |
| 400  | 0.200     |
| **600** | **0.150** |
| 800  | 0.100     |
| 1000+| 0.050     |

К iter 600 probe уже на 15%, а на 1000+ падает до 5% — exploration floor исчезает. Крупные размеры получают Q-сэмплы только через случайные probe-срабатывания, а их становится всё меньше.

Probe — **единственный** механизм, дающий крупным размерам Q-сэмплы вне текущего bias'а модели. Без probe sizing_q_net «забывает» крупные размеры, и их Q-оценки никогда не восстанавливаются.

---

## Механизм probe

```python
# src/core/deep_cfr.py — cfr_traverse_multi(), строка 1667-1680
probe_prob = self._current_sizing_probe_prob(iteration)
if self.sizing_q_enabled and random.random() < probe_prob:
    sampled_bet_size = self._sample_probe_size(slot_sizes_np)  # 40% uniform / 40% anchor / 20% jitter
    is_probe = True
```

`_sample_probe_size()` — 3-way mixture:
- **40%**: fully uniform `[0.10, 3.00]` — равные шансы всем размерам
- **40%**: random anchor из 15 фиксированных
- **20%**: anchor ± jitter 0.10

Probe влияет **только** на Q-буфер (`sizing_q_buffer`), **не** на inference-политику (`choose_action` использует `strategy_sizing_net`). Повышение probe безопасно — не ломает выбор действий в проде.

---

## Fix

**Изменён только `config.yaml`:**

```yaml
# было → стало
sizing_probe_prob_end: 0.05          → 0.15
sizing_probe_decay_iterations: 1000  → 2000
```

Новая формула:

```
probe_prob(iter) = 0.30 - 0.15 * clamp(iter / 2000, 0, 1)
```

| iter | Старый probe | Новый probe | Δ |
|------|-------------|-------------|---|
| 0    | 0.300       | 0.300       | — |
| 200  | 0.250       | 0.285       | +14% |
| 400  | 0.200       | 0.270       | +35% |
| 600  | 0.150       | **0.255**   | **+70%** |
| 800  | 0.100       | 0.240       | +140% |
| 1000 | 0.050       | 0.225       | +350% |
| 1200 | 0.050       | 0.210       | +320% |
| 2000+| 0.050       | 0.150       | +200% |

НЕ тронуто: `sizing_probe_prob_start`, probe weights (`uniform/anchor/jitter`), heat-параметры, `top_per_bucket`, SPR, EMA. Одно концептуальное изменение за раз.

---

## План верификации

1. Дообучить с checkpoint 600 (`multi_checkpoint_iter_600.pt`) на +400–600 итераций (до iter 1000–1200)
2. Снять диагностику на 800, 1000, 1200:
   ```
   py tools/checkpoint_tools.py full models/multi/multi_checkpoint_iter_800.pt --games 3000 --num-states 3000
   ```
3. Сравнить с baseline (`multi_checkpoint_iter_600_full_report.json`)

### Критерии успеха

| Метрика | Baseline (iter 600) | Цель |
|---------|---------------------|------|
| `best_sizing_anchor_dist["0.10"]` | 92.8% | Снижение, масса распределилась шире |
| `sizing_q_stats.max.mean` | -1.43 | Выросла (менее отрицательная) |
| `anchor_raw_probs["0.10"]` | 0.595 | Снизился |
| `raise_freq` | 16.8% | Не просел |
| `mean_reward` | 8.60 | Не просел |

### Тренд (не точка)

Q-буфер инертен — старый перекос в сторону `0.10` не исчезнет мгновенно. Оценивать по тренду 800 → 1000 → 1200, а не по одной точке.

---

## Ограничения

1. **Probe есть только в `cfr_traverse_multi`**, отсутствует в `_cfr_traverse_multi_outcome_node` (OS-узел). При `hybrid_os_enabled: true` часть траверсов идёт через OS-узел без probe — эффективная ставка probe ниже номинальной. Если после шага 1 крупные размеры всё ещё недосэмплированы — кандидат на шаг 2: добавить probe в OS-узел.

2. **`q_raise_vs_max_sq_diff` — не метрика.** Сравниваются разные масштабы: `q_net` в фишках vs `sizing_q_net` pot-normalized с clip ±5. Игнорировать или приводить к одному масштабу.

---

## Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `config.yaml` | `sizing_probe_prob_end: 0.05 → 0.15`, `sizing_probe_decay_iterations: 1000 → 2000` |

---

## Статус

**Открыт.** Ждёт дообучения с checkpoint 600 и снятия тренда на 800/1000/1200.

### Если probe помог лишь частично

Переход к one-step lookahead: в raise-узле для каждого размера сетки делать один `apply_action(size_k)` и оценивать `s_next` через `q_net`, записывая Q-сэмпл на каждый размер. Даёт плотное покрытие без полного обхода. Но только после проверки probe.
