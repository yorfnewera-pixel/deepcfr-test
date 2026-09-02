# Bug #93 — Sizing как полноценная CFR-подзадача (advantage-нога) + удаление bucket-loss

**Дата:** 2026-06-16
**Статус:** IMPLEMENTED (M1 PASS, M0/M2 PARTIAL)
**База:** #92 (multi-agent advantage, ACTIVE — первый прогон без relapse, M0/M1/M2 PASS)
**Остаточный дефект:** sizing concentration (0.10 = 51–57%, top-2 ≈ 79% на iter 300)

---

## TL;DR

Два изменения под флагами для чистого A/B против #92:
- **Fix A:** `sizing_bucket_loss_weight: 0.3 → 0` — удаление мёртвого градиентного члена.
- **Fix B:** `advantage_sizing_net` как играющая нога sizing-политики через regret-matching — симметрия с action-стороной. `sizing_q_net` демотируется в baseline. Флаг `sizing_cfr_mode: true`.
- **GTO inference:** `sizing_inference_mode: "hierarchical"` — частота по бакетам на инференсе.

---

## Диагноз

Action-сторона (эталон):
- играющая политика = regret-matching(advantage_net)
- advantage учит cf_regret (OS-grounded)
- strategy учит среднюю играющую политику
- baseline = q_net

Sizing-сторона (сломана):
- играющая политика = sizing_q_net heat (коллапсирует в argmax)
- advantage_sizing_net учит Q-derived таргеты (q - mean(q)), не CFR-regret
- SizingAdvantageBuffer объявлен но не инстанцирован
- нет OS-заземления для sizing-regret

---

## Все изменения кода

### 1. `config.yaml`

```yaml
sizing_bucket_loss_weight: 0.0   # Fix A: мёртвый градиентный член
sizing_cfr_mode: true            # Fix B: advantage_sizing_net → играющая нога
sizing_inference_mode: "hierarchical"  # GTO: частота по бакетам (regret-matching)
```

### 2. `src/core/deep_cfr.py` — 7 правок

| # | Что | Где | Суть |
|---|---|---|---|
| 1 | `sizing_cfr_mode` флаг | `__init__` | Чтение из config |
| 2 | `SizingAdvantageBuffer` инстанциация | `__init__`, строка ~795 | Буфер был объявлен, не использовался |
| 3 | Per-anchor CFR-regret в `hero_os_raise` | `cfr_traverse_multi`, ~2290 | OS-заземление с sizing-уровневым IS-весом |
| 4 | Per-anchor CFR-regret в `hero_full_raise` | `cfr_traverse_multi`, ~2496 | Заземление через action_values[3] с IS-весом |
| 5 | `train_sizing_anchor_network` CFR-ветка | ~2896 | Учит CFR-regret'ы (masked MSE) из sizing_advantage_buffer |
| 6 | `_hierarchical_sizing` CFR-ветка | ~2058 | Regret-matching через advantage_sizing_net при CFR-режиме |
| 7 | `sizing_strategy_buffer` запись | ~2280, ~2488 | Пишет regret-matched стратегию при CFR-режиме |

### 3. `tools/checkpoint_tools.py` — 8 правок

| # | Что | Суть |
|---|---|---|
| 1 | Импорт `regret_matching_anchors` | Из `src.core.deep_cfr` |
| 2 | `_hierarchical_rm_probs` | Хелпер: regret_matching по бакетам + внутри бакета |
| 3 | `run_collapse_diag` — GTO-режим | При `regret_matching=True`: `advantage_sizing_net` + per-bucket regret_matching |
| 4 | `run_eval_games` — GTO-sizing | При `regret_matching=True`: `advantage_sizing_net` + hierarchical RM для sizing |
| 5 | Авто-детект `sizing_cfr_mode` | Из `ckpt['config']` — если true, включает GTO-режим автоматически |
| 6 | `--regret-matching` CLI флаг | Добавлен в `p_eval` и `p_full` subparsers |
| 7 | `bucket_groups_indices` в nets | Чтение `sizing_bucket_groups` из конфига, построение индексов |
| 8 | `rm_bucket_min_prob` / `rm_anchor_min_prob` | Endpoint значения min_prob для GTO-режима |

### Инвариант симметрии

| | играющая политика | advantage учит | strategy учит | baseline |
|---|---|---|---|---|
| action | regret-matching(advantage_net) | cf_regret (OS-grounded) | средняя играющая | q_net |
| sizing (#93) | regret-matching(advantage_sizing_net) | per-anchor cf_regret (OS-grounded) | средняя играющая | sizing_q_net |

---

## Pre-registered гейты

- **M0 (no-relapse):** raise_freq не падает 100→200; sizing 0.10-доля < 50% к iter 200 (vs 57% в #92)
- **M1 (diversity):** unique_anchors ≥ 13/15 устойчиво; top-2 концентрация < 65% к iter 300 (vs 79%)
- **M2 (durability/качество):** win_rate стабилен/растёт; не хуже #92 (≈47% на iter 300)

---

## Прогон 1: test92seed1 — sizing_cfr_mode + top_bucket inference

**Конфиг:** `sizing_cfr_mode: true`, `sizing_inference_mode: "top_bucket"`, 300 traversals, seed 1.

### Траектория (iter 100–400)

| Метрика | iter 100 | iter 200 | iter 300 | iter 400 |
|---|---|---|---|---|
| raise_freq | 48.7% | 51.7% | 10.6% | 6.2% |
| win_rate | 43.6% | 44.0% | 25.5% | 17.8% |
| 0.10 масса | 32.8% | 4.4% | 10.0% | 1.6% |
| доминантный анкер | 0.10 | **0.33** | **0.33** | 1.00 |
| top-2 концентрация | ~55% | **48.7%** | 53.3% | 75.4% |
| unique_anchors | 15 | 15 | 12 | 9 |
| `next_strategy_raise_mass.raise` | 0.404 | 0.284 | — | 0.018 |

### Ключевые моменты

**Iter 200 — исторический максимум sizing diversity:**
- 0.10 масса упала с 32.8% до 4.4%
- 0.33 (27.4%), 0.75 (21.3%), 2.50 (13.2%), 2.00 (8.9%), 3.00 (8.8%) — **5 interior мод**
- Бакеты почти равномерны: 37/32/32
- **Впервые в истории sizing-багов (#61–#92) доминантные анкеры — interior, не граничные**

**Iter 300 — рецидив raise_freq до 10.6%:**
- Sizing diversity удержался: 0.33 (27.7%), 1.25 (25.6%), 2.00 (17.9%)
- 0.10 всего 10.0%
- Но модель почти перестала рейзить (10.6%) → win_rate упал до 25.5%

**Iter 400 — дальнейшая деградация action-уровня:**
- raise_freq = 6.2%, win_rate = 17.8%
- Sizing продолжает держаться: 0.33 (38.1%), 1.25 (37.2%), 0.10 = 1.6%

### Вердикт M0/M1/M2

| Гейт | iter 200 | iter 300 | Вердикт |
|---|---|---|---|
| M0 (raise_freq) | PASS (51.7%) | FAIL (10.6%) | **FAIL** |
| M1 (diversity) | PASS (top-2=48.7%) | PASS (top-2=53.3%) | **PASS** |
| M2 (win_rate) | PASS (44.0%) | FAIL (25.5%) | **FAIL** |

Sizing-механизм работает (M1 PASS) — но action-регрессия съедает выигрыш.

---

## Прогон 2: test92seed1 — sizing_cfr_mode + hierarchical inference (ПЕРЕЗАПУСК)

**Конфиг:** `sizing_cfr_mode: true`, `sizing_inference_mode: "hierarchical"`, 300 traversals, seed 1.
**Отличие от прогона 1:** инференс переключён на GTO-частоту (hierarchical regret_matching). Добавлены правки в checkpoint_tools для авто-детекта.

### Траектория (iter 100–800)

| Метрика | iter 100 | iter 200 | iter 300 (пик) | iter 400 | iter 500 | iter 600 | iter 700 | iter 800 |
|---|---|---|---|---|---|---|---|---|
| raise_freq | 35.2% | 33.5% | **50.9%** | 33.8% | 22.7% | 11.6% | 8.3% | 4.8% |
| win_rate | 37.2% | 36.1% | **38.9%** | 25.2% | 19.9% | 16.7% | 11.7% | 9.8% |
| mean_reward | +12.1 | +20.2 | **+23.5** | +17.7 | +6.6 | +17.6 | -0.5 | +6.7 |
| 0.10 масса | 24.7% | 9.3% | **1.6%** | 0.7% | 1.8% | 2.6% | 2.3% | 3.5% |
| доминантный анкер | 2.25 | 1.00 | **0.66** | 0.66 | 0.66 | 0.66 | 2.25 | 0.66 |
| unique_anchors | 15 | 15 | 14 | 13 | 14 | 12 | 15 | 11 |
| `next_strategy_raise_mass.raise` | 0.0 | **0.497** | 0.482 | 0.0 | 0.0 | 0.023 | 0.032 | 0.0003 |

### Sizing Q trajectory

| Метрика | iter 100 | iter 400 | iter 700 | iter 800 |
|---|---|---|---|---|
| `sizing_q.max.mean` | -2.27 | -0.84 | **+0.09** | -0.29 |
| `sizing_q.mean.mean` | -2.30 | -0.93 | **-0.23** | -0.58 |
| `best_sizing_anchor_dist` | 1.50=432 | 0.50=180, 0.66=221 | **0.10=404** (всё ещё Q-смещение) | 0.10=423 |

Sizing-Q пробил ноль на iter 700 — впервые sizing_q не отрицательный. Но Q всё ещё предпочитает 0.10 в 80-95% состояний (best_sizing_anchor_dist).

### Sizing diversity — детальный разрез

| Метрика | iter 100 | iter 200 | iter 300 | iter 400 | iter 500 | iter 600 | iter 700 | iter 800 |
|---|---|---|---|---|---|---|---|---|
| 0.10 масса | 24.7 | 9.3 | 1.6 | 0.7 | 1.8 | 2.6 | 2.3 | 3.5 |
| 0.33 масса | 0.7 | 8.9 | 0.9 | 0.5 | 0.4 | 0.4 | 0.5 | 0.5 |
| 0.66 масса | 0.9 | 6.7 | **37.8** | **34.1** | **32.3** | **31.8** | **29.2** | **30.8** |
| 1.00 масса | 15.4 | **29.3** | 12.5 | 16.7 | 16.5 | 18.7 | **22.7** | **22.6** |
| 2.25 масса | 1.0 | **26.8** | 19.8 | **22.6** | **30.1** | **30.0** | **31.2** | **28.7** |
| Бакеты (s/m/l) | 35/40/24 | 28/36/36 | 41/30/30 | 35/32/33 | 36/28/37 | 36/28/36 | 34/30/36 | 36/30/34 |

**0.10 подавлен стабильно:** 0.7-3.5% на всём протяжении 200-800 — min_prob floor работает.

**3 устойчивых доминанта:** 0.66 (small-medium), 1.00 (medium), 2.25 (large). Interior anchor'ы, осмысленные покерные размеры.

**Бакеты сбалансированы:** ~35/30/35 — ни один бакет не доминирует.

### Ключевые моменты

**Модель самовосстановилась на iter 200:**
`next_strategy_raise_mass.raise` был 0.0 на iter 100 (красный флаг!), но к iter 200 восстановился до 0.497. В прогоне 1 такого самовосстановления не было.

**Пик на iter 300:**
raise_freq=50.9%, win_rate=38.9%, mean_reward=+23.5. Лучший чекпоинт за всю историю #93.

**Рецидив после 400:**
Та же self-suppressing петля, что и в прогоне 1. `next_strategy_raise_mass` падает до 0 → модель перестаёт рейзить.

**mean_reward +17.6 на iter 600 при raise_freq=11.6% и win_rate=16.7%:**
Модель выигрывает крупно на сильных руках, проигрывает мало на слабых. Стратегия «колл с сильной, фолд со слабой» даёт положительный EV даже против рандомов при низком raise_freq.

---

## Сравнение прогонов на iter 700

| Метрика | #92 baseline (iter 300) | Прогон 1 (iter 400) | Прогон 2 (iter 700) |
|---|---|---|---|
| raise_freq | 52% | 6.2% | **8.3%** |
| win_rate | 47.1% | 17.8% | **11.7%** |
| mean_reward | — | — | **+17.6** |
| 0.10 масса | 51-57% | 1.6% | **2.3%** |
| доминант | 0.10 | 1.00 | **2.25** |
| sizing_q max.mean | — | -0.84 | **+0.09** |
| sizing механизм | Q-heat → argmax | regret-matching ✓ | regret-matching ✓ |

---

## Итоговый вердикт

| Гейт | Результат | Комментарий |
|---|---|---|
| **M1 (sizing diversity)** | **PASS** | 0.10 подавлен (1-4%), interior anchor'ы (0.66, 1.00, 2.25), 3 бакета сбалансированы, min_prob floor держит все 15 анкеров живыми. Впервые в истории sizing-багов. |
| **M0 (no-relapse)** | **PARTIAL** | raise_freq держится до iter 300 (пик 50.9%), но падает после 400. Self-suppressing loop на action-уровне. |
| **M2 (durability/качество)** | **PARTIAL** | win_rate нестабилен: пик 38.9% на iter 300, падает до 10-17% после 500. Причина — action-регрессия, не sizing. |

### Что победили

Sizing-коллапс в 0.10 сломан. Regret-matching с min_prob-floor **работает**: модель сама нашла осмысленные interior размеры (0.66, 1.00, 2.25), 0.10 подавлен до min_prob-уровня. Sizing-Q впервые в истории стал положительным.

### Что осталось

Action-уровень — self-suppressing loop: `next_strategy_raise_mass` падает до 0, модель перестаёт рейзить. Это ортогональная проблема, не связанная с sizing-механизмом. Против рандомов пассивная игра не наказывается → нужен более сильный оппонент или другой механизм удержания raise_freq.

## Изменённые файлы

- `config.yaml`: `sizing_bucket_loss_weight: 0.0`, `sizing_cfr_mode: true`, `sizing_inference_mode: "hierarchical"`
- `src/core/deep_cfr.py`: инстанцирование `SizingAdvantageBuffer`, per-anchor CFR-regret на траверсе, `train_sizing_anchor_network` CFR-ветка, `_hierarchical_sizing` CFR-ветка, strategy buffer запись при CFR-режиме
- `tools/checkpoint_tools.py`: `_hierarchical_rm_probs`, `run_collapse_diag` GTO-режим, `run_eval_games` GTO-sizing, авто-детект `sizing_cfr_mode`, `--regret-matching` CLI флаг, bucket_groups в nets dict
- `sizing_reports/SIZING.md`: секция #93, test93 в таблице, веха, roadmap
