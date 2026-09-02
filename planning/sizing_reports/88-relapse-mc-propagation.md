# Bug #88 — Relapse Under MC Propagation (iter 100→400, test88seed1)

**Дата:** 2026-06-14
**Статус:** Arm B' PRE-REGISTERED (test88seed1 диагностика завершена. Гипотеза: обрыв на iter 100 — от `q_target_update_interval=100`. Arm B': sync интервал 100→10. Если M0/M1 PASS но M2 FAIL → Arm C refit. Если и C FAIL → #88b discount, #88c recency.)
**Severity:** High (MC propagation устраняет #84-relapse до iter 200, но не до 400)
**Конфиг:** `sizing_q_kind_filter_enabled: false`, mask в dry_run, max_norm=100, MC propagation (#86 baseline)

---

## Краткое описание

Диагностический прогон `test88seed1` — multi-checkpoint (iter 100/200/300/400) на фиксированном наборе из 500 состояний (seed=42) для проверки durability MC-propagation baseline'а (#86). #86 показал отсутствие relapse до iter 250; test88seed1 проверяет, держится ли это до iter 400.

**Ответ: НЕТ.** Relapse возвращается на iter 200 (тот же механизм, что в #84) — advantage/regret-сеть делает flip fold↔raise, политика коллапсирует в fold+call-bot.

## Сводка по итерациям

| Итерация | Вердикт | Причина | Win Rate | Raise % | Доминирующий анкер | Unique anchors |
|----------|---------|---------|----------|---------|---------------------|----------------|
| **100** | **OK** | raise_freq=40.8%, sizes=15/15 | 42.2% | 40.8% | 0.66 (17.5%) | 15 |
| **200** | **WARN** | top-2 mass=78% >55%; low_raise=16.0% | 17.5% | 16.0% | **0.10 (55.5%)** | 8 |
| **300** | **WARN** | top-2 mass=83% >55%; low_raise=9.2% | 17.2% | 9.2% | **0.10 (57.7%)** | 7 |
| **400** | **WARN** | top-2 mass=88% >55%; low_raise=8.1% | 20.6% | 8.1% | **0.10 (64.6%)** | 6 |

## Траектория деградации (4 канала)

### A. Катастрофа raise-частоты (на фиксированных состояниях)

```
Iter 100: raise_freq = 58.5% (агрессивная, разнообразная игра)
Iter 200: raise_freq = 32.7% (потеря −25.8 п.п.)
Iter 300: raise_freq =  0.2% (почти ноль — МЕРТВА)
Iter 400: raise_freq =  0.8% (едва дышит)
```

Стратегия **умерла за 200 итераций**: от агрессивного покера до пассивного call-bot.

### B. Монополизация анкера 0.10x (fixed-state raw probs)

```
Iter 100: anchor_0.10 = 10.9%, anchor_0.66 = 17.5% (разнообразие)
Iter 200: anchor_0.10 = 55.5% (доминирует)
Iter 300: anchor_0.10 = 57.7% (доминирует)
Iter 400: anchor_0.10 = 64.6% (усиление доминирования)
```

Когда политика всё-таки рейзит — только min-рейз (0.10x). Anchor 0.10: selected_pct = 100% с iter 200.

### C. Концентрация top-2 анкеров

```
Iter 100: 32.6%  (здоровое распределение)
Iter 200: 77.9%  (тревога)
Iter 300: 82.1%  (WARN)
Iter 400: 86.7%  (WARN — ухудшается)
```

### D. Win-rate обвал

```
Iter 100: 42.2% (хорошо)
Iter 200: 17.5% (потеря 24.7 п.п.)
Iter 300: 17.2% (стагнация)
Iter 400: 20.6% (лёгкое улучшение, но далеко от iter 100)
```

---

## Механизм relapse (срез 100→200 — точка перелома)

### Первый сигнал слома: advantage/regret-сеть делает полный FLIP

| Метрика | iter 100 | iter 200 | Δ |
|---------|----------|----------|---|
| advantage fold mean | −5.10 | −1.40 | **+3.69** |
| advantage raise mean | +0.64 | +0.43 | **−0.22** |
| policy raise % | 58.5% | 32.7% | −25.8 |
| q_raise mean | −0.063 | −0.023 | +0.039 |
| q_fold mean | −0.010 | −0.011 | −0.001 |

Advantage/regret делает поворот fold←raise за 100 итераций. Q-сеть стабильна — raise_lt_fold = 77.4% → 67.5%, она ВСЕГДА считала fold > raise.

### Цепная реакция (подтверждена всеми 4 срезами)

| Срез | adv_fold Δ | adv_raise Δ | raise_freq Δ | next_raise_mass |
|------|-----------|-------------|-------------|-----------------|
| 100→200 | +3.69 | −0.22 | −25.8 п.п. | 0.37 → 0.38 |
| 200→300 | +1.93 | **−0.43** | −32.5 п.п. | 0.38 → 0.04 |
| 300→400 | −0.58 | +0.005 | +0.6 п.п. | 0.04 → 0.08 |

1. Advantage-сеть теряет предпочтение raise (H3 — накопление regret) → политика сдвигается в fold
2. Fold-сэмплы в буфере взрываются ×34 (8 402 → 284 993) → буфер отражает сдвиг стратегии
3. next_strategy.raise_mass падает 0.37 → 0.04 → bootstrap V для raise обнуляется (H2 — bootstrap усиливает)
4. target для raise-сэмплов = 0 → петля замыкается

### Сравнение с #84 relapse

| Метрика (100→200) | #84 (max_norm=100) | #88 (MC propagation) |
|--------------------|--------------------|----------------------|
| adv_fold Δ | +8.47 | **+3.69** (в 2.3× меньше) |
| adv_raise Δ | −2.48 | **−0.22** (в 11× меньше) |
| raise_freq Δ | −26.5 п.п. | −25.8 п.п. |
| q_raise Δ | +0.18 | +0.039 |
| q_fold Δ | −0.33 | −0.001 |

**MC propagation смягчает advantage-flip в 2-11×** — advantage не делает такой резкий скачок к fold, как в #84. Но raise_freq всё равно падает на те же 26 п.п. Политика гиперчувствительна к advantage: даже небольшой сдвиг преимущества вызывает массовый переход от raise к fold.

### Smoking gun: reward=0 у raise

Из buffer-диагностики: ВСЕ raise-сэмплы во всех чекпоинтах имеют `reward_mean ≈ 0`, `terminal_pct = 100%`. Raise никогда не видит терминального исхода руки — даже с MC propagation. Propagation доносит terminal value через TD-bootstrap, но исходный raise-сэмпл всё равно bootstrap-only.

---

## Гипотезы (из relapse_diff_all.json)

| Гипотеза | Описание | Статус |
|----------|----------|--------|
| H1 (buffer fold-terminal) | Буфер наполняется fold-terminal сэмплами → цели смещаются в пользу fold | **НЕ ПОДТВЕРЖДЕНА** — недостаточно данных |
| **H2 (Q-деградация)** | q_net деградирует и через bootstrap портит advantage-цели | **ПОДТВЕРЖДЕНА** — q_raise_mean отрицателен на всех итерациях |
| **H3 (regret-накопление)** | Накопление отрицательных regret за raise перевешивает и возвращает политику | **ПОДТВЕРЖДЕНА** — advantage raise падает с +0.64 до −0.003 за 200 итераций |

### Детали H2 (Q-деградация по итерациям)

```
iter 100: q_raise_mean=−0.063, q_fold_mean=−0.010, raise_lt_fold=77.4%
iter 200: q_raise_mean=−0.023, q_fold_mean=−0.011, raise_lt_fold=67.5%
iter 300: q_raise_mean=−0.010, q_fold_mean=−0.003, raise_lt_fold=87.8%
iter 400: q_raise_mean=+0.002, q_fold_mean=+0.005, raise_lt_fold=71.7%
```

Q-сеть **всегда** предпочитает fold > raise (67-88% состояний). На iter 400 Q восстанавливается и raise даже становится выше fold по mean, но raise_lt_fold всё равно 71.7% — Q слишком сжат, разница статистически незначима.

### Детали H3 (regret-накопление)

```
100→200: advantage raise  0.6441 → 0.4287  (δ=−0.2154)
           advantage fold  −5.0995 → −1.4049 (δ=+3.6946) — колоссальный сдвиг к fold
200→300: advantage raise  0.4287 → −0.0029 (δ=−0.4316) — ПЕРЕХОД В ОТРИЦАТЕЛЬНУЮ
           advantage fold  −1.4049 → +0.5267 (δ=+1.9315)
300→400: advantage raise  −0.0029 → +0.0016 (δ=+0.0045) — стабилизация около нуля
           advantage fold  +0.5267 → −0.0559 (δ=−0.5826)
```

---

## Дополнительные наблюдения

### Sizing-Q: отрицательный max на всех чекпоинтах

```
iter 100: sq_max.mean = −2.999
iter 200: sq_max.mean = −0.337 (улучшилось в 9×)
iter 300: sq_max.mean = −0.414
iter 400: sq_max.mean = −0.379
```

Max sizing-Q отрицателен на всех итерациях → argmax всегда указывает на 0.10 (первый анкер). Даже когда Q-сеть частично восстанавливается, sizing-Q остаётся монотонно убывающим по размеру.

### Q-разрыв raise vs max_sizing_q

```
iter 100: raise > max_sq в 100.0% состояний — Q_raise = −0.063 >> max_sq = −3.0
iter 200: raise > max_sq в 91.8%  состояний — разрыв сократился
iter 300: raise > max_sq в 80.2%  состояний
iter 400: raise > max_sq в 82.4%  состояний
```

Несмотря на сильный разрыв (raise имеет намного более высокий Q, чем sizing), политика всё равно не рейзит. Advantage/regret-сеть перевешивает Q-сигнал.

### Буфер: fold-взрыв (100→200)

```
fold-сэмплы:   8 402 → 284 993 (рост в 34×)
call-сэмплы: 744 037 → 359 774 (падение в 2×)
raise-сэмплы: 246 958 → 223 908 (стабильно)
```

Буфер наводняется fold-сэмплами после advantage-flip. Это НЕ первопричина, а следствие смены стратегии (H1 опровергнута).

### Target diagnostics: source2 плоский, но живой

```
source2_mean_range: 0.04-0.08 (плоский по анкорам)
source2_n_anchors: 15/15 (все анкеры представлены)
source2 clip_high/clip_low: 0 (нет клипов)
source0/1 clip_low: 17k-52k (массовые нижние клипы)
```

Source2 (lookahead) стабилен и не клипается, но его target'ы почти нулевые по всем анкорам → не даёт полезного градиента sizing-Q.

---

## Вывод

**MC propagation задерживает, но не устраняет relapse.** Advantage-flip смягчён в 2-11× относительно #84, но политика гиперчувствительна — даже небольшой сдвиг advantage вызывает массовый отказ от raise. Root cause тот же: reward-grounding asymmetry — raise никогда не получает terminal reward напрямую. MC propagation доносит terminal value через TD-bootstrap, но advantage-сеть всё равно накапливает негативный regret за raise быстрее, чем propagation успевает это компенсировать.

### Открытые вопросы

1. **Почему advantage-flip происходит несмотря на MC propagation?** MC propagation добавляет delayed reward в TD-target (через bootstrap), но `terminal_pct=100%` у raise означает, что реальный исход руки до raise-узла не доходит.
2. **Гиперчувствительность advantage→policy:** advantage raise падает всего на 0.22 (с +0.64 до +0.43), а политика с 58.5% до 32.7%. Softmax/temperature делает политику слишком чувствительной к малым изменениям advantage.
3. **Sizing-Q монотонный:** даже при восстановлении Q, sizing-Q остаётся монотонно убывающим. Это отдельный баг в sizing-Q обучении.

---

## Данные

| Файл | Содержание |
|------|-----------|
| `models/test88seed1/relapse_diff_all.json` | Полный diff 100/200/300/400 на 500 состояниях |
| `models/test88seed1/relapse_diff_100v200.json` | Детальный diff 100→200 |
| `models/test88seed1/relapse_diff_iter100.json` | Снэпшот iter 100 |
| `models/test88seed1/multi_checkpoint_iter_*_full_report.json` | Полные отчёты eval (1000 игр) |
| `models/test88seed1/multi_checkpoint_iter_*_light.pt` | Light чекпоинты |

---

## Arm B' — q_target_update_interval: 100 → 10 (PRE-REGISTERED)

### Мотивация

Референс (`DeepPDCFR-master`, `QValueTrainer.train_model`) синкает Q-target каждые **50 train-шагов внутри итерации** на свеже-инициализированной сети. Проект синкает раз в **100 CFR-итераций** (`q_target_update_interval=100`). Первый синк — на iter 100 → до этого `q_target_net.q_head = zero-init` (подтверждено в #74-v6). Это создаёт «ступеньку» на iter 100: TD-bootstrap через мёртвую сеть первые 99 итераций.

Arm B' — диагностический зонд: убирает ступеньку (sync=10 вместо 100), но сохраняет минимальную стабилизацию (10 — компромисс между sync=1 и sync=100; sync=1 на персистентной сети с 2-5 train-шагами/iter рискует осцилляцией).

### Дизайн

| Арм | Переменная | Как |
|-----|-----------|-----|
| **B'** (сейчас) | `q_target_update_interval: 100 → 10` | 1 строка в config.yaml |
| A (Polyak) | Мягкий синк `θ_t ← 0.01·θ + 0.99·θ_t` | новый метод в `_sync_q_target_net` |
| C (refit) | Re-init Q с нуля каждую итерацию + best-model | по образцу референса |

**Важно:** Arm B' — это НЕ референс. Reference синкает раз в 50 шагов внутри свежего фита с нуля. B' сохраняет персистентную сеть и просто убирает заморозку. Настоящий референс — Arm C.

### Критерии

| Gate | Условие | Где проверять |
|------|---------|--------------|
| **M0** (обрыв ушёл) | Нет ступеньки на iter 100; raise_freq(200) ≥ 30% | `checkpoint_tools.py` + `relapse_diff.py` |
| **M1** (механизм) | q_raise.mean ≥ q_fold.mean держится за iter 100; next_strategy_raise_mass не падает к 0 | `relapse_diff.py` q_diff |
| **M2** (durability) | verdict=OK на ≥3/4 чекпоинтов; raise≥30%, unique≥12, top-2≤60% | `full_report.json` verdict |

PASS = M0+M1+M2 · PARTIAL = M0+M1, durability проседает → Arm C · FAIL = M0 не выполнен → обрыв не от value-target.

### Дерево решений

```
B' (sync=10), iter 400
├── M0 FAIL → обрыв не от sync-режима → возврат к diff, копать advantage-сторону
├── M0+M1 PASS, M2 FAIL → Arm C (refit — истинный референс) ДО трогания формул
│   ├── C PASS → refit = fix, Polyak опционально
│   └── C FAIL → #88b (discount α=1.5, +1.5) → #88c (recency)
└── M0+M1+M2 PASS → Arm A (Polyak τ=0.01) для устойчивости, Arm C для сравнения
```

### Что НЕ трогаем сейчас (отдельные PR при необходимости)

- **Discount-формула advantage:** референс DCFR+ использует `(T-1)^1.5/((T-1)^1.5+1.5)`, проект `(t-1)^2.0/((t-1)^2.0+1)`. У проекта слабее форгеттинг (α=2.0 насыщается быстрее). → #88b.
- **Recency-вес стратегии:** референс `(2t/T)^(γ/2)` (линейный), проект `(t/T)^γ` (квадратичный). Проект сильнее топит ранние итерации. → #88c.

### Изменения

| Файл | Что |
|------|-----|
| `config.yaml:27` | `q_target_update_interval: 100 → 10` |

### Референс-источники

- `DeepPDCFR-master/deeppdcfr/os_deep_cumu_adv.py:768-862` — QValueTrainer.train_model (re-init + target sync /50 + best-model)
- `DeepPDCFR-master/deeppdcfr/os_deep_cumu_adv_variants.py:231-247` — VRDCFRPlus discount-формула (α=1.5, знаменатель +1.5)
- `DeepPDCFR-master/deeppdcfr/os_deep_cumu_adv_variants.py:539-543` — AvePolicyTrainer recency-вес
