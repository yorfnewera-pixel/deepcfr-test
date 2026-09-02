# 112 — Мёртвый параметр steps; 4000 шагов не сработали; план стратификации (v16)

> Дата: 2026-08-16. Следует за #111. Прогоны v14, v15, правки под v16.

---

## 1. Баг: `sizing_anchor_train_steps_per_iteration` был мёртвым параметром

`train.py:630` читал шаги так:

```python
sizing_steps = getattr(agent, 'sizing_anchor_train_steps_per_iteration',
                       getattr(agent, 'pg_train_steps_per_iteration', 5))
```

Но в `deep_cfr.py` не было `self.sizing_anchor_train_steps_per_iteration = cfg_get(...)`,
поэтому `agent` не имел этого атрибута, и `getattr` молча падал на fallback
`pg_train_steps_per_iteration` = 10. Правка `config.yaml: 10 → 4000` (v14) не имела эффекта —
в консоли v14 было `Sizing anchor loss: ... (10 steps)`, а не 4000.

### Исправление

- `deep_cfr.py` `__init__`: явное чтение `int(cfg_get('sizing_anchor_train_steps_per_iteration', 10))`.
- `train.py:630`: убран fallback целиком — `sizing_steps = agent.sizing_anchor_train_steps_per_iteration`
  (опечатка больше не будет тихой, упадёт с ошибкой).
- `config.py`: default `5 → 10` (не менять поведение старых конфигов).

Проверено: `agent.sizing_anchor_train_steps_per_iteration = 4000`.

### Соседи проверены грепом

- `strategy_sizing_train_steps` — читается явно (`deep_cfr.py:238`) ✓
- `sizing_anchor_lr` — читается явно (`deep_cfr.py:223`) ✓
- `sizing_q_train_steps_per_iteration` — читается (A/B 10→200 в v10 дал сдвиг Q) ✓

Только `sizing_anchor_train_steps_per_iteration` был мёртвым.

---

## 2. v15: 4000 шагов не сработали

### RATIO на holdout (каждые 5 итераций)

```
iter 5/10/15/20/25/30:  1.164 / 1.228 / 1.116 / 1.116 / 1.111 / 1.109
```

Стабилизировался на ~1.11, **не падает** к потолку 0.63 и не опускается ниже 1.0.

### EV гейт (`blueprint − uniform`, 400 раздач)

| iter | diff | t | вердикт |
|---|---|---|---|
| 5 | +15.59 | +1.87 | шум |
| 10 | −0.85 | −0.16 | шум |
| 20 | +1.05 | +0.90 | шум |
| 30 | +1.07 | +0.52 | шум |

Сеть ≈ uniform на всех точках. Сопутствующее: `oracle_mc − uniform` = +52.8 / +30.0 / +13.7 / +4.2
(значимо, убывает); `oracle_mc − blueprint` = +37.2 / +30.8 / +12.7 / +3.1.

### Причины расхождения с офлайн-замером

Офлайн (полный буфер v13, 521k, no-bootstrap) показывал падение holdout 1.204 → 1.038. В живом
прогоне эффекта нет. Две причины:

1. **Бутстрап активен** в живом прогоне (`target = target_net·discount + regret`), офлайн был
   `no-bootstrap`.
2. **4000 шагов = ~2–4 эпохи** на полном резервуаре (464k+), а офлайн-замеру понадобилось ~11
   эпох (~20000 шагов) для 1.038.

---

## 3. v16: стратификация свежести 50/50 + шаги 2000

### Цель

Поднять долю строк, дающих градиент: сейчас fresh 4.6% → эффективный батч ~12 из 256.

### Правки

- `buffers.py`: `SizingAdvantageBuffer.sample_fresh_stratified(batch_size, iteration)` — квота
  50/50 fresh/stale (fresh = строки текущей итерации), добивка stale при недоборе, возвращает
  `fill_count` (недобор fresh).
- `deep_cfr.py`: флаг `sizing_advantage_stratified_sampling` (default false); в
  `train_sizing_anchor_network` при флаге используется `sample_fresh_stratified`; накапливаются
  `last_sizing_fresh_share`, `last_sizing_fresh_loss`, `last_sizing_stale_fill_batches`.
- `train.py`: self-check в лог — `fresh_share`, `fresh_loss`, `stale_fill_batches`.
- `config.yaml`: `sizing_advantage_stratified_sampling: true`,
  `sizing_anchor_train_steps_per_iteration: 2000` (две переменные осознанно).

### Не трогать

`discount`, `sizing_advantage_buffer_reservoir`, `sizing_min_prob*`, `sizing_bucket_explore_prob`,
`batch_size: 256`.

### Гейты

- Основной: `Sizing holdout TV RATIO` каждые 5 итераций, движение ниже 1.0 (потолок 0.63).
- Оговорка: на iter 1–4 буфер целиком свежий, квота no-op — честное сравнение с v15 с iter 5+.
- Подтверждающий: `oracle_ab` EV на 5/10/20/30, смотреть `blueprint − uniform`.
- Рвать досрочно, если к iter 15 RATIO не сдвинулся с 1.11.

### Контроль после v16 (отдельным прогоном)

Те же 2000 шагов **без** стратификации — разделить «помогла квота» vs «помогло снижение шагов».

---

## 4. Результат v16: стратификация не сработала

### Self-check

`fresh_share = 0.5` (iter 2+), `stale_fill_batches = 0` — квота работает. Но `fresh_loss`
застрял на ~0.6–0.7 и не падает.

### RATIO на holdout

```
iter 5/10/15/20/25/30:  1.217 / 1.240 / 1.228 / 1.210 / 1.182 / 1.155
```

Не падает ниже 1.0. EV: `blueprint − uniform` = +16.31 (iter 5, t=+2.06 значимо — ранняя
точка, буфер свежий), далее +5.07 / −0.89 / −0.71 (шум).

### Leave-one-out замер (честный потолок)

| метрика | значение |
|---|---|
| LOO-RATIO (leave-one-out условное среднее) | **1.275** |
| with-leakage RATIO (прежний «потолок» — среднее включает сам визит) | 0.722 |
| within-(state, anchor) дисперсия (шумовой пол) | mean 0.441 |

**Потолок 0.63 был с утечкой.** Без утечки — 0.72, leave-one-out — 1.275. Сеть v16 (1.15)
внутри скобки [0.72, 1.275].

---

## 5. Вывод: TV против одного визита — не гейт

Таргет = один шумный визит `μ + ε`. RM от него не воспроизводим: утечка даёт оптимистичные
0.72, leave-one-out — пессимистичные 1.275, истина (`σ²`) между ними. Поэтому RATIO > 1.0 —
свойство метрики, а не сети. «Сеть на оптимуме» и «не на оптимуме» этим не доказывается.

`TV/RATIO` — третья мёртвая метрика серии (после `ptp` и `sign_agreement`), умерла по той же
причине. **Единственный живой гейт — EV через `oracle_ab`.**

### Следующий решающий замер

`sizing_oracle_ab.py --rollouts` (1 vs 10) на iter 20 и 30: `oracle_mc − uniform`. Разница
значима → шум связывает, денойзинг таргетов оправдан (начать с `sizing_target_ema_enabled`,
потом больше кредитуемых анкеров). Разницы нет → шум не узкое место, ветка S6 без запаса,
идти в 10k traversals + солвер.
