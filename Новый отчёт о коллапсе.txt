# HU D2CFR: отчёт о диагностике all-in и replay

## Цель

Понять, объясняется ли агрессивная/all-in policy тем, что all-in действительно
имеет большой counterfactual regret, либо сеть неточно восстанавливает Q, V и
regret из traversal targets.

Checkpoint: `models/HU_new/run_20260926_164346_d7a76e15/hu_checkpoint_final.pt`.
Итерация: 100. Во всех postflop-прогонах используются HU, 6 фиксированных
action slots и нормированные D2CFR targets с `advantage_reward_scale = 200`.

## Термины

- `Q_target(I, a)` — label из свежего traversal: оценка payoff при выборе
  действия `a` в infoset `I`.
- `V_target(I)` — среднее Q под текущей policy traversal.
- `R_target(I, a) = Q_target(I, a) - V_target(I)` — regret target.
- Сеть выдаёт свои `Q_hat`, `V_hat`, `R_hat = Q_hat - V_hat`.
- Regret matching получает policy только из положительных `R_hat`.

Следовательно, большой target-regret сам по себе не является ошибкой. Ошибка
есть тогда, когда сеть существенно расходится с target или меняет знак regret.

## 1. Свежие frozen postflop targets

Источник: `hu_new_frozen_target_iter100.json`.

Собрано 64 уникальных postflop-state. Для каждого сделано 64 повторных
external-sampling traversal при зафиксированных policy snapshots. Так target
noise отделён от ошибки сети.

| Действие | Q MAE сети относительно среднего target |
|---|---:|
| Raise 0.5 pot | 0.365 |
| Raise 1 pot | 0.370 |
| All-in | **0.536** |

Для all-in средний target Q standard deviation между повторными traversal был
около 0.119, то есть Q MAE 0.536 нельзя списать только на sampling noise.
Средняя L1-дистанция между regret-matching policy target и сетью: **1.363**.

Вывод: на этом postflop срезе all-in действительно аппроксимируется хуже
half-pot и pot. Это не доказывает, что all-in ошибочно выбирается во всех
раздачах, но показывает крупную ошибку NN в действии, важном для policy.

## 2. Возраст replay labels

Источник: `hu_new_buffer_age_iter100.json`.

Текущая checkpoint-сеть сравнивалась с labels, сохранёнными в replay на
разных итерациях.

| Источник labels | Общий Q MAE | All-in Q MAE | RM policy L1 |
|---|---:|---:|---:|
| Iterations 76–100 | 0.176 | 0.248 | 0.614 |
| Iterations 51–75 | 0.200 | 0.277 | 0.645 |
| Iterations 26–50 | 0.216 | 0.306 | 0.664 |
| Iterations 1–25 | 0.256 | 0.378 | 0.742 |

Вывод: сеть заметно хуже соответствует старым stored labels. Это совместимо с
изменением policy/targets по ходу обучения. Само по себе это ещё не доказывает
вред старого replay: в конфигурации есть осознанный iteration weighting, а
сеть переинициализируется на каждой итерации.

## 3. Проверка iteration weighting против uniform weights

Источник: `hu_new_weight_mode_ab_iter100.json`.

Две новые одинаково инициализированные сети получили одинаковую 20k выборку
из replay и одинаковые minibatch indices. В одной ветке применялась текущая
формула весов `source_iteration / mean(source_iteration в minibatch)`, в
другой все веса были равны единице. Оценка — на одном frozen наборе из 64
postflop-state.

| Метрика после 2000 шагов | Current iteration weights | Uniform |
|---|---:|---:|
| Eval MSE | 0.307 | **0.294** |
| Q MAE | 0.456 | **0.439** |
| RM policy L1 | 1.289 | **1.221** |
| All-in Q MAE | **0.528** | 0.570 |

Вывод: uniform немного лучше по общему fit и policy, но хуже по all-in Q MAE.
Это не делает iteration weighting доказанной причиной all-in collapse.

## 4. Full replay против последних 25 iterations

Источники:

- `hu_new_replay_freshness_iter100.json` (seed 20260926)
- `hu_new_replay_freshness_seed20260927.json`
- `hu_new_replay_freshness_seed20260928.json`
- `hu_new_replay_freshness_margin_seed20260929.json`

В каждом запуске обе сети стартовали с одинаковых случайных весов. Обе
получали по 20k labels, 2000 шагов, batch size 256 и текущую loss-функцию.
`full_replay` получал samples из iterations 1–100; `recent_window` — только
из 76–100. Оценка — на новом frozen postflop наборе из 64 states, по 64
repeated ES traversal на state.

Среднее по четырём seed:

| Метрика | Full replay | Recent 76–100 |
|---|---:|---:|
| Q MAE | 0.408 | **0.373** |
| V MAE | 0.403 | **0.370** |
| Eval MSE | 0.265 | **0.254** |
| RM policy L1 | 1.259 | **1.229** |
| All-in Q MAE | 0.550 | **0.522** |
| All-in regret MAE | **0.333** | 0.333 |
| All-in sign flip rate | **49.2%** | 54.7% |

В первых трёх seed recent window был лучше по общему Eval MSE; в seed 20260929
стал хуже (0.224 full против 0.261 recent). Поэтому корректная формулировка:
есть общий тренд в пользу свежих labels, но результат недостаточно стабилен,
чтобы прямо менять training algorithm на основе одного правила "оставлять 25
итераций".

## 5. Последняя проверка: где именно all-in regret меняет знак

Источник: `hu_new_replay_freshness_margin_seed20260929.json`.

All-in samples разделены по величине свежего `|R_target|`. Если flips были бы
только при почти нулевом target-regret, это объяснялось бы обычной
нестабильностью порога `R = 0`.

| Диапазон all-in `|R_target|` | Full replay: flips | Recent: flips |
|---|---:|---:|
| `<= 0.01` | 14 / 22 = 63.6% | 15 / 22 = 68.2% |
| `0.01–0.1` | 5 / 11 = 45.5% | 8 / 11 = 72.7% |
| `> 0.1` | **16 / 31 = 51.6%** | **14 / 31 = 45.2%** |

Вывод: flips происходят не только в пограничной зоне около нуля. В этом seed
примерно половина all-in с заметным target-regret (`|R_target| > 0.1`) получает
от сети противоположный знак. Это прямое наблюдение ошибки NN на сильном
all-in signal. Однако это пока один seed/64 states для margin-разбивки, а не
оценка всей игровой distribution.

## Что установлено и что нет

Установлено:

1. NN на свежих postflop targets заметно ошибается в Q/V/R; all-in Q в первом
   frozen-target прогоне хуже half-pot и pot.
2. Current network хуже соответствует старым replay labels, чем свежим.
3. На нескольких seed recent replay часто переносится на свежие targets лучше,
   чем full replay, но эффект не стабилен во всех запусках.
4. All-in regret sign flips встречаются и при сильном target-regret, не только
   около нуля.

## 6. Direct fit: mean targets против отдельных noisy labels

Источник: `hu_new_frozen_mean_vs_noisy_iter100.json`.

На тех же 64 frozen postflop-state собраны 64 target labels на state. Одна
новая сеть 5000 шагов обучалась на 64 усреднённых labels, вторая — на всех
4096 отдельных labels. Обе оценивались относительно усреднённого reference.

| Метрика на mean reference | Mean-target fit | Noisy-target fit |
|---|---:|---:|
| Q MAE | **0.00123** | 0.02293 |
| V MAE | **0.00212** | 0.01555 |
| Regret MAE | **0.00193** | 0.01961 |
| Eval MSE | **0.000010** | 0.001268 |
| RM policy L1 | **0.112** | 0.262 |
| Ordering all-in vs pot | **100%** (54 pairs) | 98.1% (54 pairs) |
| Ordering pot vs half-pot | **100%** (55 pairs) | 94.5% (55 pairs) |

В mean-target ветке не было ни одного sign flip при `|R_target| > 0.01`;
все 32 flips были только в зоне `|R_target| <= 0.01` и при среднем regret
error 0.00193. В noisy-target ветке тоже не было flips при `|R_target| > 0.1`
на общем наборе legal actions.

Вывод: для этих 64 state текущая архитектура, optimizer и Q/V/R loss способны
почти точно выучить усреднённый сигнал. Даже отдельные noisy traversal labels
не уничтожают ordering действий. Поэтому гипотеза "сеть принципиально не
может представить all-in Q/V/R" не подтверждается. Основной разрыв возникает
при обучении на широкой, изменяющейся истории replay, а не при direct fit
фиксированного набора.

Не установлено:

1. Что all-in action математически неверен или всегда должен быть запрещён.
2. Что old replay — единственная причина all-in-heavy policy.
3. Что замена replay на окно 25 итераций исправит обучение.
4. Что fixed-set direct fit гарантирует хорошую generalization на всей
   distribution; он проверяет capacity только на этих 64 state.

## 7. Прямое измерение дрейфа targets при смене policy

Источник: `hu_new_target_drift_iter50.json`.

Это отдельный короткий HU-training run в памяти, не изменявший checkpoint и
основное обучение. До training были зафиксированы 16 postflop-state. Затем
на итерациях 0, 10, 20, 30, 40 и 50 для каждого из тех же state считался
средний target по 32 traversal при текущем snapshot policy. Sampling самого
probe не записывался в replay (`probe_recorded_samples = 0`).

Здесь измеряется не ошибка NN, а другое: насколько меняется сам ответ
traversal на один и тот же state, когда за время training меняются policy
игроков. Все величины нормированы на `advantage_reward_scale = 200`.

| Iteration | Q target MAE от iteration 0 | Regret MAE | Изменился знак regret | All-in vs pot: прежний Q-order | Pot vs half-pot: прежний Q-order |
|---|---:|---:|---:|---:|---:|
| 10 | 0.381 | 0.172 | 37.5% | 68.8% (11/16) | 75.0% (12/16) |
| 20 | 0.297 | 0.190 | 43.8% | 62.5% (10/16) | 56.2% (9/16) |
| 30 | 0.245 | 0.177 | 50.0% | 68.8% (11/16) | 37.5% (6/16) |
| 40 | 0.328 | 0.186 | 46.9% | 62.5% (10/16) | 62.5% (10/16) |
| 50 | 0.292 | 0.176 | **50.0%** | **56.2% (9/16)** | **25.0% (4/16)** |

К iteration 50 `Q target MAE = 0.292`, то есть средний сдвиг примерно 58
фишек в исходном масштабе. `Regret MAE = 0.176` — около 35 фишек. Это намного
больше простой статистической погрешности среднего из 32 traversal: у
отдельных traversal наблюдаемый `target_q_std_mean` был 0.061–0.244, а
стандартная ошибка среднего составляет примерно `std / sqrt(32)`, то есть
порядка 0.011–0.043.

Вывод: один и тот же postflop state получает существенно другой Q/V/regret
target уже в течение короткого обучения. Поэтому ранние и поздние labels в
full replay не являются просто дополнительными независимыми примерами одного
неизменного supervised target: они описывают разные snapshots policy.

Важно: это не показывает, что all-in является единственным источником
проблемы. К iteration 50 меняется ordering и для pot/half-pot, причём на
этом маленьком срезе сильнее, чем для all-in/pot. Следовательно, корректный
вывод — общий policy/target non-stationarity является реальным bottleneck;
вывод "all-in по праву забирает весь regret" из этих данных не следует.

## Текущее объяснение простыми словами

Traversal сейчас говорит сети: "в этом state all-in лучше/хуже среднего".
Сеть должна приблизить два числа Q и V, затем policy берёт их разность R.
Если Q и V оба неточны, их разность особенно легко меняет знак. Тогда
regret matching даёт probability не тому действию, которое выбирал бы свежий
traversal target. Target-drift experiment теперь прямо показывает, что старые
labels конфликтуют с более новой policy. Но это всё ещё не доказывает, что
они — единственный источник проблемы или что окно в 25 iterations является
оптимальным решением.

## Следующее направление

Direct fit отделил базовую capacity от replay effect, а target-drift напрямую
подтвердил policy/target shift. Поэтому следующий шаг — не новая похожая
диагностика и не произвольная смена architecture/reward scale, а один
контролируемый полноценный training A/B:

1. Baseline с текущим full replay.
2. Идентичный run, где advantage-network учится только по последним 25
   iterations replay (остальная конфигурация, seed-план и evaluation одинаковы).

Сравнивать надо loss, checkpoint-vs-checkpoint, поведение в GUI и all-in
diagnostics. Такой A/B проверит причинный вопрос: улучшает ли снижение
конфликта между ранними и поздними labels реальную policy, а не только
offline fit на одном frozen наборе.
