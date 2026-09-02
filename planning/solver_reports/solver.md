# SOLVER.md - план runtime search/MMDS

> **Последнее обновление:** 2026-08-27  
> **Тема:** 6-max postflop decision-time search поверх компактного Deep CFR blueprint.  
> **Статус:** архитектурное решение принято, реализация runtime solver-а не начата.  
> **Ключевое решение:** обучение blueprint остается компактным (`fold/check/call/0.5pot/1pot/all-in`), расширенная сетка сайзингов появляется только в runtime search.

---

## 1. Итоговое решение

Мы не переносим широкую сетку сайзингов в базовое обучение Deep CFR.

Blueprint остается дешевым и стабильным:

- `fold`
- `check`
- `call`
- `raise_0.5pot`
- `raise_1pot`
- `all-in`

Runtime search работает поверх blueprint и может рассматривать больше действий только в текущей точке решения:

- первый bet на улице: `25%`, `33%`, `50%`, `75%`, `100%`, `150%`, `all-in`;
- ответ на ставку: `fold`, `call`, `raise_pot`, опционально `all-in`;
- будущие решения в rollout-ах играются через compact blueprint, чтобы дерево не раздувалось.

Причина: compact blueprint учится как общая аппроксимация стратегии, а runtime search нужен для локального decision-time improvement. Это снижает риск повторного sizing-collapse в обучении и не ломает текущий Deep CFR action space.

Называем это **decision-time policy improvement / local search**, не equilibrium solver. Гарантий для 6-max general-sum NLH нет.

---

## 2. Что зафиксировано по checkpoint-ам и blueprint

Light checkpoint содержит только `strategy_net`. Это нормально:

```text
encoded_state -> logits/probabilities по action slots
```

Heavy checkpoint нужен для продолжения обучения, light checkpoint - для inference/blueprint policy.

Правила:

- обучение продолжать только из full/heavy checkpoint;
- light checkpoint не должен грузиться через `DeepCFRAgent._load_checkpoint`, потому что там нет `advantage_net`;
- inference wrapper (`FrozenBlueprintPolicy` или `PolicyRuntimeAgent`) должен уметь грузить `strategy_only` напрямую;
- solver должен дергать blueprint как функцию `probabilities(state, player_id)` и получать весь prior `pi(a)`, а не только выбранный action.

Для оппонентов и будущих hero decisions в rollout-е blueprint вызывается с `player_id=current_player`.

---

## 3. Связь со статьей Update-Equivalence/MMDS

Используем `The Update-Equivalence Framework for Decision-Time Planning` как one-step decision-time update:

```text
1. В текущем infoset берется blueprint policy pi.
2. По belief samples и rollout-ам оцениваются action values.
3. Применяется update U(pi, values).
4. Агент играет действие из обновленной root policy.
```

Для MMDS в discrete action space:

```text
pi'(a) proportional to (pi(a) * exp(eta * q(a)) * rho(a)^(eta * alpha))^(1 / (1 + alpha * eta))
```

Где:

- `pi` - текущая root policy от blueprint, mapped на runtime actions;
- `q(a)` - математическое обозначение action value из статьи;
- `mc_values[a]` - имя в коде и диагностике для Monte Carlo estimate;
- `rho = uniform` - reference/magnet policy;
- `eta` - шаг update;
- `alpha` - сила притяжения к uniform magnet.

Важно: `rho = mapped blueprint` не используем. При `rho = pi` формула вырождается в обычный MDS:

```text
pi' proportional to pi * exp((eta / (1 + alpha * eta)) * q)
```

То есть `alpha` перестает быть magnet-регуляризацией и становится только множителем шага. Если когда-нибудь нужен `rho = pi`, надо честно убрать `alpha` и назвать метод MDS.

---

## 4. Масштаб values и численная устойчивость

Сырые poker returns измеряются в chips/BB, а в статье returns нормированы по масштабу игры. Поэтому перед MMDS фиксируем безразмерный масштаб:

```text
value_scale = max(pot, effective_remaining)
mc_values = raw_ev / value_scale
```

Где:

- `pot` - pot в текущей root-точке до применения runtime action;
- `effective_remaining` - релевантный остаток стека после обязательного call для текущего игрока;
- `value_scale` не может быть меньше малого epsilon.

Обязательная диагностика:

- логировать `raw_ev`;
- логировать `mc_values`;
- логировать `eta * mc_values` до exponentiation;
- использовать log-sum-exp, а не прямой `exp` без стабилизации;
- держать floor для полного носителя legal action set.

Нормализация через глобальный `advantage_reward_scale` не используется: она делает `eta` разным по смыслу в маленьких и больших pot-ах.

---

## 5. Старый solver

Старый `src/solver/` физически удален и не является источником новой архитектуры. Runtime search реализуется заново.

Разрешено переиспользовать только локальные идеи после отдельного ревью и тестов: CRN/paired comparison pattern, showdown helpers, диагностические шаблоны. Старый river PBS/CFR solver не восстанавливать.

---

## 6. Позиции и suit isomorphism

### 6.1 Позиции

Движку `pokers` отдельные именованные позиции не нужны: он хранит `button`, `current_player`, порядок хода и legal actions.

Нейросети позиционные признаки нужны. Оставляем hero-relative encoding:

- relative button;
- relative current player;
- player blocks в порядке от `player_id`;
- при запросе blueprint в rollout-е кодируем state относительно игрока, который сейчас ходит.

Абсолютный `player_id one-hot` не нужен по умолчанию. Seat-invariant общая стратегия предпочтительнее.

### 6.2 Suit isomorphism

MVP solver не блокируется на полном suit isomorphism.

Обязательно оставить текущую канонизацию мастей в blueprint encoding. Для runtime search полноценные canonical keys нужны как performance pass:

- кэш `blueprint_policy(state)`;
- кэш rollout/evaluation для изоморфных состояний;
- кэш encoded states.

Порядок: сначала корректный solver без сложного кэша, затем suit-isomorphic cache после latency/EV замеров.

---

## 7. Целевая архитектура

Новый код держать отдельно от обучения:

```text
src/evaluation/
  __init__.py
  blueprint_policy.py
  paired_harness.py

src/runtime_search/
  __init__.py
  actions.py
  mmds.py
  spots.py
  beliefs.py
  rollouts.py
  policy.py
```

### 7.1 `src/evaluation/blueprint_policy.py`

Отвечает за frozen inference policy:

- `FrozenBlueprintPolicy.from_checkpoint(path, num_players, device)`;
- heavy checkpoint грузить совместимым путем;
- `strategy_only` light checkpoint грузить напрямую через `PokerNetwork` + `strategy_net` state dict;
- валидировать `checkpoint_format_version == 5`;
- валидировать `action_space_version == "six_fixed_v2"`;
- `probabilities(state, player_id=None) -> np.ndarray` возвращает `NUM_ACTIONS` вероятностей с legal mask;
- `choose_action(state, rng)` нужен для paired harness и rollout continuation.

`DeepCFRAgent._load_checkpoint` не трогать: он должен оставаться strict loader-ом для full training checkpoint.

### 7.2 `src/evaluation/paired_harness.py`

Отвечает за paired evaluation:

- один schedule раздач для baseline и candidate;
- CRN для действий политик;
- seat rotation по всем игрокам при необходимости;
- identical policies должны давать ровно zero difference;
- illegal action должен падать с диагностикой engine rejection;
- `bb_per_100 = mean_difference / bb * 100`, а не `mean_difference * 100`;
- `PairedEvaluation` должен хранить `bb`, `deals`, `seats`, `samples`, raw rewards и differences.

### 7.3 `actions.py`

Отвечает только за runtime action set:

- определение первого bet на улице;
- построение legal runtime actions;
- перевод runtime action в `pkrs.Action`;
- проверка min-raise/all-in boundary через движок;
- mapping runtime action -> blueprint prior slot только для root prior на расширенной сетке.

Для continuation policy mapping не нужен: после нестандартного runtime action состояние кодируется обычным encoder-ом (`pot/bets/stage`), а blueprint выбирает среди своих compact slots.

Не менять `src/core/action_space.py` ради runtime sizes.

### 7.4 `mmds.py`

Чистая математика:

- `mmds_update(pi, mc_values, mask, eta, alpha, rho=None, floor=...)`;
- `rho = uniform` по умолчанию;
- log-sum-exp стабилизация;
- mask и floor;
- нормализация выходной policy;
- regression test: при `rho = pi` результат совпадает с MDS при `eta_eff = eta / (1 + alpha * eta)`.

Модуль не должен знать про `pokers`, checkpoint-и или rollouts.

### 7.5 `spots.py`

Отвечает за конструирование и выбор test spots:

- deterministic turn/river spots через `pkrs.State.from_mid_hand`;
- помнить, что `stake` в `from_mid_hand` - начальный стек, остаток = `stake - committed`;
- поддержать constructed spots для S5, если естественный river недостижим;
- не читать будущий board из `state.deck[:k]`.

Инвариант колоды: `state.deck` может содержать фактический будущий board. Runout для solver/eval сэмплится только из независимо перемешанного остатка после удаления известных карт.

### 7.6 `beliefs.py`

MVP:

- blocker-aware sampling скрытых рук 5 оппонентов;
- досэмпл неизвестного борда из перемешанного остатка;
- воспроизводимый `np.random.Generator`;
- uniform sampler разрешен только как debug mode, не как solver-valid режим.

Позже:

- reach-weighting по blueprint;
- particle filtering по истории действий;
- диагностика effective sample size.

### 7.7 `rollouts.py`

Отвечает за оценку `mc_values(a)`:

- структура Algorithm 1: одна `H` на particle -> все root actions;
- runout фиксирован внутри particle для всех root actions;
- root action применяется явно;
- continuation policy = compact blueprint;
- батчевый lockstep rollout через `pokers.parallel_apply_action` сразу в S3;
- батчевый blueprint inference сразу в S3;
- rollout до terminal без обычного depth limit, но с emergency depth guard и логированием.

Не добавлять `rollouts_per_action`: он провоцирует независимые rollout-ы по action и ломает CRN-сравнение.

### 7.8 `policy.py`

Публичный API runtime search:

```python
choose_action(state, hero_id, blueprint, config, rng) -> SearchDecision
```

`SearchDecision` должен содержать:

- выбранное `pkrs.Action`;
- runtime action label;
- root blueprint policy;
- root search policy;
- `raw_ev`;
- `mc_values`;
- `eta_times_values`;
- число belief particles;
- latency;
- diagnostic flags.

---

## 8. Этапы реализации

### S-1 - evaluation prerequisite

Цель: восстановить минимальную инфраструктуру prior/eval до runtime solver-а.

- починить `FrozenBlueprintPolicy.from_checkpoint`, чтобы light checkpoint грузился напрямую;
- сохранить heavy checkpoint path;
- починить `bb_per_100` с делением на `bb`;
- добавить/обновить тесты на light checkpoint;
- прогнать весь pytest.

Гейт: `FrozenBlueprintPolicy.probabilities()` работает от light checkpoint, identical paired policies дают zero diff, full test suite зеленый.

### S0 - runtime search skeleton и границы

Цель: зафиксировать контракты без изменения обучения.

- создать `src/runtime_search`;
- добавить dataclass-и для runtime actions, config и результата поиска;
- добавить конфиг runtime solver отдельно от training config;
- первые legality tests для runtime actions;
- не трогать training loop, checkpoint format и compact blueprint action space.

Гейт: unit tests проходят, обучение Deep CFR не затронуто.

### S1 - MMDS update без poker-зависимостей

Цель: проверить формулу и численную устойчивость.

- реализовать `mmds_update`;
- `rho = uniform` по умолчанию;
- зафиксировать `mc_values = raw_ev / max(pot, effective_remaining)` на уровне caller contract;
- тесты на normalization, negative values, mask, floors;
- тест `eta=0`;
- тест, что высокий `mc_values[a]` увеличивает вероятность действия;
- regression test на вырождение `rho=pi` в MDS с `eta_eff`.

Гейт: чистые математические тесты без `pokers`.

### S2 - spots и blocker belief sampler

Цель: получить корректные скрытые руки и runout cards.

- constructed turn/river spots через `from_mid_hand`;
- blocker-aware sampling;
- исключать hero cards и known board cards;
- не допускать пересечения рук оппонентов;
- runout сэмплить из перемешанного остатка;
- тест, ловящий использование `deck[:k]` как будущего борда;
- логировать причину отказа, если sample невозможен.

Гейт: property/regression tests на отсутствие duplicate cards и корректную сборку mid-hand state.

### S3 - lockstep rollout engine MVP

Цель: уметь оценить `mc_values(a)` для root actions без потери CRN.

- одна `H` на particle -> все root actions;
- фиксированный runout внутри particle;
- root action применяется явно;
- continuation policy = blueprint compact policy;
- `parallel_apply_action` и batch inference сразу в MVP;
- emergency depth guard;
- возвращать EV, дисперсию, число успешных rollouts, ошибки движка.

Гейт: deterministic smoke test на фиксированном seed и latency p50/p95.

### S4 - root policy на identity mapping

Цель: проверить MMDS без произвольного mapping расширенной сетки.

- стартовать с response-to-bet узлов, где runtime action set совпадает с blueprint set;
- root `pi` брать напрямую из `FrozenBlueprintPolicy.probabilities`;
- применить MMDS;
- выбрать legal action.

Причина: в узлах ответа на ставку набор `fold/call/raise_pot/all_in` не требует размазывания prior по новым сайзингам. Это первый честный сигнал о decision-time search без confounder-а.

Гейт: solver возвращает legal `pkrs.Action`, policy суммируется в 1, диагностика печатает `raw_ev/mc_values/policy/latency`.

### S5 - river/turn validation как диагностика

Цель: проверить механику на дешевых поздних улицах.

- замерить фактическую частоту естественных turn/river spots на финальном checkpoint;
- если river естественной игрой почти не достигается, использовать constructed spots;
- отдельно посчитать частоты действий blueprint в turn/river узлах;
- если поздний blueprint схлопнут в check/call, S5 не интерпретировать как проверку solver quality;
- paired EV на constructed spots считать диагностикой, не kill-gate.

Гейт: стабильные legal decisions, воспроизводимые diagnostics, понятный latency profile. Положительный EV пока не обязателен.

### S6 - reach-weighted beliefs

Цель: заменить uniform blocker sampling на приближение `P_pi(H | h_i)`.

- восстановить/передать историю действий;
- replay history под разными hidden hands;
- взвешивать particles вероятностью blueprint reach;
- измерить ESS;
- добавить fallback diagnostics при низком ESS.

Только после этого paired EV становится стоп-гейтом.

### S7 - расширенная сетка сайзингов

Цель: отдельно проверить, помогает ли multisizing.

- включить lead-bet runtime actions `25/33/50/75/100/150/all-in`;
- впервые добавить mapping runtime action -> blueprint prior;
- явно логировать, сколько массы prior пришло от каждого blueprint slot;
- сравнить identity response baseline против lead-bet multisizing.

Гейт: результат интерпретируется отдельно от S4/S5. Негативный результат не отменяет MMDS на blueprint action set.

### S8 - performance pass

Цель: сделать solver пригодным для real-time.

- suit-isomorphic cache keys;
- cache encoded states;
- cache blueprint policy;
- профилировка по секциям: belief/action/apply/inference/showdown/update;
- latency report с p50/p95.

Гейт: latency укладывается в целевой budget или оптимизация становится P0.

---

## 9. Конфиг solver-а

Начальные параметры:

```yaml
runtime_solver_enabled: false
runtime_solver_streets: [turn, river]
runtime_solver_belief_particles: 32
runtime_solver_eta: 10.0
runtime_solver_alpha: 0.05
runtime_solver_policy_floor: 0.001
runtime_solver_max_seconds: 2.0
runtime_solver_seed: null
runtime_solver_value_scale: max_pot_effective_remaining
runtime_solver_rho: uniform
```

Сетки:

```yaml
runtime_lead_bet_pot_fractions: [0.25, 0.33, 0.50, 0.75, 1.00, 1.50]
runtime_response_raise_fractions: [1.00]
runtime_include_all_in: true
```

Не добавлять `runtime_solver_rollouts_per_action`. Бюджет задается как:

```text
belief_particles * number_of_root_actions
```

Каждый particle обслуживает все root actions на общей hidden history/runout.

Стартовые `eta=10`, `alpha=0.05` допустимы только вместе с value scaling и diagnostics `eta * mc_values`.

---

## 10. Метрики и отчеты

Каждый этап писать отдельным файлом в `planning/solver_reports/`:

```text
001-evaluation-prerequisite.md
002-runtime-actions-and-mmds.md
003-belief-spots.md
004-lockstep-rollouts.md
005-river-turn-validation.md
...
```

Формат отчета:

```markdown
# N - короткое название

**Статус:** IMPLEMENTED / DIAGNOSIS / RESULT / FAIL / CLOSED
**Дата:** YYYY-MM-DD
**Код:** список файлов
**Команды проверки:** pytest/benchmark/eval

## Что сделано
## Почему так
## Результаты
## Риски
## Следующий шаг
```

Обязательные метрики runtime search:

- legal action failure count;
- rollout failure count;
- terminal completion rate;
- latency p50/p95;
- belief particles used;
- root actions count;
- `raw_ev_mean/raw_ev_std` по action;
- `mc_values_mean/mc_values_std` по action;
- `eta * mc_values` min/max;
- search policy entropy;
- blueprint vs search KL;
- выбранный action и его probability;
- paired EV против blueprint на фиксированном schedule;
- ESS для reach-weighted beliefs;
- частоты действий blueprint в turn/river узлах.

---

## 11. Правила остановки

Останавливаем ветку runtime search и не усложняем код, если:

- rollouts до terminal нестабильны из-за движка;
- latency не укладывается в budget даже после S8;
- после reach-weighted beliefs paired eval не показывает улучшения на двух seed-ах при работающем положительном контроле;
- `mc_values` spread меньше MC noise и не воспроизводится на holdout rollouts;
- reach-weighted beliefs дают слишком низкий ESS для практического бюджета;
- поздний blueprint настолько схлопнут, что continuation policy систематически завышает агрессивные root actions.

S5 с uniform beliefs не является kill-gate. Он диагностирует механику и слабые места, но не доказывает бесперспективность solver-а.

Если срабатывает стоп-гейт после solver-valid этапа, возвращаемся к compact blueprint без runtime search и фиксируем отчетом причину закрытия.

---

## 12. Ближайший следующий шаг

Начать с S-1/S1:

1. Починить `src/evaluation`:
   - light checkpoint path в `FrozenBlueprintPolicy`;
   - `bb_per_100` с делением на `bb`;
   - тесты на light checkpoint.
2. Реализовать чистый `mmds_update`.
3. Зафиксировать value scaling и diagnostics.
4. Записать отчет `001-evaluation-prerequisite.md`.
5. Записать отчет `002-runtime-actions-and-mmds.md`.

До этого не трогать training loop, checkpoint format и compact blueprint action space.
