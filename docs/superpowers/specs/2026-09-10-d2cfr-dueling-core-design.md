# D2CFR dueling core: дизайн

## Цель

Добавить отдельный, выключенный по умолчанию вариант обучения преимуществ
`d2cfr_dueling_v1` для текущих путей HU current-policy self-play и six-max.
Он обучает не дополнительную независимую голову, а одну dueling-сеть с общим
представлением состояния и согласованными counterfactual-целями:

```
I ──> shared encoder ──> V(I)       : [B, 1]
                       └> Q(I, a)  : [B, 6]

R(I, a) = Q(I, a) - V(I)            : [B, 6]
```

`R` — единственный выход, используемый regret matching. Поэтому при флаге
`false` обычный action-only Deep CFR остаётся байт-в-байт прежним путём, а при
`true` обе головы обучаются на одних и тех же traversal-целях.

Первое применение — отдельный HU-прогон и последующее честное A/B-сравнение с
зафиксированным HU baseline. Низкий training loss не является критерием выбора
чекпоинта или доказательством качества.

## Границы

В этот этап входят: dueling-сеть, D2 reservoir, выбор D2-ветки в HU и six-max,
полный checkpoint/resume, light checkpoint стратегии и тесты из этого
документа.

Намеренно не входят: MC rectification из статьи, outcome sampling, history-Q,
KL policy loss, D2CFR/обычный режим A/B harness, Stage B distillation,
teacher-to-student transfer `V`/`Q`/regret-голов, искусственная проекция HU на
six-max и изменение пространства шести action slots.

Метод статьи рассчитан на двух игроков с нулевой суммой. Реализация six-max
нужна как инженерный эксперимент с теми же локальными targets, но не заявляет
теоретической гарантии статьи для multi-player игры.

## Конфигурация и валидация

В `config.yaml` и defaults добавляются следующие поля:

```yaml
d2cfr_enabled: false
d2cfr_regret_loss_weight: 1.0
d2cfr_state_value_loss_weight: 1.0
d2cfr_action_value_loss_weight: 1.0
d2cfr_reinitialize_each_iteration: true
d2cfr_iteration_weight_power: 1.0
d2cfr_mc_correction_enabled: false
```

Правила:

- Все три веса неотрицательны, и хотя бы один строго положителен.
- `d2cfr_iteration_weight_power` неотрицателен.
- Значение `d2cfr_mc_correction_enabled: true` немедленно вызывает `ValueError`:
  первый этап не реализует MC correction.
- При D2CFR `advantage_regret_clip` обязан быть `null`: клиппинг regret нельзя
  согласованно представить разностью двух value-целей.
- `advantage_accumulation`, `discount_alpha`, `discount_gamma`,
  `advantage_loss` и `advantage_huber_delta` не участвуют в D2CFR-обучении.
  Их значения допускаются для совместимости общего config, но не меняют
  D2-результат.

## Сеть

Добавляется `DuelingRegretNetwork`, не изменяя контракт `PokerNetwork`.
Конструктор принимает те же `input_size`, `hidden_size`, `num_actions` и
`network_architecture`. Он поддерживает обе существующие архитектуры:

- `monolithic_v1`: общий трёхслойный `base` перед `V` и `Q`;
- `card_context_v1`: общий `card_encoder` и `context_encoder`, после их
  конкатенации — две головы.

`forward_components(x)` возвращает именованный результат `state_values`,
`action_values`, `regrets`; `forward(x)` возвращает только `regrets` для
неизменного вызова regret matching. `state_values` имеет форму `[B, 1]`, две
остальные величины — `[B, 6]`; `regrets == action_values - state_values`
проверяется тестом. Маска не подаётся в сеть и не искажает её выход: ею
исключаются только недопустимые слоты при построении target, loss и policy.

Стратегическая сеть остаётся `PokerNetwork`. D2CFR не меняет state encoder,
карточные 109 признаков и контракт `six_fixed_v2` из шести слотов.

## Traversal-цели и нормализация

В узле traverser-а существующий traversal уже вычисляет:

```
Q_raw[a] = counterfactual value после допустимого действия a
V_raw    = sum_a policy[a] * Q_raw[a]
R_raw[a] = Q_raw[a] - V_raw
```

Для D2CFR вводится общий положительный коэффициент `S(I)` и записываются:

```
Q = Q_raw / S(I)
V = V_raw / S(I)
R = (Q_raw - V_raw) / S(I)
```

`S(I)` ровно воспроизводит текущую линейную нормализацию преимуществ:
`advantage_reward_scale`, а при выбранных `pot_stack` или `per_node_max` также
соответствующий положительный знаменатель. Нелинейный clip выше запрещён.
Обычный action-only путь продолжает использовать прежний метод нормализации;
общий helper для коэффициента обязан выдавать для него тот же результат.

В D2-режиме traversal записывает D2 sample атомарно вместе со strategy sample.
Для six-max он остаётся привязан к traversing player; для HU имеются два
независимых буфера P0 и P1. Ошибка обхода откатывает все новые записи текущего
обхода, как и сейчас.

## Replay и обучение

Добавляется `DuelingAdvantageBuffer` с reservoir-семантикой. Каждая запись:
`encoded_state`, `action_values[6]`, `state_value[1]`, `regrets[6]`,
`legal_mask[6]`, `iteration`. Shape, dtype, конечность и маска валидируются до
мутации буфера.

Для batch с весом `w_i = (iteration_i / current_iteration)^p`, где
`p=d2cfr_iteration_weight_power`, используются средние, нормированные на
`sum(w_i)`, losses:

```
L = λ_R * masked_weighted_mse(R_pred, R_target)
  + λ_V * weighted_mse(V_pred, V_target)
  + λ_Q * masked_weighted_mse(Q_pred, Q_target)
```

Маски применяются к R и Q и нормируются по числу допустимых action slots;
`V` обучается для каждой записи. Для численной диагностики профиль сохраняет
три компонентных loss и суммарный loss. Проверка finite loss/gradients и
gradient clipping остаются обязательными.

При `d2cfr_reinitialize_each_iteration: true` непосредственно перед обучением
каждой advantage-ноги создаются новая `DuelingRegretNetwork` и новый `AdamW`
с текущими архитектурой и learning-rate параметрами. Исторический D2 reservoir
не очищается. Snapshot для следующего traversal создаётся только после этого
обучения. В D2-ветке отсутствуют bootstrap target-network, накопление DCFR и
обновление `advantage_target_net`; обычная ветка сохраняет их без изменений.

HU coordinator обобщается так, чтобы D2-ноге не требовался фиктивный
`advantage_target_net`: у P0 и P1 собственные dueling-сеть, optimizer, buffer
и read-only snapshot. Шесть-max использует одну advantage-ногу текущего
traversing player согласно существующей схеме.

## Checkpoint и совместимость

Полный checkpoint D2CFR получает обязательные metadata:

- `algorithm_variant: d2cfr_dueling_v1`;
- формат и параметры `DuelingRegretNetwork`, encoding, action-space и размеры;
- D2 config выше, включая нормализацию target;
- для каждой ноги: Dueling network, AdamW state и D2 reservoir при включённом
  сохранении replay;
- счётчик итераций и RNG state по прежнему strict resume-контракту.

Обычный и D2 checkpoint являются разными algorithm variants. Resume обязан
отвергать их перекрёстную загрузку до мутации runtime; silent migration нет.
HU checkpoint дополнительно обязан содержать две полные D2-ноги P0/P1.
Существующий Stage A teacher-transfer loader принимает только обычный
`hu_current_policy_self_play` checkpoint и намеренно отвергает D2 checkpoint.

Light checkpoint остаётся strategy-only артефактом и сохраняется тем же
расписанием HU; он не обещает D2 full-resume и не включает D2 advantage-головы.

## Тесты и критерии готовности

- При `d2cfr_enabled: false` все прежние HU и six-max контракты остаются
  зелёными.
- Сеть проверяется для обеих архитектур: формы, конечность, точное
  `R=Q-V`, работа card encoder и отсутствие градиента по недопустимым Q/R
  слотам.
- Буфер проверяется на strict shape/finite validation, reservoir и полную
  сериализацию.
- Synthetic batch доказывает формулу трёх loss, маски и iteration weights.
- D2-обучение не читает и не обновляет bootstrap target network; reinitialize
  создаёт новый optimizer и сохраняет reservoir.
- HU smoke проверяет раздельные P0/P1 D2 ноги, конечную нормированную policy и
  strict checkpoint/resume. Six-max smoke проверяет traversal и training без
  изменения action-space.
- Проверяются rejection `mc_correction_enabled`, clip и checkpoint другого
  algorithm variant до мутации runtime.

Критерием завершения разработки является воспроизводимый зелёный test suite и
валидный технический HU D2 checkpoint. Критерий качества появится только в
отдельном заранее фиксированном A/B-прогоне против HU baseline.
