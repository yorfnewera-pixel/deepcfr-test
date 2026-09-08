# Аудит information set в `encode_state`

**Дата:** 2026-09-09  
**Статус:** подтверждён дефект представления; код не изменён  
**Область:** blueprint Deep CFR и `PolicyRuntimeAgent`. Runtime-search history рассматривается отдельно.

## Вывод

Deep CFR не требует Transformer, RNN или передачи raw action tape в сеть. Однако конкретный encoder обязан различать наблюдаемые ситуации, когда для них может требоваться разная стратегия.

Текущий `encode_state()` этого не делает. В репозитории воспроизводятся достижимые состояния с разной публичной betting history, одинаковым tensor из 157 признаков и одинаковой маской шести action slots. Для сети это один вход, поэтому она не может выучить разные значения или стратегии для этих линий.

Это не ошибка vanilla CFR и не доказательство, что нужна последовательностная архитектура. Это доказанный lossy-слой абстракции в текущем представлении information set.

## Текущий контракт

`src/core/model.py::encode_state` кодирует:

| Блок | Размер в 6-max | Что сохраняется |
| --- | ---: | --- |
| Карты hero и board | 104 | Приватные карты hero и публичный board с канонизацией мастей |
| Улица, pot, button, current player | 18 | Снимок текущего игрового узла |
| Игроки | 24 | `active`, `bet_chips`, `pot_chips`, `stake` для каждого игрока |
| Betting snapshot | 11 | `min_bet`, pot commitment, engine legal actions, тип и размер только последнего действия |

Не сохраняются распределение ставок по завершённым улицам, автор последнего действия, авторы агрессии на завершённых улицах и последовательность betting actions. `pot_chips` агрегирует все прошлые улицы.

`State` при этом хранит `last_raise_increment`, `last_stage_action` и `from_action`. Первые два участвуют в построении action mask, но не входят в tensor. Таким образом mask может скорректировать недопустимый слот после forward pass, но не даёт сети отдельные значения для общих допустимых слотов.

## Методика

Проверка использовала настоящее Python API `pokers`, `src.core.model.encode_state`, `src.core.action_space.legal_action_mask` и только действия, разрешённые `resolve_action`.

1. Ручной regression-контрпример: heads-up, `seed=17`, `sb=1`, `bb=2`, stack `200`.
2. Ограниченный исчерпывающий перебор: heads-up, тот же seed, stack `30`, все шесть compact slots, глубина не более 12.
3. Состояния группировались по байтам `float32` результата `encode_state(state, 0)`; в bucket сравнивались полные action traces и legal masks.

Ограниченный перебор посетил 1 324 нетерминальных decision nodes, получил 418 различных tensors и 200 buckets, содержащих более одной различимой action history. Это не оценка всего 6-max game tree и не оценка EV, но достаточное доказательство существования систематических коллизий.

## Доказанные коллизии

### A. Агрессия на разных улицах

Обе линии доходят до river:

```text
A: preflop Raise(8), Call; flop Check, Check; turn Check, Check
B: preflop Call, Call; flop Raise(8), Call; turn Check, Check
```

На river совпадают board, pot `20`, stacks `190`, `pot_chips=10` у обоих игроков, текущий игрок, последний action `Check` игрока 0 и legal mask. `np.array_equal(encode_state(A, 0), encode_state(B, 0))` возвращает `True`.

Hero наблюдал, была ли агрессия preflop или flop. Для стратегии это может быть сигналом о диапазоне оппонента; tensor его теряет.

### B. Агрессор на одной улице

В ограниченном переборе найдена пара river-состояний с pot `8` и той же маской:

```text
A: ... turn Check(P1), RaiseHalfPot(P0), Call(P1)
B: ... turn RaiseHalfPot(P1), Call(P0)
```

На следующей улице агрегированные вклады равны, а `from_action` содержит одинаковый тип `Call` и размер `0`. Автор `from_action` не закодирован, поэтому tensor совпадает. Hero различает, кто был aggressor, но сеть — нет.

## Ранжирование

| Приоритет | Риск | Статус | Последствие |
| --- | --- | --- | --- |
| P0 | Потеря улицы и автора агрессии | Доказан | Blueprint не может условить стратегию на наблюдаемую betting line. |
| P0 | Тихая несовместимость весов при расширении input | Доказан в коде | Размер входа зашит в сети, буферах и runtime, а checkpoint не содержит версию encoder. |
| P1 | `last_stage_action` и `last_raise_increment` не видны сети | Поле найдено, отдельная достижимая пара ещё не выделена | Legal mask меняется после forward pass; logits общих слотов не могут условиться на эти поля. |
| P2 | Другие коллизии длинных и многорейзовых линий | Вероятен | Компактный summary не эквивалентен perfect recall; требует измерения после v3. |

## Что не доказано

- Не измерено, насколько этот дефект снижает exploitability или paired EV.
- Не доказано, что raw history либо Transformer необходимы.
- Не выполнен полный перебор 6-max tree.
- Не доказана достижимая пара с одинаковым tensor и разной mask именно из-за `last_raise_increment` или `last_stage_action`.

Эти границы важны: доказан representational defect, но не размер его практического эффекта.

## Варианты

| Вариант | Плюсы | Минусы | Решение |
| --- | --- | --- | --- |
| Ничего не менять | Нулевой риск миграции | Оставляет доказанные P0-коллизии | Отклонён |
| Только последний action и ещё несколько snapshot-полей | Малый input | Не отличает известные river-линии | Отклонён |
| `history_summary_v3` по улицам | Устраняет доказанные классы, сохраняет MLP | Осознанно не lossless | Рекомендован как первый этап |
| Полная последовательность + GRU/Transformer | Ближе к perfect recall | Большой архитектурный и memory-risk, новые эксперименты | Отложен до измерений v3 |

## Рекомендованный `history_summary_v3`

Движок должен хранить публичную history каждого валидного действия. Encoder добавляет для каждой из четырёх улиц  `2 * num_players + 8` признаков:

```text
last_actor_relative:       one-hot(num_players + 1), включая None
last_action_kind:          one-hot(5): Fold, Check, Call, Raise, None
raise_actor_relative_mask: multi-hot(num_players)
raise_count:               один нормированный scalar
raise_amount_total:        сумма raise increments / norm_unit
```

В 6-max это `4 * (12 + 8) = 80` новых признаков: базовый input меняется с 157 на 237, а multi-agent input — с 163 на 243.

Summary различает оба доказанных класса: street-specific `raise_actor_relative_mask` сохраняет улицу и aggressor, а `last_actor_relative` сохраняет автора call/check после завершения улицы. Он не пытается маскироваться под полный information set: порядок нескольких одинаковых действий и все отдельные sizing остаются намеренно сжатыми.

## Границы API

1. `pokers.State` получает `action_history: Vec<ActionRecord>` и `action_history_complete: bool`.
2. `apply_action` добавляет только валидное действие с его исходной улицей и actor. Illegal action не попадает в history.
3. `from_seed` создаёт complete history. `from_mid_hand` принимает опциональную полную history; без неё создаёт incomplete state, а v3-inference обязан завершаться явной диагностикой, а не кодировать нули как «действий не было».
4. Rollout и belief rebuild передают обе history-поля без изменений.
5. В checkpoint добавляются `encoding_version="history_summary_v3"` и `encoder_input_size`; format version повышается с 5 до 6. V2 веса и replay buffers не загружаются в v3.

## Критерии принятия

- Обе доказанные пары имеют разные v3 tensors.
- Та же history при одинаковом `player_id` даёт детерминированный tensor; смена hero корректно переводит actor positions в hero-relative координаты.
- `from_mid_hand` без полной history не позволяет выполнить v3-inference молча.
- Full, light, teacher, frozen-blueprint и policy-runtime checkpoint явно проверяют `encoding_version`, `num_players`, multi-agent mode и размер входа.
- Rust, Python и regression-тесты проходят; v3 тренируется с нуля и сравнивается с v2 в paired evaluation до замены baseline.

