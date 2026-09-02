# Bug #46 — Sizing_head изолирован от shared base через detach()

## Статус: FIXED (убран detach перед sizing_head + добавлен detach перед value_head + base добавлен в pg_optimizer)

## Суть проблемы

В `PokerNetwork.forward()` sizing_head получал **отключённые** фичи от base:

```python
# БЫЛО:
detached_features = features.detach()
sizing_params = self.sizing_head(detached_features)
```

Два следствия:

1. **Градиенты от sizing_loss не доходили до base** — sizing_loss не мог настроить shared features под sizing задачу
2. **pg_optimizer содержал только sizing_head** — даже без detach base не обновлялся бы

Features оптимизировались только под `action_head` (CFR regrets) и `value_head`. Sizing_head работал с "чужими" фичами — они различали "рейзить ли", но не были настроены различать "сколько ставить".

## Почему раньше detach был необходим

С `cf_regrets[3]` reward не зависел от sizing → sizing gradient был шумным/бессмысленным → разрешить ему менять base = испортить features для action_head.

С CSA (Bug #45) reward зависит от bet_size → sizing gradient осмысленный → можно разрешить ему влиять на base.

## Предпосылка: CSA валидирован

Fresh run с CSA показал:
- `ZMeanStd → 0.05` — sizing_head начал дифференцировать состояния на "чужих" features
- Профит растёт
- Это подтверждает: features уже содержат релевантную информацию, но sizing_head не может их оптимизировать

## Фикс: 4 файла

### 1. `src/core/model.py` — убрать detach перед sizing_head, добавить detach перед value_head

```python
# БЫЛО:
detached_features = features.detach()
sizing_params = self.sizing_head(detached_features)
value = self.value_head(features)

# СТАЛО:
sizing_params = self.sizing_head(features)
value = self.value_head(features.detach())
```

Теперь sizing_loss градиенты проходят через `sizing_head → features → base`.
Value_loss градиенты обрываются на `features.detach()` и не доходят до base.

### 2. `inference/core.py` — аналогично (консистентность)

```python
# БЫЛО:
detached_features = features.detach()
sizing_params = self.sizing_head(detached_features)
value = self.value_head(features)

# СТАЛО:
sizing_params = self.sizing_head(features)
value = self.value_head(features.detach())
```

При инференсе detach не влияет (нет backward), но для консистентности кода — применено.

### 3. `src/core/deep_cfr.py` — добавить base в pg_optimizer

```python
# БЫЛО:
self.pg_optimizer = optim.Adam(self.advantage_net.sizing_head.parameters(), lr=_pg_lr)

# СТАЛО:
self.pg_optimizer = optim.Adam([
    {"params": self.advantage_net.sizing_head.parameters(), "lr": _pg_lr},
    {"params": self.advantage_net.base.parameters(), "lr": _pg_lr * 0.1},
])
```

**Почему `lr * 0.1`:** base уже обучен под action/regret предсказание. Sizing gradient — вторичный сигнал. Маленький LR позволяет base адаптироваться под sizing без разрушения action_head функциональности.

### 4. `src/core/deep_cfr.py` — clip_grad_norm для base

```python
# БЫЛО:
torch.nn.utils.clip_grad_norm_(self.advantage_net.sizing_head.parameters(), max_norm=1.0)

# СТАЛО:
torch.nn.utils.clip_grad_norm_(self.advantage_net.sizing_head.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(self.advantage_net.base.parameters(), max_norm=0.5)
```

**Почему `max_norm=0.5` (меньше чем sizing_head=1.0):** base — shared representation, повреждение хуже чем для sizing_head. Строгий clip защищает от резких обновлений.

## Побочный эффект: RuntimeError при втором backward

Убирание detach перед sizing_head привело к тому, что **обе головы** (value_head и sizing_head) оказались на живом графе `features`. Это вызвало:

```
1. value_loss.backward() — фритит граф (включая features)
2. pg_loss.backward() — RuntimeError: backward through graph a second time
```

**До фикса #46** этого не было: sizing_head сидел на `features.detach()` (свой отдельный граф), и `pg_loss.backward()` не нуждался в оригинальном `features`.

### Решение: detach перед value_head (Вариант Д)

```python
value = self.value_head(features.detach())
```

Это **не костыль**, а точное отражение архитектуры оптимизаторов:

- `value_optimizer = Adam(value_head.parameters())` — **не содержит base**. Градиенты от value_loss до base — «фантомные»: записываются в base.grad, но value_optimizer.step() их игнорирует, а pg_optimizer.zero_grad() обнуляет.
- `features.detach()` легализует это на уровне графа: PyTorch даже не считает ненужные градиенты через base.
- value_head получает **те же входные данные** (тензор с теми же значениями), обучается идентично — просто не может протолкнуть градиент в base.

### Отклонённые альтернативы

| Вариант | Почему нет |
|---------|-----------|
| А: `retain_graph=True` | Inplace modification error если optimizer трогает base |
| В: `total_loss = value_loss + pg_loss`, один backward | При `total_loss.backward()` в base.grad накопится `∂value_loss/∂base + ∂pg_loss/∂base`. value_optimizer не обнуляет base.grad → pg_optimizer.step() обновит base по value-сигналу — ломает инвариант |
| Б: повторный forward через base | Работает и безопасно, но 2x compute. Для 3×256 неощутимо, но избыточно |

### Нюанс на будущее

value_head теперь на detached features — он не может влиять на то, какие фичи формирует base. Прямо сейчас это корректно (value_optimizer не содержит base). Если захотим True MTL (base от обоих сигналов) — придётся убрать detach перед value_head, добавить base в value_optimizer, и перейти на Вариант Б или В.

## Корректность gradient flow

В `train_sizing_network()` порядок операций:

1. `value_optimizer.zero_grad()` → zeros value_head params
2. `value_loss.backward()` → sets value_head.grad только (features.detach() обрывает граф, base.grad НЕ затрагивается)
3. `value_optimizer.step()` → updates value_head only
4. `pg_optimizer.zero_grad()` → zeros sizing_head + base params
5. `pg_loss.backward()` → sets sizing_head.grad + base.grad (чистые pg-градиенты через живой features)
6. `clip_grad_norm_(sizing_head, 1.0)`
7. `clip_grad_norm_(base, 0.5)`
8. `pg_optimizer.step()` → updates sizing_head (lr=pg_lr) + base (lr=pg_lr*0.1)

**Ключевой момент:** С `features.detach()` перед value_head шаг 2 вообще не пишет в base.grad. Граф features остаётся живым для шага 5. RuntimeError устранён. base обновляется **только** по sizing сигналу.

## Что НЕ меняется

- `action_head`, `strategy_net`, `advantage_buffer` — не тронуты
- `value_optimizer` — по-прежнему только `value_head.parameters()`
- `self.optimizer` (advantage optimizer) — по-прежнему `advantage_net.parameters()` (включая base)
- CSA-сигнал, weighted PG, temperature=2.0 — без изменений
- Return tuple, config — без изменений

## Риск и митигация

| Риск | Митигация |
|---|---|
| Sizing gradient портит features для action_head | `lr * 0.1` + `max_norm=0.5` |
| Sizing gradient шумный (variance покера) | CSA + clamp после нормализации (Bug #45 фикс B) |
| Несовместимость optimizer state с чекпоинтами | Структура модели не меняется. pg_optimizer теперь 2 param groups — при загрузке optimizer state может быть несовместим. Рекомендуется fresh training |
| value_head не может обучать base | Не риск — value_optimizer не содержит base. Если понадобится True MTL — пересмотреть архитектуру |

## Критерии что фикс улучшает sizing (fresh run, 300+ итераций)

| Метрика | С detach | Ожидание без detach |
|---|---|---|
| `Sizing/ZMeanStd` | ~0.05 | **>0.1** — finer state differentiation |
| `Sizing/Mean_Predicted_Size` | колеблется | Более стабилен, адаптирован к контексту |
| Action regrets (advantage loss) | стабильный | Не должен ухудшиться |
| Win rate vs random | положительный | Не падает |

## Связанные баги

- **Bug #43** (FIXED): log_std deadlock + z_mean runaway
- **Bug #44** (SUPERSEDED): soft weighted PG
- **Bug #45** (FIXED): CSA + clamp после нормализации + inference sync
- **Bug #46** (этот): detach изоляция sizing от base

## Консенсус

- GPT: "не убирать detach пока reward не исправлен" — CSA исправлен, precondition выполнен
- Opus: "убрать detach перед sizing + base в optimizer" — план принят
- GLM: отложил detach до валидации CSA → CSA валидирован (ZMeanStd→0.05) → теперь применяем
- Gemini: согласен с Вариантом Д (detach перед value_head) — консенсус достигнут
  - Вариант А (retain_graph) — опасен
  - Вариант В (total_loss) — ломает инвариант при асимметричных оптимизаторах
  - Вариант Б (2x forward) — избыточен
  - Вариант Д — элегантен, 1x forward, чистые градиенты, инвариант на уровне графа
