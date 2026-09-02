# План реализации: ротация checkpoint в selfplay_multi с шагом 100

> **Для агентных исполнителей:** ОБЯЗАТЕЛЬНЫЙ НАВЫК: применить `superpowers:subagent-driven-development` (рекомендуется) или `superpowers:executing-plans` и выполнять задачи по очереди. Для отслеживания используются чекбоксы.

**Цель:** Сохранять checkpoint раз в 100 CFR-итераций, формировать пул соперников по прежнему принципу на сетке checkpoint-слотов, удерживать историю с периодом 5 000 итераций, линейно разогревать advantage learning rate до 100-й итерации и выполнять random-eval раз в 10 итераций.

**Архитектура:** `src/training/train.py` остаётся единственной точкой управления жизненным циклом checkpoint. Пул строится из номеров checkpoint-слотов, а не из каждой training-итерации; он продолжает загружать только `advantage_net` тяжёлых файлов. Если ожидаемого файла нет после возобновления, используется ближайший более старый checkpoint.

**Стек:** Python 3.13, PyTorch, pytest, YAML.

**Спецификация:** Данный документ; согласованный контракт в чате от 2026-08-25.

## Глобальные ограничения

- Не менять действия за столом, формат checkpoint, буферы и модель.
- В пуле ровно пять advantage-сетей; их порядок случайно перемешивается перед назначением позициям.
- Явный `--evaluate-every` пользователя сохраняет приоритет над новым значением по умолчанию.
- `--initial-checkpoint` работает с неполным набором файлов и дублирует доступные более старые веса для недостающих мест.
- Advantage learning rate растёт линейно от 1% базового значения на итерации 1 до 100% на итерации 100; strategy learning rate не меняется.

---

## Целевое расписание

Тяжёлый и light checkpoint создаются после завершения итераций `100, 200, 300, ...`. До окончания 100-й итерации пул пуст.

| Диапазон training-итераций | Пул до случайной посадки |
| --- | --- |
| 101–200 | `100, 100, 100, 100, 100` |
| 201–300 | `200, 100, 100, 100, 100` |
| 301–400 | `300, 200, 100, 100, 100` |
| 401–500 | `400, 300, 200, 100, 100` |
| 501–600 | `500, 400, 300, 200, 100` |
| 601–700 | `600, 500, 400, 300, 200` |
| с 701 до созревания истории | последний + 4 случайных из десяти предыдущих checkpoint-слотов |
| после появления второго исторического | последний + 3 случайных из десяти предыдущих + предпоследний исторический |

Исторические checkpoint кратны 5 000. После сохранения 10 000 в пул вступает 5 000, а 10 000 ждёт до сохранения 15 000.

### Task 1: Настроить интервалы и условное сохранение

**Файлы:**

- Modify: `config.yaml:33`
- Modify: `src/training/train.py:22-25,186-211,515-521`
- Test: `tests/test_checkpoint_lifecycle.py`
- Test: `tests/test_training_regressions.py`

**Интерфейсы:**

- Produces: `_checkpoint_save_due(iteration: int, every: int) -> bool`.
- Produces: `checkpoint_save_every: 100`, `checkpoint_keep_every: 5000`.

- [ ] **Step 1: Написать падающий параметризованный тест границ**

```python
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1, False), (99, False), (100, True), (101, False), (200, True)],
)
def test_checkpoint_save_due_only_on_configured_boundaries(iteration, expected):
    assert train_mod._checkpoint_save_due(iteration, every=100) is expected
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_checkpoint_save_due_only_on_configured_boundaries -v`

Expected: FAIL — функции ещё нет.

- [ ] **Step 3: Реализовать минимальное условие**

```python
def _checkpoint_save_due(iteration: int, every: int) -> bool:
    return int(iteration) > 0 and int(iteration) % max(1, int(every)) == 0
```

Добавить в `config.yaml` ключ `checkpoint_save_every: 100`, заменить `checkpoint_keep_every: 50` на `checkpoint_keep_every: 5000` и `warmup_iterations: 12` на `warmup_iterations: 100`. В training loop вызывать обе функции сохранения и печать путей только при истинном условии.

- [ ] **Step 4: Проверить**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

- [ ] **Step 5: Добавить регрессионный тест линейного warm-up до 100-й итерации**

```python
def test_advantage_lr_warmup_reaches_base_rate_on_iteration_one_hundred():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-4)

    train_mod._apply_advantage_lr_warmup(optimizer, 50, 100, [1e-4])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

    train_mod._apply_advantage_lr_warmup(optimizer, 100, 100, [1e-4])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
```

Run: `pytest tests/test_training_regressions.py::test_advantage_lr_warmup_reaches_base_rate_on_iteration_one_hundred -v`

Expected: PASS: реализация `_apply_advantage_lr_warmup` уже имеет нужную линейную формулу, изменение состоит в новом значении конфигурации и фиксации контракта тестом.

- [ ] **Step 6: Commit**

```powershell
git add config.yaml src/training/train.py tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py
git commit -m "feat: align self-play checkpoint cadence and advantage warm-up"
```

### Task 2: Перевести warm-up и свежее окно на checkpoint-слоты

**Файлы:**

- Modify: `src/training/train.py:106-165`
- Test: `tests/test_checkpoint_lifecycle.py`

**Интерфейсы:**

- Changes: `_opponent_checkpoint_paths(iteration, checkpoint_dir, num_opponents, checkpoint_every, historical_every) -> list[Path]`.
- Consumes: `_resolve_checkpoint_path`, `_sample_checkpoint_iterations`.

- [ ] **Step 1: Написать падающий тест начального расписания**

```python
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (101, [100, 100, 100, 100, 100]),
        (201, [100, 100, 100, 100, 200]),
        (301, [100, 100, 100, 200, 300]),
        (601, [200, 300, 400, 500, 600]),
    ],
)
def test_opponent_pool_uses_checkpoint_slot_warmup(tmp_path, iteration, expected):
    for checkpoint_iteration in range(100, 701, 100):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()

    paths = train_mod._opponent_checkpoint_paths(
        iteration, tmp_path, 5, checkpoint_every=100, historical_every=5000
    )
    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == expected
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_opponent_pool_uses_checkpoint_slot_warmup -v`

Expected: FAIL — текущий код оперирует единичными итерациями.

- [ ] **Step 3: Реализовать расписание**

Вычислять последний доступный slot как:

```python
latest_slot = ((int(iteration) - 1) // checkpoint_every) * checkpoint_every
```

Явно сформировать первые шесть состояний из таблицы с шагом `checkpoint_every`. С седьмого slot формировать `[latest_slot]` и четыре случайных slot без повторов из интервала десяти предшествующих slot. Передавать `checkpoint_save_every` из `_configure_opponent_pool`.

- [ ] **Step 4: Написать и запустить тест свежего окна**

```python
def test_opponent_pool_uses_last_slot_and_previous_ten_slots(tmp_path, monkeypatch):
    for checkpoint_iteration in range(100, 1701, 100):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(1701, tmp_path, 5, 100, 5000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [1300, 1400, 1500, 1600, 1700]
```

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/training/train.py tests/test_checkpoint_lifecycle.py
git commit -m "feat: rotate self-play opponents by checkpoint slots"
```

### Task 3: Зафиксировать исторический якорь и очистку

**Файлы:**

- Modify: `src/training/train.py:42-64,106-148`
- Test: `tests/test_checkpoint_lifecycle.py`

**Интерфейсы:**

- Produces: предпоследний исторический checkpoint `h[-2]` при наличии двух checkpoint, кратных 5 000.
- Produces: хранение 11 свежих неисторических и двух последних исторических checkpoint.

- [ ] **Step 1: Добавить регрессионный тест созревания истории**

```python
def test_opponent_pool_uses_only_mature_historical_checkpoint(tmp_path, monkeypatch):
    for checkpoint_iteration in range(100, 10_001, 100):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(10_001, tmp_path, 5, 100, 5000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [5000, 9700, 9800, 9900, 10000]
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_opponent_pool_uses_only_mature_historical_checkpoint -v`

Expected: PASS после Task 2.

- [ ] **Step 3: Обновить тест очистки**

```python
def test_opponent_checkpoint_retention_keeps_eleven_recent_and_two_historical(tmp_path):
    for checkpoint_iteration in range(100, 15_001, 100):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()

    train_mod._prune_opponent_checkpoints(tmp_path, historical_every=5000)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [10000, *range(13900, 15001, 100)]
```

- [ ] **Step 4: Проверить**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS; подтверждены исторический 5 000, удержание 13 тяжёлых файлов и их совместимые light-пары.

- [ ] **Step 5: Commit**

```powershell
git add src/training/train.py tests/test_checkpoint_lifecycle.py
git commit -m "feat: retain mature five-thousand-iteration opponents"
```

### Task 4: Изменить частоту random-eval

**Файлы:**

- Modify: `src/training/train.py:421-425,564-586`
- Test: `tests/test_training_regressions.py`

**Интерфейсы:**

- Produces: `evaluate_every=10` по умолчанию в Python API и CLI.

- [ ] **Step 1: Написать падающий CLI-тест**

```python
def test_self_play_cli_evaluates_against_random_every_ten_iterations(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--self-play-multi"])

    arguments = train_mod._parse_args()

    assert arguments.evaluate_every == 10
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_training_regressions.py::test_self_play_cli_evaluates_against_random_every_ten_iterations -v`

Expected: FAIL — текущий CLI использует `default=1`.

- [ ] **Step 3: Внести минимальную правку**

Заменить `evaluate_every: int = 1` на `evaluate_every: int = 10` и CLI `default=1` на `default=10`. Условие `iteration % evaluate_every == 0` не менять.

- [ ] **Step 4: Проверить**

Run: `pytest tests/test_training_regressions.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/training/train.py tests/test_training_regressions.py
git commit -m "feat: evaluate self-play against random every ten iterations"
```

### Task 5: Итоговая проверка и smoke-test возобновления

**Файлы:**

- Verify: `config.yaml`, `src/training/train.py`, `tests/test_checkpoint_lifecycle.py`, `tests/test_training_regressions.py`.

- [ ] **Step 1: Выполнить целевой набор тестов**

Run: `pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -v`

Expected: PASS без ошибок.

- [ ] **Step 2: Проверить cadence вручную**

Запустить self-play на 101 итерацию с минимальным числом обходов и временным `--save-dir`. Должны появиться только `multi_checkpoint_iter_100.pt` и `light_checkpoint_iter_100.pt`; на 101-й итерации консоль должна показать пять копий checkpoint 100.

- [ ] **Step 3: Проверить continuation с неполным набором**

Во временной папке оставить один тяжёлый checkpoint и запустить продолжение через `--initial-checkpoint`. Пул должен использовать пять доступных старых весов и не завершаться ошибкой.

- [ ] **Step 4: Итоговая проверка diff**

```powershell
git diff --check
git status --short
```
