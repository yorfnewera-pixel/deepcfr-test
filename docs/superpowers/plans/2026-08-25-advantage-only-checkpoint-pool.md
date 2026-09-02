# Advantage-Only Checkpoint Pool Implementation Plan

> **Для агентных исполнителей:** ОБЯЗАТЕЛЬНЫЙ НАВЫК: применять `superpowers:subagent-driven-development` или `superpowers:executing-plans` по задачам последовательно.

**Goal:** Убрать загрузку replay buffers из self-play opponent pool, сохраняя для него отдельные advantage-only checkpoint и оставляя только последний full checkpoint для продолжения обучения.

**Architecture:** Full checkpoint сохраняется с прежним форматом, но после успешного сохранения удаляются все предыдущие full checkpoint. В ту же точку cadence создаётся `opponent_advantage_iter_N.pt` с одним `advantage_net` и минимальными metadata. Ротация 11 свежих и 2 исторических применяется только к advantage-only файлам; именно они загружаются в opponent pool. Light checkpoint остаётся независимым артефактом GUI/runtime.

**Tech Stack:** Python, PyTorch, pytest.

## Global Constraints

- Частота checkpoint остаётся `checkpoint_save_every: 100`.
- Исторический interval остаётся `checkpoint_keep_every: 5000`.
- Full checkpoint сохраняет оптимизаторы и buffers, но в папке прогона остаётся только самый новый.
- Advantage-only checkpoint содержит `action_space_version: "six_fixed_v2"`, `iteration` и `advantage_net`; никаких optimizer или replay-buffer полей.
- При старте/continuation без advantage-only файлов opponent pool должен безопасно использовать cold-start или доступный старый advantage-only файл; full checkpoint не загружается для игры оппонентов.

---

### Task 1: Выделить compact advantage checkpoint API

**Files:**

- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**

- Produces: `_save_opponent_advantage_checkpoint(agent, save_dir, iteration) -> Path`.
- Produces: `_load_opponent_advantage_state(path) -> dict[str, torch.Tensor]`.
- Produces: prefix `opponent_advantage_iter_`.

- [ ] **Step 1: Написать падающий тест содержимого**

```python
def test_advantage_only_checkpoint_contains_no_buffers_or_optimizers(tmp_path):
    agent = TinyAgentWithAdvantageState()
    path = train_mod._save_opponent_advantage_checkpoint(agent, tmp_path, 100)

    payload = torch.load(path, weights_only=False)

    assert payload["iteration"] == 100
    assert payload["action_space_version"] == "six_fixed_v2"
    assert payload["advantage_net"] == agent.advantage_net.state_dict()
    assert "strategy_net" not in payload
    assert "advantage_buffer" not in payload
    assert "advantage_optimizer" not in payload
```

- [ ] **Step 2: Запустить RED-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_advantage_only_checkpoint_contains_no_buffers_or_optimizers -v`

Expected: FAIL — функции ещё нет.

- [ ] **Step 3: Реализовать атомарное сохранение и строгую загрузку**

Использовать существующий `_atomic_torch_save`. При загрузке проверять action-space version и наличие `advantage_net`; некорректный файл должен дать диагностичный `ValueError`.

- [ ] **Step 4: Запустить GREEN-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

### Task 2: Перевести ротацию opponent pool на compact файлы

**Files:**

- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**

- Changes: `_opponent_checkpoint_paths` читает только `opponent_advantage_iter_*.pt`.
- Produces: `_prune_opponent_advantage_checkpoints(directory, historical_every)`.

- [ ] **Step 1: Написать падающий тест источника пула**

```python
def test_opponent_pool_ignores_full_checkpoint_files(tmp_path):
    (tmp_path / "multi_checkpoint_iter_100.pt").touch()
    (tmp_path / "opponent_advantage_iter_100.pt").touch()

    paths = train_mod._opponent_checkpoint_paths(101, tmp_path, 5, 100, 5000)

    assert paths == [tmp_path / "opponent_advantage_iter_100.pt"] * 5
```

- [ ] **Step 2: Запустить RED-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_opponent_pool_ignores_full_checkpoint_files -v`

Expected: FAIL — текущий pool ищет `multi_checkpoint_iter_*.pt`.

- [ ] **Step 3: Ввести отдельный scanner и prune**

Вынести scanner compact файлов с теми же правилами имени и номера итерации. Перенести текущую retention-логику 11 свежих + 2 исторических на этот scanner. Не читать full checkpoint как fallback для opponent pool.

- [ ] **Step 4: Обновить warm-up/history тесты**

Изменить fixtures с `multi_checkpoint_iter_` на `opponent_advantage_iter_`; оставить точные ожидания для 100-step warm-up, свежего окна и исторического checkpoint 5 000.

- [ ] **Step 5: Запустить GREEN-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

### Task 3: Оставлять только последний full checkpoint

**Files:**

- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**

- Produces: `_prune_full_checkpoints(directory) -> None`.
- Consumes: full prefix `multi_checkpoint_iter_`.

- [ ] **Step 1: Написать падающий retention-тест**

```python
def test_full_checkpoint_retention_keeps_only_latest(tmp_path):
    for iteration in (100, 200, 300):
        (tmp_path / f"multi_checkpoint_iter_{iteration}.pt").touch()

    train_mod._prune_full_checkpoints(tmp_path)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [300]
```

- [ ] **Step 2: Запустить RED-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_full_checkpoint_retention_keeps_only_latest -v`

Expected: FAIL — текущий retention хранит 13 full файлов.

- [ ] **Step 3: Реализовать retention и упорядочить save pipeline**

После атомарного full-save: удалить предыдущие full checkpoint; затем сохранить/prune advantage-only checkpoint; затем сохранить/prune light checkpoint. `_prune_light_checkpoints` должен сверять light с retained full checkpoint, поэтому в штатном режиме остаётся только последняя light-пара.

- [ ] **Step 4: Запустить GREEN-проверку**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

### Task 4: Проверить cadence и I/O contract

**Files:**

- Modify: `tests/test_training_regressions.py`
- Verify: `src/training/train.py`

- [ ] **Step 1: Добавить integration-style тест pipeline**

Проверить, что на iteration 100 созданы full, light и advantage-only файлы; на iteration 200 удалены full/light 100, но advantage-only 100 сохранён для opponent history.

- [ ] **Step 2: Добавить smoke-test загрузки**

Создать full checkpoint с крупным synthetic buffer и compact advantage-only файл. Monkeypatch `torch.load` в `_configure_opponent_pool`; проверить, что загружается только `opponent_advantage_iter_*.pt`.

- [ ] **Step 3: Выполнить проверки**

Run: `pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -v`

Expected: PASS.

Run: `python -m compileall -q src`

Expected: exit 0.

- [ ] **Step 4: Ручной smoke-run**

Запустить 201 итерацию с нулевыми traversal/eval во временный каталог. Проверить: full/light существуют только для 200, advantage-only существуют для 100 и 200; на 201-й итерации пул читает compact advantage-only файлы.

