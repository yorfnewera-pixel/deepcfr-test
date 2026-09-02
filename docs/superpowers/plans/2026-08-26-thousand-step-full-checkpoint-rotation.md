# План реализации: ротация full checkpoint с шагом 1000

> **Для агентных исполнителей:** применять `superpowers:subagent-driven-development` или `superpowers:executing-plans`; выполнять задачи последовательно и отмечать чекбоксы.

**Цель:** Отказаться от отдельных `opponent_advantage_iter_N.pt`, сохранять только полный `multi_checkpoint_iter_N.pt` и парный `light_checkpoint_iter_N.pt` раз в 1000 итераций, а opponent pool строить из full checkpoint.

**Архитектура:** Full checkpoint становится единственным источником `advantage_net` и для возобновления, и для соперников. При первом обращении к конкретному full-файлу из него извлекается только `advantage_net` и сохраняется в оперативном кэше; повторные итерации не десериализуют файл с replay-buffer. Light checkpoint остаётся strategy-only артефактом и хранится ровно для тех же номеров, что и удержанные full checkpoint.

**Технологии:** Python, PyTorch, pytest, YAML.

**Спецификация:** Согласованный контракт в чате от 2026-08-26.

## Глобальные ограничения

- `checkpoint_save_every: 1000`.
- `checkpoint_keep_every: 50000` остаётся интервалом исторических checkpoint.
- Первые 1000 training-итераций: наблюдаемый игрок играет против пяти `RandomAgent`.
- Полный checkpoint содержит сети, optimizer и replay-buffer; отдельный advantage-only checkpoint больше не создаётся.
- Ротация хранит 11 свежих неисторических full checkpoint и два последних исторических full checkpoint; light-файлы удерживаются только для этой же выборки номеров. При размере текущего full checkpoint около 556 МБ это до 13 файлов, то есть примерно 7.2 ГБ на один запуск.
- Нельзя автоматически удалять уже существующие `opponent_advantage_iter_*.pt`: новый код их игнорирует, а очистка старых файлов остаётся ручным, обратимым действием пользователя.
- Порядок пяти соперников перемешивается перед посадкой за стол. Выбор случайных historical/recent slot выполняется на каждой итерации, как и сейчас.
- Для каждого уникального full checkpoint файл читается не более одного раза за запуск: именно это не позволит 500+ МБ replay-buffer вновь съедать время каждой итерации.
- Git в проекте удалён пользователем; шаги commit из старых планов не выполнять.

---

## Целевое расписание

Checkpoint создаётся после итераций `1000, 2000, 3000, ...`. Он становится доступен пулу на следующей итерации.

| Диапазон training-итераций | Пул до случайной посадки |
| --- | --- |
| 1–1000 | пять `RandomAgent` |
| 1001–2000 | `1000, 1000, 1000, 1000, 1000` |
| 2001–3000 | `2000, 1000, 1000, 1000, 1000` |
| 3001–4000 | `3000, 2000, 1000, 1000, 1000` |
| 4001–5000 | `4000, 3000, 2000, 1000, 1000` |
| 5001–6000 | `5000, 4000, 3000, 2000, 1000` |
| 6001–7000 | `6000, 5000, 4000, 3000, 2000` |
| c 7001 до второй исторической точки | последний + 4 случайных из десяти предыдущих тысячных slot |
| с 100001 | последний + 3 случайных из десяти предыдущих slot + предпоследний исторический checkpoint |

Исторические точки: `50000, 100000, 150000, ...`. Например, на 100001 используется 50000 как исторический якорь; checkpoint 100000 остаётся текущим последним.

## Task 1: Сменить cadence и контракт хранения

**Файлы:**

- Modify: `config.yaml:32-34`
- Modify: `src/utils/config.py:39-43`
- Modify: `src/training/train.py:23-100, 233-279, 625-637`
- Test: `tests/test_checkpoint_lifecycle.py`

**Интерфейсы:**

- Keeps: `_checkpoint_save_due(iteration, every) -> bool`.
- Changes: `_prune_full_checkpoints(directory, historical_every) -> None` удерживает 11 свежих и 2 исторических full checkpoint.
- Removes: `_OPPONENT_ADVANTAGE_CHECKPOINT_PREFIX`, `_opponent_advantage_checkpoints`, `_prune_opponent_advantage_checkpoints`, `_save_opponent_advantage_checkpoint`.

- [ ] **Step 1: Написать падающие тесты границ cadence и full retention.**

```python
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(999, False), (1000, True), (1001, False), (2000, True)],
)
def test_checkpoint_save_due_every_thousand_iterations(iteration, expected):
    assert train_mod._checkpoint_save_due(iteration, every=1000) is expected


def test_full_checkpoint_retention_keeps_eleven_recent_and_two_historical(tmp_path):
    for iteration in range(1000, 100_001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{iteration}.pt").touch()

    train_mod._prune_full_checkpoints(tmp_path, historical_every=50000)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [50000, *range(89000, 101000, 1000)]
```

- [ ] **Step 2: Запустить RED-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py -k "thousand or full_checkpoint_retention" -v`

Expected: FAIL: cadence ещё равен 100, а full retention оставляет только последний файл.

- [ ] **Step 3: Изменить конфигурацию и lifecycle.**

Установить `checkpoint_save_every: 1000` и `checkpoint_keep_every: 50000` в `config.yaml` и default в `src/utils/config.py`. Заменить текущую очистку full checkpoint на общую retention-логику 11 recent + 2 historical. После full-save вызывать эту очистку, затем сохранять light checkpoint и очищать light по фактически удержанным full checkpoint. Вызов `_save_opponent_advantage_checkpoint` и консольную строку о нём удалить.

- [ ] **Step 4: Запустить GREEN-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS; в каталоге остаются только допустимые full/light файлы и не создаётся `opponent_advantage_iter_N.pt`.

## Task 2: Строить pool по full checkpoint со state-cache

**Файлы:**

- Modify: `src/training/train.py:102-217, 560-585`
- Test: `tests/test_checkpoint_lifecycle.py`

**Интерфейсы:**

- Changes: `_opponent_checkpoint_paths(...) -> list[Path]` сканирует только `multi_checkpoint_iter_*.pt`.
- Produces: `_load_full_checkpoint_advantage_state(path) -> dict[str, torch.Tensor]` с проверкой `action_space_version` и `advantage_net`.
- Produces: кэш `{Path: state_dict}` на время `train_self_play_multi`; `_configure_opponent_pool` принимает кэш и загружает только отсутствующие уникальные пути.

- [ ] **Step 1: Написать падающий тест источника pool.**

```python
def test_opponent_pool_uses_full_checkpoint_and_ignores_old_advantage_file(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    old_advantage_path = tmp_path / "opponent_advantage_iter_1000.pt"
    full_path.touch()
    old_advantage_path.touch()

    paths = train_mod._opponent_checkpoint_paths(1001, tmp_path, 5, 1000, 50000)

    assert paths == [full_path] * 5
```

- [ ] **Step 2: Написать падающий тест кэша.**

Создать реальный full checkpoint с `advantage_net`, вызвать `_configure_opponent_pool` дважды для 1001-й итерации и через monkeypatch `torch.load` зафиксировать один вызов именно для этого пути. Проверить, что агент получил пять состояний и повторная настройка не читает full-файл повторно.

- [ ] **Step 3: Запустить RED-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py -k "opponent_pool_uses_full or cache" -v`

Expected: FAIL: текущий scanner использует compact advantage-файлы и кэша нет.

- [ ] **Step 4: Реализовать scanner, loader и кэш.**

Удалить compact scanner/loader. Full loader должен десериализовать full checkpoint только при cache miss, извлечь `advantage_net`, проверить action-space и сразу отбросить весь payload. Перед `set_opponent_advantage_states` собрать состояния из кэша в порядке выбранных путей. Не держать в кэше replay-buffer или полный payload.

- [ ] **Step 5: Запустить GREEN-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS; повторные итерации одной и той же ротации не делают I/O full checkpoint.

## Task 3: Перевести warm-up и историческую ротацию на шаг 1000

**Файлы:**

- Modify: `src/training/train.py:145-191, 572-580`
- Test: `tests/test_checkpoint_lifecycle.py`
- Test: `tests/test_training_regressions.py`

**Интерфейсы:**

- Consumes: `checkpoint_save_every=1000`, `_heavy_checkpoints`.
- Keeps: пять слотов и случайная посадка.

- [ ] **Step 1: Написать падающий табличный тест начальных пяти тысячных slot.**

```python
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (1001, [1000, 1000, 1000, 1000, 1000]),
        (2001, [1000, 1000, 1000, 1000, 2000]),
        (3001, [1000, 1000, 1000, 2000, 3000]),
        (6001, [2000, 3000, 4000, 5000, 6000]),
    ],
)
def test_opponent_pool_uses_thousand_step_warmup(tmp_path, iteration, expected):
    for checkpoint_iteration in range(1000, 7001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()

    paths = train_mod._opponent_checkpoint_paths(iteration, tmp_path, 5, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == expected
```

- [ ] **Step 2: Написать тест historical-якоря.**

На наборе `1000..100000` с детерминированным `random.sample` вызвать pool для 100001 и ожидать `[50000, 97000, 98000, 99000, 100000]`. Это подтверждает: latest=100000, три recent и предпоследний historical=50000.

- [ ] **Step 3: Запустить RED-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py -k "thousand_step_warmup or historical" -v`

Expected: FAIL, пока fixtures и config используют сотый шаг или compact пути.

- [ ] **Step 4: Сменить fixtures и передаваемый interval.**

Использовать `checkpoint_save_every` в условии первых random-итераций: `iteration <= 1000`. Сообщение остаётся `Играет против: 5 RandomAgent`. Random eval продолжает запускаться раз в 10 итераций; advantage LR достигает базы на 10 000-й итерации, strategy LR не меняется.

- [ ] **Step 5: Запустить GREEN-проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -v`

Expected: PASS.

## Task 4: Убрать недействительный cold-start из лёгкой пары

**Файлы:**

- Modify: `src/core/deep_cfr.py:623-671`
- Modify: `src/training/train.py:508-550, 685-711`
- Modify: `tests/test_checkpoint_lifecycle.py`
- Modify: `tests/test_training_regressions.py`

**Интерфейсы:**

- Keeps: `--initial-checkpoint` для полноценного continuation с optimizer и buffers.
- Removes: `DeepCFRAgent.load_cold_start_pair`, `--cold-start-advantage-checkpoint`, `--cold-start-strategy-checkpoint`.

- [ ] **Step 1: Удалить тесты и код пары compact advantage + strategy.**

Удалить тесты, создающие `opponent_advantage_iter_N.pt` для восстановления, и CLI-тесты этих флагов. Удалить сам метод и аргументы, чтобы проект не рекламировал сценарий, для которого новый lifecycle больше не создаёт входной advantage-файл.

- [ ] **Step 2: Проверить normal continuation.**

Добавить или обновить тест: full checkpoint `multi_checkpoint_iter_4000.pt` через `--initial-checkpoint` выставляет `iteration_count=4000`, а следующая итерация равна 4001. Проверить, что optimizer-state присутствует в full checkpoint.

- [ ] **Step 3: Запустить проверку.**

Run: `pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -v`

Expected: PASS; доступен ровно один документированный способ восстановить обучение — полный checkpoint.

## Task 5: Итоговый smoke-test и миграция старой папки

**Файлы:**

- Verify: `config.yaml`, `src/training/train.py`, `src/core/deep_cfr.py`, тесты.

- [ ] **Step 1: Выполнить целевой набор проверок.**

Run: `pytest tests/test_action_space.py tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -q`

Run: `python -m compileall -q src policy_runtime scripts tools`

Expected: все целевые тесты проходят, компиляция завершается с кодом 0.

- [ ] **Step 2: Выполнить короткий ручной smoke-run.**

Запустить 1001 итерацию с минимально допустимым количеством traversal во временный `--save-dir`. Проверить, что после 1000 есть `multi_checkpoint_iter_1000.pt` и `light_checkpoint_iter_1000.pt`, нет `opponent_advantage_iter_1000.pt`, а в логе 1001-й итерации указаны пять оппонентов из full checkpoint 1000.

- [ ] **Step 3: Проверить восстановление.**

Запустить отдельный короткий continuation с `--initial-checkpoint <временная_папка>/multi_checkpoint_iter_1000.pt`. Проверить начало с 1001-й итерации и использование full checkpoint 1000 в пуле.

- [ ] **Step 4: Обработать старые файлы вручную.**

Старые `opponent_advantage_iter_*.pt` в уже созданных каталогах новый код игнорирует. После успешного smoke-run пользователь может удалить их вручную; автоматическое удаление не добавлять.

## Решение, необходимое до реализации

Этот контракт буквально исключает созданную ранее возможность cold-start из двух лёгких файлов: advantage-only файл больше не сохраняется. В плане поэтому оставлено только нормальное восстановление из `multi_checkpoint_iter_N.pt`. Он также требует хранить до 13 тяжёлых файлов для исторической ротации; если оставить только последний full checkpoint, строить нужный пул соперников станет не из чего.

Если всё же нужно сохранить cold-start без 500+ МБ full-файла, придётся оставить отдельный advantage-артефакт (хотя бы раз в 1000 итераций), что противоречит первому пункту нового запроса. Нужно подтвердить один из вариантов:

1. **Рекомендую для буквального нового контракта:** только full + strategy-light, хранить до 13 full файлов (около 7.2 ГБ), cold-start из лёгкой пары удалить.
2. Оставить ещё advantage-light специально для opponent pool и аварийного cold-start; это экономит диск и I/O, но отменяет требование «никаких advantage checkpoint».
