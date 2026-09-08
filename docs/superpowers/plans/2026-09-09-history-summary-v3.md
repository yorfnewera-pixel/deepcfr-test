# План реализации `history_summary_v3`

> **Для агентных исполнителей:** обязательно выполнять задачи последовательно с независимой проверкой после каждой. Для реализации использовать `superpowers:subagent-driven-development` либо `superpowers:executing-plans`.

**Цель:** устранить доказанные коллизии `encode_state` через компактный actor-aware summary публичной betting history.

**Архитектура:** Rust-движок становится единственным владельцем достоверной public action history. Python encoder строит из неё fixed-width per-street summary и остаётся MLP-входом. Новый encoder несовместим с v2 весами, buffers и inference-артефактами; совместимость проверяется метаданными checkpoint до загрузки весов.

**Стек:** Rust/PyO3, Python 3, NumPy, PyTorch, pytest, cargo test.

**Спецификация:** `docs/superpowers/specs/2026-09-09-encode-state-information-set-audit.md`

## Общие ограничения

- Версия encoder: `history_summary_v3`.
- `CHECKPOINT_FORMAT_VERSION` повышается с 5 до 6; миграции v2 весов и replay buffers нет.
- Сохраняется action space `six_fixed_v2` и шесть slots.
- В v3 запрещён silent fallback для incomplete history.
- V3 обучается с нуля; v2 остаётся отдельным baseline.

---

### Задача 1: Сохранение достоверной public action history в движке

**Файлы:**
- Изменить: `pokers/src/state.rs`
- Изменить: `pokers/src/game_logic.rs`
- Изменить: `pokers/src/state/action.rs`
- Изменить: `pokers/pokers.pyi`
- Тест: `pokers/tests/test_game_logic.py`

**Интерфейсы:**
- `State.action_history: list[ActionRecord]`
- `State.action_history_complete: bool`
- `State.from_mid_hand(..., action_history: list[ActionRecord] | None = None)`

- [ ] Добавить в `State` публичные для PyO3 поля `action_history: Vec<ActionRecord>` и `action_history_complete: bool`.
- [ ] Инициализировать `from_seed` и `from_deck` пустой complete history; `from_mid_hand` получает optional `action_history` последним аргументом и устанавливает complete только при явно переданной полной history.
- [ ] В `apply_action_internal` создать `ActionRecord` до перехода, но append выполнять лишь после проверки legal action и успешного применения; в записи сохранить actor, исходную street, action и legal actions до хода.
- [ ] Добавить Rust/Python regression: legal `Raise` затем `Call` создают две записи с правильными actor и street; illegal action не меняет length history; clone сохраняет history; `from_mid_hand` без history создаёт incomplete state.
- [ ] Запустить `cargo test` в `pokers` и `pytest pokers/tests/test_game_logic.py -q`.

### Задача 2: Детеминированный summary и анти-aliasing regression-тесты

**Файлы:**
- Изменить: `src/core/model.py`
- Изменить: `policy_runtime/core.py`
- Создать: `tests/test_information_set_encoding.py`
- Изменить: `tests/test_relative_encoding.py`

**Интерфейсы:**
- `ENCODING_VERSION = "history_summary_v3"`
- `history_summary_size(num_players: int) -> int`
- `encode_state(state, player_id=0) -> np.ndarray`

- [ ] Вынести единую формулу базового input из `DeepCFRAgent`, `model.py` и `policy_runtime` в импортируемую константу/функцию. Для `n` игроков v3 base size равен `157 + 4 * (2 * n + 8)`.
- [ ] Реализовать private helper, который валидирует `action_history_complete`, группирует `ActionRecord` по street и строит блоки в hero-relative системе координат: `last_actor_relative`, `last_action_kind`, `raise_actor_relative_mask`, `raise_count`, `raise_amount_total / norm_unit`.
- [ ] Добавить unit-тест для пустой complete history, проверки размера для 2 и 6 игроков, а также hero-relative преобразования actor id.
- [ ] Добавить regression для линии «preflop raise против flop raise»: оба состояния доходят до river с одинаковым v2 snapshot, но v3 vectors различаются.
- [ ] Добавить regression для линии «turn aggressor P0 против P1»: same river snapshot и mask, но разные v3 vectors.
- [ ] Добавить тест, что incomplete `from_mid_hand` вызывает `ValueError` с текстом об отсутствующей полной history. Не заменять исключение на нулевой summary.
- [ ] Запустить `pytest tests/test_information_set_encoding.py tests/test_relative_encoding.py tests/test_action_space.py -q`.

### Задача 3: Передача history через reconstruction и inference-пути

**Файлы:**
- Изменить: `src/runtime_search/rollouts.py`
- Изменить: `src/runtime_search/beliefs.py`
- Изменить: `policy_runtime/adapters/pokers.py`
- Изменить: `policy_runtime/core.py`
- Тест: `tests/test_runtime_search_history.py`
- Тест: `tests/test_pokers_regressions.py`

**Интерфейсы:**
- Rebuilt state получает исходные `action_history` и `action_history_complete`.
- `PolicyRuntimeAgent` и `FrozenBlueprintPolicy` не запускают v3 сеть на incomplete history.

- [ ] При `State.from_mid_hand` в rollouts и beliefs передавать clone исходной history и её флаг complete; не присваивать только `from_action`, так как этого недостаточно.
- [ ] Расширить Python protocol/adapters полями history и complete flag, не создавая фиктивную history из последнего действия.
- [ ] Добавить test, что rebuild сохраняет complete history и формирует тот же v3 tensor для неизменённой публичной части state.
- [ ] Добавить test fallback: runtime-search без полной history остаётся blueprint-only по существующей диагностике, а прямой v3 policy inference выдаёт понятную ошибку.
- [ ] Запустить `pytest tests/test_runtime_search_history.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_beliefs.py tests/test_pokers_regressions.py -q`.

### Задача 4: Версионирование input и checkpoint-контракт

**Файлы:**
- Изменить: `src/core/deep_cfr.py`
- Изменить: `policy_runtime/core.py`
- Изменить: `src/evaluation/blueprint_policy.py`
- Изменить: `tests/test_action_space.py`
- Изменить: `tests/test_checkpoint_lifecycle.py`
- Изменить: `tests/test_frozen_blueprint_policy.py`

**Интерфейсы:**
- `CHECKPOINT_FORMAT_VERSION = 6`
- checkpoint fields `encoding_version`, `encoder_input_size`, `num_players`, `use_multi_agent_advantage`

- [ ] Заменить локальные формулы input size на единый v3 helper в `DeepCFRAgent`, включая teacher, opponent networks, advantage/strategy buffers и multi-agent offset.
- [ ] Полный и light checkpoint обязаны писать `encoding_version` и `encoder_input_size` на верхнем уровне и в `config`.
- [ ] До создания сети runtime/frozen policy валидирует format version 6, action space, encoding version, declared input size, num_players и multi-agent mode. Нельзя выводить multi-agent mode только из `input_size != INPUT_SIZE`.
- [ ] `DeepCFRAgent.load_model` и teacher loader явно отклоняют format 5 или `encoding_version != history_summary_v3`; сообщение требует новый training run.
- [ ] Добавить tests, что v3 full/light checkpoints принимаются, format 5 отклоняется, mismatch encoder input size отклоняется и teacher v2 не подключается к v3 student.
- [ ] Запустить `pytest tests/test_action_space.py tests/test_checkpoint_lifecycle.py tests/test_frozen_blueprint_policy.py tests/test_training_regressions.py -q`.

### Задача 5: Обучающий gate и решение о замене baseline

**Файлы:**
- Создать: `tools/audit_information_set_aliasing.py`
- Изменить: `tests/test_information_set_encoding.py`
- Создать: `planning/solver_reports/2026-09-09-history-summary-v3-baseline.md`

**Интерфейсы:**
- CLI возвращает количество decision nodes, distinct tensors, collision buckets и примеры traces без модификации training data.

- [ ] Реализовать read-only audit CLI с параметрами `--players`, `--stack`, `--seed`, `--max-depth`; он применяет только `legal_action_mask` и `resolve_action`, группирует tensors по `float32.tobytes()` и печатает reproducible summaries.
- [ ] Добавить test с параметрами `players=2`, `stack=30`, `seed=17`, `max_depth=12`: v2 baseline имеет минимум одну коллизию, а v3 разделяет две обязательные regression-пары.
- [ ] Запустить короткий smoke training v3 с новой директорией чекпоинтов и без resume from v2; проверить сохранение и загрузку full/light checkpoint.
- [ ] Провести paired evaluation против v2 на согласованном seeds и оппонент-пуле. Заменять v2 baseline разрешено только при отсутствии регрессии по заранее выбранной primary metric и при подтверждённом устранении обязательных collision tests.
- [ ] Сохранить числа, конфигурацию, commit SHA и решение «принять/не принять» в solver report; не подменять baseline без этого отчёта.

## Самопроверка плана

- Все доказанные коллизии покрыты задачей 2.
- Все пути, которые сериализуют, реконструируют или исполняют сеть, покрыты задачами 1, 3 и 4.
- Полная perfect-recall гарантия не заявлена: v3 — измеряемый summary, а не raw sequence.
- Перечислены тесты, migration boundary и условие, запрещающее тихую загрузку v2.
