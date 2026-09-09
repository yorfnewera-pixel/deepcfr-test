# Надёжная обработка ошибок traversal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Гарантировать, что каждый root traversal либо атомарно сохраняет корректные samples, либо диагностированно отменяется без изменения replay-буферов.

**Architecture:** `DeepCFRAgent.cfr_traverse_multi()` становится единственной границей транзакции traversal. Внутренний обход поднимает классифицированные исключения вместо возврата технического `0.0`; локальный collector принимает samples до успешного commit. Тренировочный цикл обрабатывает только отменяемые ошибки в `skip_traversal`, ведёт метрики и прекращает итерацию по лимиту.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest, `pokers`, YAML.

**Spec:** `docs/superpowers/specs/2026-09-09-traversal-error-handling-design.md`

## Global Constraints

- Не менять ES/OS, action space `six_fixed_v2`, regret/discount, opponent pool, strategy-sample semantics и архитектуры сетей.
- `training_error_mode` принимает только `strict` и `skip_traversal`; по умолчанию `strict`.
- Неожиданные программные ошибки, checkpoint/model incompatibility, OOM, повреждение буфера и NaN/Inf training loss/gradient всегда прекращают процесс.
- Нельзя реализовывать откат reservoir; основной буфер не меняется до успешного commit.
- Все новые комментарии и пользовательские сообщения пишутся по-русски.

---

## File structure

- Create: `src/core/traversal_errors.py` — типы классифицированных ошибок traversal и безопасный диагностический контекст.
- Modify: `src/utils/settings.py` — актуальный getter strict-переключателя при сохранении setter.
- Modify: `src/utils/config.py` — дефолты и валидаторы параметров обработки ошибок.
- Modify: `config.yaml` — явные production-значения новых параметров.
- Modify: `src/core/deep_cfr.py` — collector, валидация, граница commit и замена silent fallbacks.
- Modify: `src/training/train.py` — счётчики отмен, лимит на итерацию, метрики и логирование.
- Modify: `tests/test_config_flags.py` — конфигурационные режимы.
- Create: `tests/test_traversal_error_handling.py` — unit/regression coverage транзакции traversal.
- Modify: `tests/test_training_regressions.py` — лимит skip traversal и защита optimizer step.

### Task 1: Контракт ошибок и конфигурации

**Files:**
- Create: `src/core/traversal_errors.py`
- Modify: `src/utils/settings.py:6-11`
- Modify: `src/utils/config.py:11-52, 68-89`
- Modify: `config.yaml:1-42`
- Modify: `tests/test_config_flags.py`

**Interfaces:**
- Produces: `TraversalFailure`, `TraversalFailureContext`, `is_skippable_traversal_failure(error)`.
- Produces: `settings.is_strict_checking() -> bool` alongside `set_strict_checking(strict_mode: bool) -> None`.
- Produces: `cfg_training_error_mode() -> str`, `cfg_training_validate_state_invariants() -> bool`, `cfg_training_max_failed_traversals_per_iteration() -> int`.

- [ ] **Step 1: Написать failing-тесты конфигурации и live strict mode**

```python
def test_strict_mode_switch_after_deep_cfr_import_changes_live_setting():
    import src.core.deep_cfr  # noqa: F401
    from src.utils import settings

    settings.set_strict_checking(False)
    assert settings.is_strict_checking() is False
    settings.set_strict_checking(True)
    assert settings.is_strict_checking() is True


@pytest.mark.parametrize("mode", ["strict", "skip_traversal"])
def test_training_error_mode_accepts_supported_values(tmp_path, mode):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"num_actions: 6\\ntraining_error_mode: {mode}\\n", encoding="utf-8")
    config_mod.load_config(config_path)
    assert config_mod.cfg_training_error_mode() == mode
```

- [ ] **Step 2: Запустить тесты и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_config_flags.py -q`

Expected: FAIL, поскольку `is_strict_checking` и `cfg_training_error_mode` ещё не существуют.

- [ ] **Step 3: Добавить минимальный типизированный контракт и валидаторы**

В `src/core/traversal_errors.py` определить неизменяемый контекст и исключение:

```python
@dataclass(frozen=True)
class TraversalFailureContext:
    iteration: int
    traversal_index: int | None
    traversing_player: int
    acting_player: int | None
    depth: int
    reason: str
    action_trace: tuple[str, ...] = ()


class TraversalFailure(RuntimeError):
    def __init__(self, context: TraversalFailureContext, cause: BaseException | None = None): ...
```

`is_skippable_traversal_failure` возвращает `True` только для `TraversalFailure`; `KeyboardInterrupt`, `SystemExit`, `MemoryError` и остальные исключения не преобразовывать.

В `settings.py` оставить существующий `STRICT_CHECKING`, добавить getter и приводить setter к `bool`:

```python
def is_strict_checking() -> bool:
    return bool(STRICT_CHECKING)
```

В `_DEFAULTS` добавить точные значения из спецификации. После `_deep_merge` валидировать mode и неотрицательный integer limit; invalid YAML должен выдавать `ValueError` с названием ключа. В `config.yaml` записать те же три ключа с документирующими русскими комментариями.

- [ ] **Step 4: Запустить конфигурационные тесты**

Run: `python -m pytest tests/test_config_flags.py -q`

Expected: PASS.

- [ ] **Step 5: Закоммитить независимый контракт**

```bash
git add src/core/traversal_errors.py src/utils/settings.py src/utils/config.py config.yaml tests/test_config_flags.py
git commit -m "feat: добавить режимы ошибок обучения"
```

### Task 2: Локальный collector и валидация samples до commit

**Files:**
- Modify: `src/core/deep_cfr.py:456-464, 510-517`
- Create: `tests/test_traversal_error_handling.py`

**Interfaces:**
- Consumes: `TraversalFailure`, `TraversalFailureContext` из Task 1.
- Produces: private `DeepCFRAgent._active_traversal_collector` и методы `_record_advantage_sample(...)`, `_record_strategy_sample(...)`, которые записывают локально во время root traversal.
- Produces: `_validate_training_sample(state, values, mask, iteration, sample_kind) -> None`.

- [ ] **Step 1: Написать failing-тест атомарности**

```python
def test_failed_root_traversal_does_not_change_full_reservoir(monkeypatch):
    agent = _agent_with_small_reservoir()
    before = _buffer_snapshot(agent.strategy_buffer)
    monkeypatch.setattr(agent, "_cfr_traverse_multi", _record_then_fail)

    with pytest.raises(TraversalFailure):
        agent.cfr_traverse_multi(_state(), iteration=1, traversing_player=0)

    assert _buffer_snapshot(agent.strategy_buffer) == before
```

Добавить второй тест: успешный root traversal вызывает ровно один commit и увеличивает оба целевых буфера ожидаемыми samples. `_buffer_snapshot` должен включать `_cur_id`, `_size` и копии использованных массивов, а не только `len(buffer)`.

- [ ] **Step 2: Запустить тест и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_traversal_error_handling.py -q`

Expected: FAIL, поскольку текущая `_record_advantage_sample` сразу вызывает `buffer.add`.

- [ ] **Step 3: Реализовать collector без копирования буферов**

Добавить приватный малый dataclass/объект collector в `deep_cfr.py`, хранящий два списка tuple-образцов. На `depth == 0` `cfr_traverse_multi` создаёт collector, присваивает его только на время вызова и вызывает `_commit_traversal_collector` лишь после успешного возврата `_cfr_traverse_multi`.

`_record_advantage_sample` сохраняет прежнее обновление счётчиков только на commit. Новый `_record_strategy_sample` заменяет прямой `self.strategy_buffer.add(...)` в traversal. Перед добавлением каждый sample проверяется:

```python
def _validate_training_sample(self, state, values, mask, iteration, sample_kind):
    assert state.shape == (self.input_size,)
    assert values.shape == (NUM_ACTIONS,)
    assert mask.shape == (NUM_ACTIONS,)
    if not (np.isfinite(state).all() and np.isfinite(values).all() and np.isfinite(mask).all()):
        raise TraversalFailure(...)
```

Заменить `assert` на явный `raise TraversalFailure`, чтобы проверки работали при `python -O`. Проверить mask: все значения 0/1, есть хотя бы один legal slot; policy дополнительно конечна, неотрицательна, нулевая на illegal slots с `atol=1e-6`, сумма legal вероятностей `np.isclose(..., atol=1e-6)`.

- [ ] **Step 4: Запустить узкие тесты collector**

Run: `python -m pytest tests/test_traversal_error_handling.py -q`

Expected: PASS.

- [ ] **Step 5: Закоммитить атомарность samples**

```bash
git add src/core/deep_cfr.py tests/test_traversal_error_handling.py
git commit -m "feat: атомарно сохранять samples traversal"
```

### Task 3: Заменить silent fallback в обходе на классифицированную ошибку

**Files:**
- Modify: `src/core/deep_cfr.py:519-628`
- Modify: `tests/test_traversal_error_handling.py`

**Interfaces:**
- Consumes: collector из Task 2 и `settings.is_strict_checking()` из Task 1.
- Produces: `_raise_traversal_failure(...) -> NoReturn` и `_validate_transition(next_state, ...) -> None`.

- [ ] **Step 1: Написать failing-регрессии для каждого ложного `0.0`**

```python
def test_apply_action_failure_raises_in_strict_mode(monkeypatch):
    agent, state = _agent_and_nonterminal_state()
    monkeypatch.setattr(state, "apply_action", _raise_runtime_error)
    settings.set_strict_checking(True)

    with pytest.raises(TraversalFailure, match="apply_action"):
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)


def test_depth_limit_is_failure_not_zero_reward(monkeypatch):
    agent, state = _agent_and_nonterminal_state()
    with pytest.raises(TraversalFailure, match="depth"):
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0, depth=201)


def test_real_terminal_zero_reward_is_returned():
    agent, state = _agent_and_terminal_zero_reward_state()
    assert agent.cfr_traverse_multi(state, iteration=1, traversing_player=0) == 0.0
```

Добавить аналогичные тесты для non-`Ok` `StateStatus`, пустой legal mask и поздней ошибочной ветки traverser. Последний должен доказывать, что policy не пересчитана на подмножестве actions и collector не закоммичен.

- [ ] **Step 2: Запустить тесты и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_traversal_error_handling.py -q`

Expected: FAIL, так как текущие ветки возвращают `0.0` или делают `continue`.

- [ ] **Step 3: Переписать только обработку ошибок в `_cfr_traverse_multi`**

Заменить импорт `from src.utils.settings import STRICT_CHECKING` на `from src.utils import settings`. Во всех ветках external policy, random agent, traverser и sampled opponent:

- преобразовать ожидаемую ошибку `action_type_to_pokers_action`, `choose_action` или `apply_action` в `TraversalFailure` c `raise ... from cause`;
- проверять `next_state.status == pkrs.StateStatus.Ok` перед рекурсией;
- при traverser не использовать `continue` и не строить `applied_mask` после единственной неудачной ветки;
- превратить `depth > 200` и пустой nonterminal `legal_slots` в failure;
- сохранить возврат terminal reward, включая `0.0`.

В `strict` root boundary всегда повторно поднимает failure. Политика режима `skip_traversal` не должна жить во внутреннем рекурсивном методе: он всегда поднимает failure, а решение продолжать принимает `train_self_play_multi` в Task 4.

- [ ] **Step 4: Запустить traversal-регрессии**

Run: `python -m pytest tests/test_traversal_error_handling.py -q`

Expected: PASS.

- [ ] **Step 5: Запустить прежние контрактные тесты traversal**

Run: `python -m pytest tests/test_action_space.py tests/test_pokers_regressions.py tests/test_training_regressions.py -q`

Expected: PASS.

- [ ] **Step 6: Закоммитить отказ от silent fallback**

```bash
git add src/core/deep_cfr.py tests/test_traversal_error_handling.py
git commit -m "fix: не маскировать ошибки обхода нулевой наградой"
```

### Task 4: Политика `skip_traversal`, лимит итерации и наблюдаемость

**Files:**
- Modify: `src/training/train.py:574-744`
- Modify: `src/core/deep_cfr.py:428-454, 510-517`
- Modify: `tests/test_training_regressions.py`

**Interfaces:**
- Consumes: `cfg_training_error_mode`, `cfg_training_max_failed_traversals_per_iteration`, `TraversalFailure`.
- Produces: `DeepCFRAgent.record_traversal_failure(error) -> dict[str, object]` и расширенный `get_traversal_stats()`.

- [ ] **Step 1: Написать failing-тесты train loop**

```python
def test_skip_traversal_continues_after_one_failure(monkeypatch, tmp_path):
    agent = _fake_agent_with_traversal_outcomes([TraversalFailure(_ctx()), None])
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)

    train_mod.train_self_play_multi(num_iterations=1, traversals_per_iteration=2,
                                    evaluate_every=0, save_dir=tmp_path)

    assert agent.traversal_attempts == 2
    assert agent.traversal_failures == 1
    assert agent.train_advantage_calls == 1


def test_skip_traversal_stops_when_iteration_limit_is_reached(monkeypatch, tmp_path):
    agent = _fake_agent_with_traversal_outcomes([TraversalFailure(_ctx()), TraversalFailure(_ctx())])
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)

    with pytest.raises(RuntimeError, match="training_max_failed_traversals_per_iteration"):
        train_mod.train_self_play_multi(num_iterations=1, traversals_per_iteration=2,
                                        evaluate_every=0, save_dir=tmp_path)
```

Добавить тест strict mode: первая `TraversalFailure` немедленно выходит из `train_self_play_multi` и не запускает обучение сетей.

- [ ] **Step 2: Запустить тесты и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_training_regressions.py -q`

Expected: FAIL, поскольку сейчас тренировка не перехватывает `TraversalFailure` и не знает лимита.

- [ ] **Step 3: Реализовать единственную границу skip policy**

В `train_self_play_multi` перед циклом traversal каждого игрока обнулить локальный `failed_traversals`. На каждый root traversal сначала увеличить attempted, затем:

```python
try:
    agent.cfr_traverse_multi(...)
    agent.record_traversal_success()
except TraversalFailure as error:
    agent.record_traversal_failure(error)
    if cfg_training_error_mode() != "skip_traversal":
        raise
    failed_traversals += 1
    if failed_traversals >= cfg_training_max_failed_traversals_per_iteration():
        raise RuntimeError("Превышен training_max_failed_traversals_per_iteration") from error
```

Не добавлять `except Exception`. В stats хранить attempted/successful/failed, `cancelled_samples`, `depth_limit_hits` и `failure_reasons`; в `_log_multi_cfr_diagnostics` отправить скалярные метрики в writer и вывести короткое русское сообщение с iteration/traversal/reason. Полный trace остаётся в error/context и локальном logger, без карт или скрытого состояния.

- [ ] **Step 4: Запустить training-регрессии**

Run: `python -m pytest tests/test_training_regressions.py -q`

Expected: PASS.

- [ ] **Step 5: Закоммитить наблюдаемую skip-политику**

```bash
git add src/training/train.py src/core/deep_cfr.py tests/test_training_regressions.py
git commit -m "feat: учитывать отменённые traversal в обучении"
```

### Task 5: Защита optimizer step от NaN/Inf

**Files:**
- Modify: `src/core/deep_cfr.py:633-724, 726-829`
- Modify: `tests/test_training_regressions.py`

**Interfaces:**
- Produces: `DeepCFRAgent._assert_finite_training_tensors(loss, parameters, stage) -> None`.

- [ ] **Step 1: Написать failing-тесты для advantage и strategy optimizers**

```python
def test_advantage_nan_loss_does_not_step_optimizer(monkeypatch):
    agent = _agent_with_one_advantage_sample()
    stepped = False
    monkeypatch.setattr(agent.optimizer, "step", lambda: pytest.fail("optimizer.step не должен вызываться"))
    monkeypatch.setattr(torch.nn.functional, "mse_loss", lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True))

    with pytest.raises(FloatingPointError, match="advantage loss"):
        agent.train_advantage_network_multi(batch_size=1)
```

Повторить для `train_strategy_network` и добавить тест нечислового gradient: подменить `loss.backward`, записав NaN в первый parameter.grad, затем проверить, что `step` не вызван.

- [ ] **Step 2: Запустить тесты и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_training_regressions.py -q`

Expected: FAIL, поскольку текущий код вызывает `optimizer.step()` без проверки конечности.

- [ ] **Step 3: Реализовать конечные проверки до step**

После вычисления loss и до `zero_grad/backward` проверять `torch.isfinite(loss)`. После `backward`, но до `clip_grad_norm_` и `optimizer.step`, обходить parameters и проверять все не-`None` `parameter.grad` через `torch.isfinite`. При failure вызывать `optimizer.zero_grad(set_to_none=True)` и поднимать `FloatingPointError` с названием stage. Не ловить это исключение в skip traversal.

- [ ] **Step 4: Запустить тесты обучения**

Run: `python -m pytest tests/test_training_regressions.py -q`

Expected: PASS.

- [ ] **Step 5: Закоммитить защиту оптимизаторов**

```bash
git add src/core/deep_cfr.py tests/test_training_regressions.py
git commit -m "fix: останавливать обучение при NaN и Inf"
```

### Task 6: Полная проверка и измерение накладных расходов

**Files:**
- Modify: `docs/superpowers/specs/2026-09-09-traversal-error-handling-design.md` только если фактические команды или результаты требуют уточнения.

**Interfaces:**
- Consumes: все завершённые изменения Tasks 1–5.
- Produces: проверенные результаты suite и smoke runs; код не меняется без отдельной причины.

- [ ] **Step 1: Запустить обязательные regression suites**

Run: `python -m pytest tests/test_config_flags.py tests/test_traversal_error_handling.py tests/test_buffer_reservoir_flags.py tests/test_action_space.py tests/test_pokers_regressions.py tests/test_training_regressions.py -q`

Expected: PASS.

- [ ] **Step 2: Выполнить HU smoke run в strict mode**

Run: `python -m src.training.train --self-play-multi --iterations 1 --traversals 2 --evaluate-every 0 --save-dir models/smoke-error-handling-hu --log-dir logs/smoke-error-handling-hu`

Expected: процесс завершается без diagnostics failure; вывести attempted/successful/failed и время traversal.

- [ ] **Step 3: Выполнить six-max smoke run в strict mode**

Run: `python -m src.training.train --self-play-multi --iterations 1 --traversals 2 --evaluate-every 0 --save-dir models/smoke-error-handling-sixmax --log-dir logs/smoke-error-handling-sixmax`

Expected: процесс завершается без diagnostics failure; записаны те же метрики.

- [ ] **Step 4: Удалить только созданные smoke artifacts после фиксации результатов**

Удалять исключительно два явно созданных каталога `models/smoke-error-handling-hu`, `models/smoke-error-handling-sixmax`, `logs/smoke-error-handling-hu`, `logs/smoke-error-handling-sixmax`, предварительно сверив их абсолютные пути и статус Git. Если их нужно сохранить для аудита, не удалять.

- [ ] **Step 5: Закоммитить документационное уточнение при наличии изменений**

```bash
git add docs/superpowers/specs/2026-09-09-traversal-error-handling-design.md
git commit -m "docs: зафиксировать проверку обработки ошибок"
```

## Self-review

- [x] Spec coverage: Task 1 реализует режимы и live strict flag; Tasks 2–3 покрывают atomic collector, policy/state failures и depth limit; Task 4 реализует skip limit, метрики и диагностику; Task 5 покрывает NaN/Inf до optimizer step; Task 6 запускает regression и HU/six-max smoke.
- [x] Explicit exclusion: state invariants по `pot`, `bet_chips`, `pot_chips` не реализуются до отдельного аудита семантики движка; флаг сохраняется и должен быть отклонён с понятной ошибкой, если включён до реализации.
- [x] Placeholder scan: в плане отсутствуют незаполненные маркеры, ссылки «как Task N» и неуточнённые тестовые действия.
- [x] Type consistency: публичные имена `TraversalFailure`, `TraversalFailureContext`, `cfg_training_error_mode` и `_assert_finite_training_tensors` определены до первого использования.
