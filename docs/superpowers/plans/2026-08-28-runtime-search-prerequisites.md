# Runtime Search Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Подготовить независимые от обучения компоненты для runtime search: inference из light checkpoint, корректную метрику парной оценки и чистое MMDS-обновление.

**Architecture:** `FrozenBlueprintPolicy` будет загружать light checkpoint непосредственно в `PokerNetwork`, не создавая `DeepCFRAgent`; heavy checkpoint останется на существующем пути. `mmds_update` будет изолированной NumPy-функцией без зависимостей от движка или checkpoint-ов.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest.

**Spec:** `planning/solver_reports/solver.md`

## Global Constraints

- Не изменять training loop, `DeepCFRAgent._load_checkpoint`, формат checkpoint и `src/core/action_space.py`.
- Принимать только checkpoint format `5` и action space `six_fixed_v2`.
- Все новые комментарии и диагностические сообщения писать по-русски.
- Не запускать полный test suite, пока пользовательский training-прогон использует вычислительные ресурсы.
- В MMDS недопустимые действия получают нулевую вероятность; допустимые получают не меньше `floor` после нормализации.

---

## File Structure

- `src/evaluation/blueprint_policy.py` — загрузка frozen strategy-only policy и получение маскированного распределения.
- `src/evaluation/paired_harness.py` — результаты paired evaluation и нормализация в bb/100.
- `src/runtime_search/__init__.py` — публичный экспорт математического ядра runtime search.
- `src/runtime_search/mmds.py` — чистая реализация MMDS.
- `tests/test_frozen_blueprint_policy.py` — интеграционные контракты light/heavy checkpoint.
- `tests/test_paired_evaluation.py` — контракт единиц измерения bb/100.
- `tests/test_mmds.py` — поведенческие тесты MMDS.
- `planning/solver_reports/001-evaluation-prerequisite.md` — отчёт S-1.
- `planning/solver_reports/002-runtime-actions-and-mmds.md` — отчёт S1.

### Task 1: Загрузка light checkpoint и bb/100

**Files:**
- Modify: `src/evaluation/blueprint_policy.py`
- Modify: `src/evaluation/paired_harness.py`
- Modify: `tests/test_frozen_blueprint_policy.py`
- Modify: `tests/test_paired_evaluation.py`

**Interfaces:**
- Consumes: light checkpoint с ключом `strategy_net`, heavy checkpoint с ключами strategy и advantage networks.
- Produces: `FrozenBlueprintPolicy.from_checkpoint(path, num_players=6, device="cpu")` для обоих допустимых видов checkpoint и `PairedEvaluation.bb_per_100` в единицах большого блайнда.

- [x] **Step 1: Написать падающий тест для light checkpoint**

Создать light checkpoint через `DeepCFRAgent.export_strategy_checkpoint`, затем вызвать `FrozenBlueprintPolicy.from_checkpoint`. Проверить `probabilities(state)` на конечные неотрицательные шесть вероятностей, сумму `1.0` и нули для illegal slots. Отдельно убедиться, что тот же API принимает heavy checkpoint из `save_model`.

- [x] **Step 2: Запустить тест и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py -q`

Expected: light checkpoint вызывает `ValueError` от строгого `DeepCFRAgent._load_checkpoint` из-за отсутствующих advantage networks.

- [x] **Step 3: Реализовать минимальную самостоятельную загрузку strategy-only checkpoint**

В `FrozenBlueprintPolicy.from_checkpoint` загрузить словарь через `torch.load(..., weights_only=False)`. Валидировать `checkpoint_format_version`, `action_space_version`, `num_actions`, ключ `strategy_net` и совместимость числа игроков. Для `checkpoint_kind == "strategy_only"` создать `PokerNetwork`, загрузить только `strategy_net`, перевести сеть в `eval()` и сохранить зависимости для существующей маски legal slots, encoding state и masked softmax. Для heavy checkpoint сохранить текущий путь через `DeepCFRAgent`.

- [x] **Step 4: Исправить контракт bb/100**

Добавить поле `bb: float` в `PairedEvaluation`, передать `bb` из `evaluate_paired`, валидировать положительность. Вернуть `mean_difference / bb * 100.0`. Добавить тест с вручную заданными `differences=np.array([4.0, 8.0])` и `bb=2.0`, ожидающий `300.0`.

- [x] **Step 5: Запустить целевые тесты**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py -q`

Expected: все тесты двух файлов проходят.

### Task 2: Чистая MMDS-математика

**Files:**
- Create: `src/runtime_search/__init__.py`
- Create: `src/runtime_search/mmds.py`
- Create: `tests/test_mmds.py`

**Interfaces:**
- Consumes: `pi`, `mc_values`, `mask`, `eta`, `alpha`, необязательный `rho`, `floor` как одномерные NumPy-массивы и скаляры.
- Produces: `mmds_update(pi, mc_values, mask, eta, alpha, rho=None, floor=1e-3) -> np.ndarray` с вероятностями только на legal actions.

- [x] **Step 1: Написать падающие тесты контракта**

Добавить независимые тесты на: сумму output `1.0`; ноль на illegal action; сохранение policy при `eta=0` и `floor=0`; рост вероятности действия с большим `mc_values`; floor на legal actions; равенство с ручным MDS при `rho=pi`, где `eta_eff = eta / (1 + alpha * eta)`.

Для MDS-эталона использовать literal-вектор `pi=[0.2, 0.3, 0.5]`, `values=[-0.4, 0.1, 0.7]`, `eta=2.0`, `alpha=0.5`; вычислить эталон через независимую формулу `pi * exp(eta_eff * values)` с вычитанием максимального logit.

- [x] **Step 2: Запустить тест и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_mmds.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'src.runtime_search'`.

- [x] **Step 3: Реализовать численно устойчивый MMDS**

Проверить одинаковую форму входов, конечность значений, неотрицательность `eta`, `alpha`, `floor` и хотя бы одно legal action. Нормализовать `pi` и `rho` только на legal support; default `rho` — uniform по legal actions. Вычислить

```python
log_weights = (
    np.log(pi_legal)
    + eta * mc_values_legal
    + eta * alpha * np.log(rho_legal)
) / (1.0 + alpha * eta)
```

Вычесть `np.max(log_weights)`, нормализовать через сумму `np.exp`, применить floor как `(1 - floor * legal_count) * policy + floor`; при `floor * legal_count > 1` выбросить `ValueError`. Вернуть `np.float64` vector исходной длины с нулями на illegal actions.

- [x] **Step 4: Запустить математические тесты**

Run: `python -m pytest tests/test_mmds.py -q`

Expected: все тесты проходят без импорта `pokers` или `torch` из `src.runtime_search.mmds`.

### Task 3: Отчёты и пропорциональная верификация

**Files:**
- Create: `planning/solver_reports/001-evaluation-prerequisite.md`
- Create: `planning/solver_reports/002-runtime-actions-and-mmds.md`

**Interfaces:**
- Consumes: результаты целевых pytest-команд.
- Produces: отчёты в формате `solver.md` с фактическими командами, результатами и рисками.

- [x] **Step 1: Заполнить отчёт S-1**

Указать затронутые файлы, факт сохранения strict heavy loader, проверку light checkpoint и формулу `mean_difference / bb * 100`.

- [x] **Step 2: Заполнить отчёт S1**

Указать чистую формулу MMDS, default `rho=uniform`, log-sum-exp стабилизацию, mask/floor и перечень тестов.

- [ ] **Step 3: Выполнить верификацию без конкуренции с training**

После завершения пользовательского прогона запустить:

```powershell
python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py -q
python -m pytest tests/test_pokers_regressions.py tests/test_training_regressions.py -q
```

Expected: exit code `0`; при любом падении зафиксировать его в отчёте и не утверждать завершение этапа.
