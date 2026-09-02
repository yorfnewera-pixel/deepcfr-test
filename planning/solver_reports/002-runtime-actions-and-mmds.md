# 002 - MMDS math core

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/runtime_search/__init__.py`, `src/runtime_search/mmds.py`, `tests/test_mmds.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py -q`

## Что сделано

- Добавлен чистый `mmds_update(pi, mc_values, mask, eta, alpha, rho=None, floor=1e-3)`.
- Default `rho` — uniform distribution на legal support.
- Обновление использует логарифмы и вычитание максимального log weight перед exponentiation.
- Illegal actions получают нулевую массу; floor задаёт положительный носитель legal actions.
- Добавлены проверки формы, конечности данных, неотрицательных параметров, пустого legal support и недопустимого floor.

## Почему так

Математика не зависит от `pokers`, checkpoint-ов или rollout-ов, поэтому её можно проверить отдельно до построения дерева поиска. Stabilized log-space update исключает численное переполнение при больших `eta * mc_values`.

## Результаты

Целевые тесты: `12 passed in 4.58s`.

Проверены normalisation, mask, `eta=0`, рост вероятности при большем value, floor и вырождение MMDS в MDS при `rho=pi`.

## Риски

Положительный floor обязателен, если входной prior содержит нулевую массу на legal action; иначе логарифм такого prior не определён. Полный regression suite отложен до окончания training-прогона.

## Следующий шаг

Использовать `mmds_update` в S4 только после появления корректных spots, beliefs и lockstep rollout values.
