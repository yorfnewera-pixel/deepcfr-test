# 006 - turn/river validation

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Checkpoint:** `models/test110/light_checkpoint_iter_2000.pt` (только чтение)
**Код:** `src/runtime_search/diagnostics.py`, `tools/run_turn_river_diagnostics.py`, `tests/test_runtime_search_diagnostics.py`
**Данные:** `planning/solver_reports/006-turn-river-validation.json`

## Что проверяется

- На natural trajectories с фиксированным seed: число решений на turn/river, средняя probability mass по шести compact slots, фактическая частота выбранных слотов, entropy и inference latency.
- На deterministic constructed turn, river и response-to-bet turn spots: legal action set, policy mass, entropy и latency независимо от того, насколько редко улица достигается естественно.
- На response-to-bet spot: один S4 decision с ограниченным числом particles, его legality, policy и latency.

## Результат checkpoint 2000

Команда:

```powershell
python tools\run_turn_river_diagnostics.py --checkpoint models\test110\light_checkpoint_iter_2000.pt --output planning\solver_reports\006-turn-river-validation.json --deals 16 --particles 2 --seed 107
```

- Natural trajectories: `16/16` completed, `0` failures, но `0` решений на turn и `0` на river. Это означает, что в этой короткой выборке игра заканчивается до поздних улиц; это не нулевой entropy и не ошибка диагностики.
- Constructed turn: legal `check/raise_0.5pot/all_in`; policy `0.00979 / 0.97416 / 0.01605`, entropy `0.13713`.
- Constructed river: legal `check/raise_0.5pot/all_in`; policy `0.03765 / 0.89777 / 0.06458`, entropy `0.39723`.
- Constructed response-to-bet turn: legal `fold/call/raise_1pot/all_in`; blueprint почти полностью выбирает `raise_1pot` (`0.999755`), entropy `0.00253`.
- S4 probe: выбран legal `all_in`, `2` particles, latency около `27.5 ms`; search policy отличалась от blueprint policy. Raw EV и policy сохранены в JSON.

## Интерпретация

Проверка доказывает работоспособность integration path для light checkpoint: допустимые слоты корректно маскируются, constructed late streets доступны, S4 возвращает legal decision и отдельный отчёт не теряется в консоли.

Она **не** доказывает качество стратегии или EV improvement. В S4 пока используется uniform belief, что явно отмечено флагом `uniform_beliefs_not_solver_valid`; поэтому raw EV probe нельзя использовать как оценку силы checkpoint. Нулевая natural frequency на 16 раздачах — основание продолжать constructed validation, а не вывод о качестве игры.

## Проверка

`python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py -q`

## Следующий шаг

S6: заменить uniform particles на opponent-conditioned beliefs и только после этого переходить к meaningful paired-EV comparison.
