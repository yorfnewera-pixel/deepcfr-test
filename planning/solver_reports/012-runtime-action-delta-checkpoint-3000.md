# Runtime Search: сравнение действий с blueprint

Дата проверки: 2026-08-28.

## Что измеряется

Перед каждым runtime search root policy сэмплирует действие frozen blueprint на том же `State`. Состояние RNG затем восстанавливается до запуска search, поэтому диагностика не меняет фактическое runtime решение и paired schedule.

Отчёт содержит:

- `action_comparisons` — число search roots, где сравнение выполнено;
- `action_changes` — число roots, где сэмплированное действие search отличается от blueprint;
- разбиение обоих счётчиков по Flop, Turn и River.

## Запуск

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_3000.pt --output planning\solver_reports\012-runtime-action-delta-checkpoint-3000.json --deals 8 --seeds 107 109 --particles 2
```

## Результат

| Seed | Samples | Search roots | Сравнений | Отличий | Flop / Turn / River roots |
|---:|---:|---:|---:|---:|---:|
| 107 | 48 | 14 | 14 | 0 | 14 / 0 / 0 |
| 109 | 48 | 15 | 15 | 0 | 15 / 0 / 0 |

На 96 samples солвер активировался 29 раз и во всех случаях использовал reach-weighted beliefs без fallback. В этих 29 сравнениях сэмплированное действие совпало с blueprint.

Это не подтверждает равенство самих распределений: две разные policy могут сэмплировать одинаковое действие при общей точке RNG. До длинной EV-проверки нужно измерить расстояние между `root_blueprint_policy` и `root_search_policy` и различие их argmax. Иначе большой прогон BB/100 может потратить время, не объяснив нулевое различие.

## Регрессия

```powershell
python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py tests/test_runtime_search_history.py tests/test_runtime_search_paired_evaluation.py -q
python -m compileall -q src tools\run_turn_river_diagnostics.py tools\run_runtime_paired_evaluation.py
```

Результат: `42 passed`.
