# Runtime Search: сравнение распределений policy

Дата проверки: 2026-08-28.

## Метрики

Для каждого runtime root сравниваются `SearchDecision.root_blueprint_policy` и `SearchDecision.root_search_policy`:

- **L1**: сумма `abs(search_policy - blueprint_policy)` по action slots; диапазон от 0 до 2;
- **L1 mean / max**: средняя и максимальная L1 на search root;
- **argmax changes**: число roots, где наиболее вероятный action slot изменился.

Эти метрики не добавляют rollout и не меняют RNG или runtime action.

## Запуск

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_3000.pt --output planning\solver_reports\013-runtime-policy-delta-checkpoint-3000.json --deals 8 --seeds 107 109 --particles 2
```

## Результат

| Seed | Search roots | L1 mean | L1 max | Argmax changes | Сэмплированные changes |
|---:|---:|---:|---:|---:|---:|
| 107 | 14 | 0.066102 | 0.173629 | 0 | 0 |
| 109 | 15 | 0.029954 | 0.101092 | 0 | 0 |

Все 29 root были на Flop и использовали reach-weighted beliefs без fallback.

## Вывод

Checkpoint 3000 не игнорирует solver: runtime search сдвигает распределения на каждой проверенной улице. Однако на данном малом sample самый вероятный slot не менялся, а значит и сэмплированное действие при common RNG осталось blueprint-действием.

Нулевой результат короткого paired EV прогона теперь объяснён: это не признак неактивного solver, а следствие малого policy shift без смены argmax. Следующая корректная проверка — увеличить число particles или MMDS `eta`, затем повторить policy-delta; в длинный BB/100 прогон имеет смысл идти, когда появятся argmax/action changes.

## Регрессия

```powershell
python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py tests/test_runtime_search_history.py tests/test_runtime_search_paired_evaluation.py -q
python -m compileall -q src tools\run_turn_river_diagnostics.py tools\run_runtime_paired_evaluation.py
```
