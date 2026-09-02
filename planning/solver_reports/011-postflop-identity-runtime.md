# Runtime Search: postflop identity roots

Дата проверки: 2026-08-28.

## Изменение

Runtime search теперь вызывается во всех compact identity-узлах postflop:

- без ставки: `check`, `raise_0.5pot`, `raise_1pot`, `all_in`;
- после ставки: `fold`, `call`, `raise_1pot`, `all_in`.

На Flop rollout доходит до showdown двумя случайными общими картами; на Turn — одной, на River — без runout. Новые размеры ставок и mapping в action space не добавлялись. Training loop, формат checkpoint и веса модели не менялись.

## Проверки

```powershell
python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py tests/test_runtime_search_history.py tests/test_runtime_search_paired_evaluation.py -q
python -m compileall -q src tools\run_turn_river_diagnostics.py tools\run_runtime_paired_evaluation.py
```

Результат: `41 passed`.

## Проверка checkpoint 3000

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_3000.pt --output planning\solver_reports\011-runtime-paired-evaluation-postflop-identity-3000.json --deals 8 --seeds 107 109 --particles 2
```

| Seed | Samples | Search decisions | Reach-weighted | Fallback | BB/100 |
|---:|---:|---:|---:|---:|---:|
| 107 | 48 | 14 | 14 | 0 | 0.0 |
| 109 | 48 | 15 | 15 | 0 | 0.0 |

Итого: 29 активаций на 96 парных samples. Это подтверждает, что checkpoint 3000 действительно доходит до runtime search в естественных postflop состояниях; прежняя response-only фильтрация больше не блокирует запуск. Нулевой BB/100 в таком коротком paired прогоне не является оценкой силы стратегии.
