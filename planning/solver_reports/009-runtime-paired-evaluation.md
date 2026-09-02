# 009 - runtime paired evaluation

**Статус:** DIAGNOSIS
**Дата:** 2026-08-28
**Checkpoint:** `models/test110/light_checkpoint_iter_2000.pt` (только чтение)
**Код:** `src/runtime_search/paired_evaluation.py`, `src/runtime_search/history.py`, `tools/run_runtime_paired_evaluation.py`, `tests/test_runtime_search_paired_evaluation.py`
**Данные:** `planning/solver_reports/009-runtime-paired-evaluation.json`

## Что сделано

- Добавлен paired-EV runner для frozen blueprint против `RuntimeSearchPolicy`.
- На каждом seed runner сохраняет `BB/100`, sample count, число runtime search roots, reach-weighted decisions и blueprint fallbacks.
- Создан CLI с явными checkpoint/output путями, двумя seed и ограниченным particle budget.

## Результат checkpoint 2000

Команда:

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_2000.pt --output planning\solver_reports\009-runtime-paired-evaluation.json --deals 8 --seeds 107 109 --particles 2
```

| Seed | Samples | BB/100 | Search roots | Reach-weighted | Fallback |
|---:|---:|---:|---:|---:|---:|
| 107 | 48 | 0.0 | 0 | 0 | 0 |
| 109 | 48 | 0.0 | 0 | 0 | 0 |

## Интерпретация

Это не доказательство нулевого EV effect. Runtime solver поддерживает только turn/river response-to-bet roots, а checkpoint 2000 в данном schedule не достиг ни одного такого узла. Поэтому candidate во всех 96 samples играл точный frozen blueprint и identical-policy paired comparison закономерно дал `0.0 BB/100`.

Положительный paired-EV gate пока не выполнен и не провален: для него нужен checkpoint/schedule с ненулевым `search_roots`, либо следующий S7 с lead-bet roots. Увеличивать число таких же preflop-terminating deals не даст дополнительного solver evidence.

## Проверка

- `38 passed in 8.90s` для целевого S1-S6, history integration и paired runner набора.
- `python -m compileall -q src tools/run_turn_river_diagnostics.py tools/run_runtime_paired_evaluation.py` завершился с exit code `0`.

## Следующий шаг

S7: расширить runtime action set для turn/river lead-bet roots и честно определить mapping runtime prior; тогда solver сможет активироваться в более широком классе поздних узлов. Альтернатива — дождаться более позднего checkpoint и повторить ту же команду, сохраняя `search_roots` как обязательный sanity metric.
