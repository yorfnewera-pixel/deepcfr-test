# Длинная paired EV проверка: checkpoint 3000, particles=2, eta=50

Дата проверки: 2026-08-28.

## Конфигурация

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_3000.pt --output planning\solver_reports\017-runtime-paired-evaluation-eta-50-checkpoint-3000.json --deals 100 --seeds 107 109 113 127 --particles 2 --eta 50
```

2 400 парных samples: по 600 на каждый из четырёх независимых seed. Runtime и frozen blueprint проходят один и тот же paired schedule.

## Результат

| Seed | BB/100 runtime − blueprint | Search roots | Argmax changes | Action changes | Fallback |
|---:|---:|---:|---:|---:|---:|
| 107 | -222.67 | 169 | 46 | 55 | 0 |
| 109 | -153.04 | 168 | 43 | 57 | 0 |
| 113 | -172.54 | 160 | 44 | 53 | 0 |
| 127 | -93.61 | 154 | 22 | 44 | 0 |
| **Среднее** | **-160.47** | **651** | **155** | **209** | **0** |

## Вывод

При `eta=50` solver действительно активно меняет policy: argmax изменился в 155 из 651 root, а фактическое действие — в 209. Однако все четыре seed показывают отрицательный paired результат против frozen blueprint.

Поэтому `eta=50` нельзя использовать как рабочую runtime-настройку. Он чрезмерно усиливает небольшие и шумные rollout EV-различия при `particles=2`. Следующая настройка должна искать промежуточное `eta` между 10 и 50 на том же schedule, например 20 и 30; `particles=8` пока исключается из-за ESS fallback.
