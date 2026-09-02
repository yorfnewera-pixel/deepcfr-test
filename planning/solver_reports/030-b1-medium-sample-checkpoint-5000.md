# Отчёт 030 — B1, средняя выборка на checkpoint 5000

Дата: 2026-08-28.

## Режим

```powershell
python tools\run_runtime_paired_evaluation.py `
  --checkpoint models\test110\light_checkpoint_iter_5000.pt `
  --output planning\solver_reports\029-b1-medium-sample-checkpoint-5000.json `
  --deals 16 --seeds 107 109 113 127 `
  --particles 128 --proposals 256 --eta 10 `
  --min-ess 8 --min-ess-ratio 0.03125 `
  --resample-ess-ratio 0.5 --min-root-gap-zscore 1
```

Это evaluation-only запуск: training, checkpoint и модель не менялись.

## Результат

| Метрика | Значение |
|---|---:|
| Paired samples | 384 |
| Search roots | 112 |
| ESS fallback | 0 |
| Noise-gate fallback | 56 |
| Roots, дошедшие до MMDS | 56 |
| Смены argmax | 15 |
| Всего B1 resampling | 384 |
| Pooled EV | 13.67 bb/100 |
| CI95 pooled EV | [-141.81, 169.16] bb/100 |

Проверка root-сигнала:

| Метрика `best_gap_zscore` | Значение |
|---|---:|
| Число конечных z-score | 94 |
| Медиана | 0.00 |
| 75-й перцентиль | 3.89 |
| `z >= 1` | 38 |
| `z < 1` | 56 |

Для ещё 18 roots разность имела нулевую ошибку, поэтому z-score был
непредставимым в JSON (`None` после защиты от `inf`); эти roots не были
отсечены noise-gate и вошли в 56 MMDS-активаций.

## Вывод

B1 устойчиво снимает ESS как ограничение: ни один root не откатился по ESS.
Ограничитель теперь именно статистический: примерно половина root не проходит
порог уверенности `z >= 1`. EV-результат всё ещё статистически незначим —
интервал включает и существенный минус, и плюс.

Следующий изолированный тест — увеличить число rollout particles с 128 до 256
при неизменных `eta`, ESS-порогах и seed. Это проверит, уменьшает ли variance
и число noise-gate fallback без смешения эффекта с MMDS температурой.
