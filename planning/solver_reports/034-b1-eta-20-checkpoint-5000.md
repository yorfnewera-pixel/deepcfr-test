# Отчёт 034 — B1: eta=20, checkpoint 5000

Дата: 2026-08-28.

## Изолируемая переменная

В сравнении с коротким B1-запуском 026 изменено только `eta: 10 -> 20`.
Остались `128` rollout particles, `256` proposals, те же seeds, deals и
пороги. При `alpha=0.05` эффективная MMDS-температура выросла с `6.67` до
`10.00`.

## Команда

```powershell
python tools\run_runtime_paired_evaluation.py `
  --checkpoint models\test110\light_checkpoint_iter_5000.pt `
  --output planning\solver_reports\033-b1-eta-20-checkpoint-5000.json `
  --deals 8 --seeds 107 109 `
  --particles 128 --proposals 256 --eta 20 `
  --min-ess 8 --min-ess-ratio 0.03125 `
  --resample-ess-ratio 0.5 --min-root-gap-zscore 1
```

## Результат

| Метрика | eta=10 (026) | eta=20 (033) |
|---|---:|---:|
| Search roots | 23 | 21 |
| ESS fallback | 0 | 0 |
| Noise fallback | 11 | 11 |
| Roots с MMDS | 12 | 10 |
| Смены argmax | 6 | 7 |
| Суммарная L1 policy delta | 10.07 | 13.84 |
| Pooled EV, bb/100 | 6.77 | 209.90 |
| CI95 EV, bb/100 | [-369.31, 382.85] | [-258.89, 678.68] |

Большая `eta` ожидаемо делает policy delta сильнее, но на этой диагностике не
показала большего числа прошедших root. EV-интервал по-прежнему включает ноль;
рост point estimate не является доказательством улучшения.

## Вывод

Не повышать `eta` дальше по текущему сигналу. Рабочая настройка остаётся
`eta=10`, B1 включён, а главный объём дальнейшей проверки должен идти не в
тюнинг температуры, а в воспроизводимый medium/large paired evaluation
выбранной фиксированной конфигурации.

Все запуски были evaluation-only и не трогали training/checkpoint.
