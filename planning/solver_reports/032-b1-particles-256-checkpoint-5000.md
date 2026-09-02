# Отчёт 032 — B1: 256 rollout particles, checkpoint 5000

Дата: 2026-08-28.

## Изолируемая переменная

В сравнении с коротким B1-запуском 026 изменено только число итоговых
rollout particles: `128 -> 256`. Число belief proposals оставлено `256`,
поэтому ESS-пороги и B1 proposal population не изменились. `eta=10`.

## Команда

```powershell
python tools\run_runtime_paired_evaluation.py `
  --checkpoint models\test110\light_checkpoint_iter_5000.pt `
  --output planning\solver_reports\031-b1-particles-256-checkpoint-5000.json `
  --deals 8 --seeds 107 109 `
  --particles 256 --proposals 256 --eta 10 `
  --min-ess 8 --min-ess-ratio 0.03125 `
  --resample-ess-ratio 0.5 --min-root-gap-zscore 1
```

## Результат

| Метрика | 128 particles (026) | 256 particles (031) |
|---|---:|---:|
| Search roots | 23 | 24 |
| ESS fallback | 0 | 0 |
| Noise fallback | 11 | 10 |
| Roots с MMDS | 12 | 14 |
| Смены argmax | 6 | 6 |
| Paired samples | 96 | 96 |
| CI95 EV, bb/100 | [-369.31, 382.85] | [-369.31, 382.85] |

Результат совместим с небольшим снижением noise-gate fallback, однако schedule
candidate меняется вместе с action policy. 96 samples недостаточно, чтобы
приписывать разницу в два roots числу particles. EV-вывод делать нельзя.

## Вывод

Увеличение particles не ухудшило B1 и дало больше MMDS-активаций в этой
диагностике, но сильного подтверждённого эффекта ещё нет. Следующая проверка
отделяет второй параметр search strength: `eta=20` при исходных 128 particles.
