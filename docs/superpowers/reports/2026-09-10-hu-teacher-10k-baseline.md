# HU teacher 10k — воспроизводимый baseline

## Артефакт

- Checkpoint: `models/hu_teacher_10k/hu_checkpoint_final.pt`.
- SHA-256: `856c5091be8fe67a87fe3a460a3ea54a1651b0670a1278d37ffa7af1277a449c`.
- Source commit на старте: `d7cb462b09db0892255c506dd2d4f0adecae7407`.
  Последнее функциональное изменение HU до запуска: `a338afbe0a16b42d666aa7703888b366ec8f7d1d`
  (`advantage_reward_scale`).
- Seed: `20260909`; iteration: `10000`; `1` traversal на iteration.

## Конфигурация и запуск

Checkpoint фиксирует `hu_current_policy_self_play=true`, `num_players=2`,
`num_trainable_players=2`, `history_summary_v3`, `card_context_v1`, hidden
size `64`, strict mode, `advantage_reward_scale=200`, AdamW (`advantage_lr=1e-4`,
`strategy_lr=5e-5`), batch size `64`, по одной эпохе, буферы `50000` и
checkpoint каждые `1000` итераций. Полная конфигурация сохранена в поле
`config` checkpoint; архитектура — в `architecture`.

Запуск воспроизводится из состояния commit выше с `config.yaml`:

```powershell
python -m src.training.train --self-play-multi --iterations 10000 --traversals 1 --save-dir models/hu_teacher_10k --log-dir logs/hu_teacher_10k --num-players 2 --trainable-players 2 --device cpu --seed 20260909
```

## Время и loss

По `logs/hu_teacher_10k/events.out.tfevents.*`: с
`2026-09-10 00:50:13 +03:00` до `07:11:33 +03:00`, wall-clock
`6:21:19.8` (среднее время iteration `2.2879 s`).

| Метрика | Последняя iteration | Среднее последних 100 | Среднее за 10 000 |
| --- | ---: | ---: | ---: |
| Advantage loss | 0.066802 | 0.104246 | 0.116736 |
| Strategy loss | 0.092659 | 0.092426 | 0.030607 |

## Resume и ограничения

Resume был проверен отдельно с `10000` до `10001`: policy P0 и P1 конечны и
нормированы. Checkpoint выбран как последний артефакт завершённого запуска, не
по training loss.

Это технически валидный, но слабый baseline: при одном traversal на iteration
его нельзя считать доказанно сильным teacher. Независимой оценки качества пока
нет; сравнение warm-start должно идти против student той же архитектуры и с
независимой оценкой, а не против training loss.
