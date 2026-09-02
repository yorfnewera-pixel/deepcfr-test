# Runtime Search: настройка силы поиска, checkpoint 3000

Дата проверки: 2026-08-28.

## Цель

Увеличить силу runtime search до появления смены `argmax` между blueprint и search policy, не меняя обучение или checkpoint.

## Одинаковый schedule

Все прогоны использовали 48 samples на каждый seed (8 deals, 6-max, seeds 107 и 109) и checkpoint `models/test110/light_checkpoint_iter_3000.pt`.

| Particles | Eta | Seed | Fallback / search | L1 mean | Argmax changes | Action changes | BB/100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 107 | 0 / 14 | 0.0661 | 0 | 0 | 0.00 |
| 2 | 10 | 109 | 0 / 15 | 0.0300 | 0 | 0 | 0.00 |
| 8 | 10 | 107 | 12 / 15 | 0.0091 | 0 | 1 | 0.00 |
| 8 | 10 | 109 | 13 / 15 | 0.0003 | 0 | 0 | 0.00 |
| 8 | 50 | 107 | 12 / 15 | 0.0332 | 0 | 1 | 0.00 |
| 8 | 50 | 109 | 13 / 15 | 0.0003 | 0 | 0 | 0.00 |
| 2 | 50 | 107 | 0 / 15 | 0.5649 | 2 | 3 | -486.46 |
| 2 | 50 | 109 | 0 / 15 | 0.3717 | 2 | 2 | 183.85 |

## Вывод

`particles=8` в этом schedule вызывает ESS fallback в 80–87% search roots, поэтому почти не меняет policy. Возврат к `particles=2` при `eta=50` оставляет fallback нулевым и даёт смену argmax на 4 из 30 root, а сэмплированного действия — на 5 из 30 root.

Контрастные BB/100 на 48 samples не являются оценкой качества: выборка слишком мала, а всего несколько изменённых решений меняют результат. Следующий измерительный шаг — длинный paired EV прогон именно с `particles=2, eta=50`, с несколькими независимыми seeds. Он не вмешивается в обучение.

## Артефакты

- `014-runtime-policy-delta-particles-8-checkpoint-3000.json`
- `015-runtime-policy-delta-particles-8-eta-50-checkpoint-3000.json`
- `016-runtime-policy-delta-particles-2-eta-50-checkpoint-3000.json`
