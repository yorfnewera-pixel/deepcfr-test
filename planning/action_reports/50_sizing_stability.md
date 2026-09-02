# Баг-репорт #50: Излишняя вариативность сайзингов в light checkpoint

**Дата:** 01.06.2026
**Приоритет:** High
**Статус:** Fixed

---

## Проблема

`sizing_q_net` оценивает Q для каждого sizing anchor. Из Q строится `slot_weights_np`. Эти веса напрямую пишутся в `sizing_strategy_buffer`. `strategy_sizing_net` учится дистиллировать эти targets через KL. Из-за noisy Q (особенно на ранних итерациях) и reservoir buffer итоговая light policy даёт слишком размазанное распределение по 15 сайзингам.

---

## Решение

Добавлены два механизма стабилизации:

### 1. EMA target smoothing по state bucket / abstraction

- Перед записью в `sizing_strategy_buffer` target проходит через EMA-сглаживание
- Ключ абстракции: `(stage, current_player, button, pot_bb_bucket, eff_stack_bb_bucket, spr_bucket, has_raise)`
- Warmup beta: линейный рост от 0.0 до 0.9 за 5000 итераций
- LRU eviction при превышении `max_size = 200000`

### 2. Sparse target: top-2 bucket / top-1 sizing внутри bucket

- После EMA-smoothing target спарсифицируется
- Top-2 sizing bucket groups по суммарной массе
- Внутри каждой группы — только best sizing anchor
- При нулевой массе — fallback на исходное распределение (логируется)

### 3. Inference mode для `choose_action`

- `mean` — взвешенное среднее по anchors (только при `deterministic=True`; при `deterministic=False` — categorical sample из всех 15 anchors)
- `argmax` — всегда выбор anchor с максимальной вероятностью
- `top_bucket` — sparse top-2 bucket / top-1 sizing внутри bucket:
  - при `deterministic=False` — частотный sampling между оставшимися представителями
  - при `deterministic=True` — argmax из sparse

**Для production / random_eval используется `top_bucket`** — частотный sampling по очищенному распределению, а не по всем 15 anchors.

### Новые config-ключи

```yaml
sizing_target_ema_enabled: true
sizing_target_ema_beta: 0.9
sizing_target_ema_min_mass: 1e-8
sizing_target_ema_warmup_steps: 5000
sizing_target_ema_beta_start: 0.0
sizing_target_ema_beta_end: 0.9
sizing_target_ema_max_size: 200000
sizing_target_sparsify_enabled: true
sizing_target_top_buckets: 2
sizing_target_top_per_bucket: 1
sizing_inference_mode: 'top_bucket'
```

---

## Изменённые файлы

| Файл | Что |
|------|-----|
| `src/utils/config.py` | +11 config-ключей |
| `src/core/deep_cfr.py` | +5 методов, +18 атрибутов, модификация 2 call sites, модификация `choose_action`, checkpoint save/load |
| `config.yaml` | `sizing_inference_mode: "top_bucket"` |

---

## Важные ограничения

- EMA/sparsify применяются **только** к target для `strategy_sizing_net`
- Q-net продолжает видеть все сайзинги без ограничений
- `sizing_inference_mode: top_bucket` — осознанный trade-off exploitability vs. practicality: частотный sampling между 2-3 anchors вместо размазанного распределения по 15
