═══════════════════════════════════════════════════════════════════
  BUG #55 — Bucket-level loss: вес + раздельное логирование + hierarchical inference
  Дата: 03.06.2026
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, src/training/train.py, config.yaml
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

Bug #53-v3 реализовал bucket-head (3 класса: small/medium/large) поверх
anchor-head (15 размеров) в обеих sizing-сетях. Однако:

1. Bucket-loss добавлен с весом 1:1 (равный с anchor-loss), без
   возможности регулировать вклад.
2. Логируется только суммарный loss без разбивки на bucket/anchor компоненты.
3. Inference (choose_action) не использует bucket_logits — они выкидываются
   через `_`. Режим `top_bucket` использует только anchor-уровень
   (`_sparsify_sizing_target_by_buckets`), а bucket_head в inference не участвует.

Эти три проблемы мешают:
  - диагностике — непонятно, учится ли bucket-head отдельно от anchor;
  - fine-tuning — bucket-target агрегированный и более грубый, 1:1 может
    перетянуть обучение в coarse-классы, ухудшив fine sizing;
  - inference — потенциал иерархической архитектуры не используется.

Решение: добавить конфигурируемый вес, раздельное логирование и
новый режим hierarchical inference.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — config.yaml: sizing_bucket_loss_weight
  Файл: config.yaml
═══════════════════════════════════════════════════════════════════

Добавлена строка после sizing_bucket_min_prob_decay_iterations (строка 58):

  sizing_bucket_loss_weight: 0.3

Значение 0.3 выбрано потому что:
  - bucket-head — вспомогательная иерархическая регуляризация;
  - основной сигнал должен идти по 15 anchors;
  - bucket-target агрегирован из anchor-target и шумнее;
  - 1:1 рискует перетянуть обучение в small/medium/large.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — deep_cfr.py __init__: загрузка параметра
  Файл: src/core/deep_cfr.py:590
═══════════════════════════════════════════════════════════════════

Добавлено после self.sizing_bucket_groups (строка 590):

  self.sizing_bucket_loss_weight = float(cfg_get('sizing_bucket_loss_weight', 0.3))

═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — train_strategy_sizing_anchor_network: вес + internal fields
  Файл: src/core/deep_cfr.py:2059 (строка loss = ((kl_per_sample + bucket_kl)...)
═══════════════════════════════════════════════════════════════════

  Было:
    loss = ((kl_per_sample + bucket_kl) * iter_weights).mean()

  Стало:
    anchor_kl_weighted = (kl_per_sample * iter_weights).mean()
    bucket_kl_weighted = (bucket_kl * iter_weights).mean()
    loss = anchor_kl_weighted + self.sizing_bucket_loss_weight * bucket_kl_weighted

    self.last_strategy_sizing_anchor_loss = float(anchor_kl_weighted.item())
    self.last_strategy_sizing_bucket_loss = float(bucket_kl_weighted.item())

Возврат total loss без изменений (среднее).

═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — train_sizing_anchor_network: вес + internal fields
  Файл: src/core/deep_cfr.py:1950 (строка loss = loss_anchor + loss_bucket)
═══════════════════════════════════════════════════════════════════

  Было:
    loss = loss_anchor + loss_bucket

  Стало:
    loss = loss_anchor + self.sizing_bucket_loss_weight * loss_bucket

    self.last_advantage_sizing_anchor_loss = float(loss_anchor.item())
    self.last_advantage_sizing_bucket_loss = float(loss_bucket.item())

═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — choose_action: hierarchical inference mode
  Файл: src/core/deep_cfr.py:2187-2215
═══════════════════════════════════════════════════════════════════

Добавлен новый режим sizing_inference_mode = "hierarchical".

Логика:
  1. Вычислить bucket_probs = softmax(bucket_logits)
  2. Вычислить anchor_probs = softmax(slot_logits)
  3. Для каждого bucket: local_probs = anchor_probs[indices] / sum(local)
  4. full_probs[indices] = bucket_probs[b] * local_probs
  5. Нормализовать full_probs → sampling/mean

Режим `hierarchical` полностью независим от `top_bucket` — переключение
через конфиг, без риска сломать текущую стратегию.

Исправлена структура if/elif в choose_action: hierarchical не
проваливается в общий else deterministic.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 6 — train.py: раздельное TensorBoard логирование
  Файл: src/training/train.py
═══════════════════════════════════════════════════════════════════

Во всех местах, где логируется Loss/SizingAnchor и Loss/StrategySizingAnchor,
добавлены раздельные метрики:

  Loss/SizingAdvantageAnchor   (из last_advantage_sizing_anchor_loss)
  Loss/SizingAdvantageBucket   (из last_advantage_sizing_bucket_loss)
  Loss/SizingStrategyAnchor    (из last_strategy_sizing_anchor_loss)
  Loss/SizingStrategyBucket    (из last_strategy_sizing_bucket_loss)
  Train/SizingBucketLossWeight (значение sizing_bucket_loss_weight)

Это позволяет отслеживать:
  - учится ли bucket-head независимо от anchor-head;
  - не перетягивает ли bucket-loss обучение;
  - текущее значение weight в конфиге.

═══════════════════════════════════════════════════════════════════
  ACCEPTANCE CRITERIA
═══════════════════════════════════════════════════════════════════

1. bucket_head получает градиент в обоих sizing train-методах.
2. bucket_loss не NaN/Inf.
3. Strategy sizing loss не взрывается.
4. Sizing распределение перестаёт быть полностью uniform.
5. В eval: unique_anchors > 3, raise_freq > 5%.
6. В TensorBoard видны 4 раздельные loss-метрики.
7. Режим sizing_inference_mode: "hierarchical" работает без ошибок.
