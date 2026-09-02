═══════════════════════════════════════════════════════════════════
  BUG #53-v3 — Иерархический bucket head для bet sizing
  Дата: 02.06.2026 (обновлено)
  Статус: Реализовано (включая bucket loss + иерархический infer)
  Файлы: src/core/model.py, src/core/deep_cfr.py, inference/core.py,
         config.yaml, checkpoint_test/check_ckpt.py
  Связан: Bug #53 (per-sample MSE), Bug #53-v2 (гиперпараметры)
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

Sizing-сети получили bucket head: выбор размера рейза стал
иерархическим — сначала класс размера (small/medium/large),
потом конкретный анкер внутри класса.

Было (плоский, 15 анкеров):
  state → base → anchor_head(15) → regret_matching → sample

Стало (иерархический):
  state → base → bucket_head(3)            → bucket_probs
               → anchor_head(15)           → anchor_probs
  bucket_probs × anchor_probs = full_15_probs → sample

Главный эффект: min_prob применяется к 5 анкерам внутри bucket,
а не ко всем 15. При min_prob=0.05: 5×5%=25% floor вместо 75%.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — Модели: bucket_head в sizing-сетях
═══════════════════════════════════════════════════════════════════

  src/core/model.py:
    StrategySizingNet: + bucket_head = Linear(hidden, 3)
      forward → (bucket_logits, slot_logits, scalar_bet)
    SizingAnchorNet:  + bucket_head = Linear(hidden, 3)
      forward → (bucket_logits, anchor_logits)

  inference/core.py:
    StrategySizingNet: идентичные изменения
      forward → (bucket_logits, slot_logits, scalar_bet)

  Совместимость: старые чекпоинты загружаются с strict=False.
  bucket_head инициализируется normal(0, 0.01), bias=0.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — Иерархический sizing: _hierarchical_sizing()
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:1327-1390

  Новый метод _hierarchical_sizing(state_tensor, iteration):

  1. advantage_sizing_net → (bucket_logits[3], anchor_logits[15])

  2. Q-путь (если sizing_q готов):
     bucket_q[i] = max(q_vals[indices_i])
     bucket_probs = heat_weights(bucket_q) + bucket_floor
     local_probs[i] = heat_weights(q_vals[indices_i])
     full_probs[indices_i] = bucket_probs[i] * local_probs[i]

  3. Fallback (Q не готов):
     bucket_probs = regret_matching_anchors(bucket_logits, bucket_floor)
     local_probs = regret_matching_anchors(logits[indices], anchor_min_prob)
     full_probs = hierarchy combination

  4. return (sampled_bet_size, full_15_probs)

  Вызывается в:
    - cfr_traverse_multi (full traversal, строка ~1640)
    - _cfr_traverse_multi_outcome_node (OS path, строка ~1442)

  _apply_sizing_bucket_floor больше не нужен — иерархия
  естественно разделяет bucket и anchor exploration.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — backward-совместимость при инференсе
═══════════════════════════════════════════════════════════════════

  choose_action (deep_cfr.py + inference/core.py):
  bucket_logits пока игнорируются — sizing идёт через плоский
  slot_logits + _sparsify_sizing_target_by_buckets (top_bucket mode).
  Поведение при игре не изменилось.

  opponent sizing: распаковывается 3 значения, slot_logits
  используются как раньше.

  training: advantage_sizing_net и strategy_sizing_net
  распаковываются в 2-3 значения, bucket_logits зарезервированы
  для будущего bucket-level loss.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — Конфиг + чекпоинты
═══════════════════════════════════════════════════════════════════

  config.yaml:
    + sizing_num_buckets: 3

  Чекпоинты: bucket_head веса сохраняются автоматически через
  state_dict(). strict=False при загрузке обеспечивает
  совместимость со старыми чекпоинтами.

  check_ckpt.py: load_state_dict(sizing_sd, strict=False).


═══════════════════════════════════════════════════════════════════
  ЭФФЕКТ НА SIZING MIN_PROB
═══════════════════════════════════════════════════════════════════

  До (плоский):
    min_prob=0.05 × 15 анкеров = 75% uniform floor

  После (иерархический):
    min_prob=0.05 × 5 анкеров внутри bucket = 25% floor

  При текущем значении 0.015:
    5 × 1.5% = 7.5% floor (было 22.5% на 15)


═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — Bucket-level loss для обучения sizing-сетей
═══════════════════════════════════════════════════════════════════

  train_strategy_sizing_anchor_network (строка 2028-2049):
    target_bucket[b] = sum(target_weights[indices_b])  # из таргета
    bucket_kl = KL(log_softmax(bucket_logits), target_bucket)
    loss = (kl_anchor + bucket_kl) * iter_weights

  train_sizing_anchor_network (строка 1931-1951):
    bucket_target[b] = max(target_adv[indices_b])       # max Q-преимущества
    loss = MSE(slot_logits, target_adv) + MSE(bucket_logits, bucket_target)


═══════════════════════════════════════════════════════════════════
  ПУНКТ 6 — Иерархический infer в inference/core.py
═══════════════════════════════════════════════════════════════════

  choose_action (строка 472-486):
    bucket_probs = softmax(bucket_logits)
    for each bucket b:
        local = anchor_probs[indices_b] / sum(anchor_probs[indices_b])
        full_probs[indices_b] = bucket_probs[b] * local
    bet = sample(full_probs) или expected_size (deterministic)

  InferenceAgent._load: добавлено self.sizing_bucket_indices
  с fallback на дефолтные 3 группы для старых чекпоинтов.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 7 — Фикс check_ckpt.py
═══════════════════════════════════════════════════════════════════

  _choose_action_regret_matching:
    slot_logits, _ → _, slot_logits, _  (3 значения вместо 2)


═══════════════════════════════════════════════════════════════════
  ЧТО НЕ СДЕЛАНО (низкий приоритет)
═══════════════════════════════════════════════════════════════════

  1. Метрики bucket distribution в TensorBoard
     (bucket_probs/small/medium/large, entropy, chosen bucket)


═══════════════════════════════════════════════════════════════════
  СВЯЗАННЫЕ БАГИ
═══════════════════════════════════════════════════════════════════

  Bug #53    — per-sample MSE + вес (t/T)^γ
  Bug #53-v2 — гиперпараметры: буферы, batch_size, epochs, epsilon
  Bug #52    — утилита проверки чекпоинтов
═══════════════════════════════════════════════════════════════════
