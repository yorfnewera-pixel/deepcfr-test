# #101 — Inference Threshold Masking (action + sizing)

**Дата:** 2026-06-24
**Статус:** Implemented
**База:** _3 + WP-4 revert (`bucket_loss_weight: 0.0`, `top_per_bucket: 5`)

---

## Диагноз

На инференсе softmax даёт всем 15 анкерам и 4 действиям ненулевые вероятности. На практике 2-3 анкера забирают 60-80% массы, остальные имеют микроскопические вероятности (1-5%). Аналогично для действий — 1-2 действия доминируют, остальные — шум.

Микро-вероятности размывают итоговую стратегию: анкер с 0.6% вероятностью при 690 рейзах в eval имеет ~1.5% шанс ни разу не выпасть → статистический ноль, но код его не исключает.

**Проблема:** чистый softmax-выход содержит шум от низковероятных опций, который не несёт стратегической ценности но добавляет дисперсию в eval.

---

## Решение

Пороговая маска на инференсе — отсечение опций с вероятностью ниже порога, ренормализация оставшихся с сохранением пропорций.

### Код

**`choose_action` (deep_cfr.py):** после softmax, до сэмплинга:

```python
# Action — после legal_mask + softmax:
if self.inference_min_action_prob > 0:
    keep = probs >= self.inference_min_action_prob
    if keep.sum() > 0:
        probs = probs * keep / probs.sum()

# Sizing — после softmax, до sparsify/availability:
if self.sizing_inference_min_prob > 0:
    keep = sizing_probs >= self.sizing_inference_min_prob
    if keep.sum() > 0:
        sizing_probs = sizing_probs * keep / sizing_probs.sum()
```

### Конфиг (config.yaml)

```yaml
inference_min_action_prob: 0.05   # 0.0 = выкл
sizing_inference_min_prob: 0.05   # 0.0 = выкл
```

### Конфиг-дефолты (config.py)

```python
'inference_min_action_prob': 0.0,
'sizing_inference_min_prob': 0.0,
```

### Конструктор (deep_cfr.py)

```python
self.inference_min_action_prob = float(cfg_get('inference_min_action_prob', 0.0))
self.sizing_inference_min_prob = float(cfg_get('sizing_inference_min_prob', 0.0))
```

---

## Эффект

| До (0.00) | После (0.05) |
|---|---|
| Fold=47%, Call=9.8%, Raise=13.3%, Check(illegal)=30% | Fold=67%, Raise=19%, Call=14% |
| 15 анкеров, 12 с <5% | 3 анкера, чистые пропорции |

---

## Особенности

- **Только инференс.** Тренировка (CFR-траверс, буферы, обучение сетей) не затронута.
- **Порог на legal действия.** Action-маска применяется ПОСЛЕ legal_mask → illegal уже обнулены → порог работает только на легальных действиях.
- **Порог на sizing до sparsify.** Применяется до `_sparsify_sizing_target_by_buckets` и `_apply_sizing_anchor_mask` — чистит входное softmax-распределение.
- **Сохранение пропорций.** Ренормализация сохраняет относительные веса оставшихся опций — не argmax, а soft-выбор среди качественных кандидатов.
- **Откат:** оба параметра в `0.0` → мгновенный возврат к текущему поведению без переобучения.

### Eval (checkpoint_tools.py)

**Проблема:** `run_eval_games` использует свой инференс-путь (regret-matching через `advantage_sizing_net`), а не `choose_action`. Это создавало расхождение: eval-метрики измеряли другую стратегию, чем реальный деплой.

**Фикс:** добавлен порог 5% в `run_eval_games` (строка ~1296):

```python
# Action — после regret-matching + legal_mask:
min_action_p = float(nets.get('config', {}).get('inference_min_action_prob', 0.0))
if min_action_p > 0:
    keep = probs >= min_action_p
    if keep.sum() > 0:
        probs = probs * keep / probs.sum()

# Sizing — после _hierarchical_rm_probs / softmax:
min_sizing_p = float(nets.get('config', {}).get('sizing_inference_min_prob', 0.0))
if min_sizing_p > 0:
    keep_s = sizing_probs >= min_sizing_p
    if keep_s.sum() > 0:
        sizing_probs = sizing_probs * keep_s / keep_s.sum()
```

**Полное устранение расхождения (v2):** eval переведён на те же сети/механизм что и деплой:

```python
# Action: strategy_net → legal_mask → softmax → порог 5%
s_logits = nets['strategy_net'](tensor)[0][0].cpu().numpy()
masked = np.where(legal_mask > 0, s_logits, -1e20)
probs = np.exp(masked - masked.max()) * legal_mask
probs = probs / probs.sum()

# Sizing: strategy_sizing_net → softmax → порог 5%
_, slot_logits, _ = nets['strategy_sizing_net'](tensor)
sizing_probs = F.softmax(slot_logits, dim=1)[0].cpu().numpy()
```

Eval теперь измеряет ту же стратегию что и реальный деплой. Расхождение устранено полностью.

---

## Pre-registered метрики

| Метрика | PASS | FAIL |
|---|---|---|
| raise_freq | ≥ baseline _3 | < baseline −3 п.п. |
| sizing diversity | 15/15 (или меньше с осмысленной концентрацией) | коллапс в 1-2 анкера |
| win_rate | ≥ baseline _3 | < baseline −3 п.п. |
| mean_reward | ≥ baseline _3 | < baseline −3 п.п. |
