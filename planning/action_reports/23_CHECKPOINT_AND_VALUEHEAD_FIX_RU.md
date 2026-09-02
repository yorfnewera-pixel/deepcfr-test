# Баг-репорт #24: Итоговый фикс — налаживание чекпоинтов + аудит Value Head

## Статус: ИСПРАВЛЕНО

## Обзор

Два связанных фикса, выполненных совместно:
1. **Checkpoint fix** (#23) — сохранение optimizer states + унификация всех точек save/load
2. **Value Head audit** (#22) — подтверждение что value_head не мёртвый код, а активный стабилизатор sizing-пути

## Часть 1: Checkpoint Fix

### Проблема
При resume обучения с чекпоинта терялись:
- 4 optimizer state_dicts (Adam momentum/variance)
- pg_baseline_mean/var, target_mean/var
- min/max_bet_size (в зависимости от точки сохранения)

8 точек `torch.save()` в проекте использовали **разные наборы ключей** — чекпоинт из одной функции не мог корректно загрузиться через другую.

### Реализация

**`_build_checkpoint()`** — единый метод формирования чекпоинта:
- Все 4 optimizer state_dicts
- Все скаляры обучения (pg_baseline, target_mean/var, min/max_bet_size)
- config, seed, git_hash

**`_load_checkpoint()`** — единый метод загрузки:
- `try/except` для каждого optimizer — fallback на холодный старт
- Обратная совместимость со старыми чекпоинтами (без optimizer states)
- `weights_only=False` для PyTorch 2.6+

**`save_model()`** — переписан через `_build_checkpoint()` + лёгкий файл:
- Полный чекпоинт (~7 МБ) — для resume
- Лёгкий файл `_light.pt` (~1.2 МБ) — только strategy_net, для игры/инференса

**`load_model()`** — переписан через `_load_checkpoint()`

**train.py** — 6 точек `torch.save()` → через `_build_checkpoint()`, 3 точки `torch.load()` → через `_load_checkpoint()`

**Лёгкий файл** теперь сохраняется **на каждой контрольной точке** (каждые 100 итераций) — не только через `save_model()`, но и во всех 5 точках `torch.save()` в train.py. Формат: `checkpoint_iter_100_light.pt` — только `{iteration, strategy_net}`

### Дополнительные фиксы
- `weights_only=False` — во всех `torch.load()` (visualize_tournament.py, opponent_modeling)
- `strict=False` — во всех `load_state_dict()` (opponent_modeling, diagnose_checkpoint, visualize_tournament)
- Хрупкий парсинг `split('iter_')` → `agent.iteration_count` (train_mixed_with_opponent_modeling.py)

### Верификация
- Roundtrip save/load с optimizer states — подтверждён
- Обратная совместимость со старыми чекпоинтами — подтверждена (INFO о холодном старте)
- pokers-тесты — проходят

## Часть 2: Value Head Audit

### Результат
Value Head **не мёртвый код**. Активно используется в `train_sizing_network()`:
- `advantages = regret_tensors - values.squeeze(1).detach()` — baseline для PG
- `value_loss = F.mse_loss(values.squeeze(1), regret_tensors.detach())` — обучение критика
- Отдельный оптимизатор `value_optimizer` с `value_lr=1e-3`

Двухуровневый advantage:
1. CFR: `regret_raise = action_values[2] - ev` — преимущество Raise над стратегией
2. Value Head: `V(s)` предсказывает ожидаемое преимущество Raise
3. PG Sizing: `advantages = regret - values.detach()` — насколько конкретный сайзинг лучше среднего рейза

**В advantage path (train_advantage_network)** Value Head не используется — это корректно, CFR считает regrets аналитически.

## Решение по обучению

Iter_4700 содержит корректные веса (обучен на исправленном коде), но решено начать **обучение с нуля** для чистоты эксперимента. Все новые чекпоинты будут содержать optimizer states — resume будет бесшовным.

## Примечание: инцидент с Serena

При вставке `_build_checkpoint`/`_load_checkpoint` через `insert_before_symbol` Serena ошибочно поместила код внутрь метода `choose_action`, разорвав его на строке `if action_type == 2:`. Тело восстановлено вручную, верифицировано тестами. Причина: баг Serena `insert_before_symbol` — некорректное определение позиции символа.
