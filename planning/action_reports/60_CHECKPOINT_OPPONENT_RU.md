# Баг-репорт #60: checkpoint_test — поддержка чекпоинта-оппонента

## Статус: ИСПРАВЛЕНО

## Проблема

Скрипт `checkpoint_test/check_ckpt.py` умел тестировать чекпоинт только против `RandomAgent`. 
Не было возможности:
- Протестировать чекпоинт против другого чекпоинта (например, iter_100 vs iter_600)
- Протестировать чекпоинт в self-play (против самого себя)

## Реализация

### Файл: `checkpoint_test/check_ckpt.py`

**1. Новый CLI-аргумент `--opponent-checkpoint`**
- Опциональный путь к `.pt` чекпоинту оппонента
- Если не указан — сохраняется старое поведение (RandomAgent на 5 слотах)
- Если указан — чекпоинт-оппонент занимает все 5 слотов (`i != MODEL_PLAYER_ID`)
- Self-play достигается указанием того же пути: `--opponent-checkpoint <тот же .pt>`

**2. `run_eval_games()` — новый параметр `opponent_checkpoint_path`**
- При указании: загружается `InferenceAgent` (strategy_net, без advantage_net)
- Создаётся словарь `opponent_agents = {i: opponent_agent for i != model_player_id}`
- В игровом цикле: для оппонента создаётся `PokersGameState(state, player_id=cp)`, вызывается `.choose_action(gs, player_id=cp, deterministic=deterministic)`, результат конвертируется через `action_to_pokers()`
- Для `RandomAgent` (fallback) — сохраняется вызов `choose_action(state)` напрямую

**3. `_build_mode_label()` — параметр `opponent_label`**
- `"strategy_net (sampling) vs Random"` — без оппонента
- `"strategy_net (sampling) vs multi_checkpoint_iter_600_light.pt"` — с оппонентом

**4. `print_report()` и `save_json_report()`**
- Добавлено поле `Opponent` в консольный отчёт
- Добавлено поле `"opponent"` в JSON-отчёт

### Примеры использования

```bash
# Старое поведение (RandomAgent)
python checkpoint_test/check_ckpt.py models/multi/multi_checkpoint_iter_600_light.pt

# Против другого чекпоинта
python checkpoint_test/check_ckpt.py models/multi/multi_checkpoint_iter_600_light.pt \
  --opponent-checkpoint models/multi/multi_checkpoint_iter_100_light.pt

# Self-play (чекпоинт против себя)
python checkpoint_test/check_ckpt.py models/multi/multi_checkpoint_iter_600_light.pt \
  --opponent-checkpoint models/multi/multi_checkpoint_iter_600_light.pt

# С regret-matching для основного чекпоинта
python checkpoint_test/check_ckpt.py models/multi/multi_checkpoint_iter_600.pt \
  --regret-matching \
  --opponent-checkpoint models/multi/multi_checkpoint_iter_100_light.pt
```

### Граничные случаи
- Оппонент всегда через `InferenceAgent` (strategy_net), даже если основной чекпоинт в режиме `--regret-matching`
- Если файл оппонента не найден — `sys.exit(1)` с сообщением об ошибке
- `compute_verdict()` вычисляется по действиям `model_player_id` (основного чекпоинта), без изменений
