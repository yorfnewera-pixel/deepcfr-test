# Баг-репорт: TypeError при запуске GUI — PyO3 enum'ы несовместимы с dict/атрибутами

## Серьёзность
**Высокая** — GUI полностью неработоспособен

## Окружение
- ОС: Windows 11
- Python: 3.11 (venv)
- PyQt5: установлен через pip
- pokers: Rust-модуль через PyO3

## Описание

При запуске графического интерфейса:

```bash
python -m scripts.poker_gui --models_folder models/mixed7
```

Приложение падает с одним из двух исключений (в зависимости от того, какое срабатывает первым):

```
TypeError: unhashable type: 'builtins.Stage'
AttributeError: module 'pokers' has no attribute 'CardRank'
```

## Два связанных бага

### Баг 1: Stage — unhashable type (строки 361-368)

Метод `PokerTable.update_stage()` использует Rust-enum `pkrs.Stage` как ключи dict:

```python
stage_names = {
    pkrs.Stage.Preflop: "Preflop",   # TypeError!
    pkrs.Stage.Flop: "Flop",
    ...
}
```

Rust-enum `Stage` через PyO3 **не реализует** `__hash__`, поэтому не может быть ключом dict.

### Баг 2: CardRank/CardSuit — AttributeError (строки 53-78)

Метод `CardWidget.update_display()` обращается к `pkrs.CardRank` и `pkrs.CardSuit`:

```python
rank_map = {
    pkrs.CardRank.R2: "2",   # AttributeError!
    ...
}
suit_map = {
    pkrs.CardSuit.Clubs: ("♣", "black"),  # AttributeError!
    ...
}
```

Rust-enum'ы `CardRank` и `CardSuit` **не экспортируются** из модуля `pokers`. Они существуют как `builtins.CardRank` и `builtins.CardSuit` (доступны через `type(card.rank)`), но не доступны через `pkrs.CardRank`.

## Корневая причина

Оба бага — следствие одного архитектурного решения: GUI использует Rust-enum'ы из PyO3 напрямую, предполагая что они ведут себя как Python-enum'ы. Но PyO3:

1. **Не реализует `__hash__`** для enum'ов → нельзя использовать как ключи dict
2. **Не экспортирует все enum'типы** через `pkrs.*` → `dir(pkrs)` содержит только: `Action, ActionEnum, ActionRecord, Card, PlayerState, Stage, State, StateStatus, parallel_apply_action, pokers, visualize_state, visualize_trace`. `CardRank` и `CardSuit` отсутствуют.

CLI-версия (`scripts/play.py`) **не имеет этих багов** — она использует int-ключи и `int()`-преобразование:

```python
# play.py — правильный подход
suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
ranks = {0: "2", 1: "3", ..., 12: "A"}
rank_text = ranks[int(card.rank)]
suit_text = suits[int(card.suit)]
```

## Исправление

### Баг 1 — poker_gui.py:361-368

```python
# Было:
stage_names = {pkrs.Stage.Preflop: "Preflop", ...}
self.stage_label.setText(f"Stage: {stage_names.get(stage, str(stage))}")

# Стало:
stage_names = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River", 4: "Showdown"}
self.stage_label.setText(f"Stage: {stage_names.get(int(stage), str(stage))}")
```

### Баг 2 — poker_gui.py:53-78

```python
# Было:
rank_map = {pkrs.CardRank.R2: "2", ...}
suit_map = {pkrs.CardSuit.Clubs: ("♣", "black"), ...}
rank_text = rank_map[self.card.rank]
suit_text, color = suit_map[self.card.suit]

# Стало:
rank_map = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
            7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
suit_map = {0: ("♣", "black"), 1: ("♦", "red"), 2: ("♥", "red"), 3: ("♠", "black")}
rank_text = rank_map[int(self.card.rank)]
suit_text, color = suit_map[int(self.card.suit)]
```

Примечание: `ActionEnum` **экспортируется** из `pkrs` и работает корректно — использования `pkrs.ActionEnum.Fold` в `in`-проверках и `==`-сравнениях не требуют изменений.

## Дополнительные замечания

- Рекомендуется добавить регрессионный тест, проверяющий что GUI запускается без исключений
- PyO3 по умолчанию не генерирует `__hash__` для enum'ов — это может измениться в будущих версиях, но полагаться на это не стоит
- Правило: при работе с PyO3 enum'ами из `pokers` — всегда приводить к `int()` перед использованием как dict-ключ

## Статус
**Исправлено** — оба бага залатаны в `scripts/poker_gui.py`
