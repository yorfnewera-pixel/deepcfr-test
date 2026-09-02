# Баг-репорт: Raise amount — абсолютная сумма вместо инкремента над min_bet

**Проект**: deepcfr-test  
**Дата**: 2026-05-10  
**Серьёзность**: Critical  
**Статус**: Fixed  

---

## Симптомы

В GUI при рейзе: игрок вводит «9 фишек», но pokers ставит 9 фишек **поверх колла**, а не 9 фишек всего. Например: текущая ставка 7, игрок хочет поставить 9 — в итоге ставится 7 + 2 (колл) + 9 (инкремент) = 18 фишек вместо 9.

Аналогичная проблема в CLI (`play.py`) — кнопки half-pot и pot, а также ручной ввод передают абсолютную сумму вместо инкремента.

## Корневая причина

Патченный `pokers` (`game_logic.rs:193`) интерпретирует `Action(Raise, amount)` как **инкремент сверх min_bet**:

```rust
let mut bet = (self.min_bet - self.players_state[player].bet_chips) + action.amount;
```

Но GUI и CLI передают **абсолютную сумму** рейза, а не инкремент. Несоответствие семантики `amount`.

## Затронутые места (7 исправлений)

| Файл | Строка | Описание |
|------|--------|----------|
| `poker_gui.py` | 873 | `human_action()` — абсолютная сумма → инкремент |
| `poker_gui.py` | 903 | `handle_half_pot()` — pot*0.5 как абсолютная → инкремент |
| `poker_gui.py` | 909 | `handle_pot()` — pot как абсолютная → инкремент |
| `poker_gui.py` | 998-1038 | `RandomAgent` — raise_options как абсолютные → инкременты |
| `play.py` | 133-141 | half-pot как абсолютная → инкремент |
| `play.py` | 144-152 | full-pot как абсолютная → инкремент |
| `play.py` | 154-164 | ручной ввод абсолютной суммы → инкремент |

## Фикс

Во всех местах: перед `pkrs.Action(Raise, amount)` конвертировать:

```python
call_amount = max(0, state.min_bet - player_state.bet_chips)
increment = desired_total - call_amount  # абсолютная → инкремент
```

Также добавлен расчёт `min_raise_increment` с учётом `state.last_raise_increment` (новое поле из наших патчей).

GUI `human_action()` теперь:
- Принимает абсолютную сумму от спиннера
- Конвертирует в инкремент: `increment = amount - call_amount`
- Логирует: "You raise $2.0 (total bet: $9.0)"

## Верификация

- Оба файла проходят `py_compile` — синтаксис OK
- Семантика `Action(Raise, increment)` теперь совпадает с pokers

## Не затронутые места (корректные)

- `deep_cfr.py:285` — `action_type_to_pokers_action()` уже передаёт `additional` (инкремент)
- `train_mixed_with_opponent_modeling.py:41` — пробрасывает `original_action.amount` (инкремент)
