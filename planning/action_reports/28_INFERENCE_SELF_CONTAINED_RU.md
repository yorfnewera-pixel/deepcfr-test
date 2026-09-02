# Баг-репорт #28 (обновление): inference — DictGameState + choose_action_from_dict + CLI

**Дата обновления:** 2026-05-10  
**Статус:** Реализовано и протестировано  
**Изменённые файлы:** `inference/core.py`, `inference/adapters/pokers.py`, `inference/__init__.py`  
**Новые классы:** DictCard, _DictAction, _DictActionRecord, DictFromAction, DictPlayerState, DictGameState

---

## Что добавлено (обновление от оригинального #28)

### 1. Классы-обёртки (Вариант А — Adapter/Wrapper)

Duck typing: обёртки имитируют атрибуты оригинальных pokers-объектов, но считывают данные из Python dict. `encode_state()` работает без изменений — даже не знает что данные из dict, а не из pokers.

#### DictCard

```python
class DictCard:
    __slots__ = ('suit', 'rank')
    def __init__(self, d):
        # принимает {"suit": 2, "rank": 12} или pokers Card
        self.suit = int(d.get('suit', 0))  # или int(d.suit)
        self.rank = int(d.get('rank', 0))
```

Дефолты: suit=0 (Clubs), rank=0 (2). Позволяет `int(card.suit)` и `int(card.rank)` работать.

#### _DictAction, _DictActionRecord, DictFromAction

Цепочка для `state.from_action.action.action` / `state.from_action.action.amount`:

```
pokers:    from_action → ActionRecord → Action → ActionEnum(int) / amount(float)
обёртка:   from_action → DictFromAction → _DictActionRecord → _DictAction → int / float
```

`_DictAction.__int__()` — позволяет `int(state.from_action.action.action)` работать.
`_DictActionRecord.amount` — проксирование, позволяет `from_action.action.amount` работать напрямую.

```python
# from_action = [2, 8.0] → Raise на 8.0
# from_action = None → начало улицы, encode_state проверяет is not None
# from_action отсутствует в dict → None (дефолт)
```

#### DictPlayerState

```python
class DictPlayerState:
    __slots__ = ('hand', 'active', 'bet_chips', 'pot_chips', 'stake')
    # Дефолты: hand=[], active=True, bet_chips=0.0, pot_chips=0.0, stake=0.0
```

#### DictGameState

```python
class DictGameState:
    __slots__ = ('bb', 'pot', 'min_bet', 'button', 'current_player', 'stage',
                 'legal_actions', 'public_cards', 'from_action',
                 'players_state', 'last_raise_increment')
    # Дефолты: bb=2.0, pot=0.0, min_bet=0.0, button=0, current_player=0,
    #          stage=0, legal_actions=[1], public_cards=[], from_action=None,
    #          last_raise_increment=0.0, players=[]
```

### 2. choose_action_from_dict()

```python
def choose_action_from_dict(self, data_dict: dict,
                            player_id: int = 0,
                            deterministic: bool = False) -> dict:
    gs = DictGameState(data_dict)
    action_type, multiplier = self.choose_action(gs, player_id, deterministic)
    return {"action_type": action_type, "multiplier": multiplier}
```

Принимает сырой dict → обёртка → choose_action → результат dict.

### 3. CLI

```bash
py inference/core.py model_light.pt --input state.json --player 0 --deterministic
```

Вывод: `{"action_type": 2, "multiplier": 1.5}`

### 4. PokersGameState/PokersPlayerState → adapters/pokers.py

Вынесены из core.py. core.py теперь **полностью без pokers зависимостей**.

---

## Формат ввода (dict/JSON)

```json
{
    "bb": 2.0,
    "pot": 30.0,
    "min_bet": 8.0,
    "button": 2,
    "current_player": 3,
    "stage": 1,
    "legal_actions": [1, 3],
    "public_cards": [{"suit": 1, "rank": 10}, {"suit": 0, "rank": 9}],
    "from_action": [2, 8.0],
    "last_raise_increment": 6.0,
    "players": [
        {"hand": [{"suit": 2, "rank": 12}, {"suit": 3, "rank": 11}], "active": true, "bet_chips": 0, "pot_chips": 5, "stake": 195},
        {"active": true, "bet_chips": 0, "pot_chips": 5, "stake": 195}
    ]
}
```

**legal_actions:** `[0=Fold, 1=Check, 2=Call, 3=Raise]` (pokers ActionEnum int значения)
**public_cards / hand:** `[{"suit": 0-3, "rank": 0-12}]` (0=Clubs..3=Spades, 0=2..12=Ace)
**from_action:** `[action_type, amount]` или `null` / отсутствует
**players:** hand только для hero (player_id), остальные — без hand

---

## Ключевые решения (согласовано с Gemini)

| Решение | Обоснование |
|---------|-------------|
| Вариант А (обёртки) вместо B (hasattr) или C (два encode_state) | Gemini: «encode_state остаётся девственно чистым», «Вариант B замедлит traversals», «Вариант C — забудешь обновить один» |
| `__slots__` во всех обёртках | Gemini: «критично если прогонять миллионы состояний» |
| `.get()` с дефолтами | Gemini: «в JSON может чего-то не хватать, враппер должен отдавать 0.0 вместо KeyError» |
| `from_action = None` (не заглушка) | encode_state уже проверяет `is not None` — самый простой и надёжный путь |
| `_DictAction.__int__()` | Позволяет `int(state.from_action.action.action)` работать |
| `_DictActionRecord.amount` проксирование | Позволяет `from_action.action.amount` работать (как в pokers Action) |

---

## Тестирование

```
✅ choose_action_from_dict с полным dict + from_action=[2,8.0]
✅ choose_action_from_dict с from_action=None
✅ choose_action_from_dict с минимальным dict (дефолты)
✅ CLI: py inference/core.py model.pt --input state.json --player 0
✅ pokers адаптер: wrap_state + choose_action — не сломался
✅ test_field_verification.py — все 6 тестов проходят
```

---

## Структура файлов (финальная)

```
inference/
├── __init__.py          # Экспорт: InferenceAgent, DictGameState, encode_state, ...
├── core.py              # Ядро: обёртки + encode_state + PokerNetwork + InferenceAgent + CLI
└── adapters/
    ├── __init__.py
    └── pokers.py        # pokers: PokersGameState, PokersPlayerState, action_to_pokers, play_hand
```

**core.py зависимости:** torch, numpy, json, argparse — **без pokers**

---

## Standalone использование

```bash
# Скопировать 2 файла + веса:
inference/core.py
inference/__init__.py  (опционально)
model_light.pt

# Программно:
from core import InferenceAgent
agent = InferenceAgent("model_light.pt")
result = agent.choose_action_from_dict({...}, player_id=0)

# CLI:
py core.py model_light.pt --input state.json --player 0 --deterministic
```

## Обновление (2026-05-10): Revert 164 → 157

`INPUT_SIZE` в `inference/core.py` обновлён с 164 на 157. Убраны `eff_spr_enc` из per-player фич (5→4 на игрока) и `stack_depth`. `FEATURE_SPEC` и `PER_PLAYER_SPEC` обновлены. Старые чекпоинты (164 фичи) несовместимы — требуется переобучение.
