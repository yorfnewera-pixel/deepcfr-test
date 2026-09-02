# БАГ #26: Полный аудит GUI/CLI/Model — 13 исправлений

**Дата:** 2026-05-10  
**Серьёзность:** Critical → High → Medium  
**Статус:** ✅ Исправлено  
**Файлы:** `poker_gui.py`, `play.py`, `src/agents/random_agent.py`, `src/core/model.py`, `src/core/deep_cfr.py`

---

## Суть

Полный аудит найденных несоответствий между pokers (Rust), GUI, CLI и Model выявил 11 багов. 9 касаются GUI/CLI, 2 — обучающего пайплайна (требуют переобучения с нуля).

---

## Исправленные баги (GUI/CLI)

### M1.1 [CRITICAL] — GUI spin box: диапазон в increment, отображение в total

**Было:** `update_raise_limits(min_raise, max_raise)` где min/max — increment (доп. ставка сверх колла). Spin box показывает increment, но пользователь видит число без контекста.

**Стало:** Spin box показывает **TOTAL** (итоговая ставка за раунд). `update_raise_limits(min_total, max_total)` где total = current_bet + call_amount + increment. Пользователь видит итоговую сумму — как в реальном покере.

**Файл:** `poker_gui.py:762-790`

### M1.1b [CRITICAL] — GUI: конверсия TOTAL→increment не вычитает bet_chips

**Было:** `increment = max(0, amount - call_amount)` — spin box после фикса M1.1 показывает TOTAL, но конверсия не вычитает `bet_chips`. Каждый рейз завышен на текущую ставку игрока.

**Пример:** bet_chips=$4, call=$4, TOTAL=$10. Было: increment=10-4=**$6** → pokers: total=$14. Стало: increment=10-4-4=**$2** → pokers: total=$10 ✅

**Это было причиной "агрессии GUI после фикса"** — каждый рейз человека завышался на bet_chips.

**Стало:** `increment = max(0, amount - player_state.bet_chips - call_amount)`

**Файл:** `poker_gui.py:905`

### M1.2 [HIGH] — GUI: min_raise игнорирует last_raise_increment

**Было:** `min_raise = max(1.0, self.state.bb)` — учитывает только BB, но не предыдущий рейз.

**Стало:** `min_increment = max(bb, last_raise_increment)` — соблюдается правило покера: минимальный рейз = max(BB, предыдущий инкремент).

**Файл:** `poker_gui.py:768-772`

### M2.1 [HIGH] — GUI: spin box позволяет выбрать increment > remaining

**Было:** Нет проверки когда remaining_stake < min_raise. Spin box показывает недоступные суммы.

**Стало:** Если remaining_after_call < min_increment → spin box показывает только all-in. Иначе — нормальный диапазон min_total..max_total.

**Файл:** `poker_gui.py:775-780`

### MX.1 [CRITICAL] — GUI: opponent_id = own player_id для OM-агента

**Было:** `agent.choose_action(self.state, opponent_id=current_player)` — передаёт **свой** ID как opponent_id. OM-агент моделирует самого себя вместо оппонента.

**Стало:** Выбирает случайного активного оппонента: `random.choice([ps.player for ps in players_state if ps.active and ps.player != current_player])`.

**Файл:** `poker_gui.py:843-847`

### M1.4 [MEDIUM] — GUI: лог AI-рейза показывает increment как абсолют

**Было:** `"Player X raises $Y"` где Y — increment. Пользователь думает это итоговая ставка.

**Стало:** `"Player X raises +$Y (total: $Z)"` — явно помечен как increment + показан total.

**Файл:** `poker_gui.py:855`

### M1.3 [HIGH] — play.py: "Raise to X" — increment показан как абсолют

**Было:** `get_action_description()` → `"Raise to {amount:.2f}"` — amount это increment, но "to" подразумевает итоговую сумму.

**Стало:** `"Raise +{amount:.2f} (total: {total:.2f})"` — increment + total в скобках.

**Файл:** `play.py:14-28`

### M8.1 [HIGH] — play.py: RandomAgent не учитывает last_raise_increment

**Было:** `min_raise = state.bb` — только BB, без last_raise_increment.

**Стало:** `min_increment = max(bb, last_raise_increment)` — как в каноничном `src/agents/random_agent.py`.

**Файл:** `play.py:402-407` (класс удалён, импортируется из `src/agents/random_agent.py`)

### M9.1 [HIGH] — GUI: glob ловит _light.pt без advantage_net

**Было:** `glob.glob("*.pt")` — ловит и `_light.pt` файлы, которые не содержат `advantage_net` → crash при загрузке.

**Стало:** Исключает `_light.pt` из поиска; fallback на `_light.pt` только если нет полных чекпоинтов.

**Файл:** `poker_gui.py:649-653`

### M9.2 [HIGH] — GUI: OM-модель определяется по "om" в имени файла

**Было:** `is_om_model = "om" in filename.lower()` — ненадёжно, зависит от конвенции именования.

**Стало:** Инспектирует checkpoint: `is_om_model = 'history_encoder' in ckpt['advantage_net']` — точно определяет тип по структуре модели.

**Файл:** `poker_gui.py:679-684`

### M10.1 [CRITICAL] — _load_checkpoint: crash при загрузке _light.pt

**Было:** `self.advantage_net.load_state_dict(checkpoint['advantage_net'])` — KeyError для `_light.pt`, который содержит только `strategy_net`. `_light.pt` — валидный формат для инференса (advantage_net не используется при игре), но `_load_checkpoint` его отвергал. Результат: GUI fallback на RandomAgent, который может пойти оллин на префлопе.

**Стало:** `if 'advantage_net' in checkpoint: self.advantage_net.load_state_dict(...)` — advantage_net загружается только при наличии. Скаляры (min_bet_size etc) не перезаписываются на None. `_light.pt` загружается корректно через strategy_net.

**Файл:** `src/core/deep_cfr.py:924-951`

---

## Исправленные баги (обучающий пайплайн — требует переобучения)

### M3.1 [HIGH] — encode_state: нормализация на текущий stake вместо начального

**Было:** `initial_stake = state.players_state[0].stake` — берёт **текущий** остаток, который уменьшается по ходу раздачи. Фичи pot, bet_chips, stake "плывут" — одна ситуация кодируется по-разному на разных стадиях.

**Стало:** `initial_stake = stake + bet_chips + pot_chips` — инвариант даёт точный начальный стек (доказан из Rust: `stake + bet_chips + pot_chips = buy_in` на всём протяжении раздачи).

**Влияние:** Существующие чекпоинты несовместимы. Требуется переобучение с нуля. Фичи станут консистентными: pot=30 всегда 0.15 при стеке 200.

**Файл:** `src/core/model.py:148-150`

### M3.2 [HIGH] — encode_state: SPR не наблюдаем сетью

**Было:** До фикса M3.1 `stake_enc ≈ 1.0` для игрока 0 (делили stake на stake) → SPR = stake/pot был ненаблюдаем. После M3.1 stake_enc стал корректным, SPR теоретически выводим из `stake_enc / pot_enc`, но MLP плохо аппроксимирует деление — тратит ёмкость на имитацию операции через логарифмы/экспоненты.

**Стало:** Добавлена 157-я фича: `spr_feat = stake_enc / (stake_enc + pot_enc)` ∈ [0, 1]. Формула Stake/(Stake+Pot): SPR=0 → 0.0 (all-in), SPR=1 → 0.5 (ключевая точка), SPR→∞ → ~1.0. Не нужен epsilon, clip, защита от деления на 0. Естественное сжатие высоких SPR (сигмоидальный характер). input_size: 156 → 157.

**Обоснование:** Консенсус с Gemini. SPR влияет на стратегию нелинейно — критические пороги SPR<1 (committed), SPR 1-4 (пуш-фолд), SPR>10 (глубокая игра). Явная фича позволяет сети сразу строить логику на этом коэффициенте, вместо того чтобы "изобретать деление" в каждой итерации.

**Влияние:** Требует переобучения с нуля (совместно с M3.1). Ускорит сходимость — сети не нужно аппроксимировать деление.

**Файл:** `src/core/model.py:174-176`, `src/core/deep_cfr.py:170`

## DRY: Консолидация RandomAgent

**Было:** 3 копии RandomAgent в `play.py`, `poker_gui.py`, `src/agents/random_agent.py`.

**Стало:** Каноничная реализация в `src/agents/random_agent.py` (с last_raise_increment). `play.py` и `poker_gui.py` импортируют `from src.agents.random_agent import RandomAgent`.

---

## Верификация поля

Создан `tests/test_field_verification.py` — smoke test подтверждает:

| Проверка | Результат |
|----------|-----------|
| Инвариант stake + bet_chips + pot_chips = buy_in | ✅ Выполняется на всей раздаче |
| Нормализация encode_state на начальный стек | ✅ pot/200 = 0.015 совпадает |
| SPR фича: stake/(stake+pot) ∈ [0,1] | ✅ SPR≈66 → 0.985, SPR=1 → 0.5 |
| Raise increment семантика (pokers) | ✅ Action(Raise, 5.0) → bet_chips-min_bet=5.0 |
| action_type_to_pokers_action | ✅ pot × multiplier = increment |
| GUI конверсия TOTAL→increment | ✅ Старый метод завышал на bet_chips |

---

## Не исправлено (Low / не ломает)

| ID | Описание | Почему не трогаем |
|----|----------|-------------------|
| M4.1 | play.py get_human_action не показывает min raise | Косметика |
| M5.1 | GUI не показывает pot odds | Улучшение UX |
| M6.1 | play.py no --max-iterations | Улучшение CLI |

---

## Безопасность для обучения

Исправления GUI/CLI не влияют на обучение. M3.1 + M3.2 в model.py улучшают качество фичей (стабильная нормализация + явный SPR), но требуют переобучения с нуля. Конверсия данных (increment семантика) **единообразна** на всём протяжении: pokers ↔ action_type_to_pokers_action ↔ cfr_traverse ↔ choose_action — все используют increment над call.
