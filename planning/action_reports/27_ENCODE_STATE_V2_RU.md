# Баг-репорт #27: encode_state v2 — BB-нормализация, stack_depth, eff_spr

**Дата:** 2026-05-10  
**Статус:** Исправлено → Частично откачено (требует переобучения с нуля)  
**Затронутые файлы:** `src/core/model.py`, `src/core/deep_cfr.py`, `src/opponent_modeling/deep_cfr_with_opponent_modeling.py`, `tests/test_field_verification.py`

---

## Суть проблемы

`encode_state` нормализовал все монетарные фичи на `initial_stake` (= buy_in = 200 фишек). Это работало корректно благодаря инварианту `stake + bet_chips + pot_chips = buy_in`, но привязывало сеть к конкретному лимиту. При переносе на другой стек (например, 150bb или 50bb) сеть не могла корректно работать — все нормализованные значения были бы искажены.

Кроме того, отсутствовали важные фичи:
- **stack_depth** — глубина стека относительно стандартного 100bb (позволяет сети понимать «насколько глубокая игра»)
- **eff_spr per opponent** — эффективный SPR относительно каждого оппонента (критично в 3-бет+ банках, где стеки расходятся)

---

## Внесённые исправления

### M4.1: BB-нормализация (делитель `100 * state.bb` вместо `initial_stake`)

**Было:** Все монетарные фичи делились на `initial_stake` (= 200 при bb=2, stake=200)  
**Стало:** Все монетарные фичи делятся на `norm_unit = 100 * state.bb` (= 200 при bb=2)

При текущих параметрах (100bb игра): `100 * bb = 200 = initial_stake` — **математически идентично**.  
При другом лимите (например NL50, bb=0.5): `100 * 0.5 = 50` — сеть видит правильные нормализованные значения.

Fallback: если `state.bb` недоступно или ≤ 0, используется инвариант `stake + bet_chips + pot_chips`.

```python
# Было:
initial_stake = (state.players_state[0].stake + state.players_state[0].bet_chips + state.players_state[0].pot_chips)
pot_enc = [state.pot / initial_stake]

# Стало:
norm_unit = float(state.bb) * 100.0 if hasattr(state, 'bb') and float(state.bb) > 0 else float(...)
pot_enc = [state.pot / norm_unit]
```

**Затронутые места (6 замен делителя):** pot_enc, bet_enc, pot_chips_enc, stake_enc, min_bet_enc, prev_action.amount

### M4.2: stack_depth — 158-я фича

```python
initial_stake = (state.players_state[0].stake + state.players_state[0].bet_chips + state.players_state[0].pot_chips)
stack_depth = [initial_stake / norm_unit]
encoded.append(stack_depth)
```

При 100bb: `stack_depth = 200 / 200 = 1.0` (мёртвый нейрон сейчас, но готов к другим стекам)  
При 150bb: `stack_depth = 300 / 200 = 1.5` (сеть видит глубокую игру)  
При 50bb: `stack_depth = 100 / 200 = 0.5` (сеть видит короткую игру)

### M4.3: eff_spr per opponent — фичи 159-164

В цикл per-player добавлена 5-я фича (всего 5 вместо 4):

```python
for p in range(num_players):
    player_state = state.players_state[p]
    active_enc = [1.0 if player_state.active else 0.0]
    bet_enc = [player_state.bet_chips / norm_unit]
    pot_chips_enc = [player_state.pot_chips / norm_unit]
    stake_enc = [player_state.stake / norm_unit]
    if p == player_id:
        eff_spr_enc = [0.0]  # vs себя — нет смысла
    else:
        opp_stake_raw = float(player_state.stake)
        eff_stack = min(player_stake_raw, opp_stake_raw)
        denom = eff_stack + pot_raw
        eff_spr_enc = [eff_stack / denom if denom > 0 else 0.0]
    encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc, stake_enc, eff_spr_enc]))
```

Формула: `eff_spr = min(my_stake, opp_stake) / (min(my_stake, opp_stake) + pot)` ∈ [0,1]

Согласована с Gemini — сигмоидальная компрессия, нет epsilon, нет clip. Согласуется с существующей `spr_feat`.

**Когда eff_spr ≠ spr:** В 3-бет+ банках стеки расходятся. Пример: P1=174bb, P2=140bb → spr=174/(174+pot), но eff_spr vs P2=140/(140+pot). Разница существенна при принятии решений.

### M4.4: Обновление input_size

**deep_cfr.py:**
```python
# Было (157):
input_size = 52 + 52 + 5 + 1 + self.num_players + self.num_players + self.num_players * 4 + 1 + 1 + 4 + 5

# Стало (164):
input_size = 52 + 52 + 5 + 1 + self.num_players + self.num_players + self.num_players * 5 + 1 + 1 + 1 + 4 + 5
```

**deep_cfr_with_opponent_modeling.py:** та же формула обновлена (156 → 164).

---

## Раскладка фичей (итого 164)

| Индекс | Фича | Размер | Описание |
|--------|------|--------|----------|
| 0-51 | hand_enc | 52 | Карманные карты (канонические масти) |
| 52-103 | community_enc | 52 | Общие карты |
| 104-108 | stage_enc | 5 | One-hot стадия |
| 109 | pot_enc | 1 | pot / (100*bb) |
| 110-115 | button_enc | 6 | One-hot баттон |
| 116-121 | current_player_enc | 6 | One-hot текущий игрок |
| 122-151 | per-player × 6 | 30 | [active, bet, pot_chips, stake, eff_spr] × 6 |
| 152 | min_bet_enc | 1 | min_bet / (100*bb) |
| 153 | spr_feat | 1 | stake / (stake + pot) — собственный SPR |
| 154 | stack_depth | 1 | buy_in / (100*bb) — NEW |
| 155-158 | legal_actions_enc | 4 | One-hot легальные действия |
| 159-163 | prev_action_enc | 5 | [4×one-hot] + amount / (100*bb) |

**Проверка:** 52+52+5+1+6+6+30+1+1+1+4+5 = **164** ✓

---

## Консенсус с Gemini

Обсуждено с Gemini Pro 3.1:
- **BB-нормализация:** Gemini ЗА → «Привязка к Big Blind делает encode_state математическим стандартом»
- **"Чистота градиентов" (аргумент Gemini):** Оспорено — при 100bb `initial_stake = 100*bb = 200`, оба делителя константы. Gemini согласился: «при жестко фиксированном 100bb обучении разницы в числах нет. Это чистый рефакторинг на будущее»
- **eff_spr:** Формула `min(stake) / (min(stake) + pot)` ∈ [0,1] согласована

---

## Тесты

Все 6 тестов `test_field_verification.py` пройдены:
1. ✅ Инвариант stake + bet_chips + pot_chips = 200
2. ✅ BB-нормализация: pot / 200.0 = корректно
3. ✅ Raise increment семантика
4. ✅ action_type_to_pokers_action
5. ✅ **157 фич, spr корректен, eff_spr/stack_depth отсутствуют** (тест обновлён после revert)
6. ✅ GUI конверсия TOTAL → increment

---

## Важные замечания

1. **Требует переобучения с нуля** — чекпоинт iter_4100 (164 фичи) несовместим с новой архитектурой (157 фичей)
2. При 100bb обучении: BB-нормализация математически идентична предыдущей (делитель = 200 в обоих случаях)
3. `stack_depth` откачен — всегда константа при фиксированном стеке (мёртвый нейрон). Полезен только при варьирующихся стеках, что CFR не поддерживает.
4. `eff_spr per opponent` откачен — дублирует spr_feat в ~70% рук, 1536 лишних параметров замедляют сходимость. Полезен в 3-бет+ банках (~30%), но не стоит затрат. Может быть добавлен позже при необходимости.
5. `spr_feat` оставлен — единственный SPR-фич, не избыточный, ∈[0,1]
6. `initial_stake` убран из encode_state — больше не нужен

## Revert (2026-05-10): 164 → 157

Причина: profit упал с ~6 до ~4.5 на плато после добавления 7 фичей (eff_spr: 6×256=1536 лишних параметров + stack_depth: 256). eff_spr дублирует spr_feat в большинстве рук. stack_depth — мёртвый нейрон при любом фиксированном стеке.

**Убрано:**
- `eff_spr_enc` из per-player цикла (5→4 фичи на игрока, -6 фичей)
- `stack_depth` блок и вычисление `initial_stake` (-1 фича)
- Итого: 164 → 157

**Оставлено:**
- BB-нормализация (M4.1) — правильная абстракция, нулевая стоимость при 100bb
- spr_feat — единственный SPR-фич, полезный и не избыточный
