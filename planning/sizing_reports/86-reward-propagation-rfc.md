# #86 — MC Reward Propagation RFC (дизайн ДО кода)

> **Статус:** RFC — дизайн зафиксирован, код ожидает реализации.
> **Связанные:** #84b (relapse mechanism), #85 (balanced loss FAIL), #84 (consolidation baseline)
> **База:** #84 (max_norm=100, balanced=false)

---

## 0. Мотивировка

### Проблема (smoking gun из #84b)

У ВСЕХ raise-сэмплов в q_buffer: `reward = 0.0`, `is_terminal = False`.

Подтверждённый механизм relapse:
1. **H3 (первопричина):** advantage/regret-сеть делает flip fold↔raise (adv_fold -7.93 → +0.53 за 100 итераций)
2. **Smoking gun:** raise не видит outcome руки → advantage-сеть физически не может удержать raise > fold
3. **Усиление через bootstrap:** next_raise_mass падает 0.58 → 0.006 → bootstrap V для raise обнуляется → цель для raise сэмплов = 0 → петля

Все конфиг-гипотезы исчерпаны (#82 max_norm, #83/#85 balanced loss). Дальше только код.

### Решение

**MC-return propagation:** вместо `reward = 0.0` для нетерминальных шагов — писать `v_sampled` (сэмплированный return остатка руки), уже вычисленный в момент `q_buffer.add`. Одновременно `is_terminal = True` отключает bootstrap-член в target — убирает второй канал усиления из #84b.

---

## 1. Механизм: sampled-return propagation

### Почему "sampled-return", а не строгий MC

`v_sampled` — не чистый MC (на hero_full узлах ниже по дереву это EV по стратегии, в hero_os есть importance-коррекция), но содержит реальные терминальные реворды внутри. Этого достаточно: raise-сэмплы получат знакопеременный сигнал об исходе руки.

### Точки вставки (проверено по коду deep_cfr.py, коммит 35)

Обе точки — в `cfr_traverse_multi`:

**Точка 1 — hero_os путь (~строка 2237):**
```python
# Текущий код:
if new_state.final_state:
    reward = new_state.players_state[traversing_player].reward
    is_terminal = True
else:
    reward = 0.0        # <-- smoking gun
    is_terminal = False

self.q_buffer.add(state_encoded, action_idx, reward,
                  next_state_encoded, next_policy_state_encoded,
                  next_mask, is_terminal, next_is_hero)
```

**Точка 2 — opponent путь (~строка 2513):** идентичный паттерн.

В обеих точках `v_sampled` уже вычислен (результат `cfr_traverse_multi(new_state, ...)` — строка выше `q_buffer.add`).

### Изменение (под флагом `q_reward_propagation: mc`)

```python
if self.config.q_reward_propagation == 'mc':
    reward = v_sampled       # sampled-return остатка руки
    is_terminal = True       # отключает bootstrap в target
else:
    # текущее поведение (default = 'none')
    if new_state.final_state:
        reward = ...
        is_terminal = True
    else:
        reward = 0.0
        is_terminal = False
```

### Почему на этапе ЗАПИСИ, а не чтения

Буфер — плоские numpy-массивы, circular. Связи между сэмплами одной руки не хранятся. Восстановить траекторию при чтении нельзя. Единственный вариант — записать reward сразу при `add`.

### Влияние на диагностики (отразить в репорте #86)

| Метрика | До | После | Причина |
|---------|----|-------|---------|
| `q_buffer terminal%` | ~0% для raise | **~100%** | `is_terminal=True` |
| `qdiag term_frac` | низкий | **вырастет** | больше terminal-сэмплов |
| `q_buffer reward mean` (raise) | 0.0 | **≠ 0, знакопеременное** | v_sampled содержит исход руки |
| `bootstrap V` (raise) | падает к 0 | **стабилен** | bootstrap отключён |
| `action_q_model next_raise_mass` | падает | **не влияет** | bootstrap не используется |

Это ожидаемые изменения — М1-проверка (`reward mean ≠ 0`) должна это подтвердить, а не дать ложную тревогу.

---

## 2. Флаги конфигурации

```yaml
# --- #86: Reward propagation for raise samples ---
q_reward_propagation: none    # none | mc | nstep (default none = текущее поведение)

# --- #87 (будущее): Нормализация целей ---
q_target_norm: none           # none | pot_relative | popart (default none)
```

Все изменения под флагами, дефолт = текущее поведение (обратная совместимость).

---

## 3. Нормализация целей (#87, будущее)

### Обоснование

#85 показал дрейф масштаба Q до -4.5 и взрывы градиентов до ~9500. Проблема в масштабе целей — чистый Huber не решает (только режет выбросы, масштаб не стабилизирует).

### Вариант A (рекомендуемый): pot-relative

`reward /= max(state.pot, 1.0)` на этапе записи.

- Доменно осмысленна: reward в долях пота
- Детерминирована: не зависит от бегущей статистики
- Прецедент в коде: sizing-Q УЖЕ нормирует цели по поту (`_compute_sizing_q_target_values` использует `state.pot` — строки 1909/2194/2372)
- `state.pot` доступен в обеих точках `q_buffer.add`

### Вариант B (fallback): Pop-Art

Адаптивная нормализация (running mean/std целей). Сложнее в реализации, требует хранения статистики. Использовать только если pot-relative недостаточно.

### Что НЕ брать

Чистый Huber loss — #85 показал что он не решает дрейф масштаба.

---

## 4. Pre-registered критерии для #86 (MC-propagation, одна переменная vs #84)

### Два уровня: механизм → поведение

**M1 (механизм, проверять на iter 100 ДО оценки поведения):**
- Из `full_report.json` buffer-секции: у raise `reward mean ≠ 0` и распределение знакопеременное (не константа)
- Если М1 не выполнен — реализация багована, **стоп, чинить**, не смотреть на поведение

**M2 (механизм, после iter 300):**
- `relapse_diff 100↔200↔300`: flip отсутствует
  - `adv_fold` НЕ переходит из отрицательного в положительный
  - `|delta adv_raise| < 1.0` между соседними чекпоинтами

**B (поведение, каждый чекпоинт 100/200/300):**
- `verdict = OK`, `raise_freq ≥ 40%`, `unique_anchors = 15/15`
- `top-2 sizing mass ≤ 55%`, `анкер 0.10 ≤ 15% массы`

**PASS:** M1 + M2 + B на всех чекпоинтах включая iter 300
**PARTIAL:** M1 + M2 ок, но B проседает (шум MC) → шаг 3 (#87 нормализация)
**FAIL:** M1 ок, но flip и relapse как в #84 → механизм глубже, вернуться к RFC с данными

---

## 5. План реализации

| Шаг | Что | Файлы | Статус |
|-----|-----|-------|--------|
| 0 | Гигиена: balanced=false, заполнить #85, обновить SIZING | config.yaml, *.md | **DONE** (коммит 35) |
| 1 | RFC #86 (этот документ) | sizing_reports/86-reward-propagation-rfc.md | **DONE** |
| 2 | Реализация MC-propagation в deep_cfr.py | src/core/deep_cfr.py | **TODO** |
| 3 | Прогон test86seed1 до iter 300 | train.py | **TODO** |
| 4 | full report + relapse_diff на 100/200/300, проверка M1/M2/B | checkpoint_tools.py, relapse_diff.py | **TODO** |
| 5 | Заполнить sizing_reports/86-mc-propagation.md результатами | *.md | **TODO** |
| 6 | Если PASS → консолидация seed2 + iter 400; если PARTIAL → #87 нормализация | — | **TODO** |

---

## 6. Чего НЕ делать

- Не трогать balanced loss ни в каком виде — закрыт навсегда
- Не менять больше одной переменной за эксперимент (propagation и нормализация — РАЗНЫЕ шаги)
- Не объявлять baseline без durability (iter 300+) и seed2
- Annealing/entropy-регуляризацию advantage-сети держать в запасе — не смешивать с #86/#87
