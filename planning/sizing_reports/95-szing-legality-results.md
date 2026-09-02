# #95 — Sizing Legality, Boundary All-in, Preflop no-Buckets, Q-bootstrap per-Player — РЕЗУЛЬТАТЫ

**Дата:** 2026-06-18
**Статус:** RESULT — A-core принят как baseline; A4 отклонён; C неубедителен; relapse-механизм уточнён
**База сравнения:** #92 (multi-agent, без relapse до iter 300), #93 (sizing CFR advantage leg)
**Техническая реализация:** #94

## Мотивация
Три проблемы из анализа #93:
1. Анкеры не гейтятся по легальности (0.10 на префлопе, где мин-рейз ≈ 2.25×pot) → рассинхрон selected/effective.
2. Нет оллин-кнопки: при стеке между анкерами nearest-snap ломает атрибуцию.
3. Q-bootstrap в `train_q_network` использует `advantage_nets[0]` для всех игроков → ломает per-player стратегию (латентно под `q_reward_propagation: mc`, т.к. bootstrap-член ×0).

## Реализация (см. #94 для деталей кода)
- **A1** `_anchor_availability(state)` — доступные анкеры по стеку/мин-рейзу; гард `can_raise`
- **A2** зануление+ренорм по доступным во всех точках сэмплирования sizing
- **A3** `eff_idx_cfr` и `credit_idx` = `sizing_anchor_idx` при `sizing_allin_boundary_enabled`
- **B** префлоп без бакетов (плоский regret-matching по доступным)
- **A4** `_encode_state_for_sizing` (+30 availability-фич на вход sizing-сетей)
- **C** `QValueBuffer._next_player_ids`; per-player bootstrap в `train_q_network`
- **R0-патч** размерность sizing-сетей условная от `sizing_availability_on_input`

**ВНИМАНИЕ:** первый прогон (старый `test93seed1`) шёл с БЕЗУСЛОВНЫМ +30 cold-start → артефактная поломка (raise 15.5%). Помечен INVALID, как datapoint не используется.

## Матрица прогонов (seed1, eval = 1000 игр; raise% / win% / reward, iter 100→400)

| Run | Конфиг | raise | win | reward | Итог |
|-----|--------|-------|-----|--------|------|
| R1 | все #94 OFF (baseline) | 35→35→49→34 | 37→34→36→24 | 0→8→11→12 | коллапса НЕТ |
| R2 | A-core+B (avail+boundary+preflop; input OFF) | **38→52→52→42** | 44→39→37→26 | 29→10→20→14 | **PASS — лучший** |
| R3 | +A4 (availability_on_input ON, +30 cold-start) | 59→52→43→40 | 28→16→13→15 | 6→**−4→−0.4→−3** | **FAIL — минус** |
| R4a | C control: per-player Q OFF, q_reward_propagation=none | 26→44→40→33 | 41→40→35→30 | 10→22→16→18 | здоров |
| R4b | C: per-player Q ON, q_reward_propagation=none | 38→44→34→37 | 43→40→23→24 | 13→13→5→11 | ≈R4a |

qdiag (TD): fit_ratio R4a 0.202 (σ .064), R4b 0.228 (σ .078); qpred_boot 0.0092 vs 0.0128.

#### next_strategy_raise_mass_by_action (целевая метрика C)

| | after-raise | after-call | after-check | after-fold |
|---|---|---|---|---|
| R4a (control) iter100 | 0.244 | 0.305 | 0.960 | 0.113 |
| R4a (control) iter400 | **0.000** | **0.000** | **0.000** | **0.000** |
| R4b (fix) iter100 | 0.418 | 0.279 | 1.000 | 0.217 |
| R4b (fix) iter400 | **0.104** | **0.074** | **0.045** | **0.032** |

R4a коллапсит в 0.000 везде к iter400 — `advantage_nets[0]` для всех игроков обнуляет raise-массу. R4b держит after-raise mass 0.104 — per-player bootstrap не обнуляется.

## Выводы по прогонам

### R1 (baseline) — PASS
С починенным кодом baseline НЕ коллапсит. Подтверждает: обвал старого test93seed1 = cold-start, не дизайн.

### R2 (A-core) — PASS, принять как новый baseline
- raise растёт 38→52, без relapse в eval
- reward@100=29 (vs 0 у R1)
- sizing разнообразный — top-2 = 56% (vs исторических 79%), unique 14–15
- Минор: concentration 56% > порога 55% (warning, не fail)
- Легальность+boundary+префлоп работают без cold-start

**Флаги:** `sizing_anchor_availability_enabled: true`, `sizing_allin_boundary_enabled: true`, `sizing_preflop_disable_buckets: true`, `sizing_availability_on_input: false`

### R3 (A4) — FAIL, отклонить
Observation-на-входе + cold-start → гипер-агрессия в минус (reward −4…−3, win 13–15%).

### R4a/R4b (C) — PASS на целевой метрике, низкорычажный для relapse
На win/reward разницы нет, но **на целевой метрике (`next_strategy_raise_mass`) R4b однозначно лучше**:
- R4a к iter400 коллапсит в 0.000 по всем действиям — `advantage_nets[0]` обнуляет raise-массу для всех игроков
- R4b держит after-raise mass 0.104 — каждый актор тянет из своей сети, бутстрап-нога живая
- **Принять как корректность-фикс:** `q_bootstrap_per_player_enabled: true` (активно только под TD, `q_reward_propagation: none`; под `mc` инертно — бутстрап-член ×0)
- Низкорычажный для relapse: `_first_to_break` = advantage raise shift (0.91) ≫ Q (0.03). C чинит Q-бутстрап-канал, но relapse идёт через advantage-ногу

## КРИТИЧЕСКОЕ уточнение: relapse_diff vs eval
`relapse_diff_report.json` считает на фиксированных 500 состояниях (seed=42) **текущую (greedy) advantage-политику**, а eval — **среднюю (`strategy_net`)**. Они расходятся:

| Run | greedy raise 100→400 | advantage raise (100→200→300→400) |
|-----|----------------------|-----------------------------------|
| R1 | 0.6→0.7% | −0.02→0.18→0.16→**−2.75** |
| R2 | 59→**2.8%** | 0.06→0.97→−0.81→**−4.65** |
| R3 | 99.7→41.5% | ≈0.01 (не учится) |
| R4a | 21→**0%** | 0.002→0.79→−0.50→**−1.12** |
| R4b | 70.5→**1.0%** | 0.03→0.005→0.58→**−3.00** |

**H3-флип (накопление negative-regret за raise) присутствует ВО ВСЕХ прогонах**, включая R2 и per-player-Q R4b. `_first_to_break` (R2): #1 advantage raise shift (0.91) ≫ #2 raise_freq (0.32) ≫ #3 Q raise (0.03). H1 (buffer fold-terminal) — supported:false. Q — минорный сигнал.

**Следствия:**
1. Health R2 в eval держится на **усреднении** (`strategy_net`); regret-нога под ним всё равно флипается
2. C (per-player Q) relapse не лечит — R4b флипается даже сильнее по advantage. НО на целевой метрике (`next_strategy_raise_mass`) C PASS: держит after-raise mass 0.104 vs R4a 0.000. Ось C корректна, но низкорычажна для advantage-relapse
3. A4 cold-start обнуляет обучение regret'ов (advantage ≈0.01) → шумовая политика

## Теоретическая рамка
- CFR гарантирует сходимость **усреднённой** политики, а не текущей. Осцилляция текущей (regret-matching) — норма, не несходимость. R2 average (eval) стабилен → это и есть сигнал сходимости
- `relapse_diff` меряет текущую политику → её «relapse» **может быть нормальной осцилляцией**, а не поломкой
- Если exploration-floor понадобится — он **легитимен**: доказательства MCCFR требуют epsilon-exploration (в табличном CFR полный обход даёт это бесплатно; сэмплирование теряет → floor восстанавливает). Уже работает для sizing (`min_prob`)
- Но против **рандома** фолд может быть **корректным** best-response → тогда floor искажал бы решение. Вероятный корень — **оппонент**, а не алгоритм

## Решения
- **Принять A-core (R2)** как baseline (флаги выше)
- **Отклонить A4** (`sizing_availability_on_input: false`)
- **Принять C** как корректность-фикс (`q_bootstrap_per_player_enabled: true` на TD). Низкорычажный для relapse, но на целевой метрике PASS
- ** UPD: К D1–D3 добавить D4 — разложить next_strategy_raise_mass (и raise-regret) по глубине/улице и по «после raise vs после call», чтобы прямо подтвердить: коллапс концентрируется в глубоких/редких узлах. Если да — фикс целится точечно в сэмплирование глубоких raise-линий (targeted exploration), а не глобальный floor.

Добавляем D4 к плану диагностики (D1–D4) и в таком виде запускаем, или сначала хочешь, чтобы я по уже имеющимся отчётам вытащил разбивку raise-mass по улицам (preflop/flop/turn/river), раз данные, возможно, уже есть?

## Следующие шаги (диагностика-первой, без хаков)
1. **D1** — добавить в `relapse_diff` raise_freq СРЕДНЕЙ политики (`strategy_net`). Если среднее стабильно — проблемы сходимости нет
2. **D2** — измерить реализованный EV(raise) vs EV(fold) в схлопнутых спотах против рандома. Если raise −EV → виноват оппонент
3. **D3** — self-play/league БЕЗ floor; смотрим, гасится ли осцилляция текущей политики
4. **D4** UPD: разложить next_strategy_raise_mass (и raise-regret) по глубине/улице и по «после raise vs после call»
5. Floor (epsilon-exploration) — ТОЛЬКО если D1 покажет деградацию среднего И D2 покажет raise +EV при голоде по сэмплам
6. Подтвердить R2 на seed2/seed3
