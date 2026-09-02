# #94 — Sizing Legality, Boundary All-in, Preflop no-Buckets, Q-bootstrap per-Player

**Дата:** 2026-06-17 (исправлено 2026-06-17)
**Статус:** IMPLEMENTED + ERRATA FIXED (ждёт прогона A-core без cold-start)
**Связанные:** #93 (sizing CFR advantage leg)

## Мотивация

Три независимые проблемы, обнаруженные при анализе #93:

1. **Анкеры не гейтятся по легальности:** сеть может выбрать 0.10 на префлопе (где мин-рейз = 2.25×pot), создавая рассинхрон между выбранным анкером и реальной ставкой.
2. **Оллин-кнопка отсутствует:** когда стек между анкерами, нет способа выбрать точный all-in — используется nearest-snap, что ломает атрибуцию регрета.
3. **Q-bootstrap использует advantage_nets[0] для всех игроков:** ломает per-player стратегию в multi-agent режиме.

## Решение (3 шага)

### Шаг 1 — A-core + B (без cold-start)

**A1:** `_anchor_availability(state)` — вычисляет доступные анкеры по стеку/мин-рейзу, с учётом `can_raise` (Raise в legal_actions).

**A2:** Во всех точках сэмплирования sizing'а (OS-raise, full-raise, `_hierarchical_sizing`, `choose_action`) — после вычисления probs зануляются недоступные анкеры и ренормируется.

**A3:** `eff_idx_cfr` и `credit_idx` заменены на `sizing_anchor_idx` безусловно при включённом `sizing_allin_boundary_enabled`. Старый `argmin(|anchors - eff_mult|)` оставлен только в диагностике (`eff_idx_diag`). Это чинит рассинхрон IS-веса и атрибуцию регрета.

**B:** Префлоп без бакетов — в `_hierarchical_sizing` при `sizing_preflop_disable_buckets` и `stage==0` используется плоский regret-matching по всем доступным анкерам.

**Новые флаги:** `sizing_anchor_availability_enabled`, `sizing_allin_boundary_enabled`, `sizing_preflop_disable_buckets`.

### Шаг 2 — A4 (cold-start sizing)

**A4:** `_encode_state_for_sizing(state, player_id)` — расширяет base encoding на +2×num_anchors (=30) с векторами доступности. Все sizing-сети (advantage_sizing_net, strategy_sizing_net, sizing_q_net) инициализируются с `sizing_input_size = input_size + 30`. Все sizing-буферы (sizing_q_buffer, sizing_advantage_buffer, sizing_strategy_buffer) тоже.

**Флаг:** `sizing_availability_on_input`.

**Cold-start:** Старые чекпоинты (format_version < 4) несовместимы по размерности — sizing-сети переинициализируются с нуля.

### Шаг 3 — Q-bootstrap per-player

**C:** В `QValueBuffer` добавлено поле `_next_player_ids`. При записи в `q_buffer.add()` сохраняется `int(new_state.current_player)`. В `train_q_network` и `_train_q_network_refit` — если `q_bootstrap_per_player_enabled` и `use_multi_agent`, bootstrap-стратегия вычисляется через `advantage_nets[p]` для каждой группы сэмплов.

**Флаг:** `q_bootstrap_per_player_enabled`.

## Изменённые файлы

- `src/core/deep_cfr.py` — основные изменения
- `src/training/train.py` — `q_buffer.add()` с `next_player_id`
- `tools/checkpoint_tools.py` — поддержка `q_buffer_next_player_ids`, инференс `sizing_input_size` из чекпоинта, `_ensure_sizing_dim` для нуль-паддинга в diagnostic-функциях

## New diagnostics

- `sizing_availability_diag`: `calls`, `avail_mean_sum`, `boundary_present`, `boundary_allin_idx_sum`, `boundary_allin_idx_sqsum`, `allin_selected`, `allin_total_selections`, `preflop_selections`, `postflop_selections`, `eff_idx_mismatches`, `eff_idx_total`
- `sizing_input_size` (30 extra dims для sizing-сетей)
- `q_buffer._next_player_ids` (per-player в буфере)

## Гейты прогона (pre-registered)

**Шаг 1 (A-core+B), seed1, до iter 300/400:**
- PASS: raise_freq не коллапсит 100→200
- PASS: top-2 sizing concentration < 79%
- PASS: unique_anchors ≥ 8
- PASS: 0.10-концентрация на префлопе падает (0.10 там недоступен)
- FAIL: raise < 20% или fold > raise

**Шаг 2 (A4), seed1, iter 300+:**
- Cold-start sizing ожидаем
- PASS: ассерт eff_idx == sizing_anchor_idx (mismatch = 0)

**Шаг 3 (C), с q_reward_propagation=none:**
- PASS: next_strategy_raise_mass стабильнее vs baseline (не net[0] для всех)

## Порядок
1. Прогнать A-core+B против #93 baseline (без cold-start)
2. Прогнать A4 поверх A-core (cold-start sizing, iter 300+)
3. Прогнать C отдельно (не смешивать atribution/Q-эффекты)
4. Откат: каждый блок за своим флагом

## Errata — исправление 2026-06-17

**Косяк:** `sizing_input_size = input_size + 2*num_anchors` был безусловным — sizing-сети ВСЕГДА получали +30 фич, даже при `sizing_availability_on_input: false`. План требовал: A-core (шаг 1) без смены размерности → A4 (шаг 2) отдельно с cold-start.

**Последствия:** test93seed1 (все флаги `false`) получил холодный старт sizing-сетей без включённых фич. Iter 100 выглядел здоровым из-за равномерного распределения свежеинициализированных сетей — не из-за фикса. Relapse 100→200 ускорен vs #92 baseline.

**Исправлено:**
- `sizing_input_size` условный: `input_size + 2*num_anchors` только при `sizing_availability_on_input: true`, иначе `input_size`
- `_encode_state_for_sizing` при выключенном флаге возвращает base-энкод без паддинга
- `_load_checkpoint` проверяет реальную размерность из чекпоинта, не только `format_version`

**Валидационные прогоны (после исправления):**
- A-core чистый: `sizing_anchor_availability_enabled: true`, `sizing_allin_boundary_enabled: true`, `sizing_preflop_disable_buckets: true` — БЕЗ cold-start
- A4: `sizing_availability_on_input: true` — отдельно, с cold-start
- C: `q_bootstrap_per_player_enabled: true`, `q_reward_propagation: none` — изолировать Q-эффект
