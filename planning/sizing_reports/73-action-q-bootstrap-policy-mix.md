# Bug #73: action-Q bootstrap policy mix для raise suppression

## Статус: IMPLEMENTED (требуется validation run до iter100/200)

## Контекст

#72 diagnostic run до iter900 показал, что #71 low-raise collapse уже не sizing/z-score проблема.

Ключевые метрики iter900:
- `raise_freq = 23.1%`
- `win_rate = 17.3`
- `raise_lt_call_pct = 99.36%`
- `raise_lt_fold_pct = 24.76%`
- `q_action.call.mean = -0.033`
- `q_action.raise.mean = -1.867`
- `q_buffer.raise.count = 374185`
- `q_buffer.raise.terminal_pct = 0.0%`
- `q_buffer.raise.reward = 0.0`
- `action_q_model.raise.bootstrap_value.mean = -6.465`
- `next_strategy_raise_mass_by_action.raise.mean = 0.117`

## Вывод

Raise не starving: samples много. Raise target — полностью bootstrap-only: reward=0, terminal=0%.

Bootstrap использует on-policy `next_strategy` из `advantage_net`, где raise mass подавлен — self-suppressing loop.

Механика deadly-triad:
1. `next_strategy` даёт ~0.117 массы на raise.
2. `next_v = Σ next_strategy * q_target` негативно для raise-transitions.
3. `q_net[raise]` занижен.
4. OS control variate: `value_hat[raise] = Q[raise]` для несэмплированного действия.
5. `cf_regret[raise]` становится отрицательным.
6. `advantage_net` ещё сильнее подавляет raise.
7. Bootstrap raise target падает дальше.

## Изменение #73

В `train_q_network` TD-target backup policy:

```python
uniform_strategy = next_masks_t / next_masks_t.sum(dim=1, keepdim=True).clamp(min=1)
backup_strategy = (1.0 - mix) * next_strategy + mix * uniform_strategy
next_v = (backup_strategy * next_q).sum(dim=1)
```

Mix применяется **только** в TD-target `train_q_network`, не в игре, не в inference, не в sizing.

## Флаги

Рабочий config:
```yaml
q_bootstrap_policy_mix_enabled: true
q_bootstrap_policy_uniform_mix: 0.10
```

Кодовые defaults:
```yaml
q_bootstrap_policy_mix_enabled: false
q_bootstrap_policy_uniform_mix: 0.10
```

## Файлы

- `config.yaml` — 2 новых ключа
- `src/utils/config.py` — defaults
- `src/core/deep_cfr.py` — init, `_apply_q_bootstrap_policy_mix`, `train_q_network`, checkpoint save/restore
- `tests/test_hybrid_outcome_sampling.py` — 4 теста

## Что НЕ трогалось

- sizing
- z-score flags
- src2 anneal
- inference / `choose_action`
- reward sign/scale/normalization
- q_buffer schema
- terminal handling
- target-net sync interval

## PASS критерии

Прогон from-scratch до iter100/200:

- `raise_lt_call_pct` заметно ниже 99%
- `q_action.raise.mean` ближе к call
- `raise_freq` не падает ниже #72 и желательно >25%
- `next_strategy_raise_mass_by_action.raise.mean` растёт выше 0.117
- `best_sizing_anchor_dist` не возвращается в 100% collapse
- `win_rate` не деградирует относительно #72

## Если FAIL

- mix 0.10 слишком слабый → пробовать 0.15
- если mix не помогает → #74 player-aware backup или альтернативный backup operator

## Валидация

1. Конфиг уже готов: `q_bootstrap_policy_mix_enabled: true`, `mix: 0.10`.
2. Запустить from-scratch до iter100.
3. Сгенерировать `full_report`.
4. Проверить PASS критерии.
5. Если OK на iter100 — продолжить до iter200.

## Дата: 2026-06-07
