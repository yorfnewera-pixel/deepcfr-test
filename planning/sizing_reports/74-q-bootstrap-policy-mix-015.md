# Bug #74: dose-response q_bootstrap_policy_uniform_mix 0.10 → 0.15

## Статус: IMPLEMENTED (требуется validation run до iter200)

## Контекст

#73 mix=0.10 дал сильный bump на iter100:
- `verdict=OK`
- `raise_freq=47.9%`
- `win_rate=41.8`
- `raise bootstrap_value.mean=+0.175`
- `next_strategy_raise_mass.raise.mean=0.591`

Но к iter200 self-suppressing loop вернулся:
- `verdict=WARN (low_raise)`
- `raise_freq=12.6%`
- `win_rate=19.6`
- `raise bootstrap_value.mean=-0.885`
- `next_strategy_raise_mass.raise.mean=0.098`

Вывод: mix=0.10 разрывает deadly-triad на iter100, но статичный floor недостаточен против самоусиления.

## Гипотеза

Более сильный uniform floor (0.15) должен создать более устойчивое плато и замедлить/остановить self-suppressing collapse к iter200.

## Изменение

Одна правка в `config.yaml`:

```yaml
q_bootstrap_policy_uniform_mix: 0.15   # было 0.10
```

Всё остальное без изменений:

```yaml
q_bootstrap_policy_mix_enabled: true
sizing_q_train_zscore: false
sizing_q_src2_loss_weight_enabled: true
sizing_q_src2_weight_start: 1.0
sizing_q_src2_weight_end: 0.25
sizing_q_src2_weight_decay_iterations: 400
save_q_buffer_in_checkpoint: true
```

## Что НЕ трогалось

- `_apply_q_bootstrap_policy_mix` — механизм не менялся
- код defaults в `src/utils/config.py` — остались `False / 0.10`
- sizing, z-score, src2 anneal, inference, reward, q_buffer schema

## PASS критерии (главное — iter200)

Удержание на iter200:
- `raise_freq > 25%`
- `win_rate` не рушится (выше #73 iter200 = 19.6)
- `raise bootstrap_value.mean` не уходит сильно в минус (выше -0.885)
- `next_strategy_raise_mass_by_action.raise.mean` держится выше `0.098`
- `best_sizing_anchor_dist` не коллапсирует в 0.10/3.00
- `raise_lt_call_pct` не возвращается к высоким значениям

## Валидация

1. Перенести старые `models/multi_new` в `models/multi_old_73`.
2. Запустить from-scratch до iter100 → проверить bump.
3. Продолжить до iter200 → проверить удержание.
4. Сгенерировать `full_report`.
5. Если PASS — продолжить до iter400 для стабильности.

## Если FAIL

#75: player-aware backup (B2) — добавить `next_is_hero` в `QValueBuffer` и накладывать mix только на hero-узлы.

## Дата: 2026-06-07
