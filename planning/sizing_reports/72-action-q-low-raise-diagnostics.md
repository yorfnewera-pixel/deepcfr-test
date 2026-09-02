# Bug #72: action-Q low-raise collapse diagnostics

## Статус: IMPLEMENTED (требуется diagnostic validation run до iter100)

## Контекст

#71 отключил target-only z-score (`sizing_q_train_zscore=false`) с src2 anneal как контроль.
Validation iter100 показал FAIL нового типа:
- `raise_freq=1.6%`
- `q_action.fold.mean=3.06`, `q_action.raise.mean=0.63`
- `raise_lt_fold_pct=100%`
- `best_sizing_anchor_dist`: `0.10 = 2993/2993 = 100%`

Первичная проблема теперь action-level: модель считает fold лучше raise почти всегда.
Sizing-collapse в 0.10 вторичен — raise почти не выбирается.

## Гипотезы

### H1: action-level deadly triad
raise value bootstrap-ится через `next_strategy` из advantage-сети.
Если policy уже подавляет raise, bootstrap value после raise занижается,
raise-Q падает, policy ещё реже рейзит.

### H2: starvation raise transitions
raise редко сэмплится → q_buffer получает мало raise transitions → raise-Q плохо обучен.

### H3: fold bootstrap неоднозначен
В 6-max fold не делает `final_state=True`. Fold-сэмплы получают bootstrap value
из состояния, где traversing player уже сфолдил. Масштаб/знак может раздувать fold-Q.

## Изменения #72

Diagnostic-only. Никаких правок в динамику обучения.

### 1. Конфиг-флаг

```yaml
save_q_buffer_in_checkpoint: false   # default-off
```

### 2. Сохранение action-level q_buffer в чекпоинт

При `save_q_buffer_in_checkpoint=true` в `_build_checkpoint` сохраняются:
- `q_buffer_states`
- `q_buffer_actions`
- `q_buffer_rewards`
- `q_buffer_next_states`
- `q_buffer_next_policy_states`
- `q_buffer_next_masks`
- `q_buffer_terminals`

Флаг также сохраняется/восстанавливается в `config` чекпоинта.

### 3. Диагностика в checkpoint_tools.py

`run_action_q_buffer_diagnostics(nets)` — без модели (count, terminal ratio, reward stats by action).

`run_action_q_model_diagnostics(nets)` — с моделью:
- `q_net_vs_q_target_diff_by_action`
- `bootstrap_value_stats_by_action` (next_v)
- `target_stats_by_action` (TD target)
- `next_strategy_raise_mass_by_action`

`q_target_net` теперь загружается в `load_full_checkpoint`, если присутствует.

Диагностика автоматически включается в `full_report` и `diagnose` режимах.

### Файлы

- `config.yaml` — новый флаг
- `src/utils/config.py` — default
- `src/core/deep_cfr.py` — init, checkpoint save/load
- `tools/checkpoint_tools.py` — load_full_checkpoint, diagnostics, print, json hooks
- `tests/test_hybrid_outcome_sampling.py` — 2 теста

## Что НЕ трогалось

- training dynamics
- sizing logic
- z-score flags
- src2 anneal
- inference
- reward normalization
- negative regret handling

## PASS критерии (после diagnostic run iter100)

Диагностика должна ответить на вопрос «почему fold Q >> raise Q»:

- `count_by_action.raise` — насколько мало raise-сэмплов (H2)
- `terminal_ratio_by_action` — отличается ли fold от raise по terminal/nonterminal (H3)
- `bootstrap_value_stats_by_action.fold` — получает ли fold большой positive bootstrap (H3)
- `next_strategy_raise_mass_by_action` — подавляет ли `next_strategy` raise для всех действий (H1)
- `q_net_vs_q_target_diff_by_action` — насколько q_target_net отстаёт по действиям

## Валидация

1. Включить `save_q_buffer_in_checkpoint: true` в config.yaml.
2. Запустить from-scratch прогон до iter100.
3. Сгенерировать `full_report` через `checkpoint_tools.py full --checkpoint multi_checkpoint_iter_100.pt`.
4. Проанализировать секцию `action_q_buffer` и `action_q_model` в JSON.
5. Не продолжать до iter200/400 без анализа.

## Если FAIL

- **#73**: точечный фикс по результатам диагностики (exploration, target-net sync, bootstrap correction)

## Дата: 2026-06-07
