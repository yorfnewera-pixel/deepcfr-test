# 004 - lockstep rollouts

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/evaluation/blueprint_policy.py`, `src/runtime_search/rollouts.py`, `tests/test_frozen_blueprint_policy.py`, `tests/test_runtime_search_rollouts.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py -q`

## Что сделано

- `FrozenBlueprintPolicy.probabilities_batch` выполняет один forward pass для batch states с разными current players.
- `materialize_particle_state` подставляет particle opponent hands, сохраняет hero hand и reconstructs deck как `runout + independently shuffled remainder`.
- `evaluate_root_actions` применяет все root actions через `pkrs.parallel_apply_action`, затем продолжает не-terminal сценарии lockstep batch calls до terminal state.
- Continuation action sampling использует детерминированный CRN stream из particle seed, depth и current player.
- Результат возвращает raw EV mean/std, successful rollouts, total particles, failure messages и terminal completion rate.

## Почему так

Все root actions одного particle видят одинаковую hidden history и упорядоченный runout; различие EV не зависит от случайно разданного board. Batch inference и parallel engine apply сокращают overhead до появления performance cache в S8.

## Результаты

Целевая проверка S1-S3: `21 passed in 4.45s`.

Покрыты совпадение batch/single policy inference, independent particle materialization, отсутствие зависимости от source `state.deck`, terminal rollout, CRN reproducibility и emergency depth guard.

## Риски

`State.from_mid_hand` принимает один initial stake на всех игроков, поэтому S3 явно отклоняет unequal initial stacks. Это безопаснее неявной реконструкции с неверными stack sizes; поддержка short stacks потребует отдельного расширения движка или adapter.

Uniform beliefs до S6 по-прежнему не дают solver-valid EV. Serena/Pyright не резолвит imports между новыми `src/runtime_search` файлами как namespace package, но Python импортирует их и целевые тесты проходят.

## Следующий шаг

S4: identity response-to-bet root policy. Взять legal compact prior `pi` из blueprint, нормировать `raw_ev` в `mc_values`, применить MMDS и вернуть `SearchDecision` с полной диагностикой.
