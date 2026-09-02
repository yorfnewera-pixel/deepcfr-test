# 003 - belief spots

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/runtime_search/cards.py`, `src/runtime_search/spots.py`, `src/runtime_search/beliefs.py`, `tests/test_runtime_search_spots.py`, `tests/test_runtime_search_beliefs.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py -q`

## Что сделано

- Добавлена независимая полная колода и сравнение карт по `(rank, suit)`.
- Добавлены deterministic 6-max turn/river spots через `State.from_mid_hand`.
- В constructed spots `stake=200.0`, `pot_chips=30.0` для каждого игрока, поэтому remaining stack равен `170.0` и pot равен `180.0`.
- Добавлен immutable `BeliefParticle` и blocker-aware sampler скрытых рук/runout от supplied `np.random.Generator`.
- Sampler принимает только hero hand и public board, поэтому не может использовать порядок `state.deck` как истинный будущий board.

## Почему так

`from_mid_hand` восстанавливает remaining stack как `stake - committed`; передача remaining stack вместо initial stack исказила бы размеры ставок и EV. Независимая колода предотвращает leakage будущего board в MC rollout. 

## Результаты

Объединённая целевая проверка: `17 passed in 4.22s`.

Покрыты turn/river construction, initial-versus-remaining stack, уникальность всех карт, blocker exclusion, воспроизводимость RNG и невозможный sample.

## Риски

Текущая particle distribution равномерна среди доступных карт и помечена `is_solver_valid=False`. До S6 она пригодна только для проверки механики, не для вывода об улучшении paired EV.

Serena/Pyright не резолвит imports между новыми файлами `src/runtime_search` как namespace package, но Python импортирует их и все 17 целевых тестов проходят. Полный regression suite отложен, пока идёт обучение.

## Следующий шаг

S3: lockstep rollout engine с одной hidden history/runout на particle и общим сравнением всех root actions.
