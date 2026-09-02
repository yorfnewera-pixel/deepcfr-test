# Task 6 — final-review fix

## RED

Регрессионный тест resume на `models/test109/multi_checkpoint_iter_1000.pt` падал: один и тот же full checkpoint читался дважды (`torch.load` вызывался два раза вместо одного).

## GREEN

Кэш strategy state создаётся до обработки initial checkpoint; возвращённый `strategy_net` сохраняется по `Path.resolve()`. `_configure_opponent_pool` использует resolved-ключ при чтении и передаче состояний. Регрессионный тест проходит и подтверждает пять states.

## Проверка

- Новый тест: `1 passed`
- Итоговый suite: `52 passed` (`pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py tests/test_action_space.py -q`)

## Риски

Риск ограничен изменением ключей кэша и resume-пути: strategy state не копируется и не мутируется; остальные training/warm-up/timing semantics не менялись.
