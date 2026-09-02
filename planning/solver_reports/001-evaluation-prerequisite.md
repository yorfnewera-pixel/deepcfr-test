# 001 - evaluation prerequisite

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/evaluation/blueprint_policy.py`, `src/evaluation/paired_harness.py`, `tests/test_frozen_blueprint_policy.py`, `tests/test_paired_evaluation.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py -q`

## Что сделано

- `FrozenBlueprintPolicy` загружает light checkpoint с `checkpoint_kind="strategy_only"` напрямую в `PokerNetwork`.
- Проверяются format version, action space, число действий, число игроков и наличие весов strategy network.
- Heavy checkpoint по-прежнему загружается через strict `DeepCFRAgent.load_model`; training loader не изменялся.
- `PairedEvaluation` хранит `bb`, а `bb_per_100` вычисляется как `mean_difference / bb * 100`.

## Почему так

Light checkpoint намеренно не содержит advantage networks, поэтому его нельзя передавать в strict loader, предназначенный для возобновления обучения. Нормализация paired EV на размер большого блайнда нужна, чтобы метрика была сопоставима между конфигурациями ставок.

## Результаты

Целевые тесты: `12 passed in 4.58s`.

Покрыты загрузка light checkpoint, сохранение heavy-path, маска legal actions и формула bb/100.

## Риски

Полный regression suite не запускался, чтобы не конкурировать с активным обучением за вычислительные ресурсы. LSP сообщает две существующие проблемы return type в `paired_harness.py`; этот этап их не изменял.

## Следующий шаг

После завершения training-прогона выполнить полный targeted и regression pytest из `solver.md`, затем перейти к S2.
