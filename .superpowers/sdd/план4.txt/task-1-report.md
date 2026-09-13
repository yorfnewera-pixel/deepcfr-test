# Task 1 — отчёт

## Реализовано

- Строковые идентификаторы публичных checkpoint вынесены в `src/core/checkpoint_kinds.py`.
- `FrozenBlueprintPolicy.from_checkpoint()` сначала различает generic light, HU light, HU full и legacy generic full.
- Для HU full добавлен отдельный parser: он читает только `mode`, `architecture.strategy` и `strategy.network`, валидирует metadata, encoder, actor-conditioned input, схему сети и формы tensor до строгой загрузки.
- HU full evaluation создаёт только `PokerNetwork`; `DeepCFRAgent`, replay buffers, optimizers, advantage-сети и RNG не восстанавливаются.
- Inference сохраняет P0/P1 actor one-hot и прежнее legal masking.

## Изменённые файлы

- `src/core/checkpoint_kinds.py`
- `src/core/deep_cfr.py`
- `src/training/train.py`
- `src/core/teacher_transfer.py`
- `src/evaluation/blueprint_policy.py`
- `tests/test_frozen_blueprint_policy.py`

## TDD evidence

RED до реализации:

```text
pytest -q tests/test_frozen_blueprint_policy.py -k 'hu_full'
6 failed, 13 deselected
```

Причина каждого нового сценария: текущий loader требовал отсутствующий у HU full корневой `num_players`, не доходя до HU `mode`.

GREEN после минимальной реализации:

```text
pytest -q tests/test_frozen_blueprint_policy.py -k 'hu_full'
6 passed, 13 deselected

pytest -q tests/test_frozen_blueprint_policy.py tests/test_hu_checkpoint_resume.py tests/test_teacher_transfer.py
84 passed in 4.97s

pytest -q
483 passed, 2 skipped in 15.31s
```

## Self-review

- Проверен порядок dispatch и сохранён generic full fallback.
- Проверены omitted/explicit HU player count, mismatch `6`, full/light logits и P0/P1 probabilities, mixed batch, legal mask, отсутствие training-agent/optimizer, malformed schema, неверный base input, отсутствие actor features и fusion shapes v2.
- `git diff --check` завершился без ошибок.

## Риски

- Загрузка checkpoint по-прежнему использует существующий проектный `torch.load(..., weights_only=False)` для совместимости формата; эта задача не меняет политику доверия к checkpoint.
- HU full parser намеренно принимает только двух игроков и текущий HU mode с `use_multi_agent_advantage=false`, как требует контракт self-play checkpoint.
