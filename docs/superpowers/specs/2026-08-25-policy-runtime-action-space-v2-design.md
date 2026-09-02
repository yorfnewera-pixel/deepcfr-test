# Runtime policy и action-space v2

**Дата:** 2026-08-25

## Цель

Заменить пакет `inference` на `policy_runtime`, выделить runtime-порог вероятности из Deep CFR, ввести action contract `six_fixed_v2` и сделать консольный итог итерации компактным.

## Контракт runtime

- Пакет `inference` удаляется без compatibility-обёртки.
- `InferenceAgent` и `InferenceAdapter` переименовываются в `PolicyRuntimeAgent` и `PolicyRuntimeAdapter`.
- Ключ конфигурации runtime: `policy_runtime_min_action_prob`.
- При чтении legacy-полей checkpoint допускается fallback `inference_min_action_prob`, но новые light-checkpoint записывают только новый ключ.
- `DeepCFRAgent.get_policy_distribution` не применяет runtime-порог.

## Контракт действий v2

- `ACTION_SPACE_VERSION = "six_fixed_v2"`.
- `RAISE_HALF_POT` legal только до первой добровольной ставки на улице.
- После добровольной ставки `RAISE_HALF_POT` illegal; `RAISE_POT` и `ALL_IN` сохраняются.
- Признак добровольной ставки определяется по `last_stage_action == Raise` любого игрока текущей улицы.
- Runtime-представление состояния передаёт `last_stage_action`, чтобы mirror-mask совпадала с training mask.
- Runtime и GUI принимают только `six_fixed_v2`.

## Консоль

После итерации выводится одна строка: `Time/Iteration`, `Time/Traversal`, `Loss/Advantage`, `Loss/Strategy`. TensorBoard-метрики не меняются.

## Совместимость

Старые v1-checkpoint сознательно не поддерживаются. Для v2 требуется новое обучение.

