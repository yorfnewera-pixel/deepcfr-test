# Final fix: runtime hidden_size для card_context_v1

## Причина P1

`FrozenBlueprintPolicy` и `tools/checkpoint_tools._load_full_checkpoint_agent`
читали только `network_architecture` из полного checkpoint. Затем они создавали
`DeepCFRAgent`, который брал `hidden_size` из текущего `config.yaml`. Поэтому
checkpoint card_context_v1, созданный с `hidden_size=8`, не загружался, если
runtime-конфигурация уже содержала `hidden_size=16`: PyTorch останавливался на
несовместимых формах `card_encoder`, `context_encoder` и `action_head`.

## RED

Добавлены два интеграционных теста, создающие реальный полный card_context_v1
checkpoint с `hidden_size=8`, затем меняющие конфигурацию на `hidden_size=16`:

- `test_frozen_policy_uses_declared_hidden_size_for_full_card_context_checkpoint`;
- `test_full_probe_uses_declared_hidden_size_for_card_context_checkpoint`.

До исправления оба теста падали ожидаемой ошибкой `RuntimeError` о shape mismatch
8 против 16 при `load_state_dict`.

## Исправление

Добавлен единый `full_checkpoint_network_spec` в `src/core/deep_cfr.py`.
Он до загрузки весов:

- получает архитектуру только из checkpoint metadata; отсутствие metadata
  остаётся legacy-путём только для monolithic checkpoint;
- получает `encoder_input_size` и `hidden_size` из checkpoint;
- проверяет metadata card_context_v1 и полный набор ключей и форм всех трёх
  state dict (`advantage_net`, `advantage_target_net`, `strategy_net`);
- для legacy monolithic checkpoint без `hidden_size` сохраняет совместимый
  fallback по `base.0.weight`.

`DeepCFRAgent` принимает необязательный `hidden_size`, а Frozen runtime и
full-probe передают в него объявленный checkpoint размер. Сам агент также
сверяет построенную сеть с checkpoint contract до `load_state_dict`.

Ни encoder, ни action space, ни HU teacher projection, ни transfer весов не
менялись.

## Проверка

Выполнено:

```text
python -m pytest tests/test_frozen_blueprint_policy.py::test_frozen_policy_uses_declared_hidden_size_for_full_card_context_checkpoint tests/test_checkpoint_tools.py::test_full_probe_uses_declared_hidden_size_for_card_context_checkpoint -q
2 passed

python -m pytest tests/test_card_context_architecture.py tests/test_frozen_blueprint_policy.py tests/test_checkpoint_lifecycle.py tests/test_checkpoint_tools.py tests/test_hu_checkpoint_resume.py tests/test_hu_self_play_core.py tests/test_hu_self_play_training.py -q
89 passed
```
