# Postflop Identity Runtime Search Design

## Цель

Активировать runtime search во всех compact postflop root nodes: flop, turn и river, как при отсутствии ставки, так и при ответе на ставку. Training loop, checkpoint format и шесть blueprint slots остаются неизменными.

## Root action sets

При отсутствии обязательного call runtime root — это `check`, `raise_0.5pot`, `raise_1pot`, `all-in`. При ставке против hero root — это существующий набор `fold`, `call`, `raise_1pot`, `all-in`. В обоих случаях prior получается прямой legal-mask нормализацией соответствующих blueprint slots: mapping на новые размеры не применяется.

## Streets и rollout

`Flop` использует два independently sampled runout cards, `Turn` — одну, `River` — ноль. Существующий S3 rollout получает те же reach-weighted particles и CRN semantics. Обновлённый `RuntimeSearchPolicy` направляет каждый postflop identity root в `choose_action`; остальные узлы продолжают играть frozen blueprint.

## Безопасность

Без observed history или при низком ESS сохраняется существующий blueprint fallback. Никаких изменений в `train.py`, модели или `src/core/action_space.py` не требуется. Lead sizes 25/33/75/150 процентов остаются отдельным будущим S7-multisizing этапом.

## Тестирование

Новые tests проверяют constructed flop state, root mask без ставки, корректные два runout cards, legal decision и автоматическую активацию wrapper на flop. Existing response-to-bet behavior остаётся regression-covered.
