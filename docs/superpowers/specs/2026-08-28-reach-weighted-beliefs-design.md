# Reach-Weighted Beliefs Design

## Цель

Заменить uniform blocker sampling в runtime search на приближение posterior `P_pi(H | h_i)` по наблюдаемым решениям оппонентов. Изменение не касается training loop, checkpoint формата, action space или текущего прогона обучения.

## Граница API

Новый opt-in тип `ObservedDecision` передаёт снимок `pkrs.State` до действия, `actor_id` и исполненный compact `action_slot`. Снимок используется только как публичная структура: при расчёте likelihood все hidden hands оппонентов заменяются картами particle, а deck не читается. Existing `choose_action` получает необязательный `observed_decisions`; отсутствие истории не запускает uniform fallback и возвращает blueprint prior с флагом `belief_history_missing_blueprint_fallback`.

## Формирование beliefs

1. Создать `N` blocker-aware proposal particles прежним sampler-ом.
2. Для каждого particle и каждого решения оппонента materialize state snapshot с particle hands.
3. Получить `blueprint.probabilities(snapshot, player_id=actor_id)` и добавить `log(max(pi(action), reach_probability_floor))`.
4. Нормализовать log-weights через log-sum-exp, вычислить `ESS = 1 / sum(w^2)`.
5. Resample `N` particles с replacement по нормализованным весам, пометив их `is_solver_valid=True`; один resampled particle продолжает обслуживать все root actions в S3 CRN rollout.

Hero actions не входят в likelihood: hero hand фиксирована условием, а его policy не добавляет информации о hidden hands оппонентов. Каждое observed decision валидируется: actor должен совпадать с `state.current_player`, а slot должен быть legal в snapshot.

## Low-ESS и ошибки

`RuntimeSearchConfig` получает `belief_min_ess_ratio` и `reach_probability_floor`. Если `ESS / N` ниже порога, MMDS и rollout не запускаются: возвращается root blueprint prior с `belief_low_ess_blueprint_fallback`. Если история отсутствует, результат аналогично остаётся blueprint-only. Некорректная история не маскируется fallback-ом и выбрасывает `ValueError`, чтобы caller не получил ложный solver-valid результат.

`SearchDecision` получает `belief_ess` и `belief_ess_ratio`; flags различают `reach_weighted_beliefs`, `belief_history_missing_blueprint_fallback`, `belief_low_ess_blueprint_fallback` и existing `rollout_incomplete`.

## Тестирование

Unit tests используют детерминированный blueprint, зависящий от private hand acting opponent. Они проверяют, что reach weights меняют ESS, resampled particles solver-valid, seed воспроизводим, hidden карты не дублируются и invalid observed action отклоняется. Policy tests проверяют blueprint fallback без истории и low-ESS fallback; существующий positive path получает предоставленную историю и сохраняет legal action/MMDS behavior.

## Не входит в S6

- paired-EV kill-gate;
- расширенная сетка сайзингов;
- изменение training/replay buffer для автоматического накопления истории;
- кэширование и latency optimisation S8.
