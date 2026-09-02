# Отчёт 028 — B1: последовательный particle filter

Дата: 2026-08-28.

## Цель

Убрать структурную деградацию reach-весов: раньше likelihood всех наблюдённых
действий перемножался для полного набора proposals, а resampling происходил
только один раз в самом конце. Поэтому длинная история часто давала ESS ниже
порога, и root возвращался к blueprint до rollout-а.

## Изменение

`sample_reach_weighted_particles` теперь обрабатывает действия оппонентов по
порядку:

1. обновляет importance weights одним наблюдённым действием;
2. нормализует веса и измеряет ESS;
3. если это не последнее действие и ESS/proposals ниже порога, resample-ит
   proposals; по умолчанию порог `0.5`;
4. после последнего действия сохраняет фактический posterior ESS для прежнего
   quality gate.

Значит, значение ESS в отчёте не подменяется значением сразу после resampling.
Добавлены `belief_resample_count` и флаг `reach_sequential_resampling` в JSON
root diagnostics. Новый CLI-параметр: `--resample-ess-ratio`.

Smoothing likelihood (B2), training, форматы checkpoint и сети не менялись.

## Проверка кода

Команда:

```powershell
python -m pytest tests\test_runtime_search_beliefs.py tests\test_runtime_search_rollouts.py tests\test_runtime_search_policy.py tests\test_runtime_search_history.py tests\test_runtime_search_paired_evaluation.py -q
python -m compileall -q src tools
```

Результат: `26 passed`; компиляция завершилась без ошибок.

Новый регрессионный тест воспроизводит настоящую последовательность двух
наблюдённых действий (`check -> check`) и подтверждает, что resampling
происходит именно между ними, а итоговый набор particles воспроизводим.

## Короткий runtime-замер

Общий режим обоих запусков: 8 deals на каждый seed, `256` proposals, `128`
rollout particles, `eta=10`, `min_ess=8`, `min_ess_ratio=0.03125`,
`resample_ess_ratio=0.5`, `min_root_gap_zscore=1`.

| Checkpoint | Search roots | Fallback | ESS fallback | Noise fallback | Roots с MMDS | Argmax changes | Resampling |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 23 | 11 | 0 | 11 | 12 | 6 | 89 |
| 4000 | 42 | 20 | 1 | 19 | 22 | 2 | 147 |

`Noise fallback` пока извлекается из `root_diagnostics`, а не отдельным
счётчиком runner-а. В 5000 все 11 fallback помечены
`rollout_signal_below_noise_blueprint_fallback`; в 4000 — 19 таким флагом и
один низким ESS.

Артефакты:

- `026-b1-sequential-filter-checkpoint-5000.json`;
- `027-b1-sequential-filter-checkpoint-4000.json`.

Для 5000 предыдущий baseline `025` имел 28 search roots и 27 fallback. После
B1 в сопоставимом коротком режиме ESS fallback больше не наблюдался, но
сравнение числа roots не является EV-оценкой: траектории candidate меняются
вместе с его действиями.

Pooled paired EV не имеет статистической мощности: для 5000 CI95 равен
`[-369.31, 382.85] bb/100`, для 4000 `[-200.49, 245.81] bb/100`. Вывод о силе
игры по этим значениям делать нельзя.

## Вывод

B1 снял прежний главный блокер — collapse ESS до rollout-а — и дал реальную
активацию solver policy на 4000 и 5000. Следующий измеряемый limiter теперь
noise-gate rollout-а, а не beliefs. B2 (минимальное smoothing likelihood)
пока не нужен: сначала стоит разобрать качество root signal на большем числе
deals при уже работающем B1.
