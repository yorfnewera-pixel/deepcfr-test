# Runtime Search: контроль Monte-Carlo шума и мощность evaluation

## Цель

Не позволять MMDS усиливать шумный root rollout и не интерпретировать маломощный paired EV как качество solver-а. Изменение касается только runtime search/evaluation; training loop, checkpoint format и compact action space не меняются.

## Причина изменения

Текущий `sample_reach_weighted_particles` создаёт ровно `particle_count` proposals и столько же resampled particles. Поэтому `ess_ratio = ESS / particle_count`; при двух particles ratio не может стать меньше `0.5`, а при восьми достигает `0.125`. Порог `0.25` не сравнивает качество belief одинаково для разных budget.

Одновременно MMDS получает средние EV всего по двум rollout на action. Повышение `eta` меняет policy на основе такой оценки, не измеряя её сигнал относительно MC-шума.

## Архитектура

### 1. Отдельные proposal и rollout budgets

`RuntimeSearchConfig` получает:

- `belief_proposal_count: int = 256` — число независимых blocker-aware hidden histories, на которых вычисляются reach weights;
- `belief_particles: int = 32` — число particles, resampled из posterior и переданных в lockstep rollout;
- `belief_min_ess: float = 8.0`;
- `belief_min_ess_ratio: float = 0.03125` — отношение `ESS / proposal_count`, а не к числу resampled particles.

`sample_reach_weighted_particles` принимает оба count. Он строит `proposal_count` независимых proposals, нормализует их weights, возвращает ESS и `proposal_count`, затем resample ровно `particle_count` particles. Конфигурация отклоняется, если `proposal_count < particle_count`.

Search делает blueprint fallback, если `ESS < belief_min_ess` или `ESS / proposal_count < belief_min_ess_ratio`. В диагностике сохраняются оба значения и причина fallback.

### 2. CRN-корректная оценка сигнала root action

`evaluate_root_actions` сохраняет statistics по reward каждого root action на общей последовательности particles. Помимо существующих `raw_ev_mean` и `raw_ev_std` он возвращает:

- `raw_ev_se = raw_ev_std / sqrt(successful_rollouts)` по action;
- разность EV лучшего и второго действия;
- standard error этой разности, вычисленный по попарным difference на одних и тех же particles (CRN учитывается, независимая формула из двух `raw_ev_std` не используется);
- `best_gap_zscore = gap / gap_se`.

`RuntimeSearchConfig.min_root_gap_zscore = 1.0`. Если валидный лучший-vs-второй gap не превосходит этот порог, `choose_action` возвращает blueprint fallback с флагом `rollout_signal_below_noise_blueprint_fallback`. На ошибке/неполных rollout применяется уже существующий fallback.

### 3. Прозрачная MMDS-диагностика

`SearchDecision` дополнительно содержит `raw_ev_std`, `raw_ev_se`, `best_gap`, `best_gap_se`, `best_gap_zscore` и `eta_effective`.

Для `rho=uniform`:

```text
eta_effective = eta / (1 + alpha * eta)
```

Uniform magnet добавляет одинаковую константу всем legal actions и после нормализации не меняет их относительные веса; `alpha` также temper-ит вклад prior через общий знаменатель. Поэтому отчёт обязан показывать как `eta`, так и `eta_effective`.

### 4. Paired EV статистика и правило мощности

`PairedEvaluation` и JSON получают pooled statistics по всем runs:

- `mean_difference`, `std_difference`, `standard_error`;
- `bb_per_100`, `standard_error_bb_per_100`;
- `t_statistic`, 95% normal CI;
- `power_status`: `insufficient_power` при samples `< 10_000`, иначе `reported`.

CLI печатает эти значения. Отчёты с `<10_000` samples являются диагностическими и не могут служить основанием признать конфигурацию лучше или хуже blueprint. Порог не гарантирует обнаружение любого малого эффекта: требуемый budget дополнительно рассчитывается из фактической SE и целевого MDE.

### 5. Документация и отчёты

`planning/solver_reports/020-noise-control-branch-closure.md` закрывает eta sweep 013–019 как диагностику усиления шума, а не как решение о качестве `eta=20/30`.

`planning/solver_reports/solver.md` фиксирует новые budgets, ESS contract, noise gate и обязательные paired statistics.

## Тесты

1. 256 proposals → 32 particles: воспроизводимость, card legality, `proposal_count` и ESS ratio относительно proposals.
2. ESS fallback возможен при `particles=2`, когда weights концентрированы в меньшем числе proposals.
3. CRN pairwise standard error совпадает с hand-calculated fixture; gate сохраняет blueprint при недостаточном z-score и пропускает ясный signal.
4. `eta_effective` соответствует формуле для `rho=uniform`.
5. Pooled paired statistics корректны для известных per-seed mean/std/samples; small sample получает `insufficient_power`.
6. Регрессия существующих beliefs, rollouts, policy и paired evaluation.

## Последовательность

1. Proposal/ESS contract и tests.
2. Rollout signal statistics и root noise gate.
3. Paired statistics, CLI и `solver.md`.
4. Проверить policy delta на checkpoint 3000: `proposal_count=256`, `particles=32`, исходный `eta=10`; без EV claim.
5. Только если noise gate не блокирует почти все root, запланировать paired EV с заранее заданным MDE и power budget.
