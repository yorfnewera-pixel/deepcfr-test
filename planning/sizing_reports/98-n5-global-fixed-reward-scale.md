# #98 — N5: Global Fixed Advantage-Reward Scale (фикс «слепой линейки»)

**Дата:** 2026-06-19
**Статус:** IMPLEMENTED (ждёт прогона test97seed1N5)
**База:** R2 (A-core, DCFR+) + OS off, откат N4 → `advantage_accumulation: dcfr_plus`

## Диагноз (#97)

advantage-loss ~40k–55k, таргеты ~±220 фишек. Raise +EV/безубыточен (reward_stats), но различие raise-vs-fold (единицы фишек) тонет в масштабе сотен фишек → сеть грубо фитит → бросает прибыльный raise. Корень — СЫРОЙ масштаб регрета на уровне фита.

## Почему не relative-pot

`pot_stack`/`pot` — переменный (per-state) делитель → ломает кумулятивный бутстрап (N1/N1' провалились). Нужна ОДНА ФИКСИРОВАННАЯ константа (гомогенна с накоплением, не ломает сумму).

## Изменение

Делить `cf_regrets` на фиксированную `advantage_reward_scale` ПЕРЕД записью в advantage-буфер в **full-traversal сайте** (cfr_traverse_multi, ~2653):

```python
if self.advantage_reward_scale != 1.0:
    cf_regrets = (cf_regrets / self.advantage_reward_scale).astype(np.float32)
```

**НЕ трогаем:** OS-сайт (Q-derived, уже O(1)), train.py (legacy-траверс), clamp+discount (DCFR+), traverse-return (raw), Q-путь.

## Точки изменений (5 шт., 2 файла)

| # | Файл | Локация | Что |
|---|------|---------|-----|
| 1 | `src/utils/config.py` | `_DEFAULTS`, ~150 | `'advantage_reward_scale': 1.0` |
| 2 | `src/core/deep_cfr.py` | `__init__`, после `advantage_huber_delta` | `self.advantage_reward_scale = float(cfg_get('advantage_reward_scale', 1.0))` |
| 3 | `src/core/deep_cfr.py` | `cfr_traverse_multi`, ~2653 (после `.astype`, перед `.add`) | деление cf_regrets на scale |
| 4 | `src/core/deep_cfr.py` | `_build_checkpoint`, после `advantage_accumulation` | сохранение в конфиг-словарь |
| 5 | `src/core/deep_cfr.py` | `_load_checkpoint` whitelist, перед `discount_alpha` | добавление в whitelist |

## Выбор константы

- **Старт:** `advantage_reward_scale: 200` (≈ стек в фишках при 100bb, bb=2)
- Калибровка: LossAdvantage > ~100 → увеличить; < ~0.01 → уменьшить
- Целевой диапазон LossAdvantage ∈ [~0.1, ~10]

## Конфиг прогона (test97seed1N5)

```yaml
advantage_reward_scale: 200       # единственное изменение vs R2+OS-off
advantage_accumulation: dcfr_plus # откат N4
advantage_regret_norm: none
advantage_regret_clip: null
advantage_loss: mse
advantage_huber_delta: 1.0
discount_alpha: 1.5
advantage_lr: 1e-4
advantage_weight_decay: 1e-5
advantage_epochs: 2
advantage_batch_size: 1024
os_force_full_traversal: true
sizing_anchor_availability_enabled: true
sizing_allin_boundary_enabled: true
sizing_preflop_disable_buckets: true
sizing_availability_on_input: false
sizing_cfr_mode: true
use_q_baseline: true
q_reward_propagation: mc
q_target_norm: pot_stack_relative
q_bootstrap_per_player_enabled: false
use_multi_agent_advantage: true
num_players: 6
big_blind: 2.0
checkpoint_frequency: 100
```

## Метрики/гейты

- **LossAdvantage → O(1–10)** — должен упасть ~в scale² (≈40000) раз
- **advantage raise flip** (`relapse_diff`): не уходит в сильный минус
- **raise_freq** (train PerformanceRaiseFreq + eval): пик не схлопывается 100→300
- **reward_stats_by_action**: raise остаётся ≥ fold
- **h2h late-vs-early**: late reward ≥ 0

## PASS/FAIL

- **PASS:** LossAdvantage O(1–10) И raise не схлопывается И advantage raise не флипается
- **FAIL:** raise < 10% к iter 300 (масштаб не помог → корень глубже)

## Риск и follow-up

- Глобальная константа НЕ выравнивает состояния между собой (глубокие большие банки остаются относительно крупнее). Если они доминируют остаточно → **N5b**: `advantage_loss: huber` (робастность к остаточному разбросу)
- Константу НЕ менять mid-run (свежие прогоны)
- Откат: `advantage_reward_scale: 1.0` → исходное поведение

## Причина пропуска OS-сайта

OS-регреты = `value_hat` из `q_net` (Q-baseline, нормирован `pot_stack_relative`) → уже O(1). Повторное деление на 200 занулило бы их. Масштабирование только на full-traversal сайте, где регреты сырые (O(100)).
