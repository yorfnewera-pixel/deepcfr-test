# Bug #47 — Sizing head не получает per-state counterfactual сигнал; Raise деградирует 25%→7%

## Статус: REVIEWED + REVISED (log1p удалён, EMA-нормировка введена, Q-baseline в multi-обходе убран, var(size|cluster) diagnostic добавлен)

## Суть проблемы

Raise frequency деградируется с ~25% (iter 0-200) до ~7% (iter 4100). При stochastic eval (np.random.choice) модель рейзит, при argmax — 0 рейзов (артефакт детерминированного инференса).

### Корневая причина

**Два взаимосвязанных бага — одна и та же проблема:**

1. **Single-sample variance leakage**: CFV(Raise) вычисляется по ONE случайно сэмплированному размеру ставки. Плохой размер → плохой CFV → CFR штрафует абстрактный "Raise". **Не атакован напрямую** — оставлен для следующей итерации (требует либо K traversal'ов, либо буфер mini-CFR; оба варианта отвергнуты в этом раунде).

2. **Sizing head не получает сигнал**: pg_memory содержит ONE (state, size, CFV) на Raise-узел за итерацию. Если Gaussian сколлапсировал — все размеры одинаковые, advantage = 0, sizing head не учится. **Атакован** через entropy floor + EMA-нормировку CFV + diagnostic var(size|cluster).

### 1.55x pot — артефакт zero-init

При zero-init `_squash(z=0) = min_bet + action_range * 0.5 = 1.55x pot`. Это midpoint диапазона [0.1, 3.0], не результат обучения.

---

## История фиксов

### Раунд 1 (предыдущий) — 5 изменений

1. **Q-net control variate для CFR regret** — снижение variance CFV(Raise) через `q + (traversal - q)`.
2. **Across-iteration sizing learning** — sizing head учится между итерациями, не внутри одного узла.
3. **Entropy floor** — `torch.clamp(log_std, min=-1.5)`.
4. **Per-action loss weighting** — Raise (7% частоты) получает ~5x больший вес.
5. **Diagnostic logging** — `traversal_stats['sizing_diag']`.

### Раунд 2 (этот аудит) — 4 изменения

6. **❌ Удалён `log1p`-scaling из pg_memory** (`scaled_v3 = sign(v) * log1p(|v|)`).
   - **Зачем удалили:** ломал градиент `∂CFV/∂size` на крупных размерах. При CFV=10 vs CFV=200 sigmoid-фильтр выдавал почти одинаковые веса → sizing head не различал «огромный рейз» и «средний рейз».
   - **Чем заменили:** EMA running std (см. п. 7).

7. **✅ EMA running std для CFV-нормировки** (`train_sizing_network`).
   - `cfv_running_std` обновляется как EMA с `decay=0.99` от батч-std сырых CFV.
   - `normalized_regret = regret_tensors / cfv_running_std` используется и для V-loss-таргета, и для PG-advantage, и для sigmoid-фильтра.
   - Per-batch std (`policy_advantages = advantages / advantages.std()`) оставлена как вторичная стабилизация — убирает резидуальную дисперсию между state'ами в батче.
   - V обучается на нормированном таргете → mse не взрывается при крупных CFV.

8. **❌ Удалён Q-baseline из `cfr_traverse_multi`** (multi-action обход всех действий).
   - **Зачем:** математически тождество `q + (traversal - q) ≡ traversal`. В multi-action ветке Q сокращается полностью → ноль variance reduction, лишний forward-pass Q.
   - **В outcome-sampling ветке Q-baseline сохранён** — там importance weight `1/μ ≠ 1`, и `q + w·(v - q)` ≠ `v`, control variate реально работает.

9. **✅ Diagnostic `var(bet_size | state-cluster)`** — детект policy-induced confounding.
   - Кластер: тапл квантованных первых 8 координат `encoded_state` (4 bin'а на координату).
   - Метрики: `size_std_median`, `size_std_p10`, `size_std_mean`, `cluster_count_with_data`, `cfv_running_std`.
   - Если медиана падает к нулю при >100 кластерах с данными → sizing head схлопывается в state-conditional режиме, advantage обнуляется внутри кластеров.

---

## Что НЕ трогали и почему

| Не атаковано | Причина |
|---|---|
| Single-sample variance leakage в CFR regret | Решение требует K traversal'ов (отвергнуто как «дорого, ветвится») или mini-CFR буфера (большая архитектурная правка). Сначала смотрим, что даст EMA-нормировка. |
| Q-net warm-up | Без `log1p` death spiral может вообще не повториться. Если повторится — добавим warm-up как отдельный шаг. |
| K разных размеров на одном Raise-узле | Явно отвергнуто — K=1 по дереву, аппроксимация через накопление в pg_memory за много итераций. |

---

## Файлы (актуальные строки после правок)

| Файл | Изменение |
|------|-----------|
| `src/core/model.py:65` | `torch.clamp(log_std, min=-1.5)` (entropy floor) |
| `src/core/deep_cfr.py:419-421` | EMA-поля: `cfv_running_std`, `cfv_running_std_decay`, `cfv_running_std_min` |
| `src/core/deep_cfr.py:423-427` | Поля для var(size\|cluster) diagnostic: `_size_var_cluster_keys`, `_size_var_quant_bins`, `_size_var_buckets` |
| `src/core/deep_cfr.py:436-462` | Helper-методы: `_reset_size_variance_diag`, `_state_cluster_key`, `_record_size_for_cluster` |
| `src/core/deep_cfr.py:516-517` | Сброс `_size_var_buckets` в `reset_traversal_stats` |
| `src/core/deep_cfr.py:603-628` | Расчёт `size_variance_diag` в `get_traversal_stats` |
| `src/core/deep_cfr.py:1080-1084` | outcome-node: чистый `v_sampled` в pg_memory + `_record_size_for_cluster` |
| `src/core/deep_cfr.py:1192-1201` | `cfr_traverse_multi`: Q-baseline убран (тождество `q+(t-q)≡t`), оба ветви используют чистый `traversal_cfv` |
| `src/core/deep_cfr.py:1241-1248` | multi-traversal: чистый `raise_traversal_cfv` в pg_memory + `_record_size_for_cluster` |
| `src/core/deep_cfr.py:1453-1464` | EMA `cfv_running_std` обновляется + `normalized_regret = regret_tensors / cfv_running_std` |
| `src/core/deep_cfr.py:1466-1473` | `policy_advantages` и V-loss работают на `normalized_regret` |
| `src/core/deep_cfr.py:1480-1483` | `raw_raise_signal` для sigmoid-фильтра считается от `normalized_regret`, не от сырого CFV |

## Новые конфиг-параметры (с дефолтами)

| Параметр | Default | Описание |
|---|---|---|
| `cfv_running_std_decay` | 0.99 | EMA-decay для running std CFV |
| `cfv_running_std_min` | 1e-3 | Минимальное значение running std (защита от деления на 0) |
| `size_var_cluster_keys` | 8 | Сколько первых координат encoded_state идут в cluster-key |
| `size_var_quant_bins` | 4 | Сколько бинов на координату при квантизации |

---

## Обнаруженная проблема (предыдущий раунд) — статус

**Q-net при zero-init бесполезен** для variance reduction.

| Метрика | iter 1 | iter 8 | iter 36+ | Диагноз |
|---------|--------|---------|----------|---------|
| Loss/Advantage | 10099 | 3547 | **0.001** | Regrets → 0, death spiral |
| PG/MeanAdvantage | +0.93 | **-3.66** | ~0 | "Рейзить всегда плохо" |
| FracPositiveAdvantage | 64% | 10% | 15-25% | Только 15-25% Raise выгодны |
| Sizing/LogStd | -1.0 | -1.0 | -1.0 | Entropy floor НЕ активирован (нормально: -1.0 > -1.5) |

**Гипотеза для следующего прогона:** `log1p`-сжатие могло быть **главным** драйвером death spiral, а не Q-net warm-up. Логика:
- При сыром CFV Raise = +50, Call = -2 → советующий рейз сигнал большой.
- После `log1p`: Raise = +3.93, Call = -1.10 → разница 5.0 → почти равные веса в sigmoid → policy gradient усредняется → sizing head не двигается.
- При коллапсе sizing head все размеры одинаковые → `var(size|cluster) → 0` → advantage схлопывается → `MeanAdvantage → -∞`.

С EMA-нормировкой относительные разности CFV сохраняются (делятся на ОДИН скаляр), а абсолютный масштаб остаётся стабильным между итерациями.

---

## Критерии успеха для следующего прогона

1. **Raise frequency** не падает с 25% до 7% за 4000 итераций.
2. **`size_variance_diag.size_std_median`** держится >0.1 (в pot-multipliers) после 500+ итераций — sizing head не схлопывается.
3. **`cfv_running_std`** стабилизируется (не растёт неограниченно, не падает к минимуму).
4. **`PG/MeanAdvantage`** не уходит ниже -1.5 (после нормировки большие отрицательные advantage невозможны без реального сигнала).
5. **`Sizing/LogStd`** держится выше -1.5 (entropy floor не активируется как костыль).

## Если death spiral повторится

Следующие гипотезы по приоритету:
1. **Q-net warm-up** на Q-buffer'е до старта PG-обучения (иначе Q ≈ 0 → control variate в outcome-ветке слабый).
2. **K=2 traversal в дереве на Raise** (отвергнуто, но если ничего не помогает — контролируемо).
3. **State-conditional CFV baseline** с отдельной сетью (учится на sampled CFV без mse-fit'а на тех же таргетах).

---

## Обсуждение архитектуры

Ключевые консенсусы (включая оба раунда):
- Cause 1 и 2 — одна и та же проблема (single-sample CFV = механизм sizing poisoning).
- K traversal'ов — слишком дорого, разветвляет дерево → отвергнуто.
- V(new_state) bootstrap — control variate поправка сокращается при within-state mean-centering → отвергнуто.
- Antithetic sampling — symmetry trap при inverted-U CFV(size) → отвергнуто.
- Heuristic anchor — контаминация training signal → отвергнуто (жёсткое ограничение: sizing self-learns).
- Q-net для sizing head — не знает про bet_size, предсказывает Q(s, action_type) → отделён от sizing head.
- **`log1p` scaling** — был в коде без обоснования в репорте, ломал градиент → удалён.
- **Q-net в multi-traversal обходе** — математически тождественен (`q + (t - q) ≡ t`) → удалён, оставлен только в outcome-sampling.

Открытый вопрос: подтвердится ли гипотеза о `log1p` как драйвере death spiral. Ответ — после следующего прогона.
