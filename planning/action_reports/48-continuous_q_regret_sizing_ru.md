# Bug #48 — Continuous Q/Advantage для sizing вместо scalar PG

## Статус: REPLACED — Anchor-Based Sizing (Phase 1.1) в `deepcfr-test`

**2026-05-30 (Phase 1.5)**: Исправлен баг double-counting старых regret в `train_sizing_anchor_network`: убранный в Phase 1.2 `is_fresh` возвращён — старые samples больше не добавляют один и тот же sampled regret бесконечно. Advantage w_norm=214 → ожидается снижение до ~50-70, восстановление state-variance.

**2026-05-29 (Phase 1.4)**: `train_strategy_sizing_anchor_network`: `epochs=3` (3 minibatches) заменено на `train_steps=20` (настоящие шаги). Scalar auxiliary loss отключён (`aux_scalar_weight=0.0`) — тянул shared base к global mean, гасил state-variance. `StrategySizingNet.anchor_head` init: `normal_(0, 0.01)` вместо `zeros_` — нулевой градиент в base на старте. `inference/core.py`: устаревшая scalar-only копия `StrategySizingNet` заменена на актуальную dual-head версию. `strategy_sizing_lr: 5e-5 → 1e-4`.

**2026-05-29 (Phase 1.3)**: Strategy инференс переведён с `regret_matching_anchors` на `softmax` — exploration floor 0.05 не должен применяться к average strategy. Strategy buffer теперь заполняется на КАЖДОМ raise-legal узле (а не только когда sampled action = raise). CE/KL loss: clamp + renorm target_probs для защиты от near-zero в таргетах.

**2026-05-29 (Phase 1.2)**: `sizing_target_net` теперь синхронизируется после каждого обучения (ранее был frozen с инициализации). Формула таргета унифицирована — discount применяется ко всем samples. ⚠️ **Баг**: удаление `is_fresh` привело к double-counting старых regret → advantage w_norm=214, коллапс. Исправлено в Phase 1.5. `StrategySizingNet` получил `anchor_head` (per-anchor logits) — distillation через KL-дивергенцию вместо MSE на scalar mean. Scalar `sizing_head` сохранён как auxiliary/fallback. Инференс использует `softmax` + `sample_waugh_sizing`. Лёгкий чекпоинт включает `anchors` и `num_anchors`.

**2026-05-29 (Phase 1.1)**: Исправлены корневые причины коллапса стратегии к одному sizing. Ключевое — добавлен `sizing_target_net` с DCFR+ bootstrap (не-сэмплированные anchors больше не теряют positive regret) и strategy_sizing_net переведена на скалярный выход (MSE регрессия вместо CE дистилляции).

**2026-05-29 (Phase 1)**: Sizing-подсистема полностью заменена на дискретно-непрерывный подход с фиксированными anchors + Waugh-интерполяцией. Старый continuous Q-regret (`SizingQNetwork`, AWR, Gaussian `SizingNetwork`) отключён через `sizing_q_enabled: false`, код сохранён для обратной совместимости чекпоинтов/тестов.

### Новая архитектура (Фаза 1, без drift)

| Компонент | Было | Стало |
|-----------|------|-------|
| Sizing-сеть | `SizingNetwork` (Gaussian z_mean/log_std/value) | `SizingAnchorNet` (N выходов = raw regrets/logits per anchor) |
| Advantage head | `advantage_sizing_net` + PG/REINFORCE + AWR | `SizingAnchorNet(num_anchors)` — учит regrets per anchor через DCFR+ bootstrap с `sizing_target_net` |
| Strategy head | `strategy_sizing_net` distill Gaussian (mean+log_std) | `StrategySizingNet` — anchor_head + sizing_head (aux), KL-дивергенция |
| Strategy inference | — | `softmax → Waugh` (stochastic) / `softmax → weighted_mean` (deterministic) — Phase 1.3 |
| Сэмплирование | `Normal(z_mean, std) → tanh_squash → bet_size` | `regret_matching → Waugh` (advantage), `softmax → Waugh` (strategy) |
| Exploration | ε-greedy + anchor probe decay | `MIN_PROB: 5%` константа на каждый anchor (без decay, Phase 1.1) |
| Bootstrap | только action target_net | `sizing_target_net` + синхронизация после каждого обучения (Phase 1.2) |
| Лёгкий чекпоинт | веса + anchors | веса + anchors + min_bet/max_bet — самодостаточен (Phase 1.2); deterministic: softmax→weighted_mean (Phase 1.3) |
| Движение anchors | — | Отложено на Фазу 2 |

### Ключевые функции

- `regret_matching_anchors(regrets, min_prob)` — regret matching по anchor regrets с exploration floor
- `sample_waugh_sizing(probs, anchors)` — Waugh-сэмплинг: один U ∈ [0,1] выбирает интервал и точку внутри
- `credit_assignment(size, anchors, regret)` — Leverage Rule: распределяет regret sampled sizing по двум ближайшим anchors

### Буферы

- `SizingAdvantageBuffer` — `(state, anchor_regrets[num_anchors], mask, iteration)`
- `SizingStrategyBuffer` — `(state, anchor_probs[num_anchors], iteration)`

### Тренировка

- `train_sizing_anchor_network()` — DCFR+ weighted MSE на anchor regrets
- `train_strategy_sizing_anchor_network()` — DCFR+ weighted KL-дивергенция anchor probs + auxiliary MSE scalar, с clamp/renorm target_probs (Phase 1.3)
- Оба вызываются в каждом тренировочном цикле (`train.py`)

### Конфиг (config.yaml)

```yaml
sizing_q_enabled: false
num_anchors: 4
anchor_sizes: [0.10, 1.07, 2.03, 3.00]
sizing_min_prob_start: 0.05
sizing_min_prob_end: 0.05               # Phase 1.1: без decay, константа
sizing_min_prob_decay_iterations: 1000
sizing_anchor_lr: 1e-4
sizing_advantage_buffer_size: 16384
sizing_strategy_buffer_size: 300000
```

### Файлы (все изменения)

| Файл | Изменение |
|------|-----------|
| `src/core/model.py` | `SizingAnchorNet` + `StrategySizingNet` (Phase 1.1), non-zero bias init; `StrategySizingNet.anchor_head` + tuple-возврат (Phase 1.2) |
| `src/core/deep_cfr.py` | `regret_matching_anchors`, `sample_waugh_sizing`, `credit_assignment`, `SizingAdvantageBuffer`, `SizingStrategyBuffer`, замена `SizingNetwork` → `SizingAnchorNet`/`StrategySizingNet`, `sizing_target_net` + DCFR+ bootstrap (Phase 1.1); target_net sync + унифицированный discount, KL-дивергенция, anchor-инференс, light ckpt с anchors (Phase 1.2); softmax вместо regret_matching, buffer на каждом raise-legal, clamp/renorm target_probs (Phase 1.3) |
| `src/training/train.py` | `_train_pg_and_log` → anchor training, `train_strategy_sizing_anchor_network()`, light checkpoint: +min_bet/max_bet (Phase 1.1) |
| `src/utils/config.py` | Дефолты anchor sizing, min_prob_end=0.05 (Phase 1.1) |
| `config.yaml` | `sizing_q_enabled: false`, min_prob_end=0.05 (Phase 1.1) |
| `diagnose_sizing.py` | Переработка с авто-детекцией anchor/scalar/gaussian, поддержка StrategySizingNet (Phase 1.1); dual-head детекция, tuple-совместимый `_anchor_forward` (Phase 1.2) |
| `inference/core.py` | `StrategySizingNet` (Phase 1.1), скалярный choose_action; `regret_matching_anchors`, `sample_waugh_sizing`, anchors из чекпоинта, dual-head выбор (Phase 1.2); softmax + deterministic weighted_mean / stochastic Waugh (Phase 1.3) |
| `inference/__init__.py` | Экспорт `StrategySizingNet` |
| `tests/test_anchor_sizing.py` | **Новый** — 5 тестов |
| `tests/test_sizing_q_regret.py` | 3 Gaussian-теста → anchor-версии |

### Тесты

**46/46 зелёных** (минус 4 pre-existing failures: `train_advantage_network()` NotImplementedError в `train_against_checkpoint`/`train_with_mixed_checkpoints`, SPR-assert в `test_field_verification`, `TypeError` в `DeepCFRAgentWithOpponentModeling`).

5 новых тестов (`tests/test_anchor_sizing.py`): regret matching, Waugh sampling, credit assignment, SizingAnchorNet forward.

### Утилиты (обновлены под anchor-формат)

**`diagnose_sizing.py`** — полная переработка с авто-детекцией anchor/Gaussian формата:
- Anchor: выводит anchor_logits (mean/std/min/max), bet_size, weights (anchor_head)
- Gaussian: старый вывод z_mean/log_std/bet_size
- Формат определяется по ключу `anchor_head.weight` в state_dict

**`inference/core.py`** — заменён `SizingNetwork` на `SizingAnchorNet`, добавлен `regret_matching_anchors`:
- `InferenceAgent._load` — авто-детекция формата, загрузка anchors из чекпоинта
- `choose_action` — `regret_matching_anchors(anchor_logits) → weighted_mean * anchors`
- Обратная совместимость: старые Gaussian-чепоинты → fallback midpoint

**`inference/__init__.py`** — экспорт `SizingAnchorNet` вместо `SizingNetwork`

### Проверка чекпоинта

```
models/multi/multi_checkpoint_iter_100.pt (100 итераций anchor-training):
  iteration:          100
  anchors:            [0.10, 1.07, 2.03, 3.00]
  sizing_adv_buffer:  16 384
  sizing_strat_buffer: 48 405
  advantage logits:   [-1.78, -2.34, -3.17, -1.92] — все отриц. (норм для 100 ит.)
  strategy logits:    [-0.01, +0.01, +0.01, -0.01] — почти zero-init
  strategy probs:     [0.046, 0.440, 0.467, 0.046] — масса на 1.07-2.03
  expected bet_size:  1.56 pot
  choose_action:      Call
  
  diagnose_sizing:
    strategy bet mean=1.564 std=0.0012
    advantage bet mean=1.661 std=0.295
    strategy anchor_head w_norm=0.022 (почти zero)
    advantage anchor_head w_norm=4.717 (начал обучаться)
```

### Phase 1.1 — Исправление коллапса стратегии

**Проблема**: после 800 итераций strategy схлопывается к одному anchor (bet=2.01 pot, std=0.000 для всех состояний). Action-сеть работает нормально (различает префлоп/флоп/тёрн/ривер). Причина — не sizing-head, а отсутствие counterfactual сигнала и bootstrap для sizing anchors.

**Корневые причины** (анализ после 1000 итераций):
1. `sizing_target_net` отсутствовал — не-сэмплированные anchors всегда получали target=0, теряя накопленный positive regret
2. `min_prob_end = 0.0125` слишком мал для разрыва positive feedback loop
3. Zero-init bias → 34% dead-ReLU в advantage-сети
4. Strategy через CE дистилляцию probs закрепляла доминирующий anchor

**Fix 1 — `sizing_target_net` + DCFR+ bootstrap**:
- `self.sizing_target_net` — замороженная копия `advantage_sizing_net`
- `train_sizing_anchor_network` переписан: `target = max(target_net(state), 0) * discount + new_regrets`
- Не-сэмплированные anchors сохраняют предыдущий positive regret

**Fix 2 — `min_prob_end = 0.05` константа**:
- Без decay, постоянный exploration floor 5% на каждый anchor
- config.yaml + config.py defaults

**Fix 3 — non-zero bias init**:
- `nn.init.constant_(self.anchor_head.bias, 0.01)` в `SizingAnchorNet`
- Уменьшает dead-ReLU на старте

**Fix 4 — strategy_sizing_net → скалярный выход**:
- Новая `StrategySizingNet` в `model.py`: `Linear(hidden, 1)` → scalar bet_size
- Обучение: MSE регрессия на `Σ probs_i × anchor_i` (advantage-weighted mean)
- Инференс: прямой выход → clamp — без anchors, без regret matching
- Лёгкий чекпоинт: только веса + min_bet/max_bet, самодостаточен

### План реализации

Полный план: `C:\Users\Cassmall\Desktop\ставки_новое\PLAN_anchor_sizing.md`

---

### Phase 1.2 — Целевая частота sizing + живой target_net

**Проблема**: после Phase 1.1 `sizing_target_net` инициализирован, но никогда не обновляется — остаётся frozen с init-весов. Strategy scalar-only MSE на `Σ p_i × anchor_i` сглаживает distribution в среднее → `std ≈ 0.04`, почти всегда один размер `≈1.0 pot`. Частота сайзингов теряется.

**Fix 1 — синхронизация `sizing_target_net` + унифицированный discount** (`deep_cfr.py:1756-1800`):

- После `train_sizing_anchor_network()` добавлено:
  ```python
  self.sizing_target_net.load_state_dict(self.advantage_sizing_net.state_dict())
  ```
- Формула таргета упрощена:
  ```python
  # Было:
  bootstrap_target = is_fresh * (prev_discounted + regret_tensors) + (1.0 - is_fresh) * prev_clamped
  # Стало:
  bootstrap_target = prev_discounted + regret_tensors
  ```
  Discount теперь применяется ко **всем** samples. Для старых samples `regret_tensors` уже содержит накопленный cumulative regret. Раньше `prev_clamped` без discount консервировал шум от frozen target_net.

**Fix 2 — `StrategySizingNet` с `anchor_head` + KL-дивергенция** (`model.py:46-75`, `deep_cfr.py:577-579, 1530, 1848-1884, 1995-2000, 2224-2232`, `inference/core.py:22-60, 386-395, 452-460`, `diagnose_sizing.py:84-116`):

- `StrategySizingNet` теперь имеет `num_anchors` и `anchor_head` (как `SizingAnchorNet`), плюс сохранён `sizing_head` для backward compat
- `forward()` возвращает `(anchor_logits, scalar_bet)` — tuple
- Distillation loss: **KL-дивергенция** `KL(target_probs || softmax(anchor_logits))` + auxiliary MSE на scalar (`weight=0.1`)
- Инференс (CFR-обход + `choose_action_eval`): `regret_matching(anchor_logits)` → probs → `sample_waugh_sizing(probs, anchors)` — **сохраняет частоту** сайзингов
- Лёгкий чекпоинт (`save_model`): сохранены `num_anchors`, `anchors`, `min_bet_size`, `max_bet_size`
- `inference/core.py`: загружает anchors из чекпоинта, использует `anchor_head` + regret_matching. Если `anchor_head` отсутствует в state_dict (`has_anchor_head=False`) — fallback на scalar `sizing_head`
- `diagnose_sizing.py`: детекция `is_dual_head_strat` (оба `anchor_head` + `sizing_head`), `_anchor_forward` обрабатывает tuple-возврат

**Fix 3 — exploration floor 0.05 оставлен**:
- Без изменений — `sizing_min_prob_start = sizing_min_prob_end = 0.05` уже был установлен в Phase 1.1

**⚠️ Баг Phase 1.2**: удаление `is_fresh` → double-counting старых regret. Исправлено в Phase 1.5 (`deep_cfr.py:1797`, +6 символов).

**Что НЕ тронуто**: `anchor_head.bias = 0.01`, Q-regret #48, anchors `[0.10, 1.07, 2.03, 3.00]`.

**Обновлённые файлы Phase 1.2** (исправлены в 1.5):

| Файл | Изменение |
|------|-----------|
| `src/core/model.py` | `StrategySizingNet`: `num_anchors`, `anchor_head`, tuple-возврат |
| `src/core/deep_cfr.py` | Fix 1: target_net sync + формула; Fix 2: init с num_anchors, KL-дивергенция, anchor-инференс, light чекпоинт с anchors |
| `inference/core.py` | `regret_matching_anchors`, `sample_waugh_sizing`, anchors из чекпоинта, dual-head выбор |
| `diagnose_sizing.py` | Детекция dual-head, `_anchor_forward` tuple-совместимость |

---

### Phase 1.5 — Double-counting старых regret (баг Phase 1.2)

**Проблема**: на 1400 итерациях advantage w_norm=214, probs std=[.003,.008,.006,.005], top-anchor 99% на 2.03. Advantage коллапсил в одно значение для всех стейтов.

**Корень**: Phase 1.2 убрал `is_fresh` из таргет-формулы:
```python
# Phase 1.2 (БАГ):
bootstrap_target = prev_discounted + regret_tensors  # для ВСЕХ samples
```
`buf.sample(n)` сэмплит ВЕСЬ буфер (включая старые записи с итераций 1..t−1). Для старого sample `regret_tensors` содержит regret, вычисленный НА ТОЙ итерации. Каждый train-вызов **заново добавляет старый regret** → double/triple counting → взрыв весов → коллапс.

**Fix** (`deep_cfr.py:1797`):
```python
is_fresh = (iter_tensors == t).unsqueeze(1).float()  # уже вычислялся, но не использовался
bootstrap_target = prev_discounted + is_fresh * regret_tensors  # ← 6 символов
```
| Тип sample | Формула |
|------------|---------|
| Fresh (iter=t) | `discounted_prev + новый_regret` — накопление |
| Old (iter<t) | `discounted_prev` — сохранить cumulative без пересчёта |

**Что НЕ тронуто**: strategy (жива), weight decay, clip_grad, гиперпараметры.

**Ожидаемый эффект**: advantage w_norm снизится с 214 до ~50-70, probs std восстановится, bet_corr станет положительным.

---

### Phase 1.4 — Обучение strategy: train budget + init + inference fix

**Проблема**: после Phase 1.3 strategy выучила среднее распределение (entropy 1.364), но **не различает стейты** — std probs across states = 0.002-0.020 при buffer std = 0.05-0.14.

**Корневые причины** (подтверждены анализом кода + GPT):

1. `train_strategy_sizing_anchor_network(epochs=3)` — каждый «epoch» = один `sample(batch_size=128)`. Итого 384 сэмпла за вызов, раз в 10 итераций. За 600 итераций = 7.7% буфера. Сеть видит слишком мало данных для обучения условного распределения.
2. `aux_scalar_weight=0.1` — scalar loss `MSE(scalar, Σ probs·anchor)` делит `base` с `anchor_head` и тянет shared representation к global mean — противоположно задаче сохранения state-variance.
3. `nn.init.zeros_(anchor_head.weight)` — градиент в base слои = `W^T · ∇ = 0` при W=0. Head bias обновляется, но base не получает градиента первые шаги.
4. `inference/core.py` — устаревшая scalar-only копия `StrategySizingNet` без `num_anchors`/`anchor_head`, несовместима с текущим `_load`.

**Fix 1 — `train_steps=20` вместо `epochs=3`** (`deep_cfr.py:1856-1886`, `train.py:490`):
```python
def train_strategy_sizing_anchor_network(self, batch_size=128, train_steps=20):
    for _ in range(train_steps):  # 20 настоящих random minibatches
        samples = self.sizing_strategy_buffer.sample(batch_size)
        ...
```
Новый конфиг-параметр: `strategy_sizing_train_steps: 20` (в `config.yaml` + `config.py` + `deep_cfr.py:601`).

**Fix 2 — `aux_scalar_weight=0.0`** (`deep_cfr.py:1864` удалён, scalar head не участвует в loss):
- Убран `scalar_loss` из общего loss, сохранён только `kl_loss`
- Scalar head остаётся в архитектуре для backward compat

**Fix 3 — `anchor_head` init через `normal_(0, 0.01)`** (`model.py:68-69`):
```python
nn.init.normal_(self.anchor_head.weight, mean=0.0, std=0.01)
nn.init.zeros_(self.anchor_head.bias)
```
Градиент течёт в base с первого шага. SizingAnchorNet (advantage) не затронут.

**Fix 4 — синхронизация `inference/core.py` StrategySizingNet** (`inference/core.py:291-312`):
- Устаревшая scalar-only копия заменена на dual-head версию с `num_anchors`, `anchor_head`, `sizing_head`
- `forward` возвращает `(anchor_logits, scalar_bet)` — совместимо с `_load` и `choose_action`

**Также:** `strategy_sizing_lr: 5e-5 → 1e-4` (сравнять с `sizing_anchor_lr`).

**Что НЕ тронуто:** advantage_sizing_net, SizingAnchorNet init, min_prob=0.05.

**Обновлённые файлы Phase 1.4**:

| Файл | Изменение |
|------|-----------|
| `src/core/deep_cfr.py` | Fix 1: `epochs → train_steps`, `aux_scalar_weight` удалён, `cfg_get('strategy_sizing_train_steps')` |
| `src/core/model.py` | Fix 3: `anchor_head.weight` init `normal_` вместо `zeros_` |
| `src/training/train.py` | Fix 1: `train_steps=getattr(agent, 'strategy_sizing_train_steps', 20)` |
| `inference/core.py` | Fix 4: dual-head `StrategySizingNet` с `num_anchors` |
| `config.yaml` | `strategy_sizing_lr: 1e-4`, `strategy_sizing_train_steps: 20` |
| `src/utils/config.py` | `strategy_sizing_train_steps: 20` |

---

### Phase 1.3 — Softmax для strategy + полное покрытие buffer

**Проблема**: после Phase 1.2 strategy схлопнулась к одному anchor (1.07 pot, 85%). Причины:
1. `regret_matching_anchors(min_prob=0.05)` — exploration floor для advantage, неприменим к average strategy. Negative логиты → zero в `np.maximum(regrets, 0)` → остается только floor 0.05
2. `sizing_strategy_buffer.add()` только при `sampled_action == 3` — buffer смещён к состояниям где raise выгоден, обрезает conditional distribution
3. KL с near-zero target_probs загоняет логиты в минус

**Fix 1 — softmax вместо regret_matching для strategy** (`deep_cfr.py:1528-1535, 2001-2009`, `inference/core.py:453-460`):

```python
# Было:
anchor_np = anchor_logits[0].detach().cpu().numpy()
probs = regret_matching_anchors(anchor_np, min_prob=0.05)
bet_size = sample_waugh_sizing(probs, anchors)

# Стало:
probs = F.softmax(anchor_logits, dim=1)[0].cpu().numpy()
bet_size = sample_waugh_sizing(probs, anchors)  # Waugh для stochastic
```
- `inference/core.py`: deterministic eval использует `weighted_mean`, stochastic — `sample_waugh_sizing`

**Fix 2 — strategy buffer на каждом Raise-legal узле** (`deep_cfr.py:1248-1264, 1331-1346, 1476-1488`):

- `anchor_probs` вычисляется всегда когда `3 in legal_action_types`, независимо от sampled action
- `sizing_strategy_buffer.add()` вынесен за `if sampled_action == 3`:
```python
if 3 in legal_action_types:
    anchor_probs = self._anchor_probs_from_regrets(...)
# ... позже, после обхода дерева:
if anchor_probs is not None:
    self.sizing_strategy_buffer.add(state, anchor_probs, iteration)  # всегда
```

**Fix 3 — clamp/renorm target_probs** (`deep_cfr.py:1871-1873`):

```python
target_probs = target_probs.clamp_min(1e-6)
target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
```

**Что НЕ тронуто**: advantage_sizing_net, min_prob=0.05, target_net sync, scalar head sigmoid, buffer size.

**Обновлённые файлы Phase 1.3**:

| Файл | Изменение |
|------|-----------|
| `src/core/deep_cfr.py` | Fix 1: softmax вместо regret_matching (2 места); Fix 2: anchor_probs всегда + buffer.add всегда; Fix 3: clamp/renorm |
| `inference/core.py` | Fix 1: softmax, deterministic→weighted_mean, stochastic→Waugh; `anchors_arr` |

---

## Архив: оригинальный дизайн #48 (Continuous Q-Regret)

Ниже сохранена исходная спецификация continuous Q/Advantage подхода, реализованного в MVP и заменённого на anchor-based sizing.

## Контекст

После #47-v2 sizing-сети перестали быть мёртвыми: `advantage_sizing_net` и `strategy_sizing_net` начали различать state, `strategy_sizing.log_std_head` получил ненулевые веса, distillation полного распределения работает.

Новый прогон `sizing_new` и диагностика `multi_checkpoint_iter_1600.pt` показали другую проблему: сеть жива, но сходится в перекошенный режим **"рейзить редко, но крупно"**.

### Метрики текущего прогона

TensorBoard export `sizing_new`:

| Метрика | First | AvgLast20 | Last | Диагноз |
|---|---:|---:|---:|---|
| `Sizing/ZMean` | +0.0023 | +0.5128 | +0.5275 | drift вверх, выше целевого `±0.3` |
| `Sizing/ZMeanStd` | 0.00027 | 0.0781 | 0.0869 | сеть жива, state-sensitive |
| `Sizing/LogStd` | -0.653 | -1.202 | -1.208 | exploration сжимается к floor `-1.3` |
| `Sizing/Mean_Predicted_Size` | 1.553 | 2.231 | 2.247 | bias к крупным sizing |
| `Performance/RaiseFreq` | 0.272 | 0.118 | 0.115 | ниже критерия `>=0.20` |
| `PG/MeanAdvantage` | -3.14 | +0.21 | +11.62 | шумный сигнал без устойчивой формы |

`diagnose_sizing.py --checkpoints models/multi/multi_checkpoint_iter_1600.pt`:

| Сеть | z_mean mean | z_mean std | log_std mean | bet_size mean | bet_size std |
|---|---:|---:|---:|---:|---:|
| `strategy_sizing` | +0.4219 | 0.0225 | -1.0196 | 2.1276 | 0.0273 |
| `advantage_sizing` | +0.4586 | 0.0323 | -1.1853 | 2.1714 | 0.0380 |

Вывод: #47-v2 оживил сеть, но не дал ей полноценного counterfactual-сигнала по размеру ставки. Strategy почти догоняет advantage, но обе копируют один и тот же drift к крупным рейзам.

---

## Текущая причина бага

Текущий sizing PG не является CFR/regret-matching алгоритмом для sizing.

### Что сейчас происходит

В `cfr_traverse_multi` / `_cfr_traverse_multi_outcome_node`:

1. На Raise-узле `advantage_sizing_net` сэмплирует один `z_raw`.
2. `z_raw` превращается в `bet_size ∈ [0.1, 3.0] pot`.
3. Дерево продолжается только с этим одним размером.
4. В `pg_memory` кладётся `(state, z_raw, raw_v3, bet_size)`.

В `train_sizing_network`:

1. `value_head` учит baseline `V(state)`.
2. `advantages = raw_v3 - V(state)`.
3. Policy обновляется через REINFORCE:

```python
policy_loss = -log_prob(sampled_z) * normalized_advantage
```

Это отвечает на вопрос:

> Был ли этот sampled_size лучше/хуже baseline?

Но не отвечает на CFR-вопрос:

> Насколько этот size лучше/хуже других размеров в этом же information set?

### Чего не хватает

Нет функции:

```text
Q(state, size) -> EV / CFV для конкретного размера
A(state, size) = Q(state, size) - E_{s ~ policy}[Q(state, s)]
```

Нет counterfactual-регрета по размеру:

```text
regret(size) = EV(size) - EV(current sizing strategy)
```

Поэтому сеть может честно двигаться по плохому/смещённому scalar PG-сигналу. В No-Limit это особенно опасно: крупные рейзы имеют больший absolute value / variance, и PG может спутать большой magnitude с лучшим sizing.

---

## Почему не buckets как основное решение

Дискретный sizing-regret head проще и ближе к классическому CFR:

```text
regret[0.25 pot], regret[0.5 pot], regret[1 pot], regret[2 pot], regret[3 pot]
```

Но он вводит жёсткую action abstraction. Для No-Limit агента это нежелательно: мы хотим сохранить continuous sizing и позволить сети обобщать паттерны вида:

- сет на сухой доске — крупнее;
- тонкое value / thin bluff — меньше;
- низкий SPR — ближе к all-in cap;
- мокрая доска — sizing зависит от equity denial.

Поэтому выбран второй путь: **continuous Q/Advantage approximation**.

---

## Целевой дизайн: Continuous Q-Regret Sizing

### Идея

Оставить continuous `SizingNetwork`, но перестать обновлять её напрямую от сырого scalar PG. Добавить явную модель value/advantage как функцию размера:

```text
SizingQNet(state, size) -> Q(state, size)
```

Затем обновлять policy через относительное преимущество размера внутри контекста:

```text
A(state, size) = Q(state, size) - E_{candidate sizes}[Q(state, size)]
```

Policy должна двигаться не к "sampled size получил высокий raw_v3", а к "этот size лучше других разумных sizes в этом state".

### Новые компоненты

1. **`SizingQNetwork`**
   - Вход: `encoded_state + normalized_size` или `encoded_state + z_raw`.
   - Выход: scalar `Q(state, size)`.
   - Назначение: аппроксимировать value конкретного размера рейза.

2. **`SizingQBuffer`**
   - Хранит `(state, z_raw, bet_size, target_value, iteration, sampling_log_prob?)`.
   - На первом этапе target = текущий `raw_v3` / VR-corrected value.
   - Позже target нормировать относительно pot/stack для снижения magnitude bias.

3. **Candidate-size evaluator**
   - Для каждого state генерирует набор candidate sizes.
   - Прогоняет `SizingQNetwork(state, candidate)`.
   - Считает relative advantage для policy update.

4. **Continuous policy improvement loss**
   - Обновляет `advantage_sizing_net` через advantage-weighted regression / sampled regret distillation, а не через текущий scalar REINFORCE.

---

## Candidate sizes / anchor probing

Candidate set не является финальной дискретизацией стратегии. Это рабочая сетка для оценки `Q(state, size)` и построения локального regret-сигнала.

MVP после обсуждения: вместо широкой сетки из 9 anchors используем три стабильные точки, чтобы сломать режим «один глобальный preferred size» без взрыва стоимости traversal:

```text
0.82 pot
1.55 pot
2.27 pot
```

На raise-node размер выбирается из смеси:

```text
with probability 1 - sizing_probe_prob:
    size ~ current continuous advantage_sizing policy

with probability sizing_probe_prob:
    size = random anchor from [0.82, 1.55, 2.27]
```

Стартовый режим:

```text
70% current continuous policy
10% 0.82 pot
10% 1.55 pot
10% 2.27 pot
```

Далее `sizing_probe_prob` decay-ится к малому exploration-rate, чтобы не мешать прогрессу в глубоких линиях:

```text
sizing_probe_prob_start = 0.30
sizing_probe_prob_end = 0.05
sizing_probe_decay_iterations = 1000
```

Probe anchors — это не buckets для inference. Они только дают Q-сети покрытие по трём зонам sizing. `strategy_sizing_net` всё равно копирует continuous policy.

Расширенный набор для будущей диагностики/второго этапа:

```text
0.10 pot
0.25 pot
0.50 pot
0.75 pot
1.00 pot
1.50 pot
2.00 pot
2.50 pot
3.00 pot
current policy mean
2-4 samples from current policy
```

Правило:

- anchors дают контролируемое покрытие ключевых зон sizing;
- policy samples дают локальную адаптацию;
- финальная inference-policy остаётся continuous через `SizingNetwork.mean_sizing(z_mean)` или stochastic sample при необходимости.

Важно: direct scalar PG/residual не должен учиться напрямую на probe samples как на on-policy данных. Probe samples используются для `SizingQBuffer`; policy improvement идёт через Q/AWR после coverage warmup.

---

## Обновление policy

### Вариант 48-A — Advantage Weighted Regression (рекомендуется для MVP)

Для batch states:

1. Сгенерировать candidates.
2. Посчитать `q_j = Q(state, size_j)`.
3. Посчитать baseline:

```text
baseline = mean(q_j) или weighted mean under current policy
adv_j = q_j - baseline
```

4. Выбрать positive-advantage candidates:

```text
weight_j = softmax(adv_j / temperature) for adv_j > 0
```

5. Обновить `advantage_sizing_net`, максимизируя likelihood этих sizes:

```text
loss = -sum_j stopgrad(weight_j) * log_prob_policy(size_j | state)
```

Плюсы:
- минимально инвазивно;
- сохраняет continuous policy;
- напрямую использует `Q(state, size)`;
- легко диагностировать.

Минусы:
- это не строгий CFR+;
- качество зависит от `SizingQNetwork`.

### Вариант 48-B — Sampled Continuous Regret Matching

Для candidates считать clipped positive regrets:

```text
r_plus_j = max(q_j - baseline, 0)
target_prob_j = r_plus_j / sum(r_plus)
```

Затем distill target distribution в Gaussian policy:

```text
loss = cross_entropy(target_prob, policy_prob_on_candidates)
```

Плюсы:
- ближе к regret matching;
- поведение интерпретируемо.

Минусы:
- Gaussian плохо аппроксимирует multimodal target;
- если все regrets <= 0, нужен fallback uniform/old policy.

### Вариант 48-C — Hybrid: Q-pretrain + осторожный PG residual

1. Сначала обучить `SizingQNetwork` без изменения policy.
2. Потом обновлять policy через 48-A/48-B.
3. Текущий REINFORCE оставить с маленьким весом как exploration/residual.

Плюсы:
- снижает риск сразу сломать sizing;
- можно сравнить Q-surface до policy updates.

Минусы:
- дольше до эффекта;
- больше конфигов.

---

## Связь с VR-DeepDCFR+

Из `C:\Users\Cassmall\Desktop\DeepDCFR\deeppdcfr.tex`:

1. VR-DeepDCFR+ использует sampled advantages, а не сырые counterfactual regrets.
2. Для variance reduction используется history/action value baseline `Q(h, a)`.
3. Cumulative advantage network bootstraps previous iteration:

```text
R_t(I,a) ≈ max(R_{t-1}(I,a), 0) * discount + sampled_advantage(I,a)
```

4. Strategy строится через regret matching по положительным cumulative advantages.

Для continuous sizing прямой tabular regret matching невозможен без дискретизации. Поэтому #48 переносит не форму табличного CFR дословно, а его ключевую идею:

```text
обновлять policy от относительного advantage альтернативных actions внутри information set,
а не от одного scalar sampled outcome.
```

---

## Фазы внедрения

### Фаза 0 — диагностика без изменения поведения

Добавить offline-инструмент для чекпоинта:

```text
diagnose_sizing_q_surface.py
```

До реализации Q можно построить суррогатные проверки по текущему buffer/logs:

- distribution `bet_size` в PG buffer;
- `raw_v3` vs `bet_size`;
- `raw_v3 / pot` vs `bet_size`;
- coverage по диапазону `[0.1, 3.0]`;
- сколько samples ниже `1.0 pot`, выше `2.0 pot`, около cap.

Критерий: подтвердить, что текущий PG data coverage не схлопнут полностью к `2.1 pot`.

### Фаза 1 — `SizingQNetwork` + buffer

Добавить сеть:

```python
SizingQNetwork(state_dim, hidden_size, size_embed_dim?) -> scalar
```

Минимальный вход:

```text
concat(encoded_state, normalized_bet_size)
```

Где:

```text
normalized_bet_size = (bet_size - min_bet) / (max_bet - min_bet)
```

Сбор данных:

```text
SizingQBuffer.add(state, normalized_bet_size, target_value, iteration, source, anchor_idx)
```

Где:

```text
source = 0  # current continuous policy sample
source = 1  # anchor probe sample
anchor_idx = -1 для policy sample, 0..2 для anchor probe
```

`SizingQBuffer` должен пополняться во всех активных traversal-режимах, включая обычную full traversal ветку, а не только hybrid outcome sampling. Иначе Q/AWR будет включаться на пустой или нерепрезентативной Q-сети.

Обучение:

```text
MSE(Q(state, size), normalized_target_value)
```

Target normalization MVP:

- сначала `raw_v3` как есть для сравнения;
- затем переключить на `raw_v3 / max(pot, 1)` или robust batch/EMA normalization, если magnitude bias подтвердится.

### Фаза 2 — Q-surface диагностика

Для фиксированных eval-state строить:

```text
Q(state, candidate_size)
argmax_size
advantage spread = max(Q) - mean(Q)
rank(current_policy_mean)
```

Критерии:

- Q-surface не монотонно растёт всегда к `3.0 pot`;
- argmax распределён по разным sizes;
- current policy mean не всегда около cap;
- `Q(state, size)` различает board/SPR/позицию через encoded_state.

### Фаза 3 — policy update от candidate advantages

Заменить или сильно ослабить текущий `policy_loss` в `train_sizing_network`.

MVP:

```text
loss_policy = AWR/CandidateRegretLoss
loss_q = MSE(Q(state, sampled_size), target_value)
loss_anchor = KL/L1 old anchors пока оставить
```

AWR включается только после минимального anchor coverage:

```text
min_samples_per_anchor >= sizing_q_min_samples_per_anchor
```

До этого sizing policy использует fallback/residual без прямого обучения на probe samples. Это защищает от режима, где zero-init Q даёт uniform AWR и двигает policy к искусственной candidate-сетке.

Текущий REINFORCE:

- либо отключить (`pg_reinforce_weight: 0.0`),
- либо оставить малым residual (`0.05-0.1`) на переходный период.

### Фаза 4 — strategy distillation

`strategy_sizing_net` продолжает копировать `advantage_sizing_net`, но target теперь должен отражать policy после Q-regret improvement.

Проверить:

- `strategy_sizing.bet_size_std >= 0.03`;
- gap `adv-strat` по bet mean/std;
- `strategy_sizing.log_std_head.w_norm > 0.1`.

### Фаза 5 — optional DCFR-style bootstrapping для continuous candidates

Если Q/AWR стабилизирует sizing, добавить cumulative candidate advantage approximation:

```text
R_candidate_t = max(R_prev(candidate), 0) * dcfr_discount + A_q_candidate
```

Но это второй этап, не MVP. Сначала нужно доказать, что `Q(state, size)` даёт полезную локальную форму.

---

## Новые конфиг-параметры

Предлагаемые defaults:

| Параметр | Default | Описание |
|---|---:|---|
| `sizing_q_enabled` | `false` | включить `SizingQNetwork` сбор/обучение |
| `sizing_q_lr` | `3e-4` | learning rate Q-сети (повышен с 1e-4: 2 шага/ит при 1e-4 не хватало для формирования Q-поверхности) |
| `sizing_q_hidden_size` | `128` | hidden size Q-сети |
| `sizing_q_size_embed_dim` | `16` | size embedding dim в Q-сети |
| `sizing_q_buffer_size` | `50000` | replay buffer (уменьшен с 100k для быстрого оборота и приоритета свежих данных) |
| `sizing_q_train_steps_per_iteration` | `10` | шаги обучения Q за iteration (увеличен с 2: при 2 шагах Q-spread ≈ 0.004 к ит. 200 — сеть не различала размеры) |
| `sizing_candidate_sizes` | `[0.82,1.55,2.27]` | MVP candidates для AWR/Q-сравнения |
| `sizing_anchor_sizes` | `[0.82,1.55,2.27]` | anchor-probe точки для controlled exploration |
| `sizing_policy_samples_per_state` | `4` | дополнительные samples из current policy |
| `sizing_awr_temperature` | `0.5` | temperature для advantage weights |
| `sizing_awr_uniform_mix` | `0.2` | conservative mix против раннего collapse |
| `sizing_reinforce_weight` | `0.0` | вес старого REINFORCE loss после включения Q-regret |
| `sizing_q_target_normalization` | `pot` | `raw`, `pot`, `ema`. **Важно**: `pot` теперь означает `raw_v3 / pot` (БЕЗ вычитания ev_hat), clip `[-5, 5]`. Ранее использовался `(raw_v3 - ev_hat) / pot` с clip `[-2, 2]` — это приводило к двойному baseline (Q-target уже содержал relative advantage, а AWR ещё раз вычитал mean(Q)), что системно занижало Q для крупных sizing и давало монотонно убывающую Q-поверхность |
| `sizing_probe_prob_start` | `0.30` | стартовая доля anchor probes на raise-node |
| `sizing_probe_prob_end` | `0.05` | минимальная доля anchor probes после decay |
| `sizing_probe_decay_iterations` | `1000` | длина decay probe probability |
| `sizing_q_min_samples_per_anchor` | `128` | coverage guard перед AWR |

---

## Критерии успеха

На новом прогоне 800-1600 итераций:

1. **Sizing не drift-ит к cap**:
   - `Sizing/ZMean` возвращается в диапазон `±0.3` или хотя бы ниже `+0.4`.
   - `Sizing/Mean_Predicted_Size` не держится стабильно выше `2.0 pot`.

2. **Context sensitivity растёт без collapse**:
   - `advantage_sizing.bet_size_std >= 0.05` на eval-state.
   - `strategy_sizing.bet_size_std >= 0.03`.

3. **RaiseFreq не деградирует**:
   - `Performance/RaiseFreq >= 0.20` после 500+ итераций.

4. **Q-surface осмысленная**:
   - argmax candidate sizes распределены не только в `2.5-3.0 pot`.
   - Для разных eval-state top candidates отличаются.

5. **Strategy догоняет advantage**:
   - gap `adv-strat bet mean < 0.08`.
   - gap `adv-strat z_mean std < 0.03`.

---

## Сигналы провала

1. `Q(state,size)` монотонно растёт к `3.0 pot` почти для всех states.
   - Вероятная причина: target magnitude bias, нужна pot/stack normalization или clipped target.

1b. `Q(state,size)` монотонно УБЫВАЕТ к `0.82 pot` для всех states (обнаружено на ит. 100 нового прогона).
   - Причина: двойной baseline. Target `(raw_v3 - ev_hat) / pot` + AWR `q_baseline = mean(Q)` = двойное вычитание.
   - Фикс: target = `raw_v3 / pot` (без ev_hat), AWR делает relative comparison сам. Clip расширен с `[-2, 2]` до `[-5, 5]`.

2. Policy collapses к одному size при низком `log_std`.
   - Вероятная причина: AWR temperature слишком низкая, candidates слишком узкие, entropy/anchor слабые.

3. `SizingQLoss` падает, но policy становится хуже.
   - Вероятная причина: Q overfits off-policy data; нужен coverage/exploration или conservative penalty.

4. `RaiseFreq` продолжает падать при хорошем sizing.
   - Причина уже не sizing PG, а action-regret / advantage network.

---

## Важное ограничение

Continuous Q-regret не даёт строгую гарантию CFR+ для бесконечного action space. Это инженерный перенос идеи VR-DeepDCFR+:

- явная value/advantage модель для альтернативных actions;
- variance reduction через Q;
- policy improvement от относительных advantages;
- сохранение continuous sizing.

Если нужна максимально близкая CFR-формулировка с простым regret matching, fallback — Bug #48-alt: дискретный sizing abstraction/buckets.

---

## Минимальный MVP

1. Добавить `SizingQNetwork(state, size) -> value`.
2. Писать в `SizingQBuffer` все sampled raise sizes и их normalized relative target.
3. Обучать Q каждый iteration.
4. Добавить diagnostics Q-surface по fixed eval states.
5. Добавить controlled anchor probing `[0.82, 1.55, 2.27]` с decay probability.
6. Включать policy update через candidate AWR только после anchor coverage warmup.
7. Оставить старый PG/residual за feature flag как fallback, но не учить его напрямую на probe samples.

MVP не должен менять action-policy и не должен требовать проходить все sizing ветки в дереве.
