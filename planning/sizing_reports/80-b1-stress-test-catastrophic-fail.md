# Bug #80: B1 Stress-Test Result — PRE-REGISTERED FAIL (raise_freq=2.1%), B1 input-side fix недостаточен

> **Статус:** test79seed1 (mask OFF / kind OFF / clear ON / **B1 selected-credit ON**) выполнен. Result: **PRE-REGISTERED FAIL** — raise_freq=2.1%, B1 input-side fix не сдвинул поведение.
> **Дата:** 2026-06-10 (исправлено 2026-06-10: предыдущая версия ошибочно утверждала B1 NOT PRESENT)

---

## 0. Исправление от 2026-06-10

**Предыдущая версия этого отчёта ошибочно интерпретировала прогон как "B1 NOT PRESENT / baseline variance test77". Причина ошибки:**

`tools/checkpoint_tools.py:283-293` строит `sizing_q_config` через whitelist `sizing_q_config_keys` из 9 ключей. `sizing_q_selected_credit_enabled` в этом whitelist **отсутствует** → `full_report.json` не показывал флаг, хотя он был активен.

**Доказательства активности B1:**
- `config.yaml:110`: `sizing_q_selected_credit_enabled: true` (commit 27, 29e9409)
- `src/core/deep_cfr.py:885`: читает флаг из конфига
- B1 credit-routing в callsites: `deep_cfr.py:2205-2215`, `deep_cfr.py:2384-2394`, `train.py:318-328`
- Checkpoint serialization: `deep_cfr.py:3244`, `deep_cfr.py:3292`
- Тесты: `test_sizing_q_regret.py` — routing test + checkpoint roundtrip true/false
- `sizing_q_mask_dry_run=true` относится **только** к legal-anchor mask diagnostics (#75), не к B1

**Корректная интерпретация:** B1 был ПОЛНОСТЬЮ активен. Результат — FAIL по pre-registration. `sizing_q_config` whitelist — reporting bug, не training issue.

---

## 1. Контекст

Прогон test79seed1 — B1 stress-test из #79 §8.2. Конфиг: mask OFF / kind OFF / clear ON / **B1 selected-credit ON**.

### 1.1 Конфиг (из `config.yaml` commit 27)

```yaml
sizing_q_legal_anchor_mask_enabled: false
sizing_q_kind_filter_enabled: false
sizing_q_replay_clear_once: true
sizing_q_selected_credit_enabled: true     # ← B1 ACTIVE
sizing_q_mask_diagnostics_enabled: true
sizing_q_mask_dry_run: true                # только mask, не B1
sizing_q_replay_clear_diagnostics_enabled: true
```

> **Почему full_report.json не показывает флаг:** `tools/checkpoint_tools.py:283-293` whitelist `sizing_q_config_keys` не включает `sizing_q_selected_credit_enabled`. Отчёт скрывает флаг, но B1 был активен. Это reporting bug, исправляется отдельно.

**Seed:** тот же (#78→#79). **Длина:** iter_100. **Папка:** `models/test79seed1/`. **Commit:** 29e9409 (27).

### 1.2 Pre-registered критерии (из #79 §8.3)

| Исход | Критерии | Интерпретация |
|---|---|---|
| **PASS** | raise ≥ 33%, best_sizing 0.10 < 499, raise_lt_fold < 50% | B1 fix успешен |
| **PARTIAL** | raise ≥ 27% (≥ test76) | B1 частично работает |
| **FAIL** | raise < 20%, fold > raise | B1 недостаточен или implementation bug |

---

## 2. Результат

### 2.1 Сводка — PRE-REGISTERED FAIL

| Метрика | test77 (kind OFF/clear ON, B1 OFF) | test79seed1 (kind OFF/clear ON, **B1 ON**) | Δ |
|---|---|---|---|
| `verdict` | WARN (conc 84%, raise 13%) | **WARN (raise 2.1%<5%, conc 89%)** | хуже |
| `raise_freq` | 13.0% | **2.1%** | **−10.9pp** |
| `win_rate` | 28.1 | **35.1** | **+7.0** |
| `mean_reward` | 2.06 | **−7.33** | **−9.39** |
| `unique_anchors` | 15/15 | **5/15** | −10 |
| `best_sizing_anchor_dist` | 0.10=499/499 | 0.10=**499/499** | ≡ |
| `top-2 sizing mass` | 84% | **89%** | +5pp |
| `actions` | fold=? check=? call=? raise=? | fold=30, check=548, call=**2484**, raise=65 | call-heavy |
| `sizing deployed` | — | 0.10=37, 0.75=21, 2.00=5, 1.00=1, 3.00=1 | 5/15 used |

**Verdict: PRE-REGISTERED FAIL.** `raise_freq=2.1% < 20%`, `raise_lt_fold_pct=100%`, `best_sizing_anchor_dist: 0.10=499/499`. **B1 input-side fix (credit_size=selected, credit_idx=selected) не сдвинул ни raise_freq, ни best_sizing_anchor_dist.**

### 2.2 Парадокс: win_rate=35.1 при raise_freq=2.1%

При почти нулевом рейзе модель показывает win_rate=35.1% — выше test77 (28.1%) и близко к test76 (37.7%). mean_reward=−7.33 — отрицательный несмотря на win_rate.

**Объяснение:** модель играет call-heavy (79.4% call) и почти не фолдит (30/3127=1.0% fold). Это агрессивный calling station — доходит до showdown, выигрывает 35% раздач, но проигрывает много на крупных потах где call стоит дорого.

### 2.3 Action-Q — fold-preference сохранён

| Метрика | test77 (B1 OFF) | test79seed1 (B1 ON) | Δ |
|---|---|---|---|
| `q_action.fold.mean` | +0.69 | **+0.51** | −0.18 |
| `q_action.call.mean` | −0.08 | **+0.07** | +0.15 |
| `q_action.raise.mean` | +0.21 | **+0.33** | +0.12 |
| `raise_lt_fold_pct` | 100% | **100%** | ≡ |
| `raise_lt_call_pct` | — | **0%** | raise > call |

**Action-Q стал менее экстремальным:** fold.mean упал с +0.69 до +0.51, raise.mean вырос с +0.21 до +0.33. `raise_lt_call_pct=0%` — raise > call всегда. Но fold всё ещё > raise на +0.18.

### 2.4 Self-suppressing loop — НЕ активен

| Метрика | Значение | Интерпретация |
|---|---|---|
| `bootstrap_value.raise.mean` | **+0.266** | положительный! |
| `target.raise.mean` | **+0.266** | = bootstrap (все raise — non-terminal) |
| `next_strategy_raise_mass.raise.mean` | **0.667** | next-политика хочет рейзить |
| `raise.terminal_pct` | 0.0% | все raise сэмплы bootstrap-only |

**Ключевой вывод:** self-suppressing deadly triad из #72/#73 НЕ активен. Bootstrap value положительный (+0.266), next_strategy raise-mass высокий (0.667). Политика в next-состояниях хочет рейзить, и bootstrap это ценит. Проблема НЕ в action-Q bootstrap, а в том что action-Q всё равно считает fold лучше raise (+0.51 vs +0.33).

### 2.5 Sizing funnel — здоров

| Path | test79seed1 | test77 |
|---|---|---|
| hero_os | 3 | — |
| hero_full | 805 | — |
| opponent | **7225** | 7262 |

Opponent raise funnel = 7225 — практически идентичен test77 (7262). Sizing-Q training получает много сэмплов и покрывает 15 анкеров в буфере (count_by_anchor: 0.10=17939...3.00=952). 

**Sizing-Q обучение здорово. Коллапс — на action-уровне, не на sizing-уровне.**

### 2.6 Sizing-Q inference — полный коллапс в 0.10

| Метрика | Значение |
|---|---|
| `best_sizing_anchor_dist` | **0.10: 499/499** (100% состояний) |
| `q_raise_gt_max_sq_pct` | 100% — raise Q всегда > лучший sizing Q |
| `q_call_gt_max_sq_pct` | 100% — call Q всегда > лучший sizing Q |
| `sizing_q_stats.max.mean` | −0.78 |

**B1 не сдвинул best_sizing_anchor_dist ни на одно состояние.** Несмотря на здоровый буфер (15/15 анкеров), sizing-Q на инференсе выбирает 0.10 во всех 499 состояниях. Sizing-Q отрицательный даже в максимуме (−0.78 mean).

### 2.7 Replay clear — сработал

```
sizing_q_kind_filter_diag.cleared_once: 1
sizing_q_replay_clear_diag.cleared: true
sizing_q_replay_clear_diag.before_clear.total: 3453
```

Буфер очищен при старте, refill произошёл под текущей политикой.

### 2.8 source2 — жив, ровный

```
sizing_lookahead_diag:
  q_canary_abs_sum.mean: 23.77 (не ноль → Q жива)
  using_target_net: 0 (bootstrap через live q_net)
  q_a0.mean: +1.15, q_a1.mean: +0.16, q_a2.mean: −0.19
  target.mean: +0.014, std: 0.067
source2_cross_anchor: n_anchors_represented=15, mean_std=0.012, amplification_risk=false
```

source2 работает штатно.

---

## 3. Интерпретация: почему B1 не сработал

### 3.1 B1 был активен, но не сдвинул поведение

**Факт:** `sizing_q_selected_credit_enabled: true` в `config.yaml`, B1 routing во всех 3 callsites, checkpoint serialization корректна. Прогон — настоящий B1 test, не no-op.

**Результат:** `raise_freq=2.1%`, `best_sizing_anchor_dist: 0.10=499/499` — идентично наихудшим non-B1 конфигам.

**Root cause: B1 input-side fix недостаточен.** B1 меняет `credit_size` (norm_size от selected, а не effective) и `credit_idx` (anchor_idx=selected, не effective). Но **`sizing_target` всё ещё отражает effective outcome**, а не counterfactual "что было бы при selected". Это задокументированный caveat из #79 §8.1:

> *target_value остаётся target фактически сыгранного effective raise (не counterfactual «что было бы при selected»). При forced-remap это сознательный credit-assignment: selected anchor получает credit за effective outcome. Потенциальный источник шума и причина возможного PARTIAL/FAIL.*

### 3.2 Два независимых блокера

**(a) Target-side caveat — primary для sizing-Q.** B1 починил вход (norm_size, anchor_idx), но target (`sizing_target` из `value_hat[3]`) отражает effective-сыгранный исход. Selected anchor учится "я выбрал 0.10, но получил результат effective-рейза". `best_sizing_anchor_dist: 0.10=499/499` не сдвинулся ни на одно состояние — прямое указание, что источник на target-стороне, а не на input-стороне.

**(b) Action-Q fold>raise — primary для raise_freq.** На 499 eval-состояниях `fold.mean=+0.51 > raise.mean=+0.33` на 100%. Модель рейзит 65 раз из 3127. Sizing-Q фикс физически не может проявиться, пока action-голова почти не выбирает raise — распределение сайзингов меряется по ~58 фактическим рейзам. Это **доминирующий блокер**, не второстепенный.

### 3.3 Action-Q eval-состояния не репрезентативны

q_compare использует 499 диагностических состояний. На них `q_net[fold]=+0.51` — положительный. На replay fold-сэмплах `target=-2.70`, `bootstrap=-2.28`. Разные распределения состояний. **q_compare fold — не показатель реального качества action-Q.**

### 3.4 Sizing-Q negative shift

`sizing_q_stats.max.mean = -0.78` — все sizing-Q отрицательные. Argmax коллапсирует в один анкер (наименее отрицательный). Возможная причина: target-ы в буфере mean отрицательный для большинства anchors (`mean_target_by_anchor: -0.12 до -1.21`).

### 3.5 Win-rate и raise_freq расходятся

win_rate=35.1% при raise_freq=2.1%. Модель может показывать высокий win_rate при деградировавшей стратегии. Win-rate в отрыве от raise_freq — не показатель качества.

### 3.6 High variance replay_clear_once

test77 (B1 OFF) и test79seed1 (B1 ON) — оба с конфигом kind OFF/clear ON, но результаты: raise 13.0% vs 2.1%. Replay_clear_once создаёт высокий variance из-за зависимости от ранних итераций.

---

## 4. Сравнительная таблица (все прогоны)

| Метрика | test76 (kind ON/clr ON) | test77 (kind OFF/clr ON) | test78 (kind ON/clr OFF) | test79 (**B1 ON**, kind OFF/clr ON) |
|---|---|---|---|---|
| `raise_freq` | 27.7% | 13.0% | 4.6% | **2.1%** |
| `win_rate` | 37.7 | 28.1 | 38.1 | 35.1 |
| `mean_reward` | 9.41 | 2.06 | 10.28 | −7.33 |
| `unique_anchors` | 14/15 | 15/15 | 9/15 | 5/15 |
| `best_sizing 0.10` | 499/499 | 499/499 | 499/499 | 499/499 |
| `top-2 sizing mass` | 83% | 84% | 89% | 89% |
| `action_q.raise count` | 233 746 | 147 878 | 27 474 | 86 065 |
| `opponent funnel` | 7 951 | 7 262 | 0 | 7 225 |
| `q_action.fold.mean` | −0.29 | +0.69 | +1.75 | +0.51 |
| `q_action.raise.mean` | +0.38 | +0.21 | −0.20 | +0.33 |
| `raise_lt_fold_pct` | 0% | 100% | 100% | 100% |
| `bootstrap_value.raise` | — | — | — | **+0.266** |
| `next_strategy_raise.raise` | 0.603 | 0.702 | 0.000 | 0.667 |
| **B1 active?** | no | no | no | **YES** |

---

## 5. Что это меняет

### 5.1 B1 selected/credit input-side fix — FAIL по pre-registration

B1 не сдвинул ни `raise_freq`, ни `best_sizing_anchor_dist`. **Selected/effective input/index mismatch — не единственный root cause.**

### 5.2 B1 target-side caveat подтверждён

B1 меняет input (credit_size/credit_idx), но `sizing_target` остаётся effective outcome. `best_sizing=0.10=499/499` не сдвинулся → target-side, а не input-side, является блокером для sizing-Q.

### 5.3 Action-Q — первичный фронт

`q_action.fold.mean=+0.51 > raise.mean=+0.33` на 100% eval-состояний. raise_freq=2.1%. Пока action-голова глушит raise, sizing-Q фикс не проявляется в поведении.

### 5.4 Tooling bug: report whitelist скрывает B1

`tools/checkpoint_tools.py:283-293` whitelist `sizing_q_config_keys` из 9 ключей не включает `sizing_q_selected_credit_enabled`. Это reporting-only issue, не влияет на тренировку. Исправляется отдельно.

---

## 6. Что НЕ делаем

- ❌ Включать hard legal-anchor mask или kind-filter — матрица #76-#79 показала, что они могут разрушать поведение (test78: 4.6% с kind ON, Full-D: 2.6% с mask ON)
- ❌ Повторять B1 без target-side fix — B1 input-side уже протестирован, результат FAIL
- ❌ Слепые config-ablation — 2x2 матрица закрыта

---

## 7. Следующие шаги (по приоритету)

### 7.1 P0: Action-Q fold>raise investigation

Расследовать, почему на 499 eval-состояниях `q_net[fold]=+0.51` (положительный), а на replay fold-сэмплах `target=-2.70` (отрицательный). Гипотезы:
- Distribution mismatch: eval-состояния ≠ replay-состояния
- q_net переобучился на check/call (check=509k, call=400k из 1M buffer) и даёт артефакт на fold
- Eval-состояния не содержат терминальных fold-узлов (fold → −pot/2)

### 7.2 P1: B1 target-side завершение

Рассмотреть, должен ли `sizing_target` тоже считаться под selected (counterfactual), а не effective `value_hat[3]`. Это глубже B1 и рискованнее — отдельная задача.

### 7.3 P2: Серия чекпоинтов iter 100/200/400

Подтвердить, что `raise_freq=2.1%` и `best_sizing=0.10=499` — не переходный артефакт после replay clear. Один iter 100 может быть переходным состоянием.

### 7.4 P3: Tooling fix — sizing_q_config whitelist

Добавить `sizing_q_selected_credit_enabled` в `sizing_q_config_keys` в `tools/checkpoint_tools.py:283`.

### 7.5 P4: Sizing-Q negative shift investigation

`sizing_q_stats.max.mean = -0.78`. Проверить, почему sizing-Q весь отрицательный и можно ли это исправить через target normalization/bias correction.

---

## 8. Файлы

| Файл | Изменения |
|------|-----------|
| `config.yaml` | `sizing_q_selected_credit_enabled: true` (commit 27, 29e9409) |
| `sizing_reports/80-b1-stress-test-catastrophic-fail.md` | Этот отчёт (исправлен 2026-06-10: B1 ACTIVE, не NOT PRESENT) |
| `sizing_reports/SIZING.md` | Обновлён (исправлена интерпретация #80) |
| `models/test79seed1/` | Прогон B1 stress-test с B1 ON — исходные данные |
| `tools/checkpoint_tools.py` | Reporting bug: whitelist не включает `sizing_q_selected_credit_enabled` → исправить |
| `src/core/deep_cfr.py` | B1 credit-routing (уже имплементирован в commit 27) |
| `src/training/train.py` | B1 sync-now credit-routing (уже имплементирован в commit 27) |
| `tests/test_sizing_q_regret.py` | B1 routing + checkpoint roundtrip тесты (уже есть) |
| `tools/action_q_role_legal_diag.py` | Диагностика action-Q role/legal на старом чекпоинте (#81) |
| `tests/test_action_q_role_legal_diag.py` | Тесты для role-legal diagnostics |
| `models/test79seed1/action_q_role_legal_diag_iter_100.json` | Результат role-legal диагностики |

---

## 9. Action-Q Role & Legal Diagnostics (добавлено 2026-06-10)

Результаты read-only диагностики `tools/action_q_role_legal_diag.py` на чекпоинте test79seed1 iter 100.

### 9.1 Interpretation Flags

| Flag | Value |
|---|---|
| B1 active | **true** |
| Raise bootstrap-only (0 terminal) | **true** |
| Raise target ceiling < 1.0 | **true** (max +0.418) |
| Fold positive terminal tail present | **true** (max +988) |
| Hero fold positive tail suspected | **false** (positive tail — opponent fold) |
| q_compare fold misleading (illegal fold) | **false** (fold legal 97.4%) |
| Fold>raise confirmed on LEGAL states | **true** (100% when fold legal) |
| Actor role available | **false** (next_is_hero proxy) |

### 9.2 q_compare Legal Mask

- **n_states:** 499 (raise-legal)
- **fold_legal_pct:** 97.4% — fold ЛЕГАЛЕН, не artifact
- **raise_legal_pct:** 100% (by construction)
- **raise_lt_best_legal_pct:** 97.4% — raise хуже best legal в 97.4% состояний
- **raise_lt_fold_when_fold_legal_pct:** 100.0% — raise ВСЕГДА хуже fold, когда fold легален
- **best_legal_action_dist:** `{fold: 486, raise: 13}` — fold доминирует

### 9.3 Raise Bootstrap Ceiling (TD-targets)

| Action | Terminal max | Nonterm p50 | Nonterm p90 | Nonterm p99 | Nonterm max |
|---|---|---|---|---|---|
| **raise** | — (0 terminal) | +0.32 | +0.36 | +0.38 | **+0.418** |
| fold | **+988.04** | +0.06 | — | — | +0.295 |
| call | **+1000.0** | — | — | — | +0.383 |
| check | +958.65 | — | — | — | +0.378 |

**Ключевой результат:** Raise имеет ПОТОЛОК non-terminal target ~+0.42, что логично: bootstrap value не может превысить максимальный Q в следующем состоянии. Fold/call имеют редкие terminal targets до +988/+1000, которые тянут их Q вверх. Это reward-grounding асимметрия.

При этом для NON-terminal сравнения raise (+0.42 max) ВЫШЕ fold (+0.295 max). Значит проблема не в том, что raise bootstrap undervalues — он даёт наивысший non-terminal target. Проблема в том, что fold/call получают terminal reward, который raise НИКОГДА не получает напрямую.

### 9.4 Terminal Positive Tail

- **fold terminal:** 1007 samples. `next_opponent` (opponent fold → hero wins) имеет `target_gt_100 = 21`, `target_gt_250 = 8`, `target_max = +988`.
- **call terminal:** 28896 samples. `target_gt_100 = 252`, `target_gt_250 = 100`, `target_max = +1000`.
- **check terminal:** 26724 samples. `target_gt_100 = 3`, `target_max = +959`.
- **raise terminal:** 0 samples.

**Hero fold positive tail НЕ обнаружен.** Positive fold targets — это opponent fold (нормальный poker outcome).

### 9.5 H1 vs H2 Resolution

| Гипотеза | Вердикт |
|---|---|
| **H1:** Raise структурно обездолен bootstrap-only / ceiling | **CONFIRMED.** 0 terminal samples. Nonterm target потолок ~+0.418 (ниже чем terminal tails fold/call: +988/+1000). |
| **H2:** q_compare fold>raise misleading (illegal fold) | **REJECTED.** Fold легален в 97.4% состояний. Fold>raise подтверждён на legal states. |

**Вывод:** Action-Q fold > raise — реальная проблема, не диагностический artifact.

### 9.6 Уточнение: self-suppressing loop vs reward asymmetry

Ранее мы видели `bootstrap_value.raise.mean = +0.266` и считали self-suppressing loop неактивным. Это верно: bootstrap raise положительный. Но проблема глубже:

1. **Bootstrap ceiling:** Raise target никогда не превышает ~+0.42, потому что это максимальный Q в любом next-состоянии.
2. **Terminal tails:** Fold/call изредка получают +988/+1000 terminal reward напрямую.
3. **Credit propagation:** Даже если bootstrap raise = +0.266 (выше fold bootstrap -2.28), ACTION-Q видит fold target с терминалами +988 и учится что fold может быть очень ценным.

Это не self-suppressing loop (#72/#73), а **delayed-reward ceiling + terminal-reward asymmetry**.

---

## 10. Исправленный Roadmap (post-diagnostics)

### 10.1 P0: Action-Q delayed reward propagation

Raise не может получать terminal reward напрямую (raise → без терминала). Нужен механизм, который доносит terminal value обратно через bootstrap chain:

- **n-step return** для action-Q: вместо 1-step TD(0) → n-step или TD(λ)
- **Monte Carlo target** для raise chains: аккумулировать terminal reward по всем шагам после рейза
- **Reward shaping для raise**: псевдо-награда за raise, пропорциональная expected terminal value

### 10.2 P1: Sizing-Q negative shift + ceiling

`sizing_q_stats.max.mean = -0.78` — все sizing-Q отрицательные. Даже после B1 input-side fix, sizing-Q выбирает 0.10 во всех 499 состояниях.

Возможная причина: sizing-Q тоже страдает от delayed-reward/ceiling проблемы (target никогда не положительный для sizing).

### 10.3 P2: Повтор B1 после action-Q fix

После фикса action-Q delayed reward — повторный прогон test79 конфига с B1 ON для проверки, что sizing-Q начинает использовать разнообразие анкеров.

### 10.4 P3: Terminal reward for raise?

Ввести artificial terminal reward для raise: raise → получает credit за showdown outcome. Это сделает raise напрямую чувствительным к terminal value, как fold/call.

### 10.5 Что НЕ делать (подтверждено диагностикой)

- ❌ Не включать legal-anchor mask или kind-filter
- ❌ Не повторять B1 без action-Q fix
- ❌ Не менять q_compare — fold>raise реально, не diagnostic artifact
