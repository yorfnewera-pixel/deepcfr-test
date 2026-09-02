# Bug #74 v6: Source2 Lookahead Diagnostics V4 — Root-Cause Split

> **Статус:** Diagnostics V4 deployed. Обучение НЕ изменено.
> **Цель:** определить, почему `why post-raise action-Q = [0, 0, 0, 0]` на всех lookahead состояниях.
> **Предыдущий отчёт (v3):** `q_a0..q_a3 = 0.0, strategy_sum=1.0, legal_sum=2.69, final_state=0, pot 3→1191`.
> **Следующий шаг:** диагностический прогон с включённым lookahead + новый report.
> **НЕ ОТКЛЮЧАТЬ** `sizing_lookahead_enabled` в этом прогоне.
> **Дата:** 2026-06-09

---

## 1. Контекст

Из v3-диагностики (`74-v5`): `source=2` пишет 33 665 константно-нулевых targets. `q_a0..q_a3 = 0.0, std=0, min=max=0` на всех post-raise состояниях при варьирующемся pot (3→1191). Это не terminal reward, не clipping, не mask/strategy. Проблема в том, что action-Q на post-raise = ровно ноль.

v4-диагностика проверяет три гипотезы одним прогоном:

| Гипотеза | Проверка |
|----------|----------|
| Dead-ReLU / OOD вход: base выдаёт 0, поэтому q_head(0)=bias≈0 | `base_out_abs_sum` |
| Вход вообще битый (NaN/нулевой/странный) | `q_input_*` |
| Сеть вообще не обучена / q_head мёртвая по этому пути | `q_canary_abs_sum` (ones_like(q_t)) |
| Используется не та сеть (target vs online) | `using_target_net` counter |
| Роль/перспектива | `current_player`, `traversing_player`, `next_is_hero` |

---

## 2. Что сделано (v4)

### 2.1 Новые counters

Файл: `src/core/deep_cfr.py`

```python
self.sizing_lookahead_diag = {
    'counts': {
        ...
        'using_target_net': 0,
        'q_input_has_nan': 0,
        'next_is_hero': 0,
    },
    'stats': {},
}
```

### 2.2 _bootstrap_sizing_value v4 debug

v3-блок заменён на:

```python
q_net_used = self.q_target_net if self.sizing_q_bootstrap_target_net else self.q_net
q_vals = q_net_used(q_t)
v = float((strategy * q_vals).sum().item())

if self.sizing_q_target_diagnostics_enabled:
    base_out = q_net_used.base(q_t)
    canary = q_net_used(torch.ones_like(q_t))
    self._last_bootstrap_debug = {
        'strategy_sum': ..., 'legal_sum': ..., 'q_vals': ...,
        'using_target_net': bool(self.sizing_q_bootstrap_target_net),
        'q_input_abs_sum': float(q_t.abs().sum().item()),
        'q_input_min': float(q_t.min().item()),
        'q_input_max': float(q_t.max().item()),
        'q_input_has_nan': bool(torch.isnan(q_t).any().item()),
        'base_out_abs_sum': float(base_out.abs().sum().item()),
        'q_canary_abs_sum': float(canary.abs().sum().item()),
        'current_player': int(current_player),
        'traversing_player': int(traversing_player),
        'next_is_hero': bool(int(current_player) == int(traversing_player)),
    }
```

Важно: `q_net_used.base(q_t)` использует ту же сеть, что и forward, поэтому мерит ровно промежуточный representation.

### 2.3 _perform_sizing_lookahead v4 stats

Новые метрики записываются в running aggregates:

```python
self._upd_la_stat('q_input_abs_sum', dbg['q_input_abs_sum'])
self._upd_la_stat('q_input_min', dbg['q_input_min'])
self._upd_la_stat('q_input_max', dbg['q_input_max'])
self._upd_la_stat('base_out_abs_sum', dbg['base_out_abs_sum'])
self._upd_la_stat('q_canary_abs_sum', dbg['q_canary_abs_sum'])
if dbg['using_target_net']: c['using_target_net'] += 1
if dbg['q_input_has_nan']: c['q_input_has_nan'] += 1
if dbg['next_is_hero']: c['next_is_hero'] += 1
```

### 2.4 Файлы

- Изменён: `src/core/deep_cfr.py`
- **Не изменён:** `tools/checkpoint_tools.py` — автоматически подхватывает новые метрики из `sizing_lookahead_diag['stats']`.

### 2.5 Что не менялось

- `sizing_lookahead_enabled` — ОСТАВЛЕН включён.
- Обучение, loss, sampling — без изменений.
- `sizing_q_target_diagnostics_enabled` — гейт как в v3.

---

## 3. Тесты

```text
py_compile src/core/deep_cfr.py → OK
70/70 pass (test_hybrid_outcome_sampling.py + test_sizing_q_regret.py)
```

---

## 4. Как читать новый report

```text
q_input_has_nan > 0
=> encoding/input bug. Сеть отдаёт ноль из-за мусорного входа.

q_input_abs_sum == 0 или почти 0
=> post-raise encode_state выдаёт пустые/нулевые тензоры.

base_out_abs_sum ≈ 0  И  q_canary_abs_sum != 0
=> dead-ReLU / OOD вход (основная гипотеза).
   Сеть жива (canary ненулевой), но post-raise states не проходят через ReLU base.

base_out_abs_sum != 0  И  q_a0..q_a3 == 0
=> q_head проблема. Base выдает features, но q_head мапит их в точный ноль.

q_canary_abs_sum == 0
=> q_net/q_head мёртвая даже на вектор единиц.
   Сеть не обучена или q_head инициализирован нулями и не получал градиент.

q_a0..q_a3 ненулевые после v4
=> v3 мог иметь stale/measurement bug.
   Перепроверить v_k/strategy_sum в том же прогоне.
```

---

## 5. Следующий шаг

Диагностический прогон с включённым lookahead:

- `sizing_lookahead_enabled=True`
- `sizing_q_target_diagnostics_enabled=True`

Затем:

```powershell
py -3 tools/checkpoint_tools.py full models/test4seed1/<checkpoint>.pt --games 1000 --num-states 500 --seed 1
```

Вернуть из JSON:

- `sizing_lookahead_diag.counts`
- `sizing_lookahead_diag.stats.{q_input_abs_sum, base_out_abs_sum, q_canary_abs_sum, q_a0, q_a1, q_a2, q_a3}`

После root-cause: отдельный ablation-патч (отключить source=2 или kind-filter). Не смешивать с диагностикой.

---

## 6. Прогон A: `sizing_q_bootstrap_target_net=true` (что мы уже протестировали)

> **Дата:** 2026-06-09
> **Конфиг:** `sizing_q_bootstrap_target_net: true` (дефолт в коммите 18)

### 6.1 Результат

Из `sizing_lookahead_diag`:

```text
attempts=33665, added=33665
final_state=0
bootstrap_debug_missing=0

q_input_has_nan=0
q_input_abs_sum mean=23.14
base_out_abs_sum mean=6.10
q_canary_abs_sum=0.0

q_a0=q_a1=q_a2=q_a3=0.0 (std=0)
v_k=0.0, target=0.0
using_target_net=33665 (все 100% source=2 шли через q_target_net)
next_is_hero=0
```

### 6.2 Интерпретация

| Метрика | Значение | Вывод |
|---------|----------|-------|
| `q_input_has_nan=0`, `q_input_abs_sum=23.14` | Вход нормальный | `encode_state` не NaN, не пустой |
| `base_out_abs_sum=6.10` | Base живой | ReLU/base выдаёт ненулевые features. Dead-ReLU гипотеза опровергнута |
| `q_canary_abs_sum=0.0` | Сеть на ones_like даёт 0 | `q_target_net` целиком выдаёт ноль |
| `q_a0..q_a3=0.0` | Q на post-raise = 0 | `strategy·Q = 0` → `v_k = 0` → `target = 0` |
| `using_target_net=33665` | 100% через target-net | Все lookahead bootstrap идут через `q_target_net` |

### 6.3 Root-Cause: stale `q_target_net`

```yaml
# config.yaml
q_target_update_interval: 100          # sync раз в 100 итераций
sizing_q_bootstrap_target_net: true    # bootstrap через target-net
```

Таймлайн прогона на 100 итераций:

```text
Init:    q_target_net = copy of q_net, обе q_head = zero-init → ровно 0
Iter 1-99: train_q_network обучает q_net.q_head → ненулевой
           НО _sync_q_target_net() НИ РАЗУ не вызывается (iter % 100 != 0)
           → q_target_net.q_head ОСТАЁТСЯ zero-init (ровно 0)
Lookahead: _bootstrap_sizing_value использует q_target_net
           → q_head(nonzero_base_out) = zero_init_weight @ base_out + zero_init_bias = 0
           → ровно 0 по всем 33665 сэмплам (std=0, canary=0)
Iter 100:  _sync_q_target_net() копирует обученный q_net → q_target_net становится ненулевым
           Checkpoint сохраняется ПОСЛЕ sync
           → q_compare \ action_q_model в report показывают ненулевые значения
```

Это объясняет кажущееся противоречие:
- `q_a* = 0` во время traversal (stale target-net)
- `q_compare.q_action` ненулевые в report (считается из checkpoint после sync)
- `q_net_vs_q_target_diff = 0` (sync только что сделал их идентичными)

---

## 7. Прогон B: `sizing_q_bootstrap_target_net=false` (тестируем сейчас)

> **Дата:** 2026-06-09
> **Конфиг:** `sizing_q_bootstrap_target_net: false`
> **Патч:** 1 строка в `config.yaml`
> **Всё остальное без изменений:** lookahead включён, v4-диагностика активна, loss не менялся.

### 7.1 Что меняется

```diff
- sizing_q_bootstrap_target_net: true
+ sizing_q_bootstrap_target_net: false
```

`_bootstrap_sizing_value` теперь использует online `q_net` (обученный каждую итерацию), а не stale `q_target_net`.

### 7.2 Ожидаемый результат

Если гипотеза верна:

```text
using_target_net → 0
q_canary_abs_sum → ненулевой (std > 0)
q_a0..q_a3 → не все равны 0
v_k / target → не все 0
source=2 mean_target_by_source_anchor → перестанет быть 0.0 по всем anchors
```

Если гипотеза НЕ верна:

```text
q_canary_abs_sum всё ещё 0 → q_net сам мёртвый (не обучен вообще)
base_out_abs_sum всё ещё 0 → проблемы с ReLU в online q_net
q_a* ненулевые, но v_k/target всё ещё 0 → проблема в target formation
```

### 7.3 Что делать дальше после Прогона B

| Результат Прогона B | Следующий шаг |
|---------------------|---------------|
| source=2 targets ненулевые | Причина подтверждена. Source=2 можно оставить в sizing-Q loss |
| source=2 targets всё ещё нулевые | q_net сам не обучен/мёртвый. Отключить source=2 навсегда или пересмотреть training |
| q_a* ненулевые, но target всё равно 0 | Баг в target formation для lookahead path |

### 7.4 Независимый roadmap (после Прогона B)

1. Legal-anchor mask на выборе sizing
2. Kind-filter: учить normal-Q только на `kind=NORMAL`
3. Replay hygiene: очистка/версионирование sizing-Q буфера
4. Verdict-gate по концентрации (энтропия / top-2 mass)
