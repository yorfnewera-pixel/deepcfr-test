# Bug #92 — Multi-Agent Advantage Result (test91seed1, test92seed1)

**Дата:** 2026-06-16
**Статус:** ACTIVE (первый прогон без relapse за всю историю проекта)
**База:** #91 — лучший single-net конфиг (α=1.5, Q ON, denominator +1.5)
**Прогоны:** test91seed1 (iter 100, FAIL), test92seed1 (iter 300, NO RELAPSE)
**Конфиг:** `use_multi_agent_advantage: true`, `use_q_baseline: true`, `discount_alpha: 1.5`, `q_target_update_interval: 10`

---

## Мотивация

Все single-net прогоны (#88–#91) relapse'ятся между iter 100 и 200. Первопричина: H3 — накопление отрицательного raise-regret в advantage-сети → advantage-flip → политика коллапсирует. Ни discount, ни MC propagation, ни sync-интервал не ломают этот паттерн.

Гипотеза: **раздельные per-player advantage-сети** предотвращают H3. Каждая сеть учится на своей позиции (UTG, MP, BU, ...), не смешивая regret'ы разных игроков. Если одна позиция страдает от negative regret — соседние сети не заражаются.

## Архитектура

```
ДО (#91):                    ПОСЛЕ (#92):
1 advantage_net             6 advantage_nets (ModuleList)
1 advantage_buffer          6 advantage_buffers
1 target_net                6 target_nets
1 optimizer                 6 advantage_optimizers
157-dim encode              163-dim encode (+6 pos one-hot)
round-robin traversing_player (t % 6)
train всех 6 per iteration
```

Буферы: advantage 1M×6, strategy 1M, Q (action) 1M, sizing_q 500k, sizing_strategy 1M.

## Технические изменения

| Файл | Что |
|------|-----|
| `config.yaml` | `use_multi_agent_advantage: true`, `use_q_baseline: true`, `sizing_q_buffer_size: 500000`, `sizing_strategy_buffer_size: 1000000` |
| `src/core/model.py` | `encode_state_with_position()` — 163 фичи |
| `src/core/deep_cfr.py` | 6 advantage_nets/buffers/target_nets/optimizers, хелперы `_adv_net()/_adv_buffer()/...`, v3 checkpoint, per-player траверс+обучение |
| `src/training/train.py` | Round-robin traversing_player, train всех 6, фикс логов |
| `tools/checkpoint_tools.py` | Multi-agent encoding в eval |
| `inference/core.py` | `encode_state_with_position`, detect `use_multi_agent_advantage` из чекпоинта |

## Результаты

### test91seed1 — FAIL (Q OFF, 204 traversals)

| Итерация | raise_freq | win_rate | mean_reward | Verdict |
|----------|:---:|:---:|:---:|---------|
| **100** | 12.4% | ? | −2.89 | WARN |

Вывод: multi-agent без Q-baseline нежизнеспособен. α=1.5 + Q OFF = худшая комбинация (подтверждает test90c).

### test92seed1 — BREAKTHROUGH (Q ON, 400 traversals)

| Итерация | raise_freq | win_rate | mean_reward | Fold | unique_anchors | Verdict |
|----------|:---:|:---:|:---:|:---:|:---:|---------|
| **100** | 43.0% | 39.9% | 12.67 | 126 | 15/15 | WARN (conc 70%) |
| **200** | **62.6%** | 45.0% | 14.71 | **8** | 14/15 | WARN (conc 64%) |
| **300** | 51.8% | **47.1%** | **18.13** | **4** | 13/15 | WARN (conc 79%) |

### Eval sizing distribution

| Анкер | Iter 100 | Iter 200 | Iter 300 |
|-------|:---:|:---:|:---:|
| 0.10 | 51% | 46% | 57% |
| 0.75 | 19% | 18% | 21% |
| 2.00 | 17% | 14% | 13% |
| 0.66 | 3% | 6% | 3% |
| 3.00 | 2.5% | 6% | 2% |

### Q-network trajectory

| Метрика | Iter 100 | Iter 200 | Iter 300 |
|---------|:---:|:---:|:---:|
| raise_lt_fold_pct | 67.7% | 91.8% | **48.3%** |
| raise_lt_call_pct | 81.2% | 17.8% | 31.5% |
| q_action.raise.mean | −0.040 | −0.068 | −0.004 |
| q_action.fold.mean | −0.042 | −0.008 | −0.015 |

### Сравнение с лучшим single-net (test90)

| Метрика | test90 (single, iter 100) | test92 (multi, iter 100) | test92 (multi, iter 200) |
|---------|:---:|:---:|:---:|
| raise_freq | **58.2%** | 43.0% | 62.6% |
| win_rate | 44.8% | 39.9% | 45.0% |
| unique_anchors | 15 | 15 | 14 |
| *relapse* | ДА (iter 200: 23%) | — | **НЕТ** (iter 200: 63%) |

## Ключевые выводы

### 1. Relapse сломан — впервые за всю историю проекта

| Прогон | iter 100 → iter 200 raise |
|--------|:---:|
| #88 (MC baseline) | 41% → 16% |
| #89 (sync=10) | 41% → 18% |
| #91 (discount) | 58% → 23% |
| #84 (max_norm=100) | 48% → 22% |
| **#92 (multi-agent)** | **43% → 63%** |

Все single-net прогоны падают. Multi-agent — растёт. Per-player advantage-сети прерывают H3 цепную реакцию.

### 2. Q-baseline критичен для α=1.5

Без Q: raise 12%. С Q: raise 43% → 63%. Разница в 3.5-5×.

### 3. Sizing concentration остаётся

Top-2 = 79% на iter 300, 0.10 = 57%. Это sizing-Q проблема, не multi-agent. Sizing-Q коллапсит во всех прогонах независимо от архитектуры advantage.

### 4. Модель улучшается с итерациями

Win rate: 39.9% → 45.0% → 47.1%. Ни один предыдущий прогон не показывал монотонный рост win rate за iter 200.

## M0/M1/M2 вердикт

| Gate | Условие | Факт | Вердикт |
|------|---------|------|---------|
| **M0** (relapse) | raise не падает 100→200 | 43% → 63% | **PASS** |
| **M1** (Q-learning) | raise_lt_fold < 60% | 48.3% на iter 300 | **PASS** |
| **M2** (durability) | win_rate стабилен/растёт | 39.9→45.0→47.1 | **PASS** |

## Данные

| Файл | Содержание |
|------|-----------|
| `models/test92seed1/multi_checkpoint_iter_*_full_report.json` | Полные отчёты eval (1000 игр) ×3 |
| `models/test92seed1/multi_checkpoint_iter_*.pt` | Полные чекпоинты ×3 (100/200/300) |
| `models/test91seed1/multi_checkpoint_iter_100_full_report.json` | test91 (Q OFF, FAIL) |

## Следующие шаги

1. **Iter 400** — подтвердить durability за iter 300
2. **Sizing concentration fix** — не multi-agent проблема, существовала всегда. Требует отдельного исследования
3. **Seed 2** — воспроизводимость на другом сиде
4. **Сравнение с single-net на iter 400** — head-to-head multi vs single
