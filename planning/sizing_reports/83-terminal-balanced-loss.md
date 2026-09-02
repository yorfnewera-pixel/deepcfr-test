# #83 — terminal-balanced Q-loss (ветка B после #82)

> **Статус:** CLOSED (WARN) — test81seed1: raise_freq=3.0%, unique_anchors=5/15, коллапс не сломан. Поведенческий прорыв test80b (max_norm=100, OK) пропущен в #82 — balanced loss при max_norm=1.0 не на что опереться. Будет перезапущен как #85 на базе max_norm=100.
> **Связанные:** #82 (max_norm ablation, B confirmed), #81 (reward-grounding asymmetry H1)

---

## 0. Pre-registered критерии (зафиксированы ДО запуска)

### Мотивировка

#82 показал: fit ratio `|q_pred|_term / |target|_term` стабилен на **~0.02** независимо от max_norm (1→10→100). Q-сжатие — следствие count-weighted MSE: bootstrap-сэмплы (97-99% батча) численно доминируют loss, terminal (1-3%) не могут перевесить даже без clip.

Лечим **композицию** лосса: separate loss с фиксированной альфой 0.5 — terminal получает 50% веса независимо от своей доли в батче.

### Дизайн

| Параметр | Значение |
|----------|----------|
| `q_terminal_balanced_loss_enabled` | true |
| `q_terminal_loss_alpha` | 0.5 |
| `q_grad_clip_max_norm` | 1.0 (base) |
| seed, итерации, всё остальное | = test79seed1 |

### Первичная метрика — fit ratio

```
fit_ratio = |q_pred|_term / |target|_term
```

Измеряется из `[Q-DIAG]` лога (fit_ratio поле) + верифицируется аудитом на финальном чекпоинте.

**База (test79seed1, test80a, test80b):** fit_ratio ≈ **0.02** (не зависит от max_norm).

| Исход | Условие | Интерпретация |
|-------|---------|---------------|
| **Успех** | fit_ratio ≥ 0.5 | Композиция лосса — ключевой механизм. Q масштабируется к реальным наградам. Next: Huber + нормализация для стабильности, затем n-step/MC. |
| **Частичный успех** | 0.2 ≤ fit_ratio < 0.5 | Балансировка помогает, но недостаточно. Next: alpha=0.7 ИЛИ max_norm=10 (одно из, не оба сразу). |
| **Провал** | fit_ratio < 0.1 | Механизм глубже композиции лосса. Next: нормализация targets (Pop-Art/z-score). |

### Диагностическое правило

Если в `[Q-DIAG]` raw_grad вырос в разы, а fit_ratio стоит на месте → upweighting упёрся в clip. Тогда (и только тогда) test81bseed1 с max_norm=10. **НЕ стартовать с комбинации** — одна переменная за раз.

### Ожидаемые побочные эффекты (НЕ провал)

- **q_loss вырастет на порядки** — арифметика: terminal с огромными residuals теперь весит 50%. Сравнивать q_loss с базой бессмысленно.
- **Control variate может зашуметь advantage** — несмещённость сохраняется при ЛЮБОМ Q, но variance временно возрастёт. Смотреть динамику, не точку.
- **За 100 итераций поведение (raise_freq, win_rate) может не сдвинуться** — не делать выводов о провале.

### Вторичные метрики (записать, НЕ критерий успеха)

| Метрика | Примечание |
|---------|------------|
| raise_freq | За 100 итераций может не успеть сдвинуться |
| win_rate (vs random) | |
| fold-vs-raise gap в action-Q | |
| bootstrap fit (`\|q_pred\|_boot / \|target\|_boot`) | Может ухудшиться — ожидаемо при перекосе веса на terminal |
| raw_grad_mean/max | Диагностика clip-лимита |

---

## 1. Результаты

**test81seed1: WARN.** raise_freq=3.0%, unique_anchors=5/15 — коллапс не сломан.

fit_ratio из [Q-DIAG] лога не сохранился (логирование в файл добавлено post-hoc, Step 0c).

### 1.1 Поведенческие метрики (full_report.json, iter 100)

| Метрика | test79seed1 (base) | test81seed1 |
|---------|-------------------|-------------|
| raise_freq | 1.5% | **3.0%** |
| unique_anchors | 7/15 | 5/15 |
| verdict | WARN | WARN |

---

## 2. Вывод: ПРОВАЛ

Balanced loss α=0.5 при max_norm=1.0 не сломал коллапс. Причина: после #82 стало ясно что fit_ratio ~0.02 не зависит от max_norm — Q всё ещё сжат в 40× — и на такой «глухой» Q balanced loss не даёт эффекта. Нужна живая база (max_norm=100, где Q дифференцирует действия).

---

## 3. Next Steps

Эксперимент будет перезапущен как **#85** на базе max_norm=100 после консолидации #84.

---

## Ссылки

- `sizing_reports/82-max-norm-ablation.md` — #82 результаты, обоснование ветки B
- `sizing_reports/81-action-q-role-legal-diagnostics.md` — grad-clip audit, H1/H2
- `SIZING.md` — сводная таблица
