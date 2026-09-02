# Баг-репорт #41: Hybrid OS блокируется после resume со зрелого checkpoint из-за пустого q_buffer

## Серьёзность: MEDIUM

## Категория: Логический баг / Resume training

## Дата обнаружения: 2026-05-16

---

## Описание

При продолжении обучения с зрелого checkpoint (например, итерация 6000) Q-сеть загружается с обученными весами, но `q_buffer` остаётся пустым (он не сохраняется в checkpoint).

`_should_use_outcome_sampling` требует `len(q_buffer) >= min_q_buffer_size (5000)` для включения OS. После resume с пустым buffer OS **не включится**, пока q_buffer не накопит 5000+ переходов — это 1-2 итерации полного External Sampling на мульти-пот постфлопе.

Загруженные Q-веса уже достаточно хороши для baseline, но порог по buffer не учитывает этот факт.

---

## Затронутые файлы

- `src/core/deep_cfr.py` — `__init__`, `_load_checkpoint`, `_should_use_outcome_sampling`
- `src/utils/config.py` — новые дефолты
- `config.yaml` — новые параметры
- `src/training/train.py` — диагностика
- `tests/test_hybrid_outcome_sampling.py` — 4 новых теста

---

## Фикс

### 1. Флаг `q_loaded_from_checkpoint`

Добавлен в `__init__` (`False` по умолчанию). Выставляется в `True` при загрузке q_net из checkpoint, `False` при отсутствии.

### 2. Config параметры

```yaml
hybrid_os_allow_loaded_q_without_buffer: true
hybrid_os_loaded_q_min_iteration: 5000
```

Порог `5000` — защита от "молодых" checkpoint. Если iteration_count < 5000, OS не включится только по факту loaded Q.

### 3. Два пути готовности Q в `_should_use_outcome_sampling`

```python
q_ready_from_buffer = (
    self.iteration_count >= self.hybrid_os_q_warmup_iterations
    and len(self.q_buffer) >= self.hybrid_os_min_q_buffer_size
)

q_ready_from_checkpoint = (
    self.hybrid_os_allow_loaded_q_without_buffer
    and self.q_loaded_from_checkpoint
    and self.iteration_count >= self.hybrid_os_loaded_q_min_iteration
)

if not (q_ready_from_buffer or q_ready_from_checkpoint):
    return False
```

- **С нуля**: OS ждёт warmup + buffer — поведение не меняется
- **С зрелого checkpoint**: OS включается сразу по loaded Q-весам
- **С молодого checkpoint**: OS не включается (iteration < 5000)

### 4. Диагностика

- TensorBoard: `Q/LoadedFromCheckpoint`
- Консоль: `Q buffer: N, loaded_from_ckpt=True/False`

---

## Тесты

| Тест | Проверяет |
|------|-----------|
| `test_os_enabled_with_loaded_q_and_empty_buffer` | loaded Q + пустой buffer + iteration >= 5000 → OS включён |
| `test_os_blocked_without_loaded_q_and_empty_buffer` | нет loaded Q + пустой buffer → OS выключен |
| `test_os_blocked_with_loaded_q_low_iteration` | loaded Q + iteration < 5000 → OS выключен |
| `test_q_loaded_from_checkpoint_default_false` | флаг = False по умолчанию |

---

## Критерии успеха

1. При resume с зрелого checkpoint OS включается с первой итерации
2. При обучении с нуля поведение не меняется
3. Молодой checkpoint (iteration < 5000) не активирует OS через checkpoint-путь
4. q_buffer продолжает наполняться и Q-сеть продолжает дообучаться
5. 21/21 тестов проходят
