# Runtime Search Turn/River Validation Plan

**Goal:** Получить воспроизводимую диагностику blueprint и S4 search на turn/river без изменения обучения или checkpoint формата.

**Architecture:** `diagnostics.py` собирает action frequency, probability mass, entropy и latency на естественных раздачах, а также на deterministic constructed spots. Отдельный CLI загружает только `strategy_only` checkpoint и сохраняет JSON-результат. Короткий S4 probe проверяет legal decision и latency, но не трактует uniform-belief EV как доказательство силы solver.

## Constraints

- Не менять training loop, checkpoint format, сеть или action space.
- Использовать фиксированный seed и явно отражать incomplete rollout в результате.
- Положительный paired EV до S6 не является kill-gate.

## Tasks

### Task 1: Diagnostic API и тесты

- [x] Написать тест для воспроизводимой статистики естественных поздних улиц.
- [x] Добавить dataclass-результаты, сбор natural/constructed метрик и S4 probe.
- [x] Проверить legal action, конечные probability/entropy/latency и JSON-serializable результат.

### Task 2: CLI и checkpoint-2000

- [x] Добавить CLI, читающий light checkpoint и сохраняющий отдельный JSON report.
- [x] Прогнать короткую диагностику на `models/test110/light_checkpoint_iter_2000.pt`.
- [x] Зафиксировать вывод, границы интерпретации и команду повторного запуска в solver report.
