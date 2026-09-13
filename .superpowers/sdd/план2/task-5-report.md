# Task 5 — отчёт

## Изменения

- Активный HU-конфиг переведён на `card_context_v2`.
- Явно записаны HU retention: interval `5000`, milestone `50000`, recent `2`, milestones `2`.
- Startup-лог D2CFR показывает выбранный loss и статус Huber; для активного MSE-режима: `D2CFR loss: mse`, `Huber delta: inactive`.
- `readme.md` описывает различия v1/v2, несовместимость full checkpoint, изоляцию HU full/light retention и ограничения `State.from_mid_hand`.

## Проверки

- RED: `python -m pytest -q tests/test_config_flags.py tests/test_d2cfr_hu.py` — 2 ожидаемых падения до правок.
- GREEN: `python -m pytest -q tests/test_config_flags.py tests/test_d2cfr_hu.py tests/test_card_context_architecture.py tests/test_checkpoint_lifecycle.py` — 85 passed.
- `git diff --check` — без ошибок whitespace.
