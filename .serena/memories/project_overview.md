# DeepCFR Poker AI — Обзор проекта

## Назначение
Реализация Deep Counterfactual Regret Minimization для 6-игрокового безлимитного Texas Hold'em. Построена на базе библиотеки `pokers` (патченный форк).

## Технологический стек
- **Python 3.11** (виртуальное окружение `.venv`)
- **PyTorch** — нейросети (PokerNetwork: shared body + action head + sizing head)
- **pokers** — покерный движок (Rust/Python, собирается через maturin из локального `pokers/`)
- **numpy** — вычисления
- **PyQt5** — GUI
- **tensorboard** — мониторинг тренировки
- **pytest** — тестирование

## Ключевые архитектурные решения
- 3 типа действий: Fold, Check/Call, Raise + непрерывный sizing (SAC-style Tanh-squashed Gaussian)
- PrioritizedMemory для advantage-обучения
- PolicyGradientMemory для strategy-обучения
- Опциональное opponent modeling (GRU-based ActionHistoryEncoder)

## Структура кода
```
src/
  core/          — PokerNetwork (model.py), DeepCFRAgent + памяти (deep_cfr.py)
  agents/        — RandomAgent
  training/      — train.py, train_with_opponent_modeling.py, train_mixed_with_opponent_modeling.py
  opponent_modeling/ — opponent_model.py, deep_cfr_with_opponent_modeling.py
  utils/         — settings.py (STRICT_CHECKING), logging.py
tests/           — test_pokers_regressions.py, test_training_regressions.py
scripts/         — play.py, poker_gui.py, visualize_tournament.py, telegram_notifier.py
configs/         — __init__.py (пусто)
pokers/          — патченный форк покерного движка (Rust/maturin)
models/          — сохранённые чекпойнты
logs/            — логи tensorboard
```

## Фазы тренировки
1. Phase 1: против random opponents
2. Phase 2: self-play против фиксированного чекпойнта
3. Phase 3: mixed training против пула чекпойнтов
