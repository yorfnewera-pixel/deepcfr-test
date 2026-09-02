# DeepCFR Poker

Экспериментальный проект покерного ИИ для **No-Limit Texas Hold'em** на основе Deep Counterfactual Regret Minimization (Deep CFR).

Проект обучает стратегию self-play для шести игроков, поддерживает исторический пул checkpoint-оппонентов, сохранение моделей и запуск игры против обученных агентов.

## Возможности

- Deep CFR / DCFR для 6-max Texas Hold'em;
- фиксированное пространство из шести действий: fold, check, call, raise 0.5 pot, raise pot, all-in;
- self-play с checkpoint-оппонентами;
- TensorBoard-логи и сохранение checkpoint-файлов;
- консольная игра, GUI на PyQt5 и визуализация турниров.

## Установка

Требуется Python 3.8+.

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd deepcfr-test
python -m venv .venv
```

Активируйте виртуальное окружение и установите зависимости:

```bash
python -m pip install -r requirements.txt
python -m pip install ./pokers
```

`pokers` устанавливается из локальной папки: она содержит патчи, необходимые этому проекту.

## Быстрый старт

Запуск короткого обучения на CPU:

```bash
python -m src.training.train --self-play-multi --iterations 10 --traversals 10 --save-dir models
```

Для ускорения на совместимой видеокарте укажите устройство:

```bash
python -m src.training.train --self-play-multi --device cuda
```

Продолжение обучения из полного checkpoint:

```bash
python -m src.training.train --self-play-multi --initial-checkpoint models/multi_checkpoint_iter_1000.pt
```

## Игра и оценка

Консольная игра против моделей из папки:

```bash
python -m scripts.play --models-dir models
```

Графический интерфейс:

```bash
python -m scripts.poker_gui --models_folder models
```

Турнир между checkpoint-моделями:

```bash
python -m scripts.visualize_tournament --checkpoints models/model_a.pt models/model_b.pt --num-games 100
```

## Структура

| Путь | Назначение |
| --- | --- |
| `src/core/` | Реализация Deep CFR, буферы и игровая логика |
| `src/training/` | Циклы обучения и сохранение checkpoint |
| `src/runtime_search/` | Поиск и оценка решений во время игры |
| `src/evaluation/` | Оценка стратегий и парные эксперименты |
| `scripts/` | Игра, GUI и вспомогательные сценарии |
| `tests/` | Автоматические тесты |

Параметры обучения находятся в `config.yaml`.

## Лицензия

Проект распространяется по лицензии MIT. См. [LICENSE.txt](LICENSE.txt).
