# DeepCFR Poker

Экспериментальный проект покерного ИИ для **No-Limit Texas Hold'em** на основе Deep Counterfactual Regret Minimization (Deep CFR).

Активный режим — HU D2CFR anchored: две независимые advantage-ноги собирают
historical reservoir, затем общая actor-conditioned strategy-сеть обучает
среднюю policy. Six-max остаётся экспериментальной multiplayer-адаптацией.

## Активная архитектура и совместимость

Активный HU-конфиг использует `card_context_v2`: после отдельных card- и
context-encoder применяется нелинейное fusion. `card_context_v1` сохраняется
как аддитивная архитектура только для явно выбранной загрузки legacy-артефактов.
Полные checkpoint v1 и v2 несовместимы между собой; для продолжения обучения
нужно выбирать checkpoint с совпадающей архитектурой.

`card_context_v2` — практическое улучшение реализации, а не утверждение, что
такое нелинейное fusion описано в исходной статье D2CFR.

HU сохраняет full и light checkpoint каждые 5000 итераций в раздельных
namespace. Для full checkpoint milestone-интервал — 50000 итераций; retention
оставляет два последних и два milestone checkpoint. Это не смешивается с
retention six-max.

При создании состояния через `State.from_mid_hand` необходимо явно передать
`current_player`. Неразрешённое состояние `Showdown` для такого восстановления
отклоняется.

## Возможности

- HU D2CFR anchored для Texas Hold'em;
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

Запуск короткого HU smoke на CPU (настройки HU D2CFR находятся в `config.yaml`):

```bash
python -m src.training.train --self-play-multi --iterations 10 --traversals 10 --save-dir models
```

Для ускорения на совместимой видеокарте укажите устройство:

```bash
python -m src.training.train --self-play-multi --device cuda
```

Full checkpoint сохраняет training state; light checkpoint содержит только
strategy и предназначен для runtime. После исправления правил HU начинайте
новое обучение, а не продолжайте модели, созданные до этих изменений.

HU-обучение использует фиксированный игровой контракт: SB 1, BB 2, стартовый
stack 200 (`stack_depth_bb` 100). `config.yaml` не может менять ставки или stack:
для других значений требуется изменение game-contract в коде. Старые HU
checkpoint могут быть несовместимы с этим контрактом. Параметр
`advantage_reward_scale: 200` масштабирует D2CFR targets и не задаёт stack.

Переход HU → 6-max — это новый запуск с пустыми replay-буферами. Разрешён
только warm-start `card_encoder` через `teacher_transfer_checkpoint`; HU
replay, optimizers, strategy head и номер итерации не переносятся.

Пример six-max запуска с HU teacher:

```bash
python -m src.training.train --self-play-multi --num-players 6 --trainable-players 6 \
  --teacher-transfer-enabled --teacher-transfer-checkpoint models/hu_checkpoint_final.pt
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
