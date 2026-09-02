# Диагностические инструменты

Скрипты в этой папке предназначены для проверки sizing-моделей и связанных данных без изменения checkpoint.

По умолчанию имена отчётов содержат фактическую итерацию `_iterN`, извлечённую из метаданных checkpoint или имени файла. Для `cross_board_analysis.py` итерация берётся из checkpoint либо входного JSON. Явно переданный путь вывода сохраняется без изменений.

## Требования

Используйте **полный checkpoint**, содержащий все необходимые состояния моделей и метаданные. Команды запускаются из корня проекта.

## Рекомендуемый порядок

1. Сформировать диагностику через текущий API:

```bash
python tools/diagnose_sizing_current_api.py --checkpoint models/path/to/checkpoint.pt
```

2. Выполнить cross-board анализ полученного результата:

```bash
python tools/cross_board_analysis.py --input sizing_current_api_diagnostic.json
```

Можно сразу передать checkpoint в cross-board анализ:

```bash
python tools/cross_board_analysis.py --checkpoint models/path/to/checkpoint.pt
```

## Остальные диагностики

Перед запуском уточните доступные параметры через `--help`:

```bash
python tools/diagnose_sizing_buffer_targets.py --help
python tools/diagnose_sizing_flatness.py --help
python tools/diagnose_sizing_q.py --help
python tools/diagnose_advantage_sizing_vs_raw_regrets.py --help
python tools/diagnose_prefloor_slot_logits.py --help
python tools/diagnose_sizing_current_api.py --help
python cross_board_analysis.py --help
```

Назначение:

- `diagnose_sizing_current_api.py` — формирует базовую диагностику sizing через текущий API.
- `cross_board_analysis.py` — сравнивает sizing-поведение между досками.
- `diagnose_sizing_buffer_targets.py` — проверяет целевые значения sizing в буфере.
- `diagnose_sizing_flatness.py` — анализирует чрезмерно плоское распределение sizing.
- `diagnose_sizing_q.py` — диагностирует Q-значения sizing.
- `diagnose_advantage_sizing_vs_raw_regrets.py` — сравнивает advantage-sizing с исходными regret-значениями.
- `diagnose_prefloor_slot_logits.py` — проверяет logits sizing-слотов до floor-обработки.


## Flatness aggregate reports

Run from the project root either as a file or module. Relative output paths are resolved from the project root. Without `--output`, flatness writes `sizing_flatness_iterN.json` and a matching `.txt`; an explicit output filename is preserved.

```bat
py -3 tools\diagnose_sizing_flatness.py --checkpoint models\test107v3\multi_checkpoint_iter_5.pt --output test107v3-2\sizing_flatness_iter5.json
py -3 tools\diagnose_sizing_flatness.py --checkpoint models\test107v3\multi_checkpoint_iter_10.pt --output test107v3-2\sizing_flatness_iter10.json
py -3 tools\diagnose_sizing_flatness.py --checkpoint models\test107v3\multi_checkpoint_iter_15.pt --output test107v3-2\sizing_flatness_iter15.json
```

Cross-board help works in both forms:

```bat
py -3 tools\cross_board_analysis.py --help
py -3 -m tools.cross_board_analysis --help
```
