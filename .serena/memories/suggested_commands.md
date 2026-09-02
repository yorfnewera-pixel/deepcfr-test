# Рекомендуемые команды

## Активация venv (Windows PowerShell)
```
& "C:\Users\Cassmall\Desktop\deepcfr-test\.venv\Scripts\Activate.ps1"
```

## Тестирование
```powershell
python -m pytest tests/test_pokers_regressions.py tests/test_training_regressions.py -q
```

## Сборка pokers (при изменении движка)
```powershell
cd pokers; maturin develop --release
# или: py -m pip install pokers/
```

## Тренировка
```powershell
# Phase 1: random opponents
python -m src.training.train --iterations 1000 --traversals 200 --log-dir logs/phase1 --save-dir models/phase1

# Phase 2: self-play
python -m src.training.train --checkpoint models/phase1/checkpoint_iter_1000.pt --self-play --iterations 2000 --traversals 400 --log-dir logs/selfplay --save-dir models/selfplay

# Phase 3: mixed
python -m src.training.train --mixed --checkpoint-dir models --model-prefix t_ --refresh-interval 1000 --num-opponents 5 --iterations 10000 --traversals 400 --log-dir logs/mixed --save-dir models/mixed
```

## Игра
```powershell
python scripts/play.py --models-dir models/phase1
python scripts/poker_gui.py --models_folder models/phase1
```

## Мониторинг
```powershell
tensorboard --logdir=logs
```

## Установка зависимостей
```powershell
pip install -r requirements.txt
```

## Системные утилиты (Windows PowerShell)
- `Get-ChildItem` — ls
- `Select-String` — grep
- `Set-Location` — cd
- `git` — версия контролируется
