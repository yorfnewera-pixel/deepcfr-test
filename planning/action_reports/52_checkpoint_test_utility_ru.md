═══════════════════════════════════════════════════════════════════
  BUG #52 — Утилита проверки чекпоинта: детекция смерти модели
  Дата: 01.06.2026 (обновлено)
  Статус: Реализовано
  Инструмент: checkpoint_test/check_ckpt.py
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

Создана универсальная утилита для проверки любого чекпоинта
на «смерть» модели. Поддерживает light и full форматы.

Проблема: без систематической проверки невозможно отличить живую
модель от коллапсировавшей — стохастический семплинг маскирует
деградацию стратегии шумом распределения. Хуже того: strategy_net
(argmax) может показывать DEAD, хотя CFR-ядро (advantage_net)
ещё живо — это ложное срабатывание.

Утилита запускает eval-игры против RandomAgent и собирает
распределение действий и сайзингов в 4 режимах.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — КРИТИЧЕСКИЙ
  Детерминированный режим вскрывает скрытый коллапс стратегии
═══════════════════════════════════════════════════════════════════

  Проблема: strategy_net → softmax → распределение вероятностей.
  При deterministic=False (sampling) даже мёртвая модель может
  случайно выбирать Raise из хвоста распределения (1-2% вероятности).
  Это создаёт ложное впечатление работающей стратегии.

  Пример — чекпоинт iter_300:

  Стохастика (sampling):
    Fold:       28.7%
    Check/Call: 50.3%
    Raise:      20.9%   ← выглядит живым
    Sizing:     15/15 anchors

  Детерминированный (argmax):
    Fold:        0.0%
    Check:      20.5%
    Call:       79.5%
    Raise:       0.0%   ← МЁРТВ — ни одного рейза
    Sizing:      0/15 anchors

  Модель коллапсировала в Call-станцию: всегда коллирует, никогда
  не фолдит и не рейзит. +7 bb/100 против рандома — не заслуга
  стратегии, а эксплуатация блефов случайного оппонента.

  Решение: добавлен флаг --deterministic, который использует
  argmax вместо sampling при выборе действия.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — КРИТИЧЕСКИЙ
  Regret-matching из advantage_net: детекция ложного DEAD
═══════════════════════════════════════════════════════════════════

  Проблема: strategy_net — это дистиллированная политика.
  Её argmax может показывать DEAD (0% raise), даже если CFR-ядро
  (advantage_net) ещё производит ненулевые regrets для Raise.
  Это ложное срабатывание — дистилляция сломана, но алгоритм жив.

  Доказательство — чекпоинт iter_700, все 4 режима:

  ┌───────────────────────────────┬──────────┬─────────────┐
  │ Режим                         │ Raise    │ Verdict      │
  ├───────────────────────────────┼──────────┼─────────────┤
  │ strategy_net (sampling)       │ 17.7%    │ OK           │
  │ strategy_net (argmax)         │ 0%       │ DEAD ← ложь  │
  │ advantage_net (sampling)      │ 3.9%     │ WARN         │
  │ advantage_net (argmax)        │ 4.3%     │ WARN         │
  └───────────────────────────────┴──────────┴─────────────┘

  Вывод: CFR-ядро живо (advantage_net видит ценность рейза),
  но strategy distillation коллапсировал. «DEAD» в strategy_net
  argmax — артефакт метрики, а не смерть алгоритма.

  Решение: добавлен флаг --regret-matching и поддержка полных
  чекпоинтов. При --regret-matching действия выбираются через
  regret matching из advantage_net, минуя strategy_net.

  Реализация — _choose_action_regret_matching():
    advantages = advantage_net(state)       # [fold, check, call, raise]
    pos = max(advantages, 0) * legal_mask
    probs = pos / pos.sum()                 # regret matching
    action = sample(probs) или argmax(probs)

  Сайзинг: через strategy_sizing_net (доступен в любом формате).

  Формат чекпоинта определяется автоматически:
    - Light (_light.pt): strategy_net + strategy_sizing_net
    - Full (.pt): дополнительно advantage_net
  При --regret-matching на light-чекпоинте — ошибка с пояснением.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — ВЫСОКИЙ
  Отсутствие унифицированного инструмента проверки чекпоинтов
═══════════════════════════════════════════════════════════════════

  До: каждый агент создавал свой скрипт для теста чекпоинта:
    - test_ckpt_100.py (108 строк) — только iter_100
    - test_ckpt_800_light.py (337 строк) — только iter_800
    - analyze_ckpt.py (88 строк) — сравнение sizing
    - diagnose_sizing.py (795 строк) — глубокая диагностика

  Все скрипты захардкожены под конкретный чекпоинт или аспект.

  После: checkpoint_test/check_ckpt.py — один скрипт для любого
  чекпоинта (light или full), любой итерации, с единым форматом.

  Расположение: checkpoint_test/ — отдельная папка, чтобы агенты
  не создавали новые скрипты для каждого теста.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — СРЕДНИЙ
  Критерии вердикта и пороги
═══════════════════════════════════════════════════════════════════

  ┌──────────────────────┬─────────────────────────────────────────┐
  │ Вердикт              │ Критерий                                │
  ├──────────────────────┼─────────────────────────────────────────┤
  │ DEAD                 │ raise_freq = 0% ИЛИ NaN/Inf в весах     │
  │ WARN                 │ raise_freq < 5% ИЛИ unique_sizes < 3    │
  │ OK                   │ raise_freq >= 5% И unique_sizes >= 3    │
  └──────────────────────┴─────────────────────────────────────────┘

  unique_sizes — количество различных сайзинг-анкеров (из 15),
  которые модель использовала хотя бы раз.

  При NaN/Inf в весах eval-игры пропускаются — вердикт DEAD сразу.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — НИЗКИЙ
  Структура и использование
═══════════════════════════════════════════════════════════════════

  Файл: checkpoint_test/check_ckpt.py

  Аргументы:
    checkpoint          — путь к чекпоинту (_light.pt или полный .pt)
    --games N           — количество eval-игр (по умолчанию 1000)
    --seed S            — random seed (по умолчанию 42)
    --deterministic     — argmax вместо sampling
    --regret-matching   — advantage_net + regret matching (только full)
    --output json       — сохранить JSON-отчёт
    --verbose           — показать все 15 бакетов сайзинга

  4 режима оценки:
    ┌───────────────────────────────────────┬──────────────────────┐
    │ Флаги                                 │ Действия из          │
    ├───────────────────────────────────────┼──────────────────────┤
    │ (по умолчанию)                        │ strategy_net sampling│
    │ --deterministic                       │ strategy_net argmax  │
    │ --regret-matching                     │ advantage_net RM     │
    │ --regret-matching --deterministic     │ advantage_net argmax │
    └───────────────────────────────────────┴──────────────────────┘

  Запуск:
    # Обычная проверка light:
    py -m checkpoint_test.check_ckpt models\multi\multi_checkpoint_iter_N_light.pt --games 3000

    # Детерминированный (диагностика):
    py -m checkpoint_test.check_ckpt models\multi\multi_checkpoint_iter_N_light.pt --games 3000 --deterministic

    # Regret-matching из advantage_net (нужен полный чекпоинт):
    py -m checkpoint_test.check_ckpt models\multi\multi_checkpoint_iter_N.pt --games 3000 --regret-matching

    # Regret-matching + argmax:
    py -m checkpoint_test.check_ckpt models\multi\multi_checkpoint_iter_N.pt --games 3000 --regret-matching --deterministic

    # С JSON-отчётом:
    py -m checkpoint_test.check_ckpt models\multi\multi_checkpoint_iter_N.pt --games 3000 --output json

  Этапы выполнения:
    1. Weight sanity — проверка NaN/Inf во всех тензорах
       (strategy_net, strategy_sizing_net, advantage_net)
    2. Eval vs Random — N игр против RandomAgent (6-max NLHE)
    3. Verdict — агрегация и вердикт OK/WARN/DEAD

  Пример вывода (advantage_net argmax):
    === Checkpoint Check: multi_checkpoint_iter_700.pt ===
      Iteration:     700
      Mode:          advantage_net (argmax)
      Games played:  50
      Win rate:      10.0%
      Mean reward:   +7.33 bb
    --- Actions ---
      Fold:           39  (55.7%)
      Check/Call:     28  (40.0%)
      Raise:           3  (4.3%)
    --- Sizing (3/15 anchors used) ---
        0.75       1
        3.00       1
        2.25       1
    --- Verdict ---
      Status: WARN
      Reason: raise_freq = 4.3% < 5%

  Вывод:
    - Текстовый отчёт в консоль
    - JSON-отчёт при --output json: {checkpoint_name}_check.json
      Поле mode содержит полное описание: "strategy_net (sampling)",
      "advantage_net (argmax)", и т.д.


═══════════════════════════════════════════════════════════════════
  ВЫВОДЫ
═══════════════════════════════════════════════════════════════════

  1. Детерминированный режим — обязательный этап проверки.
     Без него невозможно отличить живую модель от шума семплирования.

  2. Regret-matching из advantage_net — защита от ложного DEAD.
     strategy_net argmax может показывать 0% raise, даже если CFR-ядро
     живо. Только advantage_net даёт истинную картину.

  3. iter_700: CFR-ядро живо, strategy distillation сломан.
     advantage_net видит raise (3-4%), strategy_net — нет (0%).
     Проблема в дистилляции (train_strategy_network), не в CFR.

  4. Рекомендация: проверять каждый чекпоинт во всех 4 режимах.
     Если strategy_net argmax = DEAD, а advantage_net — нет,
     проблема в strategy distillation, а не в обучении CFR.


═══════════════════════════════════════════════════════════════════
  СВЯЗАННЫЕ БАГИ
═══════════════════════════════════════════════════════════════════

  Bug #15  — Fold Collapse: модель перестала фолдить
  Bug #47  — k-sample within-state advantage + sizing split
  Bug #50  — фиксированная сетка сайзингов (15 анкеров)
  Bug #51  — оптимизация потоков данных, сетей и буферов
  TBD #53  — Q-baseline для 6-max: отключение и переобучение
  TBD #54  — DCFR+ вес усреднения: (t/T)^γ вместо (t/T*2)^(γ/2)
═══════════════════════════════════════════════════════════════════
