# 8-max MTT postflop card abstraction V1

## Цель

Построить отдельную офлайн-абстракцию карточной силы для одного стола MTT
максимум на восемь игроков. Она группирует только `hero cards + public board`
по тому, как меняется ожидаемая доля общего банка против одного--семи
соперников.

Результат не является MTT policy, информационным состоянием, ICM-моделью или
готовой стратегией. Он будет использован отдельным будущим этапом вместе с
публичной историей, стеками, позициями и tournament value.

## Границы

Входит:

- две карты hero и board длиной 0, 3, 4 или 5;
- lossless preflop-таблица из 169 карточных классов;
- отдельные postflop-модели для flop, turn и river;
- профиль equity против 1, 2, ..., 7 active opponents;
- uniform random legal hands как единственный reference range;
- suit-isomorphism, воспроизводимое sampling, clustering, validation и
  versioned artifacts.

Не входит:

- MTT payouts, ICM, stacks, позиции, ante, pot, betting history и action mask;
- реальные range, зависящие от истории;
- side-pot utility;
- runtime bucket resolver и вызов estimator из CFR traversal;
- изменение HU game engine, его checkpoint или MTT policy.

## Совместимость и границы рефакторинга

Существующий `src/card_abstraction/` переводится на новый 8-max контракт;
сохранение HU feature, HU artifact API и CLI не требуется. Старые 27-мерные
HU artifacts несовместимы, не загружаются и не используются. Допускается
переименовать entry point в `tools/build_8max_postflop_abstraction.py`.

Новые artifacts располагаются в `artifacts/8max_postflop_abstraction/`, чтобы
не смешиваться с историческими HU файлами. Чистые карточные примитивы и
showdown evaluator переиспользуются. Это не разрешение удалять или менять
HU game engine, HU tests либо HU checkpoints: рефакторинг ограничен прежней
карточной abstraction и её artifacts.

## Вход и канонизация

`CardSituation` содержит только две hero-карты, board и street. Все известные
карты различны. Канонический ключ применяет одну из 24 общих перестановок
мастей ко всем известным картам и выбирает лексикографический минимум.

Количество соперников не входит в ключ: оно является осями единого feature.
Руки соперников -- временные сэмплы и никогда не становятся частью входа
policy или canonical key.

Preflop по-прежнему перечисляет 1326 физических рук и отображает их в 169
карточных классов (13 pair, 78 suited, 78 offsuit). Это описание hand type,
а не полное 8-max preflop information set.

## Multiway equity

`opponent_counts` означает число соперников в гипотетическом общем showdown
банке, включая all-in игроков. Это reference-сценарии, а не runtime-флаг
игрока, способного сделать новую ставку.

Для фиксированных hero и полного board один showdown sample раздаёт без
повторов 14 карт семи opponent hands. Первые 2 карты формируют сценарий
против одного opponent, первые 4 -- против двух, и так до семи. Для каждого
`k` payoff hero -- доля общего банка:

- `0`, если хотя бы один из первых `k` opponent сильнее hero;
- `1 / (1 + tied_opponents)`, если hero делит лучшую руку;
- `1`, если hero единственный победитель.

Нельзя усреднять pairwise победы. Вложенная раздача гарантирует, что в одном
sample доля банка для `k + 1` не больше доли для `k`.

На flop и turn для каждого из `runout_samples` future runout оценивается
условная common-pot equity посредством `opponent_samples` совместных раздач
семи рук. Результат имеет форму `(runout_samples, 7)`.

На river все семь координат вычисляются одной вложенной multiway
Monte-Carlo-выборкой. Exact HU river сохраняется только как независимый
validation oracle, а не как первая координата feature. River result имеет
форму `(1, 7)`; нулевой std внутри feature не означает нулевую
MC-погрешность. Последняя сохраняется в diagnostics.

Seed выводится из версии estimator, master seed, canonical key и типа
sampling. Существует отдельный поток выбора runout и отдельный
deterministic opponent stream для каждого стабильного индекса runout. Поэтому
порядок обработки, workers, chunk size и resume не меняют samples и feature;
увеличение budget сохраняет prefix consistency. Размер budget не входит в
stream seed, но входит в cache key и manifest.

## Feature

Для каждого `k in 1..7` строится блок из 27 float32:

```text
mean, std, q10, q25, q50, q75, q90, histogram[20]
```

Гистограмма описывает conditional equity по future runout, а не отдельные
win/loss showdown outcomes. Итоговый вектор имеет фиксированный layout:

```text
opponents_1[27] | opponents_2[27] | ... | opponents_7[27]
```

и форму `(189,)`. Все значения конечны, equity и quantiles лежат в `[0, 1]`,
каждая гистограмма суммируется в 1.

## Dataset и производительность

Train и holdout содержат unique canonical keys и не пересекаются. Выборка
карточная, без запуска турнирных эпизодов. Диагностический набор известных
сильных рук, слабых пар, draw, monotone/paired board и board-nuts проверяет
содержимое bucket, но не заменяет train distribution.

Estimator обязан поддерживать chunked processing, cache по canonical key и
полному spec, возобновление по готовым shard и controlled worker count.
Скорость и peak memory записываются в report. Перед production обязательно
выполняется измерительный build; Python-циклы без batch/parallel strategy не
достаточны для production budget.

## Clustering и validation

На каждой улице независимо обучаются scaler только по train и `KMeans`.
`StandardScaler` -- исходный baseline, но его влияние на summary и histogram
blocks измеряется отдельно. Политика scaling, веса блоков и обработка нулевой
или малой дисперсии фиксируются в manifest. Число clusters -- параметр; `200`
является стартовой гипотезой, а не инвариантом. Holdout получает labels от
train-модели.

Validation сохраняет:

- размеры и пустые bucket;
- train/holdout distances;
- внутрибакетную ошибку и разброс mean equity в исходных equity-единицах
  отдельно для 1--7 opponents;
- silhouette и representative examples;
- stability labels при увеличенном вложенном MC budget, включая river;
- независимую stability-проверку с отдельным validation seed;
- MC diagnostics: HU river error относительно exact oracle, error обычного
  budget относительно точного reference и ошибки mean/distribution для
  каждого `k=1..7`.

Сначала pilot build выбирает budgets, scaling и thresholds. Затем эти
параметры фиксируются до запуска production и оцениваются на отдельном
holdout. Production artifact публикуется только после прохождения этих
заранее зафиксированных quality gates. Silhouette рассчитывается на
ограниченной deterministic подвыборке и сам по себе не является условием
публикации.

## Артефакты

```text
artifacts/8max_postflop_abstraction/
  manifest.json
  preflop/lossless_classes.parquet
  flop/{model.joblib,train_assignments.parquet,validation.json}
  turn/{model.joblib,train_assignments.parquet,validation.json}
  river/{model.joblib,train_assignments.parquet,validation.json}
```

Manifest включает feature/canonicalizer/estimator versions, `max_table_players:
8`, `opponent_counts: [1..7]`, common-pot equity definition, reference range,
feature layout/dimension, sampling/cache spec, master seed, train/holdout
counts, scaler/KMeans parameters, evaluator revision, dependency versions,
quality thresholds и SHA-256 моделей.

Loader отвергает несовместимость estimator, layout, range model или versions.
Все streets строятся в temporary sibling directory и публикуются одним
атомарным каталогом только после полного success.

## Приёмка

- Suit-isomorphic input даёт тот же key и feature.
- В одном multiway sample `w(k+1) <= w(k)`; карты соперников не повторяются.
- Победа, проигрыш, split-pot и tie на 2--8 игроков покрыты unit tests.
- River MC coordinate против одного opponent согласуется с exact oracle в
  заранее зафиксированном статистическом допуске; строгая монотонность
  проверяется на individual samples и вложенных MC averages, без clipping.
- Feature имеет `(189,)`, `float32`, корректные bins и воспроизводимость.
- Train/holdout canonical keys дизъюнктны.
- Smoke создаёт весь новый формат, а rejection quality gate не публикует
  target.
- Prefix consistency не меняется при workers, chunking, resume или росте
  sampling budget.
- Завершение работы означает только построение и validation offline artifacts;
  MTT policy и runtime resolver ещё не используют abstraction.
