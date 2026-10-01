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
- изменение существующего HU pipeline, HU artifacts или checkpoint.

## Изоляция

Новая реализация располагается в `src/mtt_card_abstraction/`, CLI -- в
`tools/build_8max_postflop_abstraction.py`, artifacts -- в
`artifacts/8max_postflop_abstraction/`. Нельзя переиспользовать HU artifact
или заменять его 189-мерным feature. Общие чистые примитивы карт и showdown
evaluator допускается переиспользовать.

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

На river против одного opponent хранится exact oracle, против двух--семи --
multiway Monte Carlo. River result имеет форму `(1, 7)`; нулевой std внутри
feature не означает нулевую MC-погрешность. Последняя сохраняется в validation
diagnostics.

Seed выводится из версии estimator, master seed, canonical key и типа
sampling. Размер sampling budget не входит в stream seed, чтобы больший
budget продолжал ту же последовательность; он входит в cache key и manifest.

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
карточная, без запуска турнирных эпизодов.

Estimator обязан поддерживать chunked processing, cache по canonical key и
полному spec, возобновление по готовым shard и controlled worker count.
Скорость и peak memory записываются в report. Перед production обязательно
выполняется измерительный build; Python-циклы без batch/parallel strategy не
достаточны для production budget.

## Clustering и validation

На каждой улице независимо обучаются `StandardScaler` только по train и
`KMeans`. Число clusters -- параметр; `200` является стартовой гипотезой, а
не инвариантом. Holdout получает labels от train-модели.

Validation сохраняет:

- размеры и пустые bucket;
- train/holdout distances;
- внутрибакетную ошибку и разброс mean equity отдельно для 1--7 opponents;
- silhouette и representative examples;
- stability labels при увеличенном MC budget, включая river;
- MC diagnostics river для opponents 2--7.

Production artifact публикуется только после прохождения задокументированных
quality gates. Пороги stability и cluster count калибруются на измерительном
build; они не объявляются доказанными до этой калибровки.

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
counts, scaler/KMeans parameters, evaluator revision, dependency versions и
SHA-256 моделей.

Loader отвергает несовместимость estimator, layout, range model или versions.
Все streets строятся в temporary sibling directory и публикуются одним
атомарным каталогом только после полного success.

## Приёмка

- Suit-isomorphic input даёт тот же key и feature.
- В одном multiway sample `w(k+1) <= w(k)`; карты соперников не повторяются.
- Победа, проигрыш, split-pot и tie на 2--8 игроков покрыты unit tests.
- HU river block совпадает с exact oracle.
- Feature имеет `(189,)`, `float32`, корректные bins и воспроизводимость.
- Train/holdout canonical keys дизъюнктны.
- Smoke создаёт весь новый формат, а rejection quality gate не публикует
  target.
- Изменение нового pipeline не изменяет HU imports, HU tests или HU artifacts.
