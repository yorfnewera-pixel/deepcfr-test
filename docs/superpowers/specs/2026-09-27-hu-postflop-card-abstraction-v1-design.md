# HU postflop card abstraction V1: дизайн

## Цель

Построить воспроизводимую **карточную** абстракцию для текущего
heads-up D2CFR: lossless preflop представление из 169 классов и lossy
postflop ситуации по будущей силе руки. Работа не меняет правила игры,
action space, сеть, traversal или существующий 109-мерный card context.

Первый результат — офлайн-артефакт с тремя моделями `k-means` и отчётом о
качестве. Встраивание bucket ID в solver начинается отдельным этапом только
после прохождения критериев качества и выбора стратегии разрешения ранее не
встречавшихся карточных ситуаций.

## Зафиксированная область

Входит:

- ровно два игрока;
- lossless preflop и flop, turn, river;
- одна скрытая рука hero и открытая доска;
- равномерный диапазон всех допустимых двухкарточных рук одного opponent;
- suit-isomorphism, future-equity profile, L2 и отдельный `k-means(k=200)`
  для каждой улицы;
- детерминированное построение, сериализация и offline-валидация.

Не входит:

- six-max, несколько opponent и совместные диапазоны;
- диапазоны, зависящие от betting history, позиции или текущей стратегии;
- EMD/Sinkhorn, иерархическое кластеризование и перенос кода из reference
  репозиториев;
- изменение D2CFR targets, action space, neural network или checkpoint
  формата;
- утверждение, что результат воспроизводит внутренние признаки Pluribus.

## Обоснование

V1 должен измерять влияние одной новой сущности — сжатия card state. Если
одновременно добавить multi-player equity, history-aware range или новый
solver, результат A/B-прогона нельзя будет интерпретировать.

Равномерный диапазон делает `feature(cards)` независимой от стратегии и
истории. Поэтому одна и та же каноническая ситуация всегда даёт один и тот же
feature и один bucket. History-aware range намеренно отложен: при нём функция
становится `feature(cards, history)`, а ключ LUT обязан содержать историю.

## Термины и инварианты

`CardSituation` содержит:

```text
hero:  две различные карты hero;
board: 0, 3, 4 или 5 различных открытых карт;
street: preflop, flop, turn или river, выведенная только из длины board.
```

Все карты в `hero ∪ board` различны и принадлежат стандартной 52-карточной
колоде. Нарушение этого контракта завершается `ValueError` до любого расчёта.

`CanonicalCardKey` — неизменяемый ключ из канонических `hero`, `board` и
`street`. Порядок двух карт hero и порядок карт board незначимы. Никакой
betting history, player ID, stack, pot или action mask в ключ не входит.

## Канонизация

V1 использует простой полный перебор 24 биекций мастей, а не числовой индекс
из алгоритма Waugh:

1. Отсортировать hero и board внутри своих групп по `(rank, suit)`.
2. Применить каждую из `4!` перестановок мастей одновременно ко всем картам.
3. Вновь отсортировать каждую группу и закодировать пары `(rank, suit)` в
   фиксированный кортеж `hero | board`.
4. Выбрать лексикографически минимальный кортеж.

Это даёт одинаковый ключ всем suit-isomorphic ситуациям. Стоимость
`O(24 * 7 log 7)` постоянна относительно размера задачи и не является
критическим путём. Для preflop этот key образует ровно 169 lossless классов:
13 пар, 78 suited и 78 offsuit. Полный индекс Waugh понадобится только при
отдельной задаче полного перечисления postflop пространства.

## Preflop: lossless 169

Preflop не использует equity, sampling или `k-means`. Каждая допустимая
двухкарточная hero hand приводится к `CanonicalCardKey` с пустым board;
получается один из точных 169 классов. В artifact сохраняется полная
детерминированная таблица `1326 physical hands -> 169 canonical keys`.
Эта таблица служит только проверяемым lossless card representation; она не
добавляет в ключ history ставок и не заменяет abstract information set,
который в будущем также будет содержать betting history.

## Модель диапазона

`UniformHeadsUpRange` — единственный provider V1. После исключения `hero`,
`board` и выбранного future runout каждая неупорядоченная пара оставшихся карт
имеет одинаковый вес. Для пары рук hero/opponent payoff равен `1` при победе,
`0.5` при ничьей и `0` при проигрыше.

Provider имеет отдельный интерфейс, хотя альтернативные реализации не
включаются в V1:

```text
equity(hero, final_board, opponent_hands) -> [0.0, 1.0]
```

Вход provider не принимает action history. Добавление такого параметра —
несовместимое расширение V2.

## Equity profile

Feature описывает распределение **условной terminal equity** на будущих
runout, а не единичный rollout payoff. Он определён только для postflop;
preflop использует lossless 169 и не имеет equity feature.

Для flop и turn:

1. Из детерминированного RNG выбрать 128 допустимых complete runout с
   возвращением. Для flop runout содержит две карты, для turn — одну.
2. Для каждого runout выбрать 128 допустимых opponent hands равномерно с
   возвращением.
3. Вычислить условную equity `e_i` как среднее 128 showdown payoff.
4. Получить 128 значений `e_i` из диапазона `[0, 1]`.

Для river future runout отсутствует. Equity вычисляется точно перебором всех
допустимых неупорядоченных opponent hands; это единственное значение `e_1`.

RNG seed выводится из `SHA-256` следующих байтов:

```text
"hu_postflop_abstraction_v1" | master_seed | street | canonical_card_key
```

Поэтому повторный запуск, другой порядок обработки и suit-isomorphic вход
дают одинаковый feature. Реализация не использует глобальный `numpy.random`
или состояние игрового RNG.

Из набора `e` строится вектор из 27 `float32`:

```text
[mean, std, q10, q25, q50, q75, q90, histogram_0, ..., histogram_19]
```

`histogram_j` — нормированная частота в полуинтервале `[j/20, (j+1)/20)`;
последний bin включает `1.0`. Для river `std=0`, все квантили равны точной
equity, histogram содержит единицу в соответствующем bin. Это оставляет
непрерывный mean как различающий river признак, а не сводит river к 20
значениям.

Перед clustering каждый из 27 столбцов стандартизируется по обучающей
выборке: `(x - mean_train) / std_train`. Нулевой `std_train` заменяется на
`1.0`. Параметры стандартизации являются частью модели.

## Выборка, clustering и артефакты

На каждой улице строится независимая выборка из 100 000 уникальных
`CanonicalCardKey`, равномерно полученных через валидные card situation и
дедупликацию canonical key. Выборка также использует фиксированный
master-seed `20260927`. Недобор уникальных ключей является ошибкой процесса,
а не молчаливым уменьшением датасета.

Для каждой улицы запускается `KMeans` со следующими параметрами:

```text
n_clusters=200
init="k-means++"
n_init=20
max_iter=500
tol=1e-4
random_state=20260927
metric=L2 по стандартизированному feature
```

Метод хранит centroids, scaler и label каждой обучающей ситуации. Метки
`k-means` не имеют семантики и не сравниваются между независимыми запусками;
для сравнения моделей используются метрики расстояний, а не номер bucket.

Структура output directory:

```text
artifacts/hu_postflop_abstraction/v1/
  manifest.json
  preflop/lossless_classes.parquet
  flop/model.joblib
  flop/train_assignments.parquet
  flop/validation.json
  turn/model.joblib
  turn/train_assignments.parquet
  turn/validation.json
  river/model.joblib
  river/train_assignments.parquet
  river/validation.json
```

`manifest.json` обязан содержать версию feature, кодовую версию canonicalizer,
master-seed, sampling counts, k-means parameters, feature dimension, SHA-256
моделей и точные версии Python, NumPy, scikit-learn и hand evaluator.
Несовпадение manifest с runtime-конфигурацией запрещает загрузку артефакта.

## Граница LUT и solver

Выборка из 100 000 состояний не покрывает всё postflop пространство. Поэтому
`train_assignments.parquet` — это **не** runtime LUT и V1 не подключает bucket
ID в CFR traversal.

Допустимый resolver будущего этапа должен выбрать один явный режим:

1. полная предвычисленная LUT; либо
2. вычисление feature на cache miss с персистентным cache и предсказанием
   ближайшего centroid.

До выбора режима запрещено выдавать `bucket_id` для неизвестной ситуации,
подставлять default bucket или тайно вычислять equity в traversal. Для
экспериментальной интеграции потребуется отдельная спецификация с бюджетом
latency, cache policy и checkpoint compatibility.

## Валидация

### Модульные контракты

- Канонизатор инвариантен к перестановке карт внутри hero/board и к любой
  общей перестановке мастей.
- Полная preflop таблица содержит ровно 1326 physical hands и 169 разных
  canonical keys; 13 из них пары, 78 suited и 78 offsuit.
- Различные hero/board карты, неверная улица и дубликаты отклоняются.
- Feature имеет форму `(27,)`, dtype `float32`, конечные значения и сумму
  histogram, равную `1 ± 1e-6`.
- River feature не использует RNG и совпадает с ручным точным перебором.
- Одинаковый key даёт побитово одинаковый feature независимо от порядка
  задач и числа worker.
- Изменение master-seed не меняет canonical key, но разрешено менять sampling
  feature на flop/turn.

### Качество кластеров

Для каждой улицы строится независимый holdout из 20 000 canonical states с
другим фиксированным seed. В `validation.json` сохраняются:

- средняя L2-дистанция точки до своего centroid на train и holdout;
- размер каждого bucket, число пустых bucket и минимум/медиана/максимум
  размера;
- silhouette score на детерминированной подвыборке 10 000 holdout points;
- 20 ближайших и 20 наиболее удалённых примеров для каждого bucket;
- повторный feature на 1 000 holdout состояниях с удвоенным sampling budget
  `256 × 256` и расстояние до исходного feature.

Артефакт допускается к следующему этапу, только если:

- отсутствуют пустые bucket;
- ни один bucket не содержит более 10% train выборки;
- средняя holdout-дистанция до centroid не больше 1.15 средней train-дистанции;
- для не менее 95% stability-подвыборки ближайший centroid при обычном и
  удвоенном sampling budget совпадает.

Низкий training loss D2CFR, число пройденных traversals и визуальное сходство
примеров не являются критериями приёмки этого этапа.

## Совместимость и ошибки

V1 создаёт только новые files в `artifacts/hu_postflop_abstraction/v1` и не
изменяет существующие checkpoint. Любая ошибка сериализации, несовместимый
manifest, невалидная карта, не конечный feature или невозможность получить
200 непустых cluster завершают запуск с диагностическим сообщением и ненулевым
кодом. Частично записанный output публикуется только после атомарной замены
временной директории.

## Последовательность работ

1. Реализовать и протестировать domain types, preflop lossless-169 table,
   canonicalizer и exact river equity независимо от Deep CFR.
2. Добавить deterministic flop/turn sampler и 27-мерный feature.
3. Добавить offline dataset builder, scaler, k-means, manifest и атомарное
   сохранение.
4. Добавить validation report и выполнить 100 000-state offline запуск.
5. Только после review артефакта спроектировать отдельную интеграцию resolver
   в HU D2CFR и честный A/B-протокол.
