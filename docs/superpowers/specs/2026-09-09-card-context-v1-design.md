# Card Context V1: дизайн

## Цель

Подготовить общую, переносимую карточную часть сети для будущего warm-start
между HU teacher и six-max student. Этот этап не переносит веса и не добавляет
distillation.

## Решение

Добавляется явная архитектура `card_context_v1` наряду с существующей
`monolithic_v1`. `card_encoder` принимает ровно 109 первых признаков текущего
state encoding: 52 hero hand, 52 public cards и 5 street. Остаток вектора
получает контекстный блок. Их выходы объединяются перед action head.

```
[hero hand, board, street] -> card_encoder ---+
                                             +-> action head
[positions, stacks, bets, history, actor] -> context_encoder -+
```

## Совместимость

- Default остаётся `monolithic_v1`; старые checkpoint продолжают загружаться.
- `card_context_v1` хранится в metadata checkpoint вместе с размерами входа и
  карточного блока; runtime строит сеть только по совпадающей архитектуре.
- HU actor one-hot остаётся контекстом, а не карточным блоком.
- Checkpoint без architecture metadata не совместим с `card_context_v1`.

## Границы этапа

Не реализуются: teacher-to-student copying, HU distillation, перенос advantage
сетей, изменение action space или state encoder.

## Проверки

- одинаковые 109 card features для HU и six-max дают одинаковый card embedding;
- изменение контекста не меняет card embedding;
- runtime отклоняет несовместимую архитектуру;
- legacy monolithic путь сохраняет прежнее поведение.
