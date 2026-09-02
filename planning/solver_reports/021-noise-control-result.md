# 021 — Noise-control result

**Статус:** RESULT

Checkpoint 3000, `256 proposals -> 32 particles`, `eta=10`, 96 paired samples:

- search roots: 29;
- fallback: 20 (69%);
- argmax/action changes: 0 / 0;
- pooled EV: `0 ± 0 BB/100`, `power_status=insufficient_power`.

При одинаковом `eta=10` L1 shift на root упал примерно в 9–10 раз против прежнего `particles=2` прогона. Это закрывает eta sweep 017–019 как измерение усиления Monte-Carlo шума: `eta=15` не запускать. Малые прогоны не являются EV-вердиктом.

