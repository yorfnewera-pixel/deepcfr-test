# 020 — Noise-control implementation

**Статус:** IMPLEMENTED

Runtime search получил независимые belief proposals и rollout particles: `proposal_count` отделён от `particle_count`; ESS измеряется на proposals и проверяется абсолютным и относительным порогом. Rollout возвращает CRN-correct gap лучшего и второго root action, его standard error и z-score; policy сохраняет blueprint при signal ниже noise gate. Paired JSON содержит pooled SE, t, 95% CI и `power_status`.

Training loop, checkpoint format и action space не менялись.

