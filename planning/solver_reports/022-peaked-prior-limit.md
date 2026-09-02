# 022 — Peaked prior и ограничение MMDS

**Статус:** DIAGNOSIS

MMDS мультипликативен по blueprint prior. При крайне peaked prior изменение argmax требует EV-gap, перекрывающий `log(p_top / p_second) / eta_effective`. После noise-control оставшиеся policy shifts малы: значимый rollout signal ещё не означает достаточный gap для преодоления prior.

Следующая диагностика должна измерить на checkpoint 4000 распределение CRN-correct best gap, его z-score, log-ratio prior и влияние value scale до изменения normalisation или eta.
