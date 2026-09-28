"""Построение нормированного feature профиля будущей equity."""
from __future__ import annotations

import numpy as np

from .domain import CardSituation
from .equity import conditional_equities


def build_feature(
    situation: CardSituation,
    master_seed: int = 20260927,
    runout_samples: int = 128,
    opponent_samples: int = 128,
) -> np.ndarray:
    """Строит 27-мерный profile из summary statistics и 20-bin histogram."""
    equities = conditional_equities(situation, master_seed, runout_samples, opponent_samples)
    summaries = np.asarray(
        [
            np.mean(equities),
            np.std(equities),
            np.quantile(equities, 0.10),
            np.quantile(equities, 0.25),
            np.quantile(equities, 0.50),
            np.quantile(equities, 0.75),
            np.quantile(equities, 0.90),
        ],
        dtype=np.float64,
    )
    histogram, _ = np.histogram(equities, bins=20, range=(0.0, 1.0))
    feature = np.concatenate((summaries, histogram / len(equities)))
    if not np.all(np.isfinite(feature)):
        raise ValueError("Equity feature содержит не конечные значения")
    return feature.astype(np.float32)
