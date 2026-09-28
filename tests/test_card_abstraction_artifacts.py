from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans

from src.card_abstraction.domain import Street


def test_artifact_roundtrip_rejects_incompatible_manifest(tmp_path: Path):
    from src.card_abstraction.artifacts import (
        Manifest,
        StreetArtifact,
        load_street_artifact,
        publish_street_artifact,
    )

    model = KMeans(n_clusters=2, random_state=1, n_init=1).fit(np.array([[0.0], [1.0]]))
    manifest = Manifest(version="v1", feature_size=27, master_seed=20260927)
    artifact = StreetArtifact(model=model, scaler_mean=np.zeros(27), scaler_scale=np.ones(27))

    publish_street_artifact(tmp_path, Street.FLOP, artifact, manifest)

    assert load_street_artifact(tmp_path, Street.FLOP, manifest).model.n_clusters == 2
    with pytest.raises(ValueError, match="manifest"):
        load_street_artifact(tmp_path, Street.FLOP, Manifest("v2", 27, 20260927))
