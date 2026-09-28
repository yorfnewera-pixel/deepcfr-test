"""Офлайн-построение lossless preflop и postflop k-means артефактов."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .artifacts import Manifest, StreetArtifact, publish_street_artifact
from .dataset import sample_unique_situations
from .domain import Street
from .features import build_feature
from .preflop import build_lossless_preflop_table


def _build_street(
    street: Street, output_root: Path, sample_count: int, holdout_count: int, clusters: int, master_seed: int
) -> Path:
    train = sample_unique_situations(street, sample_count, master_seed)
    holdout = sample_unique_situations(street, holdout_count, master_seed + 1)
    train_features = np.vstack([build_feature(item, master_seed) for item in train])
    holdout_features = np.vstack([build_feature(item, master_seed) for item in holdout])
    scaler = StandardScaler().fit(train_features)
    model = KMeans(n_clusters=clusters, init="k-means++", n_init=20, max_iter=500, tol=1e-4, random_state=master_seed)
    model.fit(scaler.transform(train_features))
    artifact = StreetArtifact(model=model, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    manifest = Manifest(version="v1", feature_size=27, master_seed=master_seed)
    directory = publish_street_artifact(output_root, street, artifact, manifest)
    distances = np.linalg.norm(scaler.transform(holdout_features) - model.cluster_centers_[model.predict(scaler.transform(holdout_features))], axis=1)
    (directory / "validation.json").write_text(json.dumps({"holdout_mean_distance": float(distances.mean())}), encoding="utf-8")
    return directory


def build_all(output_root: Path, sample_count: int, holdout_count: int, clusters: int, master_seed: int = 20260927) -> dict[Street, Path]:
    """Строит smoke либо offline набор артефактов без обращения к solver."""
    if clusters < 2 or sample_count < clusters or holdout_count < 1:
        raise ValueError("Некорректные параметры выборки или числа кластеров")
    output_root = Path(output_root)
    preflop_directory = output_root / Street.PREFLOP.value
    preflop_directory.mkdir(parents=True, exist_ok=True)
    build_lossless_preflop_table().to_parquet(preflop_directory / "lossless_classes.parquet", index=False)
    result = {Street.PREFLOP: preflop_directory}
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        result[street] = _build_street(street, output_root, sample_count, holdout_count, clusters, master_seed)
    return result
