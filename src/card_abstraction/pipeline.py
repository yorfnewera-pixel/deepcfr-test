"""Офлайн-построение и валидация HU card abstraction без импорта solver."""
from __future__ import annotations

import json
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pokers as pkrs
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .artifacts import Manifest, StreetArtifact, publish_directory, publish_street_artifact, sha256_file
from .canonical import canonicalize
from .dataset import sample_unique_situations
from .domain import CardSituation, Street
from .features import build_feature
from .preflop import build_lossless_preflop_table

_FEATURE_VERSION = "hu_postflop_abstraction_v1"
_CANONICALIZER_VERSION = "v1"
_FEATURE_SIZE = 27
_KMEANS_PARAMETERS = {"init": "k-means++", "n_init": 20, "max_iter": 500, "tol": 1e-4}


def _distances(points: np.ndarray, model: KMeans) -> tuple[np.ndarray, np.ndarray]:
    labels = model.predict(points)
    return labels, np.linalg.norm(points - model.cluster_centers_[labels], axis=1)


def _examples(keys: list[str], labels: np.ndarray, distances: np.ndarray, clusters: int) -> dict[str, dict[str, list[str]]]:
    result = {}
    for bucket in range(clusters):
        indices = np.flatnonzero(labels == bucket)
        ordered = indices[np.argsort(distances[indices])]
        result[str(bucket)] = {
            "nearest": [keys[int(index)] for index in ordered[:20]],
            "furthest": [keys[int(index)] for index in ordered[-20:][::-1]],
        }
    return result


def _validation(
    street: Street,
    train: list[CardSituation],
    train_points: np.ndarray,
    holdout: list[CardSituation],
    holdout_points: np.ndarray,
    scaler: StandardScaler,
    model: KMeans,
    master_seed: int,
) -> dict[str, object]:
    train_labels, train_distances = _distances(train_points, model)
    holdout_labels, holdout_distances = _distances(holdout_points, model)
    bucket_sizes = np.bincount(train_labels, minlength=model.n_clusters)
    silhouette = None
    if len(holdout_points) >= 2 and 1 < len(np.unique(holdout_labels)) < len(holdout_labels):
        silhouette = float(silhouette_score(holdout_points[:10_000], holdout_labels[:10_000]))
    stability_count = min(1_000, len(holdout))
    stability = 1.0
    if stability_count and street is not Street.RIVER:
        refined = np.vstack([
            build_feature(item, master_seed, runout_samples=256, opponent_samples=256)
            for item in holdout[:stability_count]
        ])
        stability = float(np.mean(
            model.predict(holdout_points[:stability_count]) == model.predict(scaler.transform(refined))
        ))
    keys = [canonicalize(item).as_string() for item in train]
    return {
        "train_mean_distance": float(train_distances.mean()),
        "holdout_mean_distance": float(holdout_distances.mean()),
        "bucket_sizes": bucket_sizes.tolist(),
        "empty_bucket_count": int(np.count_nonzero(bucket_sizes == 0)),
        "bucket_size": {"min": int(bucket_sizes.min()), "median": float(np.median(bucket_sizes)), "max": int(bucket_sizes.max())},
        "silhouette_score": silhouette,
        "examples": _examples(keys, train_labels, train_distances, model.n_clusters),
        "stability": {"sample_count": stability_count, "same_nearest_centroid_fraction": stability},
    }


def _validate_quality(validation: dict[str, object], sample_count: int) -> None:
    bucket_size = validation["bucket_size"]
    stability = validation["stability"]
    if validation["empty_bucket_count"] != 0:
        raise RuntimeError("quality gate: обнаружены пустые bucket")
    if bucket_size["max"] > sample_count * 0.10:
        raise RuntimeError("quality gate: bucket содержит более 10% train выборки")
    if validation["holdout_mean_distance"] > validation["train_mean_distance"] * 1.15:
        raise RuntimeError("quality gate: holdout distance превышает допустимый порог")
    if stability["same_nearest_centroid_fraction"] < 0.95:
        raise RuntimeError("quality gate: stability ниже 95%")


def _build_street(
    street: Street, output_root: Path, sample_count: int, holdout_count: int,
    clusters: int, master_seed: int, enforce_quality: bool,
) -> tuple[Path, str]:
    train = sample_unique_situations(street, sample_count, master_seed)
    holdout = sample_unique_situations(street, holdout_count, master_seed + 1)
    train_features = np.vstack([build_feature(item, master_seed) for item in train])
    holdout_features = np.vstack([build_feature(item, master_seed) for item in holdout])
    scaler = StandardScaler().fit(train_features)
    train_points = scaler.transform(train_features)
    holdout_points = scaler.transform(holdout_features)
    model = KMeans(n_clusters=clusters, random_state=master_seed, **_KMEANS_PARAMETERS).fit(train_points)
    artifact = StreetArtifact(model=model, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    directory = publish_street_artifact(output_root, street, artifact, Manifest("v1", _FEATURE_SIZE, master_seed))
    labels, _ = _distances(train_points, model)
    pd.DataFrame({
        "canonical_key": [canonicalize(item).as_string() for item in train],
        "bucket_id": labels.astype(np.int32),
    }).to_parquet(directory / "train_assignments.parquet", index=False)
    validation = _validation(street, train, train_points, holdout, holdout_points, scaler, model, master_seed)
    if enforce_quality:
        _validate_quality(validation, sample_count)
    (directory / "validation.json").write_text(json.dumps(validation, sort_keys=True), encoding="utf-8")
    return directory, sha256_file(directory / "model.joblib")


def _global_manifest(master_seed: int, sample_count: int, holdout_count: int, clusters: int, model_sha256: dict[str, str]) -> dict[str, object]:
    return {
        "feature_version": _FEATURE_VERSION,
        "canonicalizer_version": _CANONICALIZER_VERSION,
        "master_seed": master_seed,
        "sampling": {"train_count": sample_count, "holdout_count": holdout_count},
        "kmeans": {**_KMEANS_PARAMETERS, "n_clusters": clusters, "metric": "l2_standardized"},
        "feature_dimension": _FEATURE_SIZE,
        "model_sha256": model_sha256,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "hand_evaluator": getattr(pkrs, "__version__", "local")},
    }


def build_all(
    output_root: Path, sample_count: int, holdout_count: int, clusters: int,
    master_seed: int = 20260927, enforce_quality: bool | None = None,
) -> dict[Street, Path]:
    """Строит набор артефактов и публикует его только после полной проверки."""
    if clusters < 2 or sample_count < clusters or holdout_count < 1:
        raise ValueError("Некорректные параметры выборки или числа кластеров")
    if enforce_quality is None:
        enforce_quality = clusters == 200 and sample_count >= 100_000
    output_root = Path(output_root)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        preflop_directory = temporary / Street.PREFLOP.value
        preflop_directory.mkdir(parents=True)
        build_lossless_preflop_table().to_parquet(preflop_directory / "lossless_classes.parquet", index=False)
        model_sha256 = {}
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            _, model_sha256[street.value] = _build_street(
                street, temporary, sample_count, holdout_count, clusters, master_seed, enforce_quality
            )
        (temporary / "manifest.json").write_text(
            json.dumps(_global_manifest(master_seed, sample_count, holdout_count, clusters, model_sha256), sort_keys=True),
            encoding="utf-8",
        )
        publish_directory(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {street: output_root / street.value for street in Street}
