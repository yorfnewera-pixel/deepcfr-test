"""Сериализация versioned offline артефактов абстракции."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from .domain import Street


@dataclass(frozen=True)
class Manifest:
    version: str
    feature_size: int
    master_seed: int


@dataclass
class StreetArtifact:
    model: object
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def publish_street_artifact(root: Path, street: Street, artifact: StreetArtifact, manifest: Manifest) -> Path:
    """Атомарно публикует model и manifest для одной улицы."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{street.value}-", dir=root))
    target = root / street.value
    try:
        (temporary / "manifest.json").write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")
        joblib.dump(artifact, temporary / "model.joblib")
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_street_artifact(root: Path, street: Street, expected_manifest: Manifest) -> StreetArtifact:
    """Загружает artifact только при точном совпадении manifest."""
    directory = Path(root) / street.value
    actual = Manifest(**json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
    if actual != expected_manifest:
        raise ValueError("manifest артефакта несовместим с ожидаемой конфигурацией")
    return joblib.load(directory / "model.joblib")
