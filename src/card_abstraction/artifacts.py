"""Сериализация versioned offline артефактов абстракции."""
from __future__ import annotations

import json
import hashlib
import os
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


def sha256_file(path: Path) -> str:
    """Возвращает SHA-256 содержимого файла без загрузки его целиком в память."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_directory(temporary: Path, target: Path) -> Path:
    """Публикует полностью подготовленный каталог с восстановлением старой версии."""
    temporary = Path(temporary)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.previous-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    replaced_previous = False
    try:
        if target.exists():
            target.replace(backup)
            replaced_previous = True
        temporary.replace(target)
    except Exception:
        if replaced_previous and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        if backup.exists() and target.exists():
            shutil.rmtree(backup)
    return target


def publish_street_artifact(root: Path, street: Street, artifact: StreetArtifact, manifest: Manifest) -> Path:
    """Атомарно публикует model и manifest для одной улицы."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{street.value}-", dir=root))
    target = root / street.value
    try:
        (temporary / "manifest.json").write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")
        joblib.dump(artifact, temporary / "model.joblib")
        return publish_directory(temporary, target)
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
