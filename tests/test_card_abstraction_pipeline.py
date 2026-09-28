import subprocess
import sys

import json
import pytest

from src.card_abstraction.domain import Street


def test_build_all_smoke_writes_preflop_and_three_postflop_artifacts(tmp_path):
    from src.card_abstraction.pipeline import build_all

    result = build_all(
        output_root=tmp_path,
        sample_count=5,
        holdout_count=3,
        clusters=2,
        master_seed=20260927,
    )

    assert set(result) == {Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER}
    assert (tmp_path / "preflop" / "lossless_classes.parquet").is_file()
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        assert (tmp_path / street.value / "model.joblib").is_file()
        assert (tmp_path / street.value / "train_assignments.parquet").is_file()
        assert (tmp_path / street.value / "validation.json").is_file()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_version"] == "hu_postflop_abstraction_v1"
    assert manifest["canonicalizer_version"] == "v1"
    assert set(manifest["model_sha256"]) == {"flop", "turn", "river"}


def test_cli_smoke_builds_artifacts(tmp_path):
    from tools.build_hu_postflop_abstraction import main

    exit_code = main(["--smoke", "--output", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "preflop" / "lossless_classes.parquet").is_file()


def test_cli_smoke_runs_as_script_from_project_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_hu_postflop_abstraction.py",
            "--smoke",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "preflop" / "lossless_classes.parquet").is_file()


def test_build_all_does_not_publish_when_quality_gate_rejects(tmp_path, monkeypatch):
    from src.card_abstraction import pipeline

    def reject_quality(*_args, **_kwargs):
        raise RuntimeError("quality gate отклонил артефакт")

    monkeypatch.setattr(pipeline, "_validate_quality", reject_quality, raising=False)

    with pytest.raises(RuntimeError, match="quality gate"):
        pipeline.build_all(
            output_root=tmp_path / "rejected",
            sample_count=5,
            holdout_count=3,
            clusters=2,
            enforce_quality=True,
        )

    assert not (tmp_path / "rejected").exists()
