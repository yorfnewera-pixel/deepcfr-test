import subprocess
import sys

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
        assert (tmp_path / street.value / "validation.json").is_file()


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
