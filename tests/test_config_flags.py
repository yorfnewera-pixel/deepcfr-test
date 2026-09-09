import pytest

from src.training import train as _train_mod
from src.utils import config as config_mod

del _train_mod


def test_clear_strategy_buffer_each_iteration_uses_explicit_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "num_actions: 6",
                "clear_strategy_buffer_each_iteration: true",
                "strategy_buffer_reservoir: true",
            ]
        ),
        encoding="utf-8",
    )

    try:
        config_mod.load_config(config_path)

        assert config_mod.cfg_clear_strategy_buffer_each_iteration() is True
    finally:
        config_mod.load_config("config.yaml")


def test_legacy_strategy_buffer_reservoir_maps_to_clear_strategy_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "num_actions: 6",
                "strategy_buffer_reservoir: false",
            ]
        ),
        encoding="utf-8",
    )

    try:
        config_mod.load_config(config_path)

        assert config_mod.cfg_clear_strategy_buffer_each_iteration() is True
    finally:
        config_mod.load_config("config.yaml")


def test_strict_mode_switch_after_deep_cfr_import_changes_live_setting():
    import src.core.deep_cfr  # noqa: F401
    from src.utils import settings

    try:
        settings.set_strict_checking(False)
        assert settings.is_strict_checking() is False
        settings.set_strict_checking(True)
        assert settings.is_strict_checking() is True
    finally:
        config_mod.load_config("config.yaml")
        settings.set_strict_checking(False)


@pytest.mark.parametrize("mode", ["strict", "skip_traversal"])
def test_training_error_mode_accepts_supported_values(tmp_path, mode):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"num_actions: 6\ntraining_error_mode: {mode}\n", encoding="utf-8")

    try:
        config_mod.load_config(config_path)
        assert config_mod.cfg_training_error_mode() == mode
    finally:
        config_mod.load_config("config.yaml")
        from src.utils import settings

        settings.set_strict_checking(False)


@pytest.mark.parametrize("invalid_mode", ["recover", [], {}])
def test_training_error_mode_rejects_unknown_value(tmp_path, invalid_mode):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"num_actions: 6\ntraining_error_mode: {invalid_mode}\n",
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="training_error_mode"):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")
        from src.utils import settings

        settings.set_strict_checking(False)


def test_training_max_failed_traversals_rejects_negative_limit(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "num_actions: 6\ntraining_max_failed_traversals_per_iteration: -1\n",
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="training_max_failed_traversals_per_iteration"):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")
        from src.utils import settings

        settings.set_strict_checking(False)


def test_training_validate_state_invariants_returns_bool(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "num_actions: 6\ntraining_validate_state_invariants: enabled\n",
        encoding="utf-8",
    )

    try:
        config_mod.load_config(config_path)
        assert config_mod.cfg_training_validate_state_invariants() is True
    finally:
        config_mod.load_config("config.yaml")
        from src.utils import settings

        settings.set_strict_checking(False)
