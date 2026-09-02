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
