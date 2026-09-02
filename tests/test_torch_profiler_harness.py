"""Малые контракты настроек torch.profiler harness."""

from tools.benchmark_torch_profiler import profile_options


def test_torch_profiler_modes_are_mutually_exclusive() -> None:
    assert profile_options("time") == {
        "record_shapes": False, "profile_memory": False, "with_stack": False,
    }
    assert profile_options("shapes")["record_shapes"] is True
    assert profile_options("memory")["profile_memory"] is True
    assert profile_options("stacks")["with_stack"] is True
