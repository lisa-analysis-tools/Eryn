# tests/test_flow_imports.py
"""Tests that eryn.flows imports correctly and does not pull in torch at module level."""
from __future__ import annotations

import subprocess
import sys

import pytest


EXPECTED_ALL = sorted([
    "Flow",
    "FlowHistory",
    "FlowProposalDistribution",
    "DataTransform",
    "IdentityTransform",
    "ConditioningStrategy",
    "OneHotLeafConditioning",
    "get_flow_wrapper",
    "ZukoFlow",
    "WhiteningTransform",
    "TrainerError",
    "FlowSpec",
    "TrainerExecutor",
    "InlineExecutor",
    "ProcessExecutor",
    "WorkerConfig",
])


def test_eryn_flows_importable():
    """eryn.flows can be imported without error."""
    import eryn.flows  # noqa: F401


def test_all_matches_expected():
    """__all__ contains exactly the expected public names."""
    import eryn.flows

    assert sorted(eryn.flows.__all__) == EXPECTED_ALL


def test_torch_free_at_module_level():
    """Importing eryn.flows and accessing torch-free symbols must NOT import torch."""
    code = (
        "import sys\n"
        "import eryn.flows\n"
        "_ = eryn.flows.Flow\n"
        "_ = eryn.flows.OneHotLeafConditioning\n"
        "_ = eryn.flows.IdentityTransform\n"
        "assert 'torch' not in sys.modules, 'torch was imported at module level'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_no_torch_environment():
    """With torch/zuko blocked, torch-free API works and torch-backed API raises ImportError
    with a helpful 'eryn[flow]' message."""
    code = (
        "import sys\n"
        "\n"
        "# Block torch/zuko imports via a meta path finder\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'torch' or name == 'zuko' or name.startswith('torch.') or name.startswith('zuko.'):\n"
        "            raise ImportError(f'torch/zuko blocked in test: {name}')\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "\n"
        "import eryn.flows\n"
        "\n"
        "# Torch-free classes must still construct\n"
        "cond = eryn.flows.OneHotLeafConditioning(4)\n"
        "assert cond.context_dim == 4\n"
        "tr = eryn.flows.IdentityTransform()\n"
        "assert tr.is_fitted\n"
        "\n"
        "# get_flow_wrapper('zuko') must raise ImportError mentioning eryn[flow]\n"
        "try:\n"
        "    eryn.flows.get_flow_wrapper('zuko')\n"
        "    print('FAIL: expected ImportError from get_flow_wrapper')\n"
        "    sys.exit(1)\n"
        "except ImportError as e:\n"
        "    if 'eryn[flow]' not in str(e):\n"
        "        print(f'FAIL: ImportError does not mention eryn[flow]: {e}')\n"
        "        sys.exit(1)\n"
        "\n"
        "# ZukoFlow lazy attribute access must raise ImportError mentioning eryn[flow]\n"
        "try:\n"
        "    _ = eryn.flows.ZukoFlow\n"
        "    print('FAIL: expected ImportError for ZukoFlow')\n"
        "    sys.exit(1)\n"
        "except (ImportError, AttributeError) as e:\n"
        "    if 'eryn[flow]' not in str(e):\n"
        "        print(f'FAIL: error does not mention eryn[flow]: {e}')\n"
        "        sys.exit(1)\n"
        "\n"
        "print('SENTINEL_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SENTINEL_OK" in result.stdout


def test_get_flow_wrapper_unknown_backend():
    """get_flow_wrapper with unknown backend raises ValueError mentioning 'zuko'."""
    import eryn.flows

    with pytest.raises(ValueError, match="zuko"):
        eryn.flows.get_flow_wrapper("nope")


def test_misspelled_attribute_raises_attribute_error():
    """A misspelled attribute name raises AttributeError (not ImportError)."""
    import eryn.flows

    with pytest.raises(AttributeError):
        _ = eryn.flows.NopeNope


def test_executor_names_resolve():
    """Executor names resolve from the torch-free executors module."""
    import eryn.flows

    for name in ("TrainerError", "FlowSpec", "TrainerExecutor", "InlineExecutor",
                 "ProcessExecutor", "WorkerConfig"):
        assert getattr(eryn.flows, name) is not None


def test_executors_torch_free_at_module_level():
    """Importing eryn.flows.executors and resolving executor names must NOT import torch."""
    code = (
        "import sys\n"
        "import eryn.flows\n"
        "_ = eryn.flows.TrainerExecutor\n"
        "_ = eryn.flows.InlineExecutor\n"
        "_ = eryn.flows.FlowSpec\n"
        "_ = eryn.flows.TrainerError\n"
        "_ = eryn.flows.ProcessExecutor\n"
        "_ = eryn.flows.WorkerConfig\n"
        "assert 'torch' not in sys.modules, 'torch was imported by executors'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
