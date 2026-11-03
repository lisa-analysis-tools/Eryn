# -*- coding: utf-8 -*-
"""Tests for executor checkpointing (save_path) and per-round diagnostics.

Covers both executors:
- HDF5 checkpoint written where the trained flow lives (parent for
  InlineExecutor, spawned worker for ProcessExecutor), atomic overwrite,
  full round-trip through ``ZukoFlow.load`` including the fitted transform.
- ``latest_history`` exposure (FlowHistory travels with each weights item).
- Diagnostics PNGs: per-round loss curve, running val-NLL trend, and the
  ``plot_corner`` toggle (training buffer vs flow draws).
- Fail-fast validation of the knobs.
"""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eryn.flows import OneHotLeafConditioning
from eryn.flows.executors import InlineExecutor, ProcessExecutor
from eryn.flows.torch import WhiteningTransform, ZukoFlow

NDIM = 2


def _make_tiny_flow(seed: int = 0) -> ZukoFlow:
    return ZukoFlow(
        dims=NDIM,
        device="cpu",
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        data_transform=WhiteningTransform(ndim=NDIM),
        seed=seed,
        transforms=2,
        hidden_features=(16, 16),
        bins=4,
    )


def _training_batch(n: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {0: rng.standard_normal((n, NDIM)) * 3.0 + 1.0}


def _poll_weights(ex, deadline_s: float = 120.0):
    t0 = time.monotonic()
    while time.monotonic() < t0 + deadline_s:
        lw = ex.latest_weights()
        if lw is not None:
            return lw
        time.sleep(0.1)
    return None


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------

def test_plot_corner_requires_diagnostics_dir():
    flow = _make_tiny_flow()
    with pytest.raises(ValueError, match="diagnostics_dir"):
        InlineExecutor(flow, plot_corner=True)
    with pytest.raises(ValueError, match="diagnostics_dir"):
        ProcessExecutor(flow, plot_corner=True, start=False)


def test_output_dirs_created_at_init(tmp_path):
    flow = _make_tiny_flow()
    save_path = tmp_path / "sub" / "ckpt.h5"
    diag_dir = tmp_path / "diag"
    ex = InlineExecutor(
        flow, save_path=str(save_path), diagnostics_dir=str(diag_dir)
    )
    assert save_path.parent.is_dir()
    assert diag_dir.is_dir()
    ex.shutdown()


# ---------------------------------------------------------------------------
# InlineExecutor
# ---------------------------------------------------------------------------

def test_inline_checkpoint_roundtrip_and_history(tmp_path):
    import h5py

    flow = _make_tiny_flow()
    save_path = tmp_path / "flow_ckpt.h5"
    ex = InlineExecutor(
        flow,
        fit_kwargs=dict(n_epochs=2),
        min_train_samples=10,
        save_path=str(save_path),
    )
    assert ex.latest_history is None
    assert ex.submit(_training_batch()) is True

    # checkpoint exists, carries the version attr, no leftover tmp file
    assert save_path.is_file()
    assert not os.path.exists(str(save_path) + ".tmp")
    with h5py.File(save_path, "r") as f:
        assert int(f.attrs["version"]) == 1

    # full round-trip: fitted transform + finite density
    loaded = ZukoFlow.load(str(save_path))
    assert loaded.data_transform.is_fitted
    x = _training_batch(n=16, seed=1)[0]
    assert np.all(np.isfinite(loaded.log_prob(x, context=0)))

    # loaded flow matches the executor's trained clone (same net + transform)
    version, snapshot = ex.latest_weights()
    sibling = _make_tiny_flow()
    sibling.set_weights(snapshot)
    np.testing.assert_allclose(
        loaded.log_prob(x, context=0), sibling.log_prob(x, context=0),
        rtol=1e-5, atol=1e-6,
    )

    # history exposed with the round's epochs
    hist = ex.latest_history
    assert hist is not None and len(hist.training_loss) == 2
    ex.shutdown()


def test_inline_save_every_cadence(tmp_path):
    flow = _make_tiny_flow()
    save_path = tmp_path / "flow_ckpt.h5"
    ex = InlineExecutor(
        flow,
        fit_kwargs=dict(n_epochs=1),
        min_train_samples=10,
        save_path=str(save_path),
        save_every=2,
    )
    ex.submit(_training_batch(seed=0))  # version 1 -> no save
    assert not save_path.exists()
    ex.submit(_training_batch(seed=1))  # version 2 -> save
    assert save_path.is_file()
    ex.shutdown()


def test_inline_diagnostics_pngs_and_corner_toggle(tmp_path):
    flow = _make_tiny_flow()
    diag_dir = tmp_path / "diag"
    ex = InlineExecutor(
        flow,
        fit_kwargs=dict(n_epochs=2),
        min_train_samples=10,
        diagnostics_dir=str(diag_dir),
        plot_corner=True,
        corner_max_samples=200,
    )
    ex.submit(_training_batch())
    assert (diag_dir / "flow_loss_v0001.png").is_file()
    assert (diag_dir / "flow_val_nll_trend.png").is_file()
    assert (diag_dir / "flow_corner_v0001_cond0.png").is_file()
    ex.shutdown()


def test_inline_no_corner_without_toggle(tmp_path):
    flow = _make_tiny_flow()
    diag_dir = tmp_path / "diag"
    ex = InlineExecutor(
        flow,
        fit_kwargs=dict(n_epochs=1),
        min_train_samples=10,
        diagnostics_dir=str(diag_dir),
    )
    ex.submit(_training_batch())
    assert (diag_dir / "flow_loss_v0001.png").is_file()
    assert not list(diag_dir.glob("flow_corner_*"))
    ex.shutdown()


# ---------------------------------------------------------------------------
# ProcessExecutor (checkpoint + plots produced in the spawned worker)
# ---------------------------------------------------------------------------

def test_process_checkpoint_diagnostics_and_history(tmp_path):
    import h5py

    flow = _make_tiny_flow()
    save_path = tmp_path / "flow_ckpt.h5"
    diag_dir = tmp_path / "diag"
    ex = ProcessExecutor(
        flow,
        epochs_per_round=2,
        min_train_samples=50,
        seed=11,
        save_path=str(save_path),
        diagnostics_dir=str(diag_dir),
        plot_corner=True,
        corner_max_samples=200,
    )
    try:
        assert ex.submit(_training_batch()) is True
        lw = _poll_weights(ex)
        assert lw is not None, "worker never produced weights"
        version, snapshot = lw
        assert version >= 1

        # history travels with the weights item
        hist = ex.latest_history
        assert hist is not None and len(hist.training_loss) == 2

        # the worker wrote the checkpoint + plots before pushing the weights,
        # so they must exist by now
        assert save_path.is_file()
        with h5py.File(save_path, "r") as f:
            assert int(f.attrs["version"]) >= 1
        loaded = ZukoFlow.load(str(save_path))
        assert loaded.data_transform.is_fitted
        x = _training_batch(n=16, seed=1)[0]
        assert np.all(np.isfinite(loaded.log_prob(x, context=0)))

        assert (diag_dir / f"flow_loss_v{version:04d}.png").is_file()
        assert (diag_dir / "flow_val_nll_trend.png").is_file()
        assert (diag_dir / f"flow_corner_v{version:04d}_cond0.png").is_file()
    finally:
        ex.shutdown()
