# tests/test_flow_fit.py
"""Tests for ZukoFlow.fit() — the self-contained Adam training loop.

Requires torch and zuko (``pip install eryn[flow]``).

Port of the spirit of LISAanalysistools/tests/test_ml_training.py extended
with new requirements: seeded determinism, patience early-stopping, NaN guard,
and refit_data_transform semantics.

Designed to run in under ~60 s; uses small flows (transforms=3, bins=5,
hidden_features=(64,64)) and ≤ 800 samples throughout.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")  # noqa: F841

from eryn.flows import (
    FlowHistory,
    OneHotLeafConditioning,
    ZukoFlow,
)
from eryn.flows.torch.transforms import WhiteningTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_FLOW_KW = dict(
    flow_class="NSF",
    transforms=3,
    hidden_features=(64, 64),
    bins=5,
)


def _make_unconditional(seed: int = 42, dims: int = 2) -> ZukoFlow:
    """Small unconditional ZukoFlow with IdentityTransform."""
    return ZukoFlow(dims=dims, device="cpu", seed=seed, **_SMALL_FLOW_KW)


def _make_conditional(seed: int = 7, n_conds: int = 2, dims: int = 2) -> ZukoFlow:
    """Small conditional ZukoFlow with WhiteningTransform + OneHotLeafConditioning."""
    cond = OneHotLeafConditioning(nleaves_max=n_conds)
    wt = WhiteningTransform(ndim=dims)
    return ZukoFlow(
        dims=dims,
        device="cpu",
        conditioning=cond,
        data_transform=wt,
        seed=seed,
        **_SMALL_FLOW_KW,
    )


def _gauss2d(rng, mean, cov, n: int) -> np.ndarray:
    return rng.multivariate_normal(mean, cov, size=n).astype(np.float64)


# ---------------------------------------------------------------------------
# Test 1 — loss decreases (unconditional)
# ---------------------------------------------------------------------------

def test_unconditional_loss_decreases():
    """Unconditional flow: validation loss decreases over 8 epochs (2-D Gaussian data)."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], n=600)

    flow = _make_unconditional(seed=1)
    history = flow.fit(
        data,
        n_epochs=8,
        lr=1e-2,
        batch_size=128,
        validation_fraction=0.2,
        seed=1,
        verbose=False,
    )

    assert isinstance(history, FlowHistory)
    assert len(history.training_loss) == len(history.validation_loss)
    assert len(history.validation_loss) <= 8
    assert len(history.validation_loss) >= 1
    # Validation loss must drop from first to last epoch
    assert history.validation_loss[-1] < history.validation_loss[0], (
        f"Val loss did not decrease: {history.validation_loss[0]:.4f} -> "
        f"{history.validation_loss[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — conditional multi-condition fit
# ---------------------------------------------------------------------------

def test_conditional_multi_condition_fit():
    """Conditional flow: two-condition dict fit reduces val loss and gives finite log-probs."""
    rng = np.random.default_rng(11)
    # Two Gaussians with very different means so WhiteningTransform is informative
    data_0 = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], n=400)
    data_1 = _gauss2d(rng, [5.0, 5.0], [[1.0, 0.0], [0.0, 1.0]], n=400)

    flow = _make_conditional(seed=3, n_conds=2, dims=2)
    history = flow.fit(
        {0: data_0, 1: data_1},
        n_epochs=8,
        lr=1e-2,
        batch_size=128,
        validation_fraction=0.2,
        seed=3,
        verbose=False,
    )

    assert isinstance(history, FlowHistory)
    assert len(history.validation_loss) >= 1
    # Val loss must drop
    assert history.validation_loss[-1] < history.validation_loss[0], (
        f"Conditional val loss did not decrease: {history.validation_loss[0]:.4f} -> "
        f"{history.validation_loss[-1]:.4f}"
    )

    # After fit, both contexts give finite log_probs on test data
    test_0 = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], n=32)
    lp0 = flow.log_prob(test_0, context=0)
    lp1 = flow.log_prob(test_0, context=1)
    assert np.all(np.isfinite(lp0)), f"context=0 gave non-finite log_probs: {lp0}"
    assert np.all(np.isfinite(lp1)), f"context=1 gave non-finite log_probs: {lp1}"


# ---------------------------------------------------------------------------
# Test 3 — patience early stopping
# ---------------------------------------------------------------------------

def test_patience_early_stop():
    """patience=2 stops training before n_epochs=200 on small noisy data."""
    rng = np.random.default_rng(99)
    # Very small dataset + high lr to ensure val loss oscillates / plateaus
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], n=50)

    flow = _make_unconditional(seed=5)
    history = flow.fit(
        data,
        n_epochs=200,
        lr=5e-2,          # high lr -> noisy training
        batch_size=16,
        validation_fraction=0.3,
        patience=2,
        seed=5,
        verbose=False,
    )

    assert len(history.validation_loss) < 200, (
        f"Expected early stop before 200 epochs; ran {len(history.validation_loss)} epochs"
    )
    # Stopped at best_epoch + patience
    assert len(history.validation_loss) >= 1


# ---------------------------------------------------------------------------
# Test 4 — NaN input raises ValueError
# ---------------------------------------------------------------------------

def test_nan_input_raises_value_error():
    """NaN values in samples after transform raise ValueError mentioning 'non-finite'."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=100)
    data[5, 0] = np.nan  # inject NaN

    flow = _make_unconditional(seed=1)
    # IdentityTransform passes NaNs through — the NaN guard inside fit() catches it
    with pytest.raises(ValueError, match="[Nn]on.finite"):
        flow.fit(data, n_epochs=2, seed=1)


# ---------------------------------------------------------------------------
# Test 5 — multi-condition dict without conditioning raises ValueError
# ---------------------------------------------------------------------------

def test_multi_condition_without_conditioning_raises():
    """dict with >1 keys + conditioning=None raises ValueError."""
    rng = np.random.default_rng(0)
    data_0 = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=100)
    data_1 = _gauss2d(rng, [3.0, 3.0], np.eye(2), n=100)

    flow = _make_unconditional(seed=1)  # no conditioning
    with pytest.raises(ValueError, match="conditioning"):
        flow.fit({0: data_0, 1: data_1}, n_epochs=2, seed=1)


# ---------------------------------------------------------------------------
# Test 6 — refit_data_transform semantics
# ---------------------------------------------------------------------------

def test_refit_data_transform_false_leaves_transform_unchanged():
    """refit_data_transform=False: pre-fitted WhiteningTransform state unchanged by fit."""
    rng = np.random.default_rng(7)
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], n=400)

    wt = WhiteningTransform(ndim=2)
    wt.fit({0: data})

    # Snapshot the mean vector from the centering transform before fit.
    # WhiteningTransform stores per-condition ComposeTransform objects;
    # the first part of the chain is the AffineTransform (centering), which
    # exposes a `loc` shift attribute.  ComposeTransform uses `.parts` (not
    # `.transforms`) to expose its sub-transforms.
    before_tensor = wt.transforms[0].parts[0].loc.clone()

    cond = OneHotLeafConditioning(nleaves_max=1)
    flow = ZukoFlow(
        dims=2, device="cpu", conditioning=cond, data_transform=wt, seed=1, **_SMALL_FLOW_KW
    )
    flow.fit(data, n_epochs=2, refit_data_transform=False, seed=1)

    after_tensor = wt.transforms[0].parts[0].loc
    assert torch.allclose(before_tensor, after_tensor), (
        "WhiteningTransform was re-fitted despite refit_data_transform=False"
    )


def test_refit_data_transform_false_unfitted_transform_gets_fitted():
    """refit_data_transform=False with unfitted transform: the `or not is_fitted` clause fires."""
    rng = np.random.default_rng(8)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=300)

    wt = WhiteningTransform(ndim=2)
    assert not wt.is_fitted, "WhiteningTransform should not be fitted yet"

    cond = OneHotLeafConditioning(nleaves_max=1)
    flow = ZukoFlow(
        dims=2, device="cpu", conditioning=cond, data_transform=wt, seed=2, **_SMALL_FLOW_KW
    )
    flow.fit(data, n_epochs=2, refit_data_transform=False, seed=2)

    assert wt.is_fitted, (
        "WhiteningTransform should have been fitted when refit_data_transform=False "
        "but transform was not yet fitted"
    )


# ---------------------------------------------------------------------------
# Test 7 — determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed_same_history():
    """Two fresh flows with the same seed and same data produce identical val loss sequences."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], n=400)

    # Both flows are constructed with the same seed; torch.manual_seed(seed) is
    # called inside BaseTorchFlow.__init__, so identical init weights.
    flow_a = _make_unconditional(seed=77)
    flow_b = _make_unconditional(seed=77)

    h_a = flow_a.fit(data, n_epochs=4, lr=1e-3, batch_size=64, seed=77, verbose=False)
    h_b = flow_b.fit(data, n_epochs=4, lr=1e-3, batch_size=64, seed=77, verbose=False)

    np.testing.assert_array_equal(
        h_a.validation_loss,
        h_b.validation_loss,
        err_msg="Identical flows+seed produced different validation_loss sequences",
    )
    np.testing.assert_array_equal(
        h_a.training_loss,
        h_b.training_loss,
        err_msg="Identical flows+seed produced different training_loss sequences",
    )


# ---------------------------------------------------------------------------
# Test 8 — fit returns FlowHistory type
# ---------------------------------------------------------------------------

def test_fit_returns_flow_history_instance():
    """fit() returns an instance of eryn.flows.FlowHistory."""
    from eryn.flows.base import FlowHistory as FlowHistoryBase

    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=200)

    flow = _make_unconditional(seed=1)
    history = flow.fit(data, n_epochs=2, seed=1)

    assert isinstance(history, FlowHistoryBase), (
        f"Expected FlowHistory, got {type(history)}"
    )


# ---------------------------------------------------------------------------
# Test 9 — validation_fraction range check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_fraction", [0.0, 1.0])
def test_validation_fraction_out_of_range_raises(bad_fraction):
    """validation_fraction at the closed boundaries (0.0, 1.0) raises ValueError."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=200)

    flow = _make_unconditional(seed=1)
    with pytest.raises(ValueError, match="validation_fraction"):
        flow.fit(data, n_epochs=2, validation_fraction=bad_fraction, seed=1)


# ---------------------------------------------------------------------------
# Test 10 — seed coerced at the boundary (whole float accepted)
# ---------------------------------------------------------------------------

def test_seed_whole_float_is_coerced_and_matches_int():
    """A whole-float seed (7.0) is coerced via int() and gives the same result as int seed 7."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], n=300)

    flow_int = _make_unconditional(seed=11)
    flow_float = _make_unconditional(seed=11)

    h_int = flow_int.fit(data, n_epochs=3, batch_size=64, seed=7, verbose=False)
    h_float = flow_float.fit(data, n_epochs=3, batch_size=64, seed=7.0, verbose=False)

    np.testing.assert_array_equal(
        h_int.validation_loss,
        h_float.validation_loss,
        err_msg="seed=7 and seed=7.0 produced different validation_loss sequences",
    )


# ---------------------------------------------------------------------------
# Test 11 — selectable optimizer
# ---------------------------------------------------------------------------

def test_unknown_optimizer_name_raises():
    """An unresolvable optimizer name raises ValueError naming the bad spec."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=120)

    flow = _make_unconditional(seed=1)
    with pytest.raises(ValueError, match="optimizer"):
        flow.fit(data, n_epochs=2, optimizer="notanoptimizer", seed=1)


def test_optimizer_invalid_type_raises():
    """A non-str / non-Optimizer-subclass optimizer raises TypeError."""
    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=120)

    flow = _make_unconditional(seed=1)
    with pytest.raises(TypeError, match="optimizer"):
        flow.fit(data, n_epochs=2, optimizer=123, seed=1)


def test_optimizer_string_selects_class_and_forwards_kwargs(monkeypatch):
    """optimizer='adamw' instantiates AdamW; optimizer_kwargs are forwarded."""
    import torch.optim as optim

    seen: dict = {}
    real_adamw = optim.AdamW

    def spy(params, **kw):
        opt = real_adamw(params, **kw)
        seen["cls"] = type(opt).__name__
        seen["kw"] = kw
        return opt

    # The resolver looks the class up via getattr(torch.optim, "AdamW"), so
    # patching the attribute on the module is what fit() will pick up.
    monkeypatch.setattr(optim, "AdamW", spy)

    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=200)
    flow = _make_unconditional(seed=1)
    flow.fit(
        data,
        n_epochs=1,
        lr=1e-2,
        optimizer="adamw",
        optimizer_kwargs={"weight_decay": 0.123},
        seed=1,
        verbose=False,
    )

    assert seen["cls"] == "AdamW"
    assert seen["kw"]["weight_decay"] == 0.123
    # top-level lr is passed through to the optimizer
    assert seen["kw"]["lr"] == 1e-2


def test_optimizer_class_accepted():
    """A torch.optim.Optimizer subclass can be passed directly and trains."""
    import torch as _torch

    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], n=300)
    flow = _make_unconditional(seed=1)
    history = flow.fit(
        data, n_epochs=2, lr=1e-2, optimizer=_torch.optim.AdamW, seed=1, verbose=False
    )
    assert len(history.validation_loss) >= 1


def test_default_optimizer_is_adam_with_no_extra_kwargs(monkeypatch):
    """Default optimizer='adam' builds Adam and forwards no extra kwargs (backward compat)."""
    import torch.optim as optim

    seen: dict = {}
    real_adam = optim.Adam

    def spy(params, **kw):
        seen["cls"] = "Adam"
        seen["kw"] = kw
        return real_adam(params, **kw)

    monkeypatch.setattr(optim, "Adam", spy)

    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=200)
    flow = _make_unconditional(seed=1)
    flow.fit(data, n_epochs=1, lr=1e-3, seed=1, verbose=False)

    assert seen["cls"] == "Adam"
    assert seen["kw"]["lr"] == 1e-3
    # No optimizer_kwargs supplied => only lr is passed
    assert set(seen["kw"]) == {"lr"}


def test_top_level_lr_overrides_optimizer_kwargs_lr(monkeypatch):
    """A duplicate lr in optimizer_kwargs is dropped; the top-level lr wins."""
    import torch.optim as optim

    seen: dict = {}
    real_adam = optim.Adam

    def spy(params, **kw):
        seen["kw"] = kw
        return real_adam(params, **kw)

    monkeypatch.setattr(optim, "Adam", spy)

    rng = np.random.default_rng(0)
    data = _gauss2d(rng, [0.0, 0.0], np.eye(2), n=200)
    flow = _make_unconditional(seed=1)
    flow.fit(
        data,
        n_epochs=1,
        lr=1e-3,
        optimizer_kwargs={"lr": 999.0},
        seed=1,
        verbose=False,
    )

    assert seen["kw"]["lr"] == 1e-3
