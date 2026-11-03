# tests/test_flow_zuko.py
"""Tests for ZukoFlow — the zuko-backed conditional normalizing flow.

These tests port the three core tests from LISAanalysistools/tests/test_ml_flow.py
(renamed and adapted to the eryn.flows API) and add additional coverage for the
unified context contract, config round-trips, and h5 save/load.

Requires torch and zuko (``pip install eryn[flow]``).
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")

from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning, get_flow_wrapper


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_flow(seed: int = 0) -> tuple:
    """Build a small ZukoFlow with WhiteningTransform and OneHotLeafConditioning.

    Returns (flow, samples) where samples is a (3000, 3) float64 array used to
    fit the WhiteningTransform (flow itself is not trained — weights are random).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=3000),
        (rng.normal(2.0, 0.4, size=3000)) % (2 * np.pi),
    ]).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    wt.fit({0: samples})

    flow = ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow, samples


def _make_unconditional_flow(seed: int = 42) -> "ZukoFlow":
    """Build a small unconditional ZukoFlow (no conditioning, no data_transform)."""
    torch.manual_seed(seed)
    flow = ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=None,
        conditioning=None,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow


# ---------------------------------------------------------------------------
# Ported from test_ml_flow.py — renamed FlowModel → ZukoFlow, condition→context,
# normalizer→data_transform, OneHotLeafConditioning kept.
# ---------------------------------------------------------------------------

def test_sample_shape_and_logprob_consistency():
    """sample_and_log_prob shapes and log_prob(x) == logq at sampled points (atol 1e-4)."""
    flow, _ = _make_flow()
    x, logq = flow.sample_and_log_prob(128, context=0)
    assert x.shape == (128, 3), f"expected (128, 3), got {x.shape}"
    assert logq.shape == (128,), f"expected (128,), got {logq.shape}"
    # Re-evaluate at the same points: must match sampling-time log_prob
    logq2 = flow.log_prob(x, context=0)
    np.testing.assert_allclose(logq, logq2, atol=1e-4,
                               err_msg="log_prob recomputed differs from sample_and_log_prob")


def test_logprob_is_finite_on_training_support():
    """log_prob returns finite values for all points in the training support."""
    flow, samples = _make_flow()
    lp = flow.log_prob(samples[:256], context=0)
    assert np.all(np.isfinite(lp)), f"Non-finite log-probs: {np.sum(~np.isfinite(lp))} entries"


def test_proposal_distribution_matches_flow():
    """FlowProposalDistribution.logpdf and rvs match ZukoFlow.log_prob/sample."""
    from eryn.flows import FlowProposalDistribution
    flow, _ = _make_flow()
    dist = FlowProposalDistribution(flow, condition=0)
    x = dist.rvs(64)
    assert x.shape == (64, 3)
    lp_dist = dist.logpdf(x)
    lp_flow = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_dist, lp_flow, atol=1e-5)


# ---------------------------------------------------------------------------
# Unconditional flow
# ---------------------------------------------------------------------------

def test_unconditional_log_prob_shape_and_finite():
    """Unconditional ZukoFlow (context=None) returns finite log_probs, correct shape."""
    flow = _make_unconditional_flow()
    x = np.random.default_rng(0).standard_normal((32, 3)).astype(np.float64)
    lp = flow.log_prob(x, context=None)
    assert lp.shape == (32,)
    assert np.all(np.isfinite(lp)), "Non-finite log_probs from unconditional flow"


def test_unconditional_sample_shape_and_finite():
    """Unconditional ZukoFlow sample returns correct shape and no NaNs."""
    flow = _make_unconditional_flow()
    x = flow.sample(64, context=None)
    assert x.shape == (64, 3)
    assert not np.any(np.isnan(x)), "NaNs in unconditional samples"


def test_unconditional_sample_and_log_prob_consistency():
    """Unconditional sample_and_log_prob logq matches log_prob(x) (atol 1e-4)."""
    flow = _make_unconditional_flow()
    x, logq = flow.sample_and_log_prob(64, context=None)
    logq2 = flow.log_prob(x, context=None)
    np.testing.assert_allclose(logq, logq2, atol=1e-4)


# ---------------------------------------------------------------------------
# Raw-array context
# ---------------------------------------------------------------------------

def test_raw_array_context_matches_int_context():
    """Passing conditioning.encode(0) directly gives same result as context=0 (atol 1e-5)."""
    flow, samples = _make_flow()
    raw_ctx = flow.conditioning.encode(0)  # numpy float32, shape (context_dim,)
    x = samples[:32]
    lp_int = flow.log_prob(x, context=0)
    lp_raw = flow.log_prob(x, context=raw_ctx)
    np.testing.assert_allclose(lp_int, lp_raw, atol=1e-5,
                               err_msg="int context and raw array context disagree")


def test_raw_array_context_sample_and_log_prob():
    """sample_and_log_prob with raw context vector returns valid outputs."""
    flow, _ = _make_flow()
    raw_ctx = flow.conditioning.encode(0)
    x, logq = flow.sample_and_log_prob(32, context=raw_ctx)
    assert x.shape == (32, 3)
    assert logq.shape == (32,)
    assert np.all(np.isfinite(logq))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_int_context_without_conditioning_raises():
    """Passing an int context to a flow with no conditioning raises ValueError."""
    flow = _make_unconditional_flow()
    with pytest.raises(ValueError, match="conditioning"):
        flow.log_prob(np.zeros((4, 3)), context=0)


def test_context_with_no_context_dim_raises():
    """Passing any non-None context to a flow built with context_dim=0 raises ValueError."""
    flow = _make_unconditional_flow()
    ctx_vec = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        flow.log_prob(np.zeros((4, 3)), context=ctx_vec)


# ---------------------------------------------------------------------------
# get_flow_wrapper
# ---------------------------------------------------------------------------

def test_get_flow_wrapper_returns_zukoflow_and_torch():
    """get_flow_wrapper('zuko') returns (ZukoFlow, torch)."""
    FlowCls, backend = get_flow_wrapper("zuko")
    assert FlowCls is ZukoFlow
    import torch as _torch
    assert backend is _torch


# ---------------------------------------------------------------------------
# get_weights / set_weights
# ---------------------------------------------------------------------------

def test_get_weights_cpu_and_detached():
    """get_weights() tensors are on CPU and detached; mutating them doesn't change the flow."""
    flow, _ = _make_flow()
    weights = flow.get_weights()
    assert all(isinstance(v, torch.Tensor) for v in weights.values())
    assert all(v.device.type == "cpu" for v in weights.values())

    # Mutating the returned dict must NOT change the live flow outputs
    x = np.zeros((4, 3), dtype=np.float64)
    lp_before = flow.log_prob(x, context=0)
    for v in weights.values():
        v.fill_(999.0)
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8,
                               err_msg="Mutating get_weights() output changed the flow")


def test_set_weights_restores_log_prob():
    """set_weights(get_weights()) is a no-op: log_prob is unchanged."""
    flow, _ = _make_flow()
    x = np.random.default_rng(7).standard_normal((16, 3))
    lp_before = flow.log_prob(x, context=0)
    flow.set_weights(flow.get_weights())
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8)


# ---------------------------------------------------------------------------
# config_dict round-trip
# ---------------------------------------------------------------------------

def test_config_dict_round_trip_log_prob():
    """ZukoFlow(**flow.config_dict()) + set_weights reproduces log_prob (atol 1e-6)."""
    flow, _ = _make_flow()
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    cfg = flow.config_dict()
    flow2 = ZukoFlow(**cfg)
    flow2.set_weights(flow.get_weights())

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="config_dict round-trip changed log_prob")


def test_config_dict_contains_flow_kwargs():
    """flow_kwargs like transforms, bins are flattened into config_dict() at the top level."""
    flow, _ = _make_flow()
    cfg = flow.config_dict()
    assert "transforms" in cfg, "transforms not found in config_dict()"
    assert "bins" in cfg, "bins not found in config_dict()"
    assert cfg["dims"] == 3


def test_config_dict_tuple_hidden_features_survives():
    """tuple hidden_features survives config round-trip (may come back as list; both ok)."""
    flow, _ = _make_flow()
    cfg = flow.config_dict()
    hf = cfg["hidden_features"]
    # tuple or list, both are fine — ZukoFlow accepts both
    assert tuple(hf) == (64, 64), f"hidden_features mangled: {hf}"


# ---------------------------------------------------------------------------
# HDF5 save / load round-trip
# ---------------------------------------------------------------------------

def test_h5_save_load_round_trip(tmp_path):
    """h5 save/load reproduces log_prob exactly and restores transform + conditioning."""
    flow, _ = _make_flow()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 3))

    h5_path = str(tmp_path / "flow.h5")
    flow.save(h5_path)
    flow2 = ZukoFlow.load(h5_path)

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.load(h5_path).log_prob(x, context=0)  # load again via cls method
    np.testing.assert_allclose(lp1, lp2, atol=1e-8,
                               err_msg="h5 save/load changed log_prob")

    # Conditioning must be restored
    assert flow2.conditioning is not None
    assert flow2.conditioning.context_dim == flow.conditioning.context_dim
    # data_transform must be restored and functional
    z = flow2.data_transform.forward(x, condition=0)
    assert z.shape == (64, 3)


def test_h5_save_load_with_h5py_handle(tmp_path):
    """save/load also accept an open h5py.File handle."""
    h5py = pytest.importorskip("h5py")
    flow, _ = _make_flow()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((16, 3))

    h5_path = str(tmp_path / "flow2.h5")
    with h5py.File(h5_path, "w") as f:
        flow.save(f, path="myflow")

    with h5py.File(h5_path, "r") as f:
        flow2 = ZukoFlow.load(f, path="myflow")

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-8)


# ---------------------------------------------------------------------------
# fit() smoke test (Task 4 implemented)
# ---------------------------------------------------------------------------

def test_fit_returns_flow_history():
    """fit() runs and returns a FlowHistory (no longer a stub, Task 4 implemented)."""
    from eryn.flows.base import FlowHistory
    flow, samples = _make_flow()
    history = flow.fit(samples[:200], n_epochs=2, seed=0)
    assert isinstance(history, FlowHistory)
    assert len(history.training_loss) >= 1
    assert len(history.validation_loss) >= 1


# ---------------------------------------------------------------------------
# Regression: per-point log-det (non-constant Jacobian)
# ---------------------------------------------------------------------------

class _SoftplusTransform:
    """Custom DataTransform with a non-constant Jacobian for regression testing.

    forward:  z_i = log(1 + exp(x_i))   (softplus, element-wise)
    inverse:  x_i = log(exp(z_i) - 1)   (inverse softplus)
    log|det|: sum_i log(sigmoid(x_i)) = sum_i log(1 / (1 + exp(-x_i)))

    The log-det depends on x, so it differs between any two distinct x points.
    This means the old probe-point cache would return the same scalar for every
    point and give WRONG densities for at least one of them.
    """

    def fit(self, samples) -> None:  # no-op
        pass

    def forward(self, x, condition: int = 0):
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(x))  # softplus

    def inverse(self, z, condition: int = 0):
        z = np.asarray(z, dtype=np.float64)
        return np.log(np.expm1(z))  # inverse softplus

    def log_abs_det_jacobian(self, x, z, condition: int = 0):
        x = np.asarray(x, dtype=np.float64)
        # d(softplus)/dx_i = sigmoid(x_i); log-det = sum_i log sigmoid(x_i)
        log_sigmoid = -np.log1p(np.exp(-x))  # numerically stable log sigmoid
        return log_sigmoid.sum(axis=-1)  # shape (N,)

    @property
    def is_fitted(self) -> bool:
        return True


def test_log_prob_per_point_logdet_nonconstant_jacobian():
    """log_prob uses the per-point log-det for a non-constant-Jacobian transform.

    Regression test for the removed probe-point cache.  The old cache would
    evaluate log_abs_det_jacobian at a zero probe and return a single scalar
    for all points.  For _SoftplusTransform the Jacobian varies with x, so:

      - cached (old) implementation: wrong for any x != 0
      - per-point (new) implementation: matches analytic per-point logdet

    We verify that flow.log_prob(x) matches:
        flow_net_log_prob(z) + analytic_logdet(x)
    at two distinct x points, and that the two expected values differ by
    different amounts than the cached constant would produce.
    """
    import torch as _torch

    tr = _SoftplusTransform()
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=tr,
        conditioning=None,
        seed=7,
        flow_class="NSF",
        transforms=2,
        hidden_features=(32, 32),
        bins=4,
    )

    rng = np.random.default_rng(42)
    # Two clearly different x values so logdet(x0) != logdet(x1)
    x0 = rng.uniform(0.5, 1.5, size=(1, 2))
    x1 = rng.uniform(3.0, 5.0, size=(1, 2))

    # Analytic expected: flow-net log_prob at z + per-point logdet
    def expected_lp(x_np):
        z_np = tr.forward(x_np)
        z_t = _torch.as_tensor(z_np.astype(np.float32))
        with _torch.no_grad():
            flow_lp = flow._flow().log_prob(z_t).cpu().numpy()  # shape (1,)
        logdet = tr.log_abs_det_jacobian(x_np, z_np)           # shape (1,)
        return (flow_lp + logdet).astype(np.float64)

    exp0 = expected_lp(x0)
    exp1 = expected_lp(x1)

    got0 = flow.log_prob(x0)
    got1 = flow.log_prob(x1)

    np.testing.assert_allclose(got0, exp0, atol=1e-5,
        err_msg="log_prob(x0) does not match per-point expected (non-constant Jacobian)")
    np.testing.assert_allclose(got1, exp1, atol=1e-5,
        err_msg="log_prob(x1) does not match per-point expected (non-constant Jacobian)")

    # Confirm the two analytic logdets differ — otherwise the test doesn't
    # distinguish cached from per-point.
    logdet0 = tr.log_abs_det_jacobian(x0, tr.forward(x0))[0]
    logdet1 = tr.log_abs_det_jacobian(x1, tr.forward(x1))[0]
    assert abs(logdet0 - logdet1) > 0.5, (
        f"Test misconfigured: logdet0={logdet0:.4f} logdet1={logdet1:.4f} "
        "are too close to discriminate cached vs per-point."
    )


def test_sample_and_log_prob_per_point_logdet_nonconstant_jacobian():
    """sample_and_log_prob logq matches log_prob(x) with non-constant Jacobian (atol 1e-4).

    Regression companion to the log_prob test above: verifies that
    sample_and_log_prob also uses per-point log-det (not a cached scalar).
    If both methods are consistent the cached implementation would pass here
    only if both were equally wrong — but the log_prob test above catches that.
    """
    tr = _SoftplusTransform()
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=tr,
        conditioning=None,
        seed=13,
        flow_class="NSF",
        transforms=2,
        hidden_features=(32, 32),
        bins=4,
    )

    x, logq = flow.sample_and_log_prob(64)
    logq2 = flow.log_prob(x)
    np.testing.assert_allclose(logq, logq2, atol=1e-4,
        err_msg="sample_and_log_prob logq differs from log_prob(x) with non-constant Jacobian")


# ---------------------------------------------------------------------------
# I1 — h5 overwrite: save larger flow, then smaller flow to same path
# ---------------------------------------------------------------------------

def test_h5_overwrite_smaller_flow_no_orphan_keys(tmp_path):
    """Saving flow B (fewer transforms) over flow A must not leave orphan weight datasets.

    Regression for: save into an existing group with require_group left stale
    weight datasets, causing load() to fail with 'Unexpected key(s)'.
    """
    import h5py as h5py_mod

    h5_path = str(tmp_path / "overwrite.h5")

    # Flow A — larger (transforms=4)
    flow_a = ZukoFlow(
        dims=3, device="cpu", conditioning=None, seed=1,
        flow_class="NSF", transforms=4, hidden_features=(32, 32), bins=4,
    )
    flow_a.save(h5_path)

    # Flow B — smaller (transforms=2)
    flow_b = ZukoFlow(
        dims=3, device="cpu", conditioning=None, seed=2,
        flow_class="NSF", transforms=2, hidden_features=(32, 32), bins=4,
    )
    flow_b.save(h5_path)  # overwrite into the same path

    # load() must return B without KeyError / Unexpected-key error
    flow_loaded = ZukoFlow.load(h5_path)

    # Weight count must match B (not A)
    assert len(flow_loaded.get_weights()) == len(flow_b.get_weights()), (
        "loaded flow has a different number of weight tensors than flow B — "
        "orphan keys from flow A survived the overwrite"
    )

    # log_prob must match B at the same points
    rng = np.random.default_rng(77)
    x = rng.standard_normal((16, 3))
    np.testing.assert_allclose(
        flow_loaded.log_prob(x),
        flow_b.log_prob(x),
        atol=1e-6,
        err_msg="loaded flow log_prob differs from flow B after overwrite",
    )


# ---------------------------------------------------------------------------
# I2 — numpy-boundary validation
# ---------------------------------------------------------------------------

def test_log_prob_empty_x_returns_empty():
    """log_prob with shape (0, dims) returns shape (0,) without error."""
    flow = _make_unconditional_flow()
    x = np.empty((0, 3), dtype=np.float64)
    result = flow.log_prob(x)
    assert result.shape == (0,), f"expected (0,), got {result.shape}"
    assert result.dtype == np.float64


def test_sample_n_zero_returns_empty():
    """sample(0) returns shape (0, dims) without error."""
    flow = _make_unconditional_flow()
    x = flow.sample(0)
    assert x.shape == (0, 3), f"expected (0, 3), got {x.shape}"
    assert x.dtype == np.float64


def test_sample_and_log_prob_n_zero_returns_empty():
    """sample_and_log_prob(0) returns (0, dims) samples and (0,) logq without error."""
    flow = _make_unconditional_flow()
    x, logq = flow.sample_and_log_prob(0)
    assert x.shape == (0, 3), f"expected x shape (0, 3), got {x.shape}"
    assert logq.shape == (0,), f"expected logq shape (0,), got {logq.shape}"
    assert x.dtype == np.float64
    assert logq.dtype == np.float64


def test_log_prob_1d_x_raises():
    """Passing a 1-D array to log_prob raises ValueError (not a silent 0-d scalar)."""
    flow = _make_unconditional_flow()
    x_1d = np.zeros(3, dtype=np.float64)  # shape (3,) — wrong
    with pytest.raises(ValueError, match=r"x must have shape"):
        flow.log_prob(x_1d)


def test_log_prob_wrong_dims_raises():
    """Passing x with wrong feature dim (N, 5) for a dims=3 flow raises ValueError."""
    flow = _make_unconditional_flow()  # dims=3
    x_bad = np.zeros((4, 5), dtype=np.float64)
    with pytest.raises(ValueError, match=r"x must have shape"):
        flow.log_prob(x_bad)


def test_sample_negative_n_raises():
    """sample(-1) raises ValueError."""
    flow = _make_unconditional_flow()
    with pytest.raises(ValueError, match=r"non-negative"):
        flow.sample(-1)


def test_wrong_length_raw_context_raises():
    """Passing a raw context vector with wrong length raises a clear ValueError."""
    flow, _ = _make_flow()  # context_dim = 1 (OneHotLeafConditioning, nleaves_max=1)
    x = np.zeros((4, 3), dtype=np.float64)
    bad_ctx = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # length 3, not 1
    with pytest.raises(ValueError, match=r"Raw context vector"):
        flow.log_prob(x, context=bad_ctx)


# ---------------------------------------------------------------------------
# Multi-condition inference: log_prob differs across conditions
# ---------------------------------------------------------------------------

def test_multi_condition_log_prob_differs_across_conditions():
    """log_prob(x, context=0) differs from log_prob(x, context=1) after fitting different transforms.

    Uses OneHotLeafConditioning(2) + WhiteningTransform fit on two datasets
    with very different scales/offsets.  Even with random (untrained) flow
    weights, the whitening guarantees that log_prob values differ between
    conditions on the same x.
    """
    from eryn.flows import WhiteningTransform

    rng = np.random.default_rng(55)
    # Condition 0: samples near [0, 0, 0] with unit scale
    samples_0 = rng.multivariate_normal(
        [0.0, 0.0, 0.0], np.eye(3), size=500
    ).astype(np.float64)
    # Condition 1: samples near [10, 10, 10] with scale 5 — very different
    samples_1 = rng.multivariate_normal(
        [10.0, 10.0, 10.0], 25 * np.eye(3), size=500
    ).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=2)
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: samples_0, 1: samples_1})

    flow = ZukoFlow(
        dims=3, device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=42,
        flow_class="NSF",
        transforms=2, hidden_features=(32, 32), bins=4,
    )

    x = rng.standard_normal((8, 3))
    lp0 = flow.log_prob(x, context=0)
    lp1 = flow.log_prob(x, context=1)

    # With different whitening the log_prob values MUST differ on the same x
    assert not np.allclose(lp0, lp1, atol=1e-3), (
        "log_prob(x, context=0) and log_prob(x, context=1) are unexpectedly equal; "
        "the multi-condition whitening is not being applied."
    )


# ---------------------------------------------------------------------------
# fit(): val_split="temporal" and train_noise
# ---------------------------------------------------------------------------

def test_fit_invalid_val_split_and_noise_raise():
    flow, samples = _make_flow()
    with pytest.raises(ValueError, match="val_split"):
        flow.fit(samples[:100], n_epochs=1, val_split="foo")
    with pytest.raises(ValueError, match="train_noise"):
        flow.fit(samples[:100], n_epochs=1, train_noise=-0.1)


def test_fit_temporal_split_holds_out_newest_rows():
    """Temporal split validates on the NEWEST rows.

    Buffer whose newest 30% come from a far-shifted distribution: the
    temporal-split validation loss must be dominated by those outliers,
    while a random split (val ~ same mix as train) reports a much lower
    value. This is the property that lets early stopping detect a flow
    that no longer covers fresh chain positions.
    """
    from eryn.flows import ZukoFlow

    rng = np.random.default_rng(0)
    old = rng.standard_normal((700, 3))
    new = rng.standard_normal((300, 3)) + 12.0  # newest rows, far away
    buf = np.concatenate([old, new], axis=0)  # temporal order: old -> new

    def _flow():
        torch.manual_seed(1)
        return ZukoFlow(
            dims=3, device="cpu", data_transform=None, conditioning=None,
            seed=1, flow_class="NSF", transforms=2,
            hidden_features=(32, 32), bins=4,
        )

    h_rand = _flow().fit(buf, n_epochs=3, seed=0, validation_fraction=0.3,
                         val_split="random")
    h_temp = _flow().fit(buf, n_epochs=3, seed=0, validation_fraction=0.3,
                         val_split="temporal")
    # identical nets/data; only the val membership differs. The temporal val
    # set is pure outliers -> much larger NLL than the mixed random val set.
    assert h_temp.validation_loss[-1] > h_rand.validation_loss[-1] + 5.0


def test_fit_train_noise_trains_and_smooths():
    """train_noise > 0 still trains, and both losses stay finite."""
    flow, samples = _make_flow(seed=3)
    history = flow.fit(
        samples[:400], n_epochs=4, seed=3, train_noise=0.2,
        val_split="temporal", validation_fraction=0.25,
    )
    assert np.isfinite(history.training_loss).all()
    assert np.isfinite(history.validation_loss).all()
    # training still reduces the loss despite the jitter
    assert history.training_loss[-1] < history.training_loss[0]
