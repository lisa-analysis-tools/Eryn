# tests/test_whitening_periodic_cholesky.py
"""Tests for ``WhiteningTransform(periodic_in_cholesky=True)``.

Background
----------
The default (``periodic_in_cholesky=False``) scheme whitens block-diagonally:
full Cholesky for the non-periodic dims, marginal-std scaling for the
(post-``CircularShift``) periodic dims.  Cross-correlations between periodic
angles and non-periodic dims therefore survive whitening and must be learned
raw by the downstream flow.  ``periodic_in_cholesky=True`` instead folds the
shifted periodic dims into a single full-covariance Cholesky, decorrelating
them from the rest at the transform level.

Pinned contract (do NOT test around this)
------------------------------------------
``CircularShiftTransform._inverse`` deliberately folds draws past the wrap cut
into ``[0, T)``; round-trip error is only guaranteed small for the *typical*
(non-cut-crossing) draw.  Tests here assert median round-trip error is tiny
and cut-crossers are rare — never ``allclose`` over all rows.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")  # noqa: F841

from eryn.flows import OneHotLeafConditioning, WhiteningTransform, ZukoFlow  # noqa: E402

PERIOD = 2 * np.pi


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _correlated_samples(
    n: int = 20_000, seed: int = 0, center: float = 3.0, slope: float = 0.3
) -> np.ndarray:
    """(n, 2) float64 samples: non-periodic dim 0, periodic dim 1 strongly
    correlated with it.  ``center`` is chosen well clear of the wrap boundary
    (0 / 2*pi) so the antimode cut lands away from the bulk and wrap-crossers
    stay rare.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n) * 2.0
    noise = rng.normal(0.0, 0.05, n)
    theta = (center + slope * x0 + noise) % PERIOD
    return np.column_stack([x0, theta]).astype(np.float64)


def _fitted_pair(seed: int = 0):
    """Return (wt_false, wt_true, samples) fit on the same correlated data."""
    samples = _correlated_samples(seed=seed)
    wt_false = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)}, periodic_in_cholesky=False)
    wt_true = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)}, periodic_in_cholesky=True)
    wt_false.fit({0: samples})
    wt_true.fit({0: samples})
    return wt_false, wt_true, samples


# ---------------------------------------------------------------------------
# 1. Cross-block decorrelation
# ---------------------------------------------------------------------------

def test_periodic_in_cholesky_false_retains_cross_correlation():
    """Default (block-diagonal) whitening leaves the periodic/non-periodic
    cross-correlation intact — the bug this feature fixes."""
    wt_false, _, samples = _fitted_pair()
    z = np.asarray(wt_false.forward(samples, condition=0), dtype=np.float64)
    cov = np.cov(z.T)
    # strongly correlated input (slope 0.3, x0 std 2) -> whitened cross term
    # should remain large, not collapse toward 0.
    assert abs(cov[0, 1]) > 0.5, f"expected retained correlation, got cov={cov}"


def test_periodic_in_cholesky_true_decorrelates_cross_block():
    """periodic_in_cholesky=True whitens the full vector (including the
    shifted periodic dim) to ~identity covariance, cross terms included."""
    _, wt_true, samples = _fitted_pair()
    z = np.asarray(wt_true.forward(samples, condition=0), dtype=np.float64)
    cov = np.cov(z.T)
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.1)
    assert abs(cov[0, 1]) < 0.05, f"expected near-zero cross term, got cov={cov}"


# ---------------------------------------------------------------------------
# 2. Round-trip: median error tiny, cut-crossers rare
# ---------------------------------------------------------------------------

def test_round_trip_median_error_tiny_crossers_rare():
    """forward -> inverse median error is tiny; wrap-cut crossers are rare.

    Per the pinned CircularShiftTransform contract, do NOT assert allclose
    over all rows -- only the median (typical draw) and crosser rarity.
    """
    _, wt_true, samples = _fitted_pair()
    z = wt_true.forward(samples, condition=0)
    x_back = np.asarray(wt_true.inverse(z, condition=0))

    err0 = np.abs(x_back[:, 0] - samples[:, 0])
    dphi = np.abs((x_back[:, 1] - samples[:, 1] + np.pi) % PERIOD - np.pi)

    assert np.median(err0) < 1e-4
    assert np.median(dphi) < 1e-4

    crossers = dphi > 0.1
    assert crossers.mean() < 0.01, f"too many wrap-cut crossers: {crossers.mean():.4f}"


# ---------------------------------------------------------------------------
# 3. Log-det exactness
# ---------------------------------------------------------------------------

def test_log_det_forward_inverse_sum_to_zero():
    """forward log_abs_det_jacobian + inverse log_abs_det_jacobian ~ 0 pointwise
    (affine map: exact inverses, so this should hold to float64 precision)."""
    _, wt_true, samples = _fitted_pair()
    x = torch.as_tensor(samples[:32], dtype=torch.float64)
    z = wt_true.transforms[0](x)

    ladj_fwd = wt_true.log_abs_det_jacobian(x, z, condition=0)
    ladj_inv = wt_true.transforms[0].inv.log_abs_det_jacobian(z, x)

    total = ladj_fwd + ladj_inv
    np.testing.assert_allclose(total.numpy(), 0.0, atol=1e-8)


def test_log_det_matches_analytic_cholesky_diagonal():
    """The forward log-det is a constant (affine map) equal to the log|det| of
    the whitening matrix -- validate against slogdet of the raw matrix."""
    _, wt_true, samples = _fitted_pair()
    x = torch.as_tensor(samples[:10], dtype=torch.float64)
    z = wt_true.transforms[0](x)
    ladj = wt_true.log_abs_det_jacobian(x, z, condition=0)

    # matrix_transform is always the last part of the composed transform;
    # CircularShift is volume-preserving (log-det 0) and centering is a
    # unit-scale affine (log-det 0), so the analytic total is slogdet(matrix).
    matrix = wt_true.transforms[0].parts[-1].matrix
    analytic = torch.linalg.slogdet(matrix)[1]

    # constant across samples
    np.testing.assert_allclose(ladj.numpy(), analytic.item(), atol=1e-8)


def test_log_det_matches_numerical_jacobian():
    """Analytic log|det J| matches a float64 finite-difference estimate."""
    _, wt_true, samples = _fitted_pair()
    pts = samples[:5].copy()
    eps = 1e-5

    def fwd64(x):
        return wt_true.transforms[0](torch.as_tensor(x, dtype=torch.float64)).numpy()

    z0 = fwd64(pts)
    analytic = wt_true.log_abs_det_jacobian(
        torch.as_tensor(pts), torch.as_tensor(z0), condition=0
    ).numpy()

    for k in range(pts.shape[0]):
        J = np.zeros((2, 2))
        for j in range(2):
            xp = pts[k].copy()
            xm = pts[k].copy()
            xp[j] += eps
            xm[j] -= eps
            zp = fwd64(xp[None])[0]
            zm = fwd64(xm[None])[0]
            J[:, j] = (zp - zm) / (2 * eps)
        numerical = np.linalg.slogdet(J)[1]
        assert np.isclose(analytic[k], numerical, atol=1e-3), (
            f"point {k}: analytic {analytic[k]} vs numerical {numerical}"
        )


# ---------------------------------------------------------------------------
# 4. shared=False: per-condition whitening under different correlation
#    structures, each whitens to ~identity under its own map.
# ---------------------------------------------------------------------------

def test_per_condition_whitens_to_identity_under_own_map():
    """Two conditions with different correlation structures each whiten to
    ~identity covariance under their own (per-condition) fitted map."""
    samples_a = _correlated_samples(seed=1, center=2.0, slope=0.5)
    samples_b = _correlated_samples(seed=2, center=4.5, slope=-0.4)

    wt = WhiteningTransform(
        ndim=2, periodic={1: (0.0, PERIOD)}, shared=False, periodic_in_cholesky=True
    )
    wt.fit({0: samples_a, 1: samples_b})

    za = np.asarray(wt.forward(samples_a, condition=0), dtype=np.float64)
    zb = np.asarray(wt.forward(samples_b, condition=1), dtype=np.float64)

    cov_a = np.cov(za.T)
    cov_b = np.cov(zb.T)

    np.testing.assert_allclose(np.diag(cov_a), 1.0, atol=0.1)
    np.testing.assert_allclose(np.diag(cov_b), 1.0, atol=0.1)
    assert abs(cov_a[0, 1]) < 0.05, f"condition 0 not decorrelated: {cov_a}"
    assert abs(cov_b[0, 1]) < 0.05, f"condition 1 not decorrelated: {cov_b}"


# ---------------------------------------------------------------------------
# 5. Pickle back-compat: old checkpoint (no attribute) loads as False
# ---------------------------------------------------------------------------

def test_pickle_roundtrip_preserves_true_behavior():
    """A transform fitted with periodic_in_cholesky=True survives a normal
    pickle round-trip with identical forward / log-det output."""
    _, wt_true, samples = _fitted_pair()
    pts = samples[:10]
    z_before = wt_true.forward(pts, condition=0)
    ladj_before = wt_true.log_abs_det_jacobian(torch.as_tensor(pts), z_before, condition=0)

    wt2 = pickle.loads(pickle.dumps(wt_true))

    z_after = wt2.forward(pts, condition=0)
    ladj_after = wt2.log_abs_det_jacobian(torch.as_tensor(pts), z_after, condition=0)

    assert torch.equal(z_before, z_after)
    assert torch.allclose(ladj_before, ladj_after)


def test_pickle_missing_attribute_behaves_as_false():
    """Simulate an old checkpoint pickled before periodic_in_cholesky existed:
    delete the attribute from __dict__, pickle round-trip, then refit -- must
    behave exactly like an explicit periodic_in_cholesky=False instance.
    """
    samples = _correlated_samples(seed=3)

    wt_old_shape = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})
    wt_old_shape.fit({0: samples})
    # Simulate a pre-feature pickle: the attribute was never set.
    del wt_old_shape.__dict__["periodic_in_cholesky"]

    blob = pickle.dumps(wt_old_shape)
    wt_restored = pickle.loads(blob)
    assert "periodic_in_cholesky" not in wt_restored.__dict__

    # Forward/inverse must still work (no AttributeError) before any refit.
    z = wt_restored.forward(samples[:5], condition=0)
    assert torch.isfinite(z).all()

    # Refit (simulating a later re-fit call on the restored object) must
    # behave as periodic_in_cholesky=False, matching an explicit instance.
    wt_restored.fit({0: samples})
    wt_explicit_false = WhiteningTransform(
        ndim=2, periodic={1: (0.0, PERIOD)}, periodic_in_cholesky=False
    )
    wt_explicit_false.fit({0: samples})

    z_restored = wt_restored.forward(samples[:20], condition=0)
    z_explicit = wt_explicit_false.forward(samples[:20], condition=0)
    assert torch.allclose(z_restored, z_explicit, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. End-to-end CPU smoke test: tiny ZukoFlow NSF, 2-condition dataset
# ---------------------------------------------------------------------------

def test_end_to_end_flow_smoke():
    """A tiny ZukoFlow NSF fits a 2-condition dataset with
    periodic_in_cholesky=True; samples stay within periodic bounds and land
    near their own condition's center."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    center_0, center_1 = 1.0, 4.5
    n = 600
    data_0 = np.column_stack([
        rng.normal(0.0, 1.0, n),
        (rng.normal(center_0, 0.2, n)) % PERIOD,
    ]).astype(np.float64)
    data_1 = np.column_stack([
        rng.normal(5.0, 1.0, n),
        (rng.normal(center_1, 0.2, n)) % PERIOD,
    ]).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=2)
    wt = WhiteningTransform(
        ndim=2, periodic={1: (0.0, PERIOD)}, periodic_in_cholesky=True
    )
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=0,
        flow_class="NSF",
        transforms=2,
        hidden_features=(32, 32),
        bins=4,
    )
    history = flow.fit(
        {0: data_0, 1: data_1},
        n_epochs=5,
        lr=1e-2,
        batch_size=128,
        validation_fraction=0.2,
        seed=0,
        verbose=False,
    )
    assert len(history.validation_loss) >= 1
    assert wt.is_fitted

    samples_0 = flow.sample(200, context=0)
    samples_1 = flow.sample(200, context=1)

    # periodic dim stays within bounds
    assert np.all(samples_0[:, 1] >= 0.0) and np.all(samples_0[:, 1] < PERIOD)
    assert np.all(samples_1[:, 1] >= 0.0) and np.all(samples_1[:, 1] < PERIOD)

    # per-condition draws land near their own condition's center (loose,
    # circular-distance check; a handful of untrained-flow outliers tolerated
    # via median rather than mean).
    def circ_dist(a, b):
        return np.abs((a - b + np.pi) % PERIOD - np.pi)

    assert np.median(circ_dist(samples_0[:, 1], center_0)) < 1.0
    assert np.median(circ_dist(samples_1[:, 1], center_1)) < 1.0
    assert np.median(np.abs(samples_0[:, 0] - 0.0)) < 3.0
    assert np.median(np.abs(samples_1[:, 0] - 5.0)) < 3.0
