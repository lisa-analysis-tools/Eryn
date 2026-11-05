# tests/test_flow_transforms.py
"""Tests for the torch-backed WhiteningTransform and its helper transforms.

Guards
------
``torch`` and ``zuko`` are optional extras; the entire module is skipped when
either is absent.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from eryn.flows.torch.transforms import (  # noqa: E402
    CircularShiftTransform,
    LogTransform,
    WhiteningTransform,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_samples(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Return (n, 3) float64 samples: two correlated Gaussians + one periodic angle."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    g = rng.multivariate_normal([2.0, -1.0], cov, size=n)
    ang = (rng.normal(1.0, 0.3, size=n)) % (2 * np.pi)  # bulk away from 0/2pi wrap
    return np.column_stack([g, ang]).astype(np.float64)


def _fitted_wt() -> tuple[WhiteningTransform, np.ndarray]:
    """Return a fitted WhiteningTransform and the training samples."""
    samples = _make_samples()
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    wt.fit({0: samples})
    return wt, samples


# ---------------------------------------------------------------------------
# is_fitted
# ---------------------------------------------------------------------------

def test_is_fitted_false_before_fit():
    """is_fitted is False before fit() is called."""
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    assert not wt.is_fitted


def test_is_fitted_true_after_fit():
    """is_fitted is True after fit() is called."""
    wt, _ = _fitted_wt()
    assert wt.is_fitted


# ---------------------------------------------------------------------------
# forward / inverse round-trip
# ---------------------------------------------------------------------------

def test_forward_inverse_roundtrip():
    """forward then inverse reproduces original samples (periodic dim compared mod period)."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples, condition=0)
    x_back = wt.inverse(z, condition=0)
    # non-periodic dims
    assert np.allclose(x_back[:, :2], samples[:, :2], atol=1e-4)
    # periodic dim — compare modulo period
    d = np.abs((x_back[:, 2] - samples[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    assert np.max(d) < 1e-4


def test_forward_returns_float32_tensor():
    """forward() must return a float32 torch.Tensor (flow-input contract)."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples[:10], condition=0)
    assert isinstance(z, torch.Tensor)
    assert z.dtype == torch.float32


def test_forward_output_shape():
    """forward() output shape matches input shape."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples[:17], condition=0)
    assert z.shape == (17, 3)


# ---------------------------------------------------------------------------
# log_abs_det_jacobian shape — regression test
# ---------------------------------------------------------------------------

def test_log_abs_det_jacobian_shape_with_periodic():
    """Regression: with a periodic dim the log-det must stay (N,), not broadcast to (N, N).

    The PartialTransform wrapping the circular shift returned a per-dim (N, k) term that
    ComposeTransform summed against the (N,) terms, silently producing an (N, N) Jacobian.
    """
    wt, samples = _fitted_wt()
    x = samples[:7]
    z = wt.forward(x, condition=0)
    ladj = wt.log_abs_det_jacobian(torch.tensor(x), z, condition=0)
    assert tuple(ladj.shape) == (7,), f"expected (7,), got {tuple(ladj.shape)}"


# ---------------------------------------------------------------------------
# log_abs_det_jacobian correctness — float64 finite-difference gate
# ---------------------------------------------------------------------------

def test_log_abs_det_jacobian_matches_numerical():
    """Analytic log|det J| must match a float64 finite-difference estimate.

    This is the headline correctness gate: catches a wrong/missing periodic-angle
    Jacobian — the silent-bias hole described in the spec.
    """
    wt, samples = _fitted_wt()
    pts = samples[:5].copy()  # bulk points, away from wrap
    eps = 1e-5

    # Compute latents in float64 directly via the internal transform.
    # ``wt.forward`` casts to float32, which would wreck the finite-difference
    # derivative below via catastrophic cancellation in (zp - zm) / 2eps.
    # The analytic log|det| is float64 and exact.
    def fwd64(x):
        return wt.transforms[0](torch.as_tensor(x, dtype=torch.float64)).numpy()

    z0 = fwd64(pts)
    analytic = wt.log_abs_det_jacobian(
        torch.as_tensor(pts), torch.as_tensor(z0), condition=0
    ).numpy()

    for k in range(pts.shape[0]):
        J = np.zeros((3, 3))
        for j in range(3):
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
# fit with dict of conditions
# ---------------------------------------------------------------------------

def test_fit_dict_of_conditions():
    """fit() accepts a dict[int, ndarray] and builds per-condition transforms."""
    rng = np.random.default_rng(42)
    s0 = rng.standard_normal((500, 3))
    s1 = rng.standard_normal((500, 3)) + 5.0
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: s0, 1: s1})
    assert wt.is_fitted
    assert 0 in wt.transforms
    assert 1 in wt.transforms
    z0 = wt.forward(s0[:10], condition=0)
    z1 = wt.forward(s1[:10], condition=1)
    assert z0.shape == (10, 3)
    assert z1.shape == (10, 3)


def test_fit_array_uses_condition_zero():
    """fit() with a plain array stores the transform under condition 0."""
    wt = WhiteningTransform(ndim=3)
    samples = _make_samples(n=200)
    wt.fit(samples)
    assert 0 in wt.transforms


# ---------------------------------------------------------------------------
# pickle round-trip
# ---------------------------------------------------------------------------

def test_pickle_roundtrip():
    """WhiteningTransform must survive pickle.loads(pickle.dumps(wt)) identically.

    Load-bearing for Task-9 multiprocessing spawn: the worker process receives
    the transform via pickle and must produce bit-identical forward / log-det.
    """
    wt, samples = _fitted_wt()
    pts = samples[:10]
    z_before = wt.forward(pts, condition=0)
    ladj_before = wt.log_abs_det_jacobian(
        torch.as_tensor(pts), z_before, condition=0
    )

    wt2 = pickle.loads(pickle.dumps(wt))

    z_after = wt2.forward(pts, condition=0)
    ladj_after = wt2.log_abs_det_jacobian(
        torch.as_tensor(pts), z_after, condition=0
    )

    assert torch.equal(z_before, z_after), "forward output differs after pickle round-trip"
    assert torch.allclose(ladj_before, ladj_after), (
        "log_abs_det_jacobian differs after pickle round-trip"
    )


# ---------------------------------------------------------------------------
# periodic_indices / non_periodic_indices properties
# ---------------------------------------------------------------------------

def test_periodic_and_non_periodic_indices():
    """periodic_indices and non_periodic_indices partition range(ndim)."""
    wt = WhiteningTransform(ndim=5, periodic={1: (0.0, 1.0), 3: (0.0, 2 * np.pi)})
    assert sorted(wt.periodic_indices) == [1, 3]
    assert sorted(wt.non_periodic_indices) == [0, 2, 4]


# ---------------------------------------------------------------------------
# unknown condition raises ValueError
# ---------------------------------------------------------------------------

def test_forward_unknown_condition_raises():
    """forward() with an unknown condition raises ValueError."""
    wt, _ = _fitted_wt()
    with pytest.raises(ValueError, match="condition"):
        wt.forward(_make_samples(n=5), condition=99)


# ---------------------------------------------------------------------------
# CircularShiftTransform round-trip semantics — explicit invariants
# ---------------------------------------------------------------------------

def test_circular_shift_round_trip_consistency():
    """Pin the exact round-trip math of CircularShiftTransform.

    Two identities hold for the circular shift (forward ``_call`` wraps into the
    fundamental window ``[-T/2, T/2)``; inverse ``_inverse`` canonicalises into
    ``[0, T)``):

      1.  ``_inverse(_call(x)) == x (mod T)`` for every real ``x`` (a dense
          coords grid spanning several periods).
      2.  ``_call(_inverse(z)) == wrap_{[-T/2, T/2)}(z)`` for every real ``z``
          (a dense latent grid spanning ``[-T, T]``).  In particular it equals
          ``z`` exactly when ``z`` already lies in ``[-T/2, T/2)``, and folds it
          back by a full period otherwise.

    Identity (2) is the documented limitation behind the periodic
    sample/log_prob aliasing: the forward modulo is what makes the map robust to
    arbitrary external coords, and it is precisely what prevents ``_call`` from
    recovering a latent the flow drew outside the window.
    """
    period = 2 * np.pi
    half = period / 2.0
    shift = torch.tensor(0.7, dtype=torch.float64)
    t = CircularShiftTransform(shift, period)

    # (1) inverse(call(x)) == x (mod period), dense coords grid over 4 periods.
    x = torch.linspace(-2 * period, 2 * period, 20001, dtype=torch.float64)
    x_rt = t._inverse(t._call(x))
    d = (x_rt - x + half) % period - half
    assert torch.max(torch.abs(d)).item() < 1e-9, "inverse(call(x)) != x (mod T)"

    # (2) call(inverse(z)) == wrap into [-half, half), dense latent grid.
    z = torch.linspace(-period, period, 20001, dtype=torch.float64)
    z_rt = t._call(t._inverse(z))
    z_wrapped = ((z + half) % period) - half
    assert torch.max(torch.abs(z_rt - z_wrapped)).item() < 1e-9, (
        "call(inverse(z)) != wrap_{[-half,half)}(z)"
    )
    # Exact identity on the fundamental window.
    z_in = torch.linspace(-half, half - 1e-9, 20001, dtype=torch.float64)
    assert torch.max(torch.abs(t._call(t._inverse(z_in)) - z_in)).item() < 1e-9, (
        "call(inverse(z)) != z on the fundamental window [-half, half)"
    )


def test_circular_shift_inverse_in_canonical_bounds():
    """_inverse always returns canonical coords in [0, period).

    The canonical representative keeps log_prob a proper (periodic) density on
    the circle; dropping it would make log_prob(theta) != log_prob(theta + T)
    and bias the ConditionalFlowMove Hastings factor.  This pins the [0, T) convention.
    """
    period = 2 * np.pi
    t = CircularShiftTransform(torch.tensor(1.3, dtype=torch.float64), period)
    z = torch.linspace(-3 * period, 3 * period, 5001, dtype=torch.float64)
    x = t._inverse(z)
    assert torch.all(x >= 0.0) and torch.all(x < period), (
        "inverse output left the canonical [0, period) window"
    )


# ---------------------------------------------------------------------------
# LogTransform smoke test
# ---------------------------------------------------------------------------

def test_log_transform_round_trip_and_ladj():
    """LogTransform: forward/inverse round-trip and log_abs_det_jacobian finite on positive inputs."""
    t = LogTransform()
    x = torch.tensor([0.5, 1.0, 2.0, 10.0], dtype=torch.float32)
    y = t(x)
    x_back = t.inv(y)
    assert torch.allclose(x, x_back, atol=1e-6), "round-trip failed"
    ladj = t.log_abs_det_jacobian(x, y)
    assert torch.all(torch.isfinite(ladj)), "log_abs_det_jacobian not finite"


# ---------------------------------------------------------------------------
# numpy-array boundary coercion (periodic dims)
# ---------------------------------------------------------------------------

def test_forward_accepts_numpy_with_periodic():
    """forward() accepts a numpy array when periodic dims are present (shape + finiteness)."""
    wt, samples = _fitted_wt()
    x_np = samples[:8]  # plain ndarray
    z = wt.forward(x_np, condition=0)
    assert z.shape == (8, 3)
    assert torch.all(torch.isfinite(z))


def test_log_abs_det_jacobian_accepts_numpy_with_periodic():
    """log_abs_det_jacobian() accepts numpy arrays and matches tensor-input result."""
    wt, samples = _fitted_wt()
    x_np = samples[:8]
    z_np = wt.forward(x_np, condition=0).numpy().astype(np.float64)

    # tensor inputs (reference)
    ladj_tensor = wt.log_abs_det_jacobian(
        torch.as_tensor(x_np), torch.as_tensor(z_np), condition=0
    )
    # numpy inputs
    ladj_numpy = wt.log_abs_det_jacobian(x_np, z_np, condition=0)

    assert ladj_numpy.shape == (8,)
    assert torch.all(torch.isfinite(ladj_numpy))
    assert torch.allclose(ladj_tensor.float(), ladj_numpy.float(), atol=1e-5)


def test_inverse_unknown_condition_raises():
    """inverse() with an unknown condition raises ValueError (symmetry with forward)."""
    wt, samples = _fitted_wt()
    z_dummy = wt.forward(samples[:5], condition=0)
    with pytest.raises(ValueError, match="condition"):
        wt.inverse(z_dummy, condition=99)


def test_forward_no_userwarning_on_tensor_input():
    """forward() must not emit UserWarning when given an existing tensor."""
    wt, samples = _fitted_wt()
    x_tensor = torch.as_tensor(samples[:5], dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Should not raise — as_tensor avoids the copy-construct warning
        z = wt.forward(x_tensor, condition=0)
    assert z.shape == (5, 3)


# ---------------------------------------------------------------------------
# shared mode
# ---------------------------------------------------------------------------

def _shared_wt():
    """Shared WhiteningTransform fit on two differently-located leaf clouds."""
    rng = np.random.default_rng(1)
    a = np.column_stack([rng.normal(0.0, 1.0, (1500, 2)),
                         rng.normal(1.0, 0.3, 1500) % (2 * np.pi)])
    b = np.column_stack([rng.normal(5.0, 2.0, (1500, 2)),
                         rng.normal(4.0, 0.5, 1500) % (2 * np.pi)])
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)}, shared=True)
    wt.fit({0: a, 1: b})
    return wt, np.concatenate([a, b], axis=0)


def test_shared_uses_one_map_for_every_condition():
    """A shared transform whitens identically regardless of the condition arg."""
    wt, pooled = _shared_wt()
    x = pooled[:32]
    z0 = wt.forward(x, condition=0)
    z1 = wt.forward(x, condition=1)
    assert torch.equal(z0, z1)


def test_shared_accepts_unseen_condition():
    """The key robustness property: a condition never seen at fit time still works
    (a leaf that becomes active only after the one-time fit)."""
    wt, pooled = _shared_wt()
    x = pooled[:16]
    z = wt.forward(x, condition=99)  # leaf 99 never appeared in fit()
    assert z.shape == (16, 3) and torch.isfinite(z).all()
    ld = wt.log_abs_det_jacobian(x, z, condition=99)
    assert ld.shape == (16,) and torch.isfinite(ld).all()
    back = wt.inverse(z, condition=99)
    assert np.allclose(back % (2 * np.pi), x % (2 * np.pi), atol=1e-4) or \
        np.allclose(back[:, :2], x[:, :2], atol=1e-4)


def test_shared_round_trip_and_logdet_shape():
    wt, pooled = _shared_wt()
    x = pooled[:64]
    z = wt.forward(x, condition=0)
    back = wt.inverse(z, condition=0)
    assert np.allclose(back[:, :2], x[:, :2], atol=1e-4)
    assert wt.log_abs_det_jacobian(x, z, condition=0).shape == (64,)


def test_shared_is_picklable():
    wt, pooled = _shared_wt()
    x = pooled[:16]
    wt2 = pickle.loads(pickle.dumps(wt))
    assert wt2.shared is True and wt2.is_fitted
    assert torch.equal(wt.forward(x, 0), wt2.forward(x, 0))


def test_shared_fit_array_or_dict_equivalent():
    """Pooling a dict equals fitting the concatenation directly."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((500, 2))
    b = rng.standard_normal((500, 2)) + 3.0
    wt_dict = WhiteningTransform(ndim=2, shared=True)
    wt_dict.fit({0: a, 1: b})
    wt_arr = WhiteningTransform(ndim=2, shared=True)
    wt_arr.fit(np.concatenate([a, b], axis=0))
    x = np.concatenate([a, b], axis=0)[:20]
    assert torch.allclose(wt_dict.forward(x, 0), wt_arr.forward(x, 0), atol=1e-10)


def test_non_shared_still_raises_unseen_condition():
    """Default (per-condition) behavior is unchanged: unknown condition raises."""
    wt, _ = _fitted_wt()  # shared=False
    x = np.zeros((4, 3))
    with pytest.raises(ValueError, match="[Cc]ondition"):
        wt.forward(x, condition=99)


# ---------------------------------------------------------------------------
# periodic_cut: wrap-cut placement
# ---------------------------------------------------------------------------

PERIOD = 2 * np.pi


def _fitted_cut(wt, dim: int, condition: int = 0) -> float:
    """Extract the wrap-cut position of periodic dim ``dim`` from a fitted transform.

    The fitted per-condition ComposeTransform is
    [ComposeTransform(PartialTransform(CircularShiftTransform), ...), Affine, Linear];
    the cut of a CircularShiftTransform sits at ``shift + period / 2``.
    """
    for part in wt.transforms[condition].parts[0].parts:
        if part.dimensions == [dim]:
            cs = part.transform
            return (float(cs.shift) + cs.period / 2.0) % cs.period
    raise AssertionError(f"no circular shift found for dim {dim}")


def _circ_dist(a: float, b: float, period: float = PERIOD) -> float:
    return abs((a - b + period / 2.0) % period - period / 2.0)


def test_periodic_cut_antimode_avoids_antipodal_modes():
    """With two antipodal angle modes the cut must land BETWEEN them.

    The legacy circular-mean placement is noise-dominated here (the resultant
    vector of antipodal modes is ~0) and can put the cut on a mode; the
    antimode placement must keep it away from both bulks.
    """
    rng = np.random.default_rng(0)
    m1, m2, std = 1.0, 1.0 + np.pi, 0.3
    ang = np.concatenate([
        rng.normal(m1, std, 3000), rng.normal(m2, std, 3000)
    ]) % PERIOD
    samples = np.column_stack([rng.standard_normal(6000), ang])

    wt = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})
    wt.fit({0: samples})
    cut = _fitted_cut(wt, dim=1)

    # antipodal modes leave two gaps at m1 +/- pi/2; the cut must sit in one,
    # i.e. well clear (>4 sigma) of both mode centres
    assert _circ_dist(cut, m1) > 4 * std, f"cut {cut} on mode 1"
    assert _circ_dist(cut, m2) > 4 * std, f"cut {cut} on mode 2"


def test_periodic_cut_antimode_matches_antipode_when_unimodal():
    """For a unimodal angle the antimode cut reproduces the legacy placement."""
    rng = np.random.default_rng(1)
    bulk = 2.5
    ang = rng.normal(bulk, 0.5, 5000) % PERIOD
    samples = np.column_stack([rng.standard_normal(5000), ang])

    wt_new = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})
    wt_new.fit({0: samples})
    wt_old = WhiteningTransform(
        ndim=2, periodic={1: (0.0, PERIOD)}, periodic_cut="circular_mean"
    )
    wt_old.fit({0: samples})

    cut_new = _fitted_cut(wt_new, dim=1)
    cut_old = _fitted_cut(wt_old, dim=1)
    assert _circ_dist(cut_old, (bulk + np.pi) % PERIOD) < 0.1  # sanity: legacy = antipode
    assert _circ_dist(cut_new, cut_old) < 0.4  # antimode agrees within histogram noise


def test_periodic_cut_roundtrip_and_logdet_unchanged():
    """Antimode placement keeps the forward/inverse round-trip and (N,) log-det."""
    rng = np.random.default_rng(2)
    ang = np.concatenate([
        rng.normal(1.0, 0.3, 1000), rng.normal(1.0 + np.pi, 0.3, 1000)
    ]) % PERIOD
    samples = np.column_stack([rng.standard_normal(2000), ang])
    wt = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})
    wt.fit({0: samples})

    x = samples[:64]
    z = wt.forward(x, 0)
    x_back = wt.inverse(z, 0)
    np.testing.assert_allclose(x_back[:, 0], x[:, 0], atol=1e-6)
    dphi = np.abs((x_back[:, 1] - x[:, 1] + np.pi) % PERIOD - np.pi)
    np.testing.assert_allclose(dphi, 0.0, atol=1e-6)
    assert wt.log_abs_det_jacobian(x, z, 0).shape == (64,)


def test_periodic_cut_invalid_value_raises():
    with pytest.raises(ValueError, match="periodic_cut"):
        WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)}, periodic_cut="foo")


# ---------------------------------------------------------------------------
# Relative eps regularization (multi-scale whitening)
# ---------------------------------------------------------------------------

def test_whitening_unit_std_across_scales():
    """Dims whose variance is below the old absolute eps floor still whiten to ~1.

    Regression for the EMRI case: eccentricity std ~3e-5 (variance ~1e-9)
    was swamped by the previous absolute ``eps=1e-8`` covariance floor and
    whitened to z-std ~0.04 instead of 1.
    """
    rng = np.random.default_rng(3)
    n = 20_000
    stds = np.array([3e-5, 1e-4, 1.0, 5e3])
    samples = rng.standard_normal((n, 4)) * stds
    # add a periodic angle with tiny spread as well
    ang = rng.normal(1.0, 5e-5, size=n) % (2 * np.pi)
    samples = np.column_stack([samples, ang])

    wt = WhiteningTransform(ndim=5, periodic={4: (0.0, 2 * np.pi)})
    wt.fit({0: samples})
    z = np.asarray(wt.forward(samples, 0))
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=0.05)


def test_whitening_constant_dim_does_not_crash():
    """A zero-variance (constant) column must not break the Cholesky."""
    rng = np.random.default_rng(4)
    samples = np.column_stack([
        rng.standard_normal(500),
        np.full(500, 3.7),                      # constant non-periodic dim
        np.full(500, 1.2) % (2 * np.pi),        # constant periodic dim
    ])
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    wt.fit({0: samples})
    z = np.asarray(wt.forward(samples, 0))
    assert np.isfinite(z).all()
    x_back = np.asarray(wt.inverse(z, 0))
    np.testing.assert_allclose(x_back[:, 0], samples[:, 0], atol=1e-6)
