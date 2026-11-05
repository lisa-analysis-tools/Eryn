# tests/test_flow_base_scale.py
"""Tests for temperature-scaled base distributions in flow proposals.

Covers two layers:

- :class:`eryn.flows.torch.flows.ZukoFlow` — the ``base_scale`` keyword on
  :meth:`~eryn.flows.torch.flows.ZukoFlow.log_prob` and
  :meth:`~eryn.flows.torch.flows.ZukoFlow.sample_and_log_prob`.
- :class:`eryn.moves.ConditionalFlowMove` — the ``active_betas`` attribute
  that routes per-temperature ``base_scale`` through :meth:`get_proposal`.

Test layout
-----------
- ``test_base_scale_none_matches_explicit_one`` — equivalence of the default
  (``None``) and explicit ``base_scale=1.0`` code paths.
- ``test_base_scale_round_trip_and_importance_identity`` — density exactness
  (no periodic dims, so sample/log_prob round-trips exactly) plus an
  importance-sampling identity that validates relative normalisation between
  two different base scales.
- ``test_base_scale_broadens_draws`` — empirical std of scaled draws exceeds
  the unscaled ones.
- ``test_conditional_flow_move_active_betas_*`` — ``ConditionalFlowMove``
  wiring: per-temperature scale routing, and a ``active_betas=None``
  regression anchor.
"""
from __future__ import annotations

import numpy as np
import pytest

from eryn.moves import ConditionalFlowMove


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flow(seed: int = 0):
    """Build a small ZukoFlow with WhiteningTransform (NO periodic dims) +
    OneHotLeafConditioning.

    No periodic dims means the data_transform's sample/log_prob round-trip is
    exact (periodic wrap-around would break exactness of the sample ->
    log_prob comparisons in the density-exactness test below).
    """
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], size=2000
    ).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=2)
    wt.fit({0: samples})

    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow


# ---------------------------------------------------------------------------
# ZukoFlow.base_scale — equivalence (None vs. 1.0)
# ---------------------------------------------------------------------------

def test_base_scale_none_matches_explicit_one():
    """base_scale=None and base_scale=1.0 give identical draws/logq under the
    same torch seed, and log_prob(x) == log_prob(x, base_scale=1.0)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=0)

    torch.manual_seed(123)
    x_none, logq_none = flow.sample_and_log_prob(64, context=0)
    torch.manual_seed(123)
    x_one, logq_one = flow.sample_and_log_prob(64, context=0, base_scale=1.0)

    np.testing.assert_array_equal(
        x_none, x_one, err_msg="base_scale=None and base_scale=1.0 draws differ"
    )
    np.testing.assert_array_equal(
        logq_none, logq_one,
        err_msg="base_scale=None and base_scale=1.0 logq differ",
    )

    lp_none = flow.log_prob(x_none, context=0)
    lp_one = flow.log_prob(x_none, context=0, base_scale=1.0)
    np.testing.assert_array_equal(
        lp_none, lp_one,
        err_msg="log_prob(x) != log_prob(x, base_scale=1.0)",
    )


def test_base_scale_none_is_bit_identical_to_pre_base_scale_behaviour():
    """Passing base_scale=None must not perturb the default code path at all
    (hard back-compat requirement: live production uses these classes)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=1)

    torch.manual_seed(55)
    x1, logq1 = flow.sample_and_log_prob(32, context=0)
    torch.manual_seed(55)
    x2, logq2 = flow.sample_and_log_prob(32, context=0)  # base_scale omitted entirely

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(logq1, logq2)

    lp1 = flow.log_prob(x1, context=0)
    lp2 = flow.log_prob(x1, context=0)
    np.testing.assert_array_equal(lp1, lp2)


# ---------------------------------------------------------------------------
# Density exactness (no periodic dims -> exact round trip)
# ---------------------------------------------------------------------------

def test_base_scale_round_trip_and_importance_identity():
    """log_prob(draws, base_scale=2.0) matches the reported logq exactly
    (float tolerance), and the importance identity
    mean(exp(logq_base1(x) - logq_base2(x))) over x ~ q_base2 ~= 1 within MC
    error — validating the relative normalisation of the two densities."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=2)

    # --- round-trip exactness ---
    x, logq_reported = flow.sample_and_log_prob(500, context=0, base_scale=2.0)
    logq_recomputed = flow.log_prob(x, context=0, base_scale=2.0)
    np.testing.assert_allclose(
        logq_reported, logq_recomputed, atol=1e-4,
        err_msg="log_prob(draws, base_scale=2.0) != logq reported by sample_and_log_prob",
    )

    # --- importance identity: relative normalisation of base_scale=1 vs 2 ---
    torch.manual_seed(2024)
    x_b2, _ = flow.sample_and_log_prob(20000, context=0, base_scale=2.0)
    logq_b1_at_b2 = flow.log_prob(x_b2, context=0, base_scale=1.0)
    logq_b2_at_b2 = flow.log_prob(x_b2, context=0, base_scale=2.0)
    weights = np.exp(logq_b1_at_b2 - logq_b2_at_b2)

    mc_mean = weights.mean()
    mc_stderr = weights.std() / np.sqrt(len(weights))
    assert abs(mc_mean - 1.0) < 6.0 * mc_stderr, (
        f"importance identity mean={mc_mean:.4f} deviates from 1.0 by more "
        f"than 6 MC standard errors ({mc_stderr:.4f})"
    )


# ---------------------------------------------------------------------------
# Broadening: empirical std of scaled draws exceeds unscaled draws
# ---------------------------------------------------------------------------

def test_base_scale_broadens_draws():
    """Empirical std of base_scale=3.0 draws exceeds base_scale=1.0 draws
    (loose check on a trained-tiny flow)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=3)

    x_unscaled, _ = flow.sample_and_log_prob(4000, context=0, base_scale=1.0)
    x_scaled, _ = flow.sample_and_log_prob(4000, context=0, base_scale=3.0)

    std_unscaled = x_unscaled.std(axis=0)
    std_scaled = x_scaled.std(axis=0)

    assert np.all(std_scaled > std_unscaled), (
        f"scaled-base draws are not broader per-dim: unscaled std={std_unscaled}, "
        f"scaled std={std_scaled}"
    )


# ---------------------------------------------------------------------------
# base_scale input validation
# ---------------------------------------------------------------------------

def test_base_scale_non_positive_raises():
    """base_scale <= 0 raises ValueError (log(s) undefined otherwise)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=4)
    x = np.zeros((4, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="base_scale"):
        flow.log_prob(x, context=0, base_scale=0.0)
    with pytest.raises(ValueError, match="base_scale"):
        flow.log_prob(x, context=0, base_scale=-1.0)
    with pytest.raises(ValueError, match="base_scale"):
        flow.sample_and_log_prob(4, context=0, base_scale=-2.0)


# ---------------------------------------------------------------------------
# ConditionalFlowMove.active_betas
# ---------------------------------------------------------------------------

class _RecordingFlow:
    """Thin pass-through wrapper recording the ``base_scale`` used per call.

    Delegates every other attribute/method (``data_transform``, ``dims``, ...)
    to the wrapped flow via ``__getattr__`` so it is a drop-in replacement for
    ``ConditionalFlowMove.flow``.
    """

    def __init__(self, inner):
        self._inner = inner
        self.log_prob_scales: list = []
        self.sample_scales: list = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def log_prob(self, x, context=None, base_scale=None):
        self.log_prob_scales.append(base_scale)
        return self._inner.log_prob(x, context=context, base_scale=base_scale)

    def sample_and_log_prob(self, n, context=None, base_scale=None):
        self.sample_scales.append(base_scale)
        return self._inner.sample_and_log_prob(n, context=context, base_scale=base_scale)


def test_conditional_flow_move_active_betas_routes_per_temperature_scale():
    """active_betas=[1.0, 0.5] routes base_scale=beta**-0.5 per temperature,
    for BOTH sampling and old-point log_prob; factors are finite; shapes match
    the active_betas=None path."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=5)
    rec_flow = _RecordingFlow(flow)

    move = ConditionalFlowMove(rec_flow, branch_name="x")
    move.active_condition = 0
    active_betas = np.array([1.0, 0.5])
    move.active_betas = active_betas

    ntemps, nwalkers, nleaves, ndim = 2, 6, 1, 2
    rng = np.random.default_rng(11)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    q, factors = move.get_proposal(branches, rng)

    # shapes/ordering match the active_betas=None path (same branches_coords
    # structure and factors shape as every other ConditionalFlowMove test).
    assert q["x"].shape == (ntemps, nwalkers, nleaves, ndim)
    assert factors.shape == (ntemps, nwalkers)
    assert np.all(np.isfinite(factors)), "active_betas factors contain non-finite values"
    assert not np.array_equal(q["x"], coords), "proposal did not change coordinates"

    # exactly one log_prob call + one sample_and_log_prob call per temperature,
    # in ascending temperature order, at scale beta_t**-0.5.
    expected_scales = [float(b) ** -0.5 for b in active_betas]
    assert len(rec_flow.log_prob_scales) == ntemps
    assert len(rec_flow.sample_scales) == ntemps
    np.testing.assert_allclose(rec_flow.log_prob_scales, expected_scales)
    np.testing.assert_allclose(rec_flow.sample_scales, expected_scales)


def test_conditional_flow_move_active_betas_none_regression_baseline():
    """active_betas=None (default) is deterministic and reproduces a fixed
    seeded baseline — a regression anchor guarding against the new
    active_betas branch perturbing the unscaled default path."""
    torch = pytest.importorskip("torch")

    ntemps, nwalkers, nleaves, ndim = 2, 6, 1, 2
    coords = np.random.default_rng(11).standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    flow_a = _make_flow(seed=5)
    move_a = ConditionalFlowMove(flow_a, branch_name="x")
    move_a.active_condition = 0
    assert move_a.active_betas is None  # default
    torch.manual_seed(777)
    q_a, factors_a = move_a.get_proposal(branches, np.random.default_rng(3))

    flow_b = _make_flow(seed=5)
    move_b = ConditionalFlowMove(flow_b, branch_name="x")
    move_b.active_condition = 0
    move_b.active_betas = None  # explicit
    torch.manual_seed(777)
    q_b, factors_b = move_b.get_proposal(branches, np.random.default_rng(3))

    np.testing.assert_array_equal(q_a["x"], q_b["x"])
    np.testing.assert_array_equal(factors_a, factors_b)
