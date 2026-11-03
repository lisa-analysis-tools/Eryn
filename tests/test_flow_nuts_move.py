# tests/test_flow_nuts_move.py
"""Tests for FlowNUTSMove and the flow gradient API (log_prob_and_grad).

Coverage:

1. Gradient correctness: ``ZukoFlow.log_prob_and_grad`` against central finite
   differences of ``log_prob``, through a fitted ``WhiteningTransform`` (with
   and without a periodic dimension) and through ``IdentityTransform``; the
   returned log density must be bit-identical to ``log_prob``.
2. The ``Flow`` ABC default raises ``NotImplementedError``.
3. Move guards: pre-fit identity pass-through; construction-time ``TypeError``
   for a flow without a gradient implementation.
4. Whitening-derived mass matrix: value (= inverse sample covariance on the
   non-periodic block) and the hot-reload/condition rebuild trigger.
5. End-to-end no-bias gates through EnsembleSampler: matched flow (high
   acceptance), deliberately WRONG flow (must still recover the true target —
   the surrogate-kernel MH correction is what this file exists to pin), and a
   periodic target with in-bounds proposals.
6. Executor integration: hot-reload bumps the metric.

Requires torch and zuko (``pip install eryn[flow]``).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")

from scipy import stats  # noqa: E402

from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.prior import ProbDistContainer, uniform_dist  # noqa: E402
from eryn.state import State  # noqa: E402
from eryn.utils import PeriodicContainer  # noqa: E402

from eryn.flows import (  # noqa: E402
    ZukoFlow,
    WhiteningTransform,
    OneHotLeafConditioning,
    InlineExecutor,
)
from eryn.flows.base import Flow  # noqa: E402
from eryn.flows.transforms import IdentityTransform  # noqa: E402
from eryn.moves import ConditionalFlowMove, FlowNUTSMove  # noqa: E402

PERIOD = 2 * np.pi

TINY_NSF = dict(flow_class="NSF", transforms=2, hidden_features=(16, 16), bins=4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr_gauss_samples(n, seed, mean=(1.0, -0.5), cov=((1.0, 0.6), (0.6, 1.5))):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n).astype(np.float64)


def _periodic_samples(n, seed):
    """(N, 2): Normal(1, 0.7) x wrapped Normal(2.5, 0.5) on [0, 2pi)."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(1.0, 0.7, size=n)
    x1 = (rng.normal(2.5, 0.5, size=n)) % PERIOD
    return np.column_stack([x0, x1]).astype(np.float64)


def _make_flow(samples, periodic=None, seed=0, fit_transform=True):
    """Small ZukoFlow (random weights) with a WhiteningTransform."""
    ndim = samples.shape[1]
    wt = WhiteningTransform(ndim=ndim, periodic=periodic)
    if fit_transform:
        wt.fit({0: samples})
    return ZukoFlow(
        dims=ndim,
        device="cpu",
        data_transform=wt,
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        seed=seed,
        **TINY_NSF,
    )


def _fd_grad(flow, x, context, eps=1e-4):
    """Central finite differences of flow.log_prob at each row of x."""
    grad = np.zeros_like(x)
    for j in range(x.shape[1]):
        xp = x.copy()
        xm = x.copy()
        xp[:, j] += eps
        xm[:, j] -= eps
        grad[:, j] = (
            flow.log_prob(xp, context=context) - flow.log_prob(xm, context=context)
        ) / (2 * eps)
    return grad


def _check_grad(flow, x, context):
    logq, grad = flow.log_prob_and_grad(x, context=context)
    # density must be the SAME density the Hastings factors use — bit-identical
    np.testing.assert_array_equal(logq, flow.log_prob(x, context=context))
    fd = _fd_grad(flow, x, context)
    # float32 net => ~1e-3 noise floor on O(1) gradients
    np.testing.assert_allclose(grad, fd, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 1-2. Gradient correctness
# ---------------------------------------------------------------------------

def test_log_prob_and_grad_matches_fd_whitening_periodic():
    samples = _periodic_samples(3000, seed=1)
    flow = _make_flow(samples, periodic={1: (0.0, PERIOD)}, seed=1)
    # bulk points away from the periodic cut (bulk at 2.5, cut near 2.5 + pi)
    x = samples[:32]
    _check_grad(flow, x, context=0)


def test_log_prob_and_grad_matches_fd_whitening_gaussian():
    samples = _corr_gauss_samples(3000, seed=2)
    flow = _make_flow(samples, periodic=None, seed=2)
    _check_grad(flow, samples[:32], context=0)


def test_log_prob_and_grad_identity_transform():
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=IdentityTransform(),
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        seed=3,
        **TINY_NSF,
    )
    x = np.random.default_rng(3).standard_normal((32, 2))
    _check_grad(flow, x, context=0)


def test_log_prob_and_grad_empty_input():
    samples = _corr_gauss_samples(500, seed=4)
    flow = _make_flow(samples, seed=4)
    logq, grad = flow.log_prob_and_grad(np.empty((0, 2)), context=0)
    assert logq.shape == (0,) and grad.shape == (0, 2)


class _NoGradFlow(Flow):
    """Minimal concrete Flow that does NOT override log_prob_and_grad."""

    def log_prob(self, x, context=None):
        return np.zeros(len(x))

    def sample(self, n, context=None):
        return np.zeros((n, self.dims))

    def sample_and_log_prob(self, n, context=None):
        return np.zeros((n, self.dims)), np.zeros(n)

    def fit(self, samples, **kwargs):
        pass

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


def test_flow_abc_default_raises():
    f = _NoGradFlow(dims=2)
    with pytest.raises(NotImplementedError, match="log_prob_and_grad"):
        f.log_prob_and_grad(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# 3. Move guards
# ---------------------------------------------------------------------------

def test_prefit_identity_passthrough_and_type_guard():
    samples = _corr_gauss_samples(100, seed=7)
    flow = _make_flow(samples, seed=7, fit_transform=False)
    move = FlowNUTSMove(flow, "x")
    assert isinstance(move, ConditionalFlowMove)

    coords = np.random.default_rng(7).standard_normal((2, 5, 1, 2))
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(7))
    np.testing.assert_array_equal(q["x"], coords)
    np.testing.assert_array_equal(factors, np.zeros((2, 5)))

    with pytest.raises(TypeError, match="log_prob_and_grad"):
        FlowNUTSMove(_NoGradFlow(dims=2), "x")


def test_proposal_moves_and_wraps():
    samples = _periodic_samples(3000, seed=8)
    flow = _make_flow(samples, periodic={1: (0.0, PERIOD)}, seed=8)
    move = FlowNUTSMove(flow, "x", step_size=0.2)
    coords = samples[:40].reshape(2, 20, 1, 2).copy()

    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(8))
    assert q["x"].shape == coords.shape
    assert factors.shape == (2, 20)
    assert np.isfinite(factors).all()
    assert not np.allclose(q["x"], coords)
    # untouched dims of other walkers must be preserved exactly where NUTS
    # rejected — but at minimum the periodic dim must land inside its bounds
    ang = q["x"][..., 1]
    assert ((ang >= 0.0) & (ang < PERIOD)).all()


def test_deepcopy_roundtrip():
    samples = _corr_gauss_samples(500, seed=9)
    flow = _make_flow(samples, seed=9)
    move = copy.deepcopy(FlowNUTSMove(flow, "x"))
    coords = samples[:20].reshape(1, 20, 1, 2).copy()
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(0))
    assert np.isfinite(factors).all()


# ---------------------------------------------------------------------------
# 4. Whitening-derived mass matrix
# ---------------------------------------------------------------------------

def test_whitening_mass_matrix():
    samples = _corr_gauss_samples(4000, seed=5)
    flow = _make_flow(samples, seed=5)
    move = FlowNUTSMove(flow, "x")

    move._maybe_refresh_metric()
    mass = move._nuts._mass_matrix
    # z = (x - mu) @ M with M = chol(cov)^{-T}  =>  mass = M M^T = cov^{-1}
    # (regularization is relative: eps * var_i added per dimension)
    cov = np.cov(samples.T)
    expected = np.linalg.inv(
        cov + flow.data_transform.eps * np.diag(np.diag(cov))
    )
    np.testing.assert_allclose(mass, expected, rtol=1e-8, atol=1e-12)

    # same fitted transform object -> no rebuild
    move._maybe_refresh_metric()
    assert move._nuts._mass_matrix is mass

    # snapshot install with a refit transform (what a hot-reload does) -> rebuild
    samples2 = _corr_gauss_samples(
        4000, seed=6, mean=(0.0, 0.0), cov=((2.0, 0.0), (0.0, 0.5))
    )
    wt2 = WhiteningTransform(ndim=2)
    wt2.fit({0: samples2})
    flow.set_weights({"net": flow.get_weights(), "data_transform": wt2})
    move._maybe_refresh_metric()
    cov2 = np.cov(samples2.T)
    expected2 = np.linalg.inv(cov2 + wt2.eps * np.diag(np.diag(cov2)))
    np.testing.assert_allclose(move._nuts._mass_matrix, expected2, rtol=1e-8)


# ---------------------------------------------------------------------------
# 5-7. End-to-end no-bias gates (EnsembleSampler)
# ---------------------------------------------------------------------------

MEAN = np.array([1.0, -0.5])
COV = np.array([[1.0, 0.6], [0.6, 1.5]])
COV_INV = np.linalg.inv(COV)
SIG = np.sqrt(np.diag(COV))


def _log_gauss_target(x):
    x = np.atleast_2d(x)
    d = x - MEAN
    return -0.5 * np.einsum("ni,ij,nj->n", d, COV_INV, d)


def _log_periodic_target(x):
    """Normal(1, 0.7) x wrapped Normal(2.5, 0.5) (nearest image)."""
    x = np.atleast_2d(x)
    lp0 = -0.5 * ((x[:, 0] - 1.0) / 0.7) ** 2
    d = (x[:, 1] - 2.5 + np.pi) % PERIOD - np.pi
    lp1 = -0.5 * (d / 0.5) ** 2
    return lp0 + lp1


def _train_flow(samples, periodic=None, seed=0, steps=400):
    """Train a small NSF (same budget as test_flow_detailed_balance)."""
    torch.manual_seed(seed)
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        data_transform=WhiteningTransform(ndim=2, periodic=periodic),
        seed=seed,
        flow_class="NSF",
        transforms=4,
        hidden_features=(64, 64),
        bins=6,
    )
    flow.fit(
        samples,
        n_epochs=steps,
        lr=1e-3,
        batch_size=len(samples),  # full-batch: one gradient step per epoch
        validation_fraction=0.1,
        seed=seed,
    )
    return flow


def _run_sampler(flow, log_target, priors_in, start_coords, nsteps, burn,
                 move_kwargs=None, periodic=None):
    """Run EnsembleSampler with FlowNUTSMove as the only move.

    Returns (chain (nsteps*nwalkers, 2), mean acceptance fraction).
    """
    nwalkers = start_coords.shape[0]
    move = FlowNUTSMove(flow, branch_name="x", **(move_kwargs or {}))
    move.active_condition = 0

    sampler = EnsembleSampler(
        nwalkers, {"x": 2}, log_target,
        {"x": ProbDistContainer(priors_in)},
        tempering_kwargs=dict(ntemps=1), vectorize=True,
        periodic=periodic, moves=[move], branch_names=["x"],
    )
    sampler.random_state = np.random.RandomState(0).get_state()
    start = State({"x": start_coords.reshape(1, nwalkers, 1, 2)})
    sampler.run_mcmc(start, nsteps, burn=burn, progress=False)

    chain = sampler.get_chain()["x"][:, 0, :, 0, :].reshape(-1, 2)
    return chain, float(np.mean(sampler.acceptance_fraction))


@pytest.mark.slow
def test_recovers_matched_gaussian_target():
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        train = _corr_gauss_samples(8000, seed=1, mean=MEAN, cov=COV)
        flow = _train_flow(train, seed=0)

        chain, acc = _run_sampler(
            flow, _log_gauss_target,
            {0: uniform_dist(-6.0, 8.0), 1: uniform_dist(-7.0, 6.0)},
            _corr_gauss_samples(64, seed=7, mean=MEAN, cov=COV),
            nsteps=400, burn=150,
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    assert acc > 0.5, f"matched flow should accept often, got {acc}"
    np.testing.assert_allclose(chain.mean(axis=0), MEAN, atol=0.1 * SIG.max())
    np.testing.assert_allclose(chain.std(axis=0), SIG, rtol=0.10)
    corr = np.corrcoef(chain.T)[0, 1]
    assert abs(corr - COV[0, 1] / (SIG[0] * SIG[1])) < 0.1, f"corr {corr}"


@pytest.mark.slow
def test_wrong_flow_still_unbiased():
    """THE gate: a deliberately wrong surrogate must not bias the chain.

    The flow is trained on shifted (+0.5 sigma) and narrowed (0.6 sigma) data;
    NUTS dynamics therefore aim at the wrong region with the wrong scale.  The
    surrogate-kernel MH correction must still deliver the TRUE target moments —
    any sign/convention error in the factors fails here.
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        raw = _corr_gauss_samples(8000, seed=2, mean=MEAN, cov=COV)
        wrong = MEAN + 0.5 * SIG + (raw - MEAN) * 0.6
        flow = _train_flow(wrong, seed=0)

        chain, acc = _run_sampler(
            flow, _log_gauss_target,
            {0: uniform_dist(-6.0, 8.0), 1: uniform_dist(-7.0, 6.0)},
            _corr_gauss_samples(64, seed=7, mean=MEAN, cov=COV),
            nsteps=800, burn=200,
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    assert acc > 0.05, f"sampler frozen (acc={acc}); cannot assess bias"
    np.testing.assert_allclose(chain.mean(axis=0), MEAN, atol=0.15 * SIG.max())
    np.testing.assert_allclose(chain.std(axis=0), SIG, rtol=0.15)


@pytest.mark.slow
def test_periodic_target_recovery():
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        train = _periodic_samples(8000, seed=1)
        flow = _train_flow(train, periodic={1: (0.0, PERIOD)}, seed=0)

        chain, acc = _run_sampler(
            flow, _log_periodic_target,
            {0: uniform_dist(-5.0, 7.0), 1: uniform_dist(0.0, PERIOD)},
            _periodic_samples(100, seed=7),
            nsteps=600, burn=200,
            periodic=PeriodicContainer({"x": {1: PERIOD}}),
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    # the wrap keeps every stored sample inside the declared bounds
    assert ((chain[:, 1] >= 0.0) & (chain[:, 1] < PERIOD)).all()

    ks_p = stats.kstest((chain[:, 0] - 1.0) / 0.7, "norm").pvalue
    assert ks_p > 1e-3, f"non-periodic marginal off (KS p={ks_p})"

    ang = chain[:, 1]
    cmean = np.angle(np.mean(np.exp(1j * ang))) % PERIOD
    cstd = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * ang)))))
    assert abs(((cmean - 2.5 + np.pi) % PERIOD) - np.pi) < 0.15, f"circ mean {cmean}"
    assert abs(cstd - 0.5) < 0.15, f"circ std {cstd}"


# ---------------------------------------------------------------------------
# 8. Executor integration
# ---------------------------------------------------------------------------

def test_executor_hot_reload_updates_metric():
    wt = WhiteningTransform(ndim=2)  # unfitted: move starts as identity
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=wt,
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        seed=10,
        **TINY_NSF,
    )
    ex = InlineExecutor(
        flow,
        fit_kwargs=dict(n_epochs=3, batch_size=256, lr=1e-3),
        min_train_samples=200,
    )
    move = FlowNUTSMove(flow, "x", executor=ex, harvest_every=None)
    coords = _corr_gauss_samples(20, seed=10).reshape(1, 20, 1, 2)

    # pre-training: identity pass-through, no metric yet
    q, _ = move.get_proposal({"x": coords}, np.random.RandomState(1))
    np.testing.assert_array_equal(q["x"], coords)
    assert move._metric_source is None

    # round 1: submit -> InlineExecutor trains synchronously -> hot-reload
    move.submit_by_leaf({0: _corr_gauss_samples(500, seed=11)})
    move.setup({"x": coords})
    assert move.loaded_version == 1
    assert move.flow.data_transform.is_fitted

    move.get_proposal({"x": coords}, np.random.RandomState(2))
    src1 = move._metric_source
    assert src1 is not None and move._nuts._mass_matrix is not None

    # round 2: new snapshot ships a fresh transform copy -> metric rebuilt
    move.submit_by_leaf({0: _corr_gauss_samples(500, seed=12)})
    move.setup({"x": coords})
    assert move.loaded_version == 2

    move.get_proposal({"x": coords}, np.random.RandomState(3))
    assert move._metric_source is not src1
