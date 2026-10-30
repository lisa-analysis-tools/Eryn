# tests/test_flow_gaussian_move.py
"""Tests for FlowGaussianMove: a flow on a subset of the columns, a Gaussian walk on the rest."""
from __future__ import annotations

import numpy as np
import pytest

from eryn.flows.executors import TrainerExecutor
from eryn.moves import FlowGaussianMove
from eryn.moves.gaussian import _make_proposal
from eryn.utils import PeriodicContainer


class _StubExecutor(TrainerExecutor):
    """Records submits; never has weights."""

    def __init__(self):
        self.submits = []

    def submit(self, samples_by_condition):
        self.submits.append(samples_by_condition)
        return True

    def latest_weights(self):
        return None

    @property
    def version(self):
        return 0

    def shutdown(self, timeout: float = 10.0):
        pass


@pytest.fixture(autouse=True)
def _one_torch_thread():
    """Tiny flows: torch's default thread pool (one per core) makes each call ~100x slower."""
    torch = pytest.importorskip("torch")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _make_flow(seed: int = 0, nleaves: int = 1, mean=(0.0, 0.0), scale: float = 1.0, fit: bool = True):
    """A small 2-D ZukoFlow (random weights) whose whitening is fitted to N(mean, scale**2 C)."""
    torch = pytest.importorskip("torch")
    from eryn.flows import OneHotLeafConditioning, WhiteningTransform, ZukoFlow

    torch.manual_seed(seed)
    wt = WhiteningTransform(ndim=2, shared=True)
    if fit:
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(mean, scale**2 * np.array([[1.0, 0.3], [0.3, 1.0]]), size=2000)
        wt.fit({leaf: samples for leaf in range(nleaves)})
    return ZukoFlow(
        dims=2, device="cpu", data_transform=wt, conditioning=OneHotLeafConditioning(nleaves_max=nleaves),
        seed=seed, flow_class="NSF", transforms=2, hidden_features=(32, 32), bins=4,
    )


MASK = np.array([True, False, True, False])  # flow on columns 0, 2; Gaussian on 1, 3


def _coords(ntemps=2, nwalkers=6, nleaves=1, seed=1):
    return np.random.default_rng(seed).standard_normal((ntemps, nwalkers, nleaves, len(MASK)))


def test_rejects_bad_setups():
    flow = _make_flow()
    with pytest.raises(ValueError, match="some but not all"):
        FlowGaussianMove(flow, "x", np.ones(4, dtype=bool), cov=1.0)
    with pytest.raises(ValueError, match="flow.dims"):
        FlowGaussianMove(flow, "x", np.array([True, False, False, False]), cov=1.0)
    with pytest.raises(ValueError, match="cov is required"):
        FlowGaussianMove(flow, "x", MASK)
    with pytest.raises(ValueError, match="boolean mask"):
        FlowGaussianMove(flow, "x", np.array([1, 0, 1, 0]), cov=1.0)


def test_factor_is_flow_only_and_columns_split():
    flow = _make_flow()
    move = FlowGaussianMove(flow, "x", MASK, cov=np.diag([0.1, 0.2]))
    coords = _coords()
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(0))

    old, new = coords.reshape(-1, 4), q["x"].reshape(-1, 4)
    assert np.all(new[:, ~MASK] != old[:, ~MASK]) and np.all(new[:, MASK] != old[:, MASK])
    expected = flow.log_prob(old[:, MASK], context=0) - flow.log_prob(new[:, MASK], context=0)
    np.testing.assert_allclose(factors, expected.reshape(factors.shape), atol=1e-5)


def test_gaussian_columns_match_gaussian_move_logic():
    """The Gaussian columns are exactly the GaussianMove proposal (same RNG stream)."""
    cov = np.array([[0.2, 0.05], [0.05, 0.1]])
    move = FlowGaussianMove(_make_flow(fit=False), "x", MASK, cov=cov, mode="am")
    coords = _coords()
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(7))

    ref, _ = _make_proposal(cov, "am")(coords.reshape(-1, 4)[:, ~MASK], np.random.RandomState(7))
    np.testing.assert_array_equal(q["x"].reshape(-1, 4)[:, ~MASK], ref)
    # unfitted flow: flow columns fixed, zero factor
    np.testing.assert_array_equal(q["x"][..., MASK], coords[..., MASK])
    assert np.all(factors == 0.0)


@pytest.mark.parametrize("mode, cov", [("vector", np.array([0.1, 0.2])), ("random", np.array([0.1, 0.2])),
                                       ("sequential", 0.3), ("vector", np.diag([0.1, 0.2]))])
def test_other_gaussian_modes(mode, cov):
    """Diagonal covariances used to crash GaussianMove's proposal setup (Cholesky of a vector)."""
    move = FlowGaussianMove(_make_flow(fit=False), "x", MASK, cov=cov, mode=mode, factor=None if mode == "vector" else 2.0)
    coords = _coords()
    q, _ = move.get_proposal({"x": coords}, np.random.RandomState(0))
    ref, _ = _make_proposal(cov, mode, None if mode == "vector" else 2.0)(
        coords.reshape(-1, 4)[:, ~MASK], np.random.RandomState(0)
    )
    np.testing.assert_array_equal(q["x"].reshape(-1, 4)[:, ~MASK], ref)


def test_no_cov_yet_moves_flow_columns_only():
    flow = _make_flow()
    move = FlowGaussianMove(flow, "x", MASK, adapt_cov=True)
    coords = _coords()
    q, _ = move.get_proposal({"x": coords}, np.random.RandomState(0))
    np.testing.assert_array_equal(q["x"][..., ~MASK], coords[..., ~MASK])
    assert not np.array_equal(q["x"][..., MASK], coords[..., MASK])


def test_inactive_leaves_and_other_branches_untouched():
    flow = _make_flow()
    move = FlowGaussianMove(flow, "x", MASK, cov=0.1)
    coords = _coords(nleaves=2)
    inds = np.ones(coords.shape[:-1], dtype=bool)
    inds[:, :, 1] = False
    other = np.zeros((2, 6, 1, 3))
    q, factors = move.get_proposal(
        {"x": coords, "y": other}, np.random.RandomState(0), branches_inds={"x": inds, "y": inds[..., :1]}
    )
    np.testing.assert_array_equal(q["x"][:, :, 1], coords[:, :, 1])
    np.testing.assert_array_equal(q["y"], other)
    old, new = coords[:, :, 0], q["x"][:, :, 0]
    expected = flow.log_prob(old[..., MASK].reshape(-1, 2), context=0) - flow.log_prob(
        new[..., MASK].reshape(-1, 2), context=0
    )
    np.testing.assert_allclose(factors, expected.reshape(factors.shape), atol=1e-5)


def test_active_betas_scale_flow_columns():
    flow = _make_flow()
    move = FlowGaussianMove(flow, "x", MASK, cov=0.1)
    move.active_betas = np.array([1.0, 0.25])
    coords = _coords()
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(0))
    for t, beta in enumerate(move.active_betas):
        s = beta**-0.5
        old, new = coords[t, :, 0, MASK].T, q["x"][t, :, 0, MASK].T
        expected = flow.log_prob(old, context=0, base_scale=s) - flow.log_prob(new, context=0, base_scale=s)
        np.testing.assert_allclose(factors[t], expected, atol=1e-5)


def test_per_leaf_covariance():
    move = FlowGaussianMove(_make_flow(fit=False, nleaves=2), "x", MASK, cov={None: 1e-2, 1: 1e2})
    coords = _coords()
    move.active_condition = 0
    q0, _ = move.get_proposal({"x": coords}, np.random.RandomState(0))
    move.active_condition = 1
    q1, _ = move.get_proposal({"x": coords}, np.random.RandomState(0))
    np.testing.assert_allclose(q1["x"] - coords, 100 * (q0["x"] - coords))


def test_adapt_cov_and_executor_slicing():
    ex = _StubExecutor()
    move = FlowGaussianMove(_make_flow(nleaves=2), "x", MASK, adapt_cov=True, adapt_min_samples=50, executor=ex)
    rng = np.random.default_rng(0)
    rows0 = rng.standard_normal((200, 4)) * [1.0, 2.0, 1.0, 0.5]
    rows1 = rng.standard_normal((10, 4))  # too few rows: no adaptation for leaf 1
    move.submit_by_leaf({0: rows0, 1: rows1})

    np.testing.assert_allclose(move.gaussian_proposal(0).scale, np.cov(rows0[:, ~MASK], rowvar=False), rtol=1e-9)
    assert move.gaussian_proposal(1) is None
    sent = ex.submits[-1]
    np.testing.assert_array_equal(sent[0], rows0[:, MASK])
    np.testing.assert_array_equal(sent[1], rows1[:, MASK])

    with pytest.raises(ValueError, match="full ndim"):
        move.submit_by_leaf({0: rows0[:, MASK]})


def test_adapt_cov_without_executor():
    move = FlowGaussianMove(_make_flow(), "x", MASK, cov=1.0, adapt_cov=True, adapt_min_samples=20)
    rows = np.random.default_rng(0).standard_normal((100, 4))
    move.submit_by_leaf({0: rows})  # executor None: still adapts
    np.testing.assert_allclose(move.gaussian_proposal(0).scale, np.cov(rows[:, ~MASK], rowvar=False), rtol=1e-9)
    np.testing.assert_array_equal(move.gaussian_proposal(1).scale, np.eye(2))  # the default entry is kept


def test_cov_shape_checked():
    with pytest.raises(ValueError, match="cov has shape"):
        FlowGaussianMove(_make_flow(), "x", MASK, cov=np.eye(3))


def test_periodic_wraps_gaussian_columns_only():
    flow = _make_flow()
    move = FlowGaussianMove(flow, "x", MASK, cov=0.1)
    move.periodic = PeriodicContainer({"x": {0: 0.5, 1: 0.5}})
    coords = _coords() + 10.0  # far outside [0, 0.5)
    q, factors = move.get_proposal({"x": coords}, np.random.RandomState(0))
    new = q["x"].reshape(-1, 4)
    assert np.all((new[:, 1] >= 0) & (new[:, 1] < 0.5))  # Gaussian periodic column wrapped
    # flow column 0 is not wrapped: it is the value whose log q is in the factor
    old = coords.reshape(-1, 4)
    expected = flow.log_prob(old[:, MASK], context=0) - flow.log_prob(new[:, MASK], context=0)
    np.testing.assert_allclose(factors, expected.reshape(factors.shape), atol=1e-5)


def test_sampler_recovers_correlated_target():
    """End-to-end detailed balance: only this move, a 3-D correlated Gaussian target.

    The flow is deliberately a poor match to the target marginal of its columns (shifted,
    broader, untrained), so a missing or wrong Hastings factor would bias the chain toward it.
    """
    pytest.importorskip("torch")
    from eryn.ensemble import EnsembleSampler
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State

    mean = np.array([0.5, -0.3, 1.0])
    cov = np.array([[1.0, 0.4, 0.5], [0.4, 1.0, 0.3], [0.5, 0.3, 1.0]])
    icov = np.linalg.inv(cov)

    def log_like(x):
        d = np.atleast_2d(x) - mean
        return -0.5 * np.einsum("ni,ij,nj->n", d, icov, d)

    mask = np.array([True, True, False])  # flow on 0, 1; Gaussian on 2
    flow = _make_flow(seed=3, mean=(0.0, 0.2), scale=1.6)
    move = FlowGaussianMove(flow, "x", mask, cov=0.5)

    nwalkers, nsteps, burn = 32, 1500, 200
    priors = {"x": ProbDistContainer({i: uniform_dist(-15.0, 15.0) for i in range(3)})}
    sampler = EnsembleSampler(
        nwalkers, {"x": 3}, log_like, priors, tempering_kwargs=dict(ntemps=1), vectorize=True,
        moves=[move], branch_names=["x"],
    )
    np.random.seed(11)
    start = State({"x": np.random.multivariate_normal(mean, cov, size=(1, nwalkers, 1))})
    sampler.run_mcmc(start, nsteps, burn=burn, progress=False)

    chain = sampler.get_chain()["x"][:, 0, :, 0].reshape(-1, 3)
    assert 0.05 < np.mean(sampler.acceptance_fraction) < 0.95
    # ~ nsteps * nwalkers / tau effective samples; tolerances are several sigma
    np.testing.assert_allclose(chain.mean(axis=0), mean, atol=0.08)
    np.testing.assert_allclose(np.cov(chain, rowvar=False), cov, atol=0.1)
