# tests/test_flow_move.py
"""Tests for ConditionalFlowMove and IndependentProposalMove.

Test layout
-----------
- ``test_independent_proposal_factor_identity`` — numpy-only (no torch).
- ``test_flow_move_factor_identity`` — requires torch; ``importorskip``'d.
- ``test_flow_move_multileaf_accumulates_factors`` — requires torch.
- ``test_flow_move_sampler_smoke`` — requires torch; 50-step EnsembleSampler smoke test.
"""
from __future__ import annotations

import numpy as np
import pytest

from eryn.flows.executors import TrainerError, TrainerExecutor
from eryn.moves import ConditionalFlowMove, IndependentProposalMove


# ---------------------------------------------------------------------------
# Helpers shared across flow tests
# ---------------------------------------------------------------------------

def _make_flow(seed: int = 0):
    """Build a small ZukoFlow with WhiteningTransform + OneHotLeafConditioning.

    Returns the fitted (but not trained) flow.  Weights are random — that is
    fine for factor-identity tests; only the smoke test needs some acceptance.
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
# IndependentProposalMove — factor identity (numpy only, no torch required)
# ---------------------------------------------------------------------------

class _GaussianDist:
    """Simple isotropic Gaussian proposal for testing (no torch, no scipy)."""

    def __init__(self, ndim: int, seed: int = 0):
        self.ndim = ndim
        self._rng = np.random.default_rng(seed)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # log N(0, I): -0.5 * sum(x**2) - ndim/2 * log(2pi)
        return -0.5 * np.sum(x ** 2, axis=-1) - 0.5 * self.ndim * np.log(2 * np.pi)

    def rvs(self, size: int) -> np.ndarray:
        return self._rng.standard_normal((size, self.ndim))


def test_independent_proposal_factor_identity():
    """factors == log q(old) - log q(new) for IndependentProposalMove (numpy only)."""
    ndim = 3
    ntemps, nwalkers, nleaves = 2, 8, 1
    rng = np.random.default_rng(42)

    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}
    dist = _GaussianDist(ndim, seed=7)
    move = IndependentProposalMove(dist, branch_name="x")

    q, factors = move.get_proposal(branches, rng)

    assert q["x"].shape == (ntemps, nwalkers, nleaves, ndim)
    assert factors.shape == (ntemps, nwalkers)

    # Recompute expected factors independently
    old = coords.reshape(-1, ndim)
    new = q["x"].reshape(-1, ndim)
    logq_old = dist.logpdf(old).reshape(ntemps, nwalkers)
    logq_new = dist.logpdf(new).reshape(ntemps, nwalkers)
    expected = logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-10,
                               err_msg="IndependentProposalMove factors != log q(old) - log q(new)")


def test_independent_proposal_multileaf_accumulates_factors():
    """Multi-leaf: factors sum per-leaf contributions (np.add.at, not last-write-wins)."""
    ndim = 3
    ntemps, nwalkers, nleaves = 1, 6, 2
    rng = np.random.default_rng(99)

    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}
    dist = _GaussianDist(ndim, seed=5)
    move = IndependentProposalMove(dist, branch_name="x")

    q, factors = move.get_proposal(branches, rng)

    # Recompute expected by summing over leaves
    expected = np.zeros((ntemps, nwalkers))
    for leaf in range(nleaves):
        old_l = coords[:, :, leaf, :].reshape(-1, ndim)
        new_l = q["x"][:, :, leaf, :].reshape(-1, ndim)
        expected += (dist.logpdf(old_l) - dist.logpdf(new_l)).reshape(ntemps, nwalkers)

    np.testing.assert_allclose(factors, expected, atol=1e-10,
                               err_msg="Multi-leaf factors not summed correctly")


def test_independent_proposal_non_target_branch_unchanged():
    """Branches other than branch_name are copied unchanged."""
    ndim = 2
    rng = np.random.default_rng(11)
    coords_x = rng.standard_normal((1, 4, 1, ndim))
    coords_y = rng.standard_normal((1, 4, 1, ndim))
    branches = {"x": coords_x, "y": coords_y}
    dist = _GaussianDist(ndim, seed=3)
    move = IndependentProposalMove(dist, branch_name="x")

    q, _ = move.get_proposal(branches, rng)
    np.testing.assert_array_equal(q["y"], coords_y,
                                  err_msg="Non-target branch 'y' was mutated")


# ---------------------------------------------------------------------------
# ConditionalFlowMove — factor identity (requires torch)
# ---------------------------------------------------------------------------

def test_flow_move_factor_identity():
    """factors == log q(old) - log q(new) for ConditionalFlowMove (recomputed via flow.log_prob)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=0)

    ntemps, nwalkers, nleaves, ndim = 1, 8, 1, 2
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    move = ConditionalFlowMove(flow, branch_name="x")
    move.active_condition = 0

    q, factors = move.get_proposal(branches, rng)

    assert q["x"].shape == (ntemps, nwalkers, nleaves, ndim)
    assert factors.shape == (ntemps, nwalkers)

    # Recompute expected factors via flow.log_prob independently
    old = coords.reshape(-1, ndim)
    new = q["x"].reshape(-1, ndim)
    logq_old = flow.log_prob(old, context=0).reshape(ntemps, nwalkers)
    logq_new = flow.log_prob(new, context=0).reshape(ntemps, nwalkers)
    expected = logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-5,
                               err_msg="ConditionalFlowMove factors != log q(old) - log q(new)")


def test_flow_move_unfitted_transform_is_identity_noop():
    """A ConditionalFlowMove over a flow whose WhiteningTransform is UNFITTED must not raise:
    it proposes an identity move (old coords, zero factors) until the transform
    is fitted (e.g. by the trainer's first snapshot).  This is what lets the move
    be mixed in from step 0 of a lazy/no-pre-fit online run."""
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    wt = WhiteningTransform(ndim=2, shared=True)  # UNFITTED
    assert wt.is_fitted is False
    flow = ZukoFlow(
        dims=2, device="cpu", data_transform=wt,
        conditioning=OneHotLeafConditioning(1), seed=0,
        flow_class="NSF", transforms=2, hidden_features=(16,), bins=3,
    )
    move = ConditionalFlowMove(flow, branch_name="x")
    move.active_condition = 0

    ntemps, nwalkers, nleaves, ndim = 1, 6, 1, 2
    rng = np.random.default_rng(3)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))

    q, factors = move.get_proposal({"x": coords}, rng)  # must NOT raise
    np.testing.assert_array_equal(q["x"], coords)       # identity proposal
    assert np.all(factors == 0.0) and factors.shape == (ntemps, nwalkers)

    # once the transform is fitted, the move proposes for real (non-identity)
    wt.fit({0: rng.standard_normal((500, 2))})
    q2, factors2 = move.get_proposal({"x": coords}, rng)
    assert not np.array_equal(q2["x"], coords)


def test_flow_move_multileaf_accumulates_factors():
    """Regression: multi-leaf factors must SUM per-leaf contributions (np.add.at)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=2)

    ntemps, nwalkers, nleaves, ndim = 1, 6, 2, 2
    rng = np.random.default_rng(3)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    move = ConditionalFlowMove(flow, branch_name="x")
    move.active_condition = 0

    q, factors = move.get_proposal(branches, rng)

    # Recompute by summing per-leaf contributions
    expected = np.zeros((ntemps, nwalkers))
    for leaf in range(nleaves):
        old_l = coords[:, :, leaf, :].reshape(-1, ndim)
        new_l = q["x"][:, :, leaf, :].reshape(-1, ndim)
        logq_old = flow.log_prob(old_l, context=0).reshape(ntemps, nwalkers)
        logq_new = flow.log_prob(new_l, context=0).reshape(ntemps, nwalkers)
        expected += logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-5,
                               err_msg="Multi-leaf ConditionalFlowMove factors not summed correctly")


# ---------------------------------------------------------------------------
# ConditionalFlowMove sampler smoke test (requires torch)
# ---------------------------------------------------------------------------

def _log_like_gauss_vectorized(x):
    """Vectorized 2-D standard-normal log-likelihood (shape (N, ndim) -> (N,))."""
    x = np.atleast_2d(x)
    return -0.5 * np.sum(x ** 2, axis=-1)


def test_flow_move_sampler_smoke():
    """50-step EnsembleSampler smoke test with ConditionalFlowMove on a 2-D Gaussian target.

    The flow is frozen (not trained); acceptance fraction > 0 just means the
    move is wired correctly — detailed balance is validated by the factor-identity
    test above.
    """
    torch = pytest.importorskip("torch")

    from eryn.ensemble import EnsembleSampler
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State

    ndim = 2
    ntemps = 1
    nwalkers = 16

    flow = _make_flow(seed=0)
    move = ConditionalFlowMove(flow, branch_name="x")
    move.active_condition = 0

    priors = {
        "x": ProbDistContainer(
            {0: uniform_dist(-10.0, 10.0), 1: uniform_dist(-10.0, 10.0)}
        )
    }

    sampler = EnsembleSampler(
        nwalkers,
        {"x": ndim},
        _log_like_gauss_vectorized,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        vectorize=True,
        moves=[move],
        branch_names=["x"],
    )

    rng = np.random.default_rng(42)
    start_coords = rng.standard_normal((ntemps, nwalkers, 1, ndim))
    start = State({"x": start_coords})

    sampler.run_mcmc(start, 50, burn=0, progress=False)

    chain = sampler.get_chain()["x"]
    # Shape: (nsteps, ntemps, nwalkers, nleaves_max, ndim)
    assert chain.shape[0] == 50, f"Unexpected chain length: {chain.shape[0]}"
    assert np.all(np.isfinite(chain)), "Chain contains non-finite values"

    # With ntemps=1 the sampler reports acceptance per move; at least one
    # proposal must have been accepted over 50 steps with 16 walkers.
    acceptance = sampler.acceptance_fraction
    # acceptance_fraction is (ntemps, nwalkers); check at least one acceptance
    assert np.any(acceptance > 0), (
        f"No proposals accepted in 50 steps (acceptance={acceptance}); "
        "the move is likely mis-wired."
    )


# ---------------------------------------------------------------------------
# Missing-branch guard — both move classes
# ---------------------------------------------------------------------------

def test_independent_proposal_missing_branch_raises():
    """get_proposal raises KeyError when branch_name is not in branches_coords."""
    ndim = 2
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((1, 4, 1, ndim))
    branches = {"x": coords}

    dist = _GaussianDist(ndim, seed=0)
    move = IndependentProposalMove(dist, branch_name="typo_branch")

    with pytest.raises(KeyError, match="typo_branch"):
        move.get_proposal(branches, rng)


def test_flow_move_missing_branch_raises():
    """ConditionalFlowMove.get_proposal raises KeyError when branch_name is not in branches_coords."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)

    ndim = 2
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((1, 4, 1, ndim))
    branches = {"x": coords}

    move = ConditionalFlowMove(flow, branch_name="typo_branch")

    with pytest.raises(KeyError, match="typo_branch"):
        move.get_proposal(branches, rng)


# ---------------------------------------------------------------------------
# active_condition routing — ConditionalFlowMove (requires torch)
# ---------------------------------------------------------------------------

def _make_flow_two_conditions(seed: int = 0):
    """Build a ZukoFlow with OneHotLeafConditioning(2) and two very different scales.

    Condition 0 is fit on tight samples (scale 0.1); condition 1 on wide samples
    (scale 10).  A flow that ignores the condition would produce the same logq for
    both calls, making the factors identical — the test asserts they differ.
    """
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Very different scales so the whiten transform encodes condition-dependent info
    samples_0 = rng.multivariate_normal(
        [0.0, 0.0], [[0.01, 0.0], [0.0, 0.01]], size=2000
    ).astype(np.float64)
    samples_1 = rng.multivariate_normal(
        [0.0, 0.0], [[100.0, 0.0], [0.0, 100.0]], size=2000
    ).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=2)
    wt = WhiteningTransform(ndim=2)
    wt.fit({0: samples_0, 1: samples_1})

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


def test_flow_move_active_condition_routing():
    """Changing active_condition actually routes to different flow branches.

    Build a flow whose WhiteningTransform is fit on two very different scales.
    Run get_proposal twice with active_condition=0 and active_condition=1 on
    identical coordinates.  The resulting factors must differ — a hard-coded
    condition value of 0 in get_proposal would cause both to return the same
    factors, failing this test.
    """
    pytest.importorskip("torch")
    flow = _make_flow_two_conditions(seed=42)

    ntemps, nwalkers, nleaves, ndim = 1, 8, 1, 2
    rng = np.random.default_rng(7)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    move0 = ConditionalFlowMove(flow, branch_name="x")
    move0.active_condition = 0
    _, factors0 = move0.get_proposal(branches, rng)

    # Fresh move with identical coords but condition=1
    move1 = ConditionalFlowMove(flow, branch_name="x")
    move1.active_condition = 1
    _, factors1 = move1.get_proposal(branches, np.random.default_rng(7))

    assert not np.allclose(factors0, factors1, atol=1e-6), (
        "factors identical for condition=0 and condition=1 — "
        "active_condition may not be reaching the flow."
    )


# ---------------------------------------------------------------------------
# ConditionalFlowMove online-training hooks (setup) — stub executor (numpy only)
# ---------------------------------------------------------------------------

class _StubExecutor(TrainerExecutor):
    """Records submits and serves canned (version, weights) on demand.

    Subclasses the :class:`TrainerExecutor` ABC so a signature change to the
    seam contract breaks loudly here instead of drifting silently.
    """

    def __init__(self, canned=None, raise_on_poll=False):
        self.submits = []
        self._canned = canned  # tuple[int, dict] or None
        self._raise_on_poll = raise_on_poll
        self.shutdown_called = 0

    def submit(self, samples_by_condition):
        self.submits.append(samples_by_condition)
        return True

    def latest_weights(self):
        if self._raise_on_poll:
            raise TrainerError("stub trainer died")
        return self._canned

    @property
    def version(self):
        return 0 if self._canned is None else self._canned[0]

    def shutdown(self, timeout: float = 10.0):
        self.shutdown_called += 1

    def serve(self, version, weights):
        self._canned = (version, weights)


def _setup_branches(ntemps=2, nwalkers=4, nleaves=1, ndim=2, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    return {"x": coords}


def test_flowmove_executor_none_setup_is_noop():
    """executor=None: setup does nothing, raises nothing, loaded_version stays 0."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    move = ConditionalFlowMove(flow, branch_name="x")  # no executor
    branches = _setup_branches()
    move.setup(branches)  # must be a pure no-op
    assert move.loaded_version == 0


def test_flowmove_harvest_every_honored():
    """harvest_every=3 → submit only on every 3rd setup call; poll every call."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    ex = _StubExecutor(canned=None)
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=3)
    branches = _setup_branches()

    for _ in range(6):
        move.setup(branches)

    # 6 calls / harvest_every=3 → 2 submits.
    assert len(ex.submits) == 2


def test_flowmove_harvest_submits_cold_chain_per_leaf():
    """Harvest submits one (nwalkers, ndim) cold-chain block per leaf id.

    Since the per-leaf conditioning redesign, ``ConditionalFlowMove.submit`` keys the
    training batch by LEAF index (the flow's condition id) rather than pooling
    all leaves under ``active_condition`` — each leaf trains its own
    conditional slice of the flow.
    """
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    ex = _StubExecutor(canned=None)
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=1)
    ntemps, nwalkers, nleaves, ndim = 3, 5, 2, 2
    branches = _setup_branches(ntemps, nwalkers, nleaves, ndim)

    move.setup(branches)

    assert len(ex.submits) == 1
    submitted = ex.submits[0]
    assert set(submitted.keys()) == set(range(nleaves))
    for leaf in range(nleaves):
        arr = submitted[leaf]
        # cold chain (temp 0) of this leaf: nwalkers rows, ndim columns
        assert arr.shape == (nwalkers, ndim)
        np.testing.assert_array_equal(arr, branches["x"][0][:, leaf, :])


def test_flowmove_submit_by_leaf_flattens_leading_axes():
    """(n_snapshots, nwalkers, ndim) values reach the executor as (n_snapshots * nwalkers, ndim) rows."""
    pytest.importorskip("torch")
    ex = _StubExecutor(canned=None)
    move = ConditionalFlowMove(_make_flow(seed=0), branch_name="x", executor=ex)
    snaps = np.random.default_rng(0).standard_normal((3, 5, 2))
    move.submit_by_leaf({0: snaps, 1: snaps[0]})
    np.testing.assert_array_equal(ex.submits[-1][0], snaps.reshape(-1, 2))
    np.testing.assert_array_equal(ex.submits[-1][1], snaps[0])


def test_flowmove_hot_reload_advances_loaded_version_and_changes_outputs():
    """Serving newer weights via the stub hot-loads them and changes flow outputs."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    other = _make_flow(seed=123)  # different random weights
    new_weights = other.get_weights()

    ex = _StubExecutor(canned=None)
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=1)
    branches = _setup_branches(ndim=2)

    x = np.full((4, 2), 0.5)
    before = flow.log_prob(x, context=0).copy()

    # No weights yet: poll returns None, version stays 0.
    move.setup(branches)
    assert move.loaded_version == 0

    # Serve version 1: setup loads it.
    ex.serve(1, new_weights)
    move.setup(branches)
    assert move.loaded_version == 1
    after = flow.log_prob(x, context=0)
    assert not np.allclose(before, after), "hot-reloaded weights did not change outputs"

    # Serving the SAME version again must not reload (no-op when not strictly newer).
    after2 = flow.log_prob(x, context=0)
    move.setup(branches)
    assert move.loaded_version == 1
    np.testing.assert_array_equal(after, after2)


def test_flowmove_executor_swap_resets_version_and_applies_new_weights():
    """Swapping in a NEW executor resets loaded_version so its restarted counter is honoured.

    Regression for I1: the per-executor ``version`` counter restarts at 1 for a
    replacement executor.  ConditionalFlowMove gates reload on ``version > loaded_version``,
    so without a reset, executor B's version 1 would be ignored against the
    remembered version 3 from executor A — silently freezing hot-reload.  The
    ``executor`` setter resets ``loaded_version`` (and the harvest counter)
    whenever a genuinely new executor object is assigned.
    """
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)

    # Distinct weights so we can prove B's weights were actually applied.
    weights_a = _make_flow(seed=111).get_weights()
    weights_b = _make_flow(seed=222).get_weights()

    branches = _setup_branches(ndim=2)
    x = np.full((4, 2), 0.5)

    # Executor A serves version 3 → loaded_version advances to 3.
    ex_a = _StubExecutor(canned=(3, weights_a))
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex_a, harvest_every=1)
    move.setup(branches)
    assert move.loaded_version == 3
    after_a = flow.log_prob(x, context=0).copy()

    # Swap in a NEW executor B serving version 1.  The setter must reset
    # loaded_version (and the harvest counter) on this real swap.
    ex_b = _StubExecutor(canned=(1, weights_b))
    move.executor = ex_b
    assert move.loaded_version == 0, "executor swap must reset loaded_version"
    assert move._setup_calls == 0, "executor swap must reset the harvest counter"

    # Next setup honours B's restarted version 1 and applies B's weights.
    move.setup(branches)
    assert move.loaded_version == 1
    after_b = flow.log_prob(x, context=0)
    assert not np.allclose(after_a, after_b), (
        "executor B's weights (version 1) were not applied after the swap"
    )

    # Re-assigning the SAME object is a no-op: counters are preserved.
    move.executor = ex_b
    assert move.loaded_version == 1
    assert move._setup_calls == 1


def test_flowmove_trainer_error_propagates_from_setup():
    """A failed trainer surfaces as TrainerError out of setup (not swallowed)."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    ex = _StubExecutor(raise_on_poll=True)
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=1)
    branches = _setup_branches()

    with pytest.raises(TrainerError, match="stub trainer died"):
        move.setup(branches)


def test_flowmove_setup_missing_branch_raises():
    """setup raises KeyError when branch_name absent (same loud guard as get_proposal)."""
    pytest.importorskip("torch")
    flow = _make_flow(seed=0)
    ex = _StubExecutor(canned=None)
    move = ConditionalFlowMove(flow, branch_name="typo_branch", executor=ex)
    branches = _setup_branches()

    with pytest.raises(KeyError, match="typo_branch"):
        move.setup(branches)


def test_flowmove_inline_executor_end_to_end_sampler():
    """Smoke test: ConditionalFlowMove + real InlineExecutor through a short EnsembleSampler run."""
    pytest.importorskip("torch")

    from eryn.ensemble import EnsembleSampler
    from eryn.flows import InlineExecutor
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State

    ndim = 2
    ntemps = 1
    nwalkers = 16

    flow = _make_flow(seed=0)
    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=1), min_train_samples=0,
                        train_every=5)
    move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=2)
    move.active_condition = 0

    priors = {
        "x": ProbDistContainer(
            {0: uniform_dist(-10.0, 10.0), 1: uniform_dist(-10.0, 10.0)}
        )
    }

    sampler = EnsembleSampler(
        nwalkers,
        {"x": ndim},
        _log_like_gauss_vectorized,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        vectorize=True,
        moves=[move],
        branch_names=["x"],
    )

    rng = np.random.default_rng(42)
    start_coords = rng.standard_normal((ntemps, nwalkers, 1, ndim))
    start = State({"x": start_coords})

    sampler.run_mcmc(start, 20, burn=0, progress=False)
    ex.shutdown()

    chain = sampler.get_chain()["x"]
    assert chain.shape[0] == 20
    assert np.all(np.isfinite(chain))
    # The executor trained at least once and the move hot-loaded new weights.
    assert ex.version >= 1
    assert move.loaded_version >= 1
