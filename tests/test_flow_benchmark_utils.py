# tests/test_flow_benchmark_utils.py
"""Tests for eryn.flows.benchmark — ESS, eval counting, targets, GMM proposal.

scipy and scikit-learn are optional eryn dependencies but are present in the
dev venv.  Tests that require them are guarded with ``pytest.importorskip`` so
upstream CI without those packages still gets a clean SKIP rather than FAIL.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Subprocess test — must come first (no eryn import yet in this process)
# ---------------------------------------------------------------------------

def test_benchmark_no_torch_sklearn_scipy_at_module_level():
    """Importing eryn.flows.benchmark must NOT import torch/sklearn/scipy."""
    code = (
        "import sys\n"
        "import eryn.flows.benchmark\n"
        "bad = [m for m in ('torch', 'sklearn', 'scipy') if m in sys.modules]\n"
        "assert not bad, f'unexpected imports: {bad}'\n"
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


# ---------------------------------------------------------------------------
# ESS tests
# ---------------------------------------------------------------------------

def test_ess_iid_close_to_n():
    """ESS of i.i.d. samples is close to N."""
    from eryn.flows.benchmark import effective_sample_size

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 3))  # (nsteps, ndim) — iid -> ESS ~ N
    ess = effective_sample_size(x)
    assert ess.shape == (3,)
    assert np.all(ess > 0.6 * 20000)


def test_ess_correlated_much_less_than_n():
    """ESS of a highly correlated AR(1) chain is << N."""
    from eryn.flows.benchmark import effective_sample_size

    rng = np.random.default_rng(0)
    n = 20000
    x = np.zeros((n, 1))
    for i in range(1, n):
        x[i, 0] = 0.95 * x[i - 1, 0] + rng.normal()  # AR(1), high autocorr
    ess = effective_sample_size(x)
    assert ess[0] < 0.1 * n


def test_ess_ensemble_shape():
    """ESS accepts (nsteps, nwalkers, ndim) ensembles and returns shape (ndim,)."""
    from eryn.flows.benchmark import effective_sample_size

    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 8, 3))  # (nsteps, nwalkers, ndim)
    ess = effective_sample_size(x)
    assert ess.shape == (3,)
    assert np.all(ess > 0)


def test_ess_ensemble_vs_single_chain():
    """Ensemble ESS (total n*nwalkers samples) >= single-chain ESS."""
    from eryn.flows.benchmark import effective_sample_size

    rng = np.random.default_rng(7)
    # single chain
    x_single = rng.normal(size=(1000, 2))
    ess_single = effective_sample_size(x_single)
    # ensemble (same independent draws, reshaped)
    x_ens = rng.normal(size=(1000, 4, 2))
    ess_ens = effective_sample_size(x_ens)
    # ensemble total samples = 4x, so ESS should be larger
    assert np.all(ess_ens > ess_single)


# ---------------------------------------------------------------------------
# EvalCountingTarget tests
# ---------------------------------------------------------------------------

def test_eval_counter_counts_rows():
    """EvalCountingTarget counts the number of point evaluations."""
    from eryn.flows.benchmark import EvalCountingTarget

    target = EvalCountingTarget(lambda a: -0.5 * np.sum(np.atleast_2d(a) ** 2, axis=1))
    target(np.zeros((7, 2)))
    target(np.zeros((3, 2)))
    assert target.n_evals == 10


def test_eval_counter_reset():
    """EvalCountingTarget.reset() zeroes the counter."""
    from eryn.flows.benchmark import EvalCountingTarget

    target = EvalCountingTarget(lambda a: np.zeros(len(np.atleast_2d(a))))
    target(np.zeros((5, 2)))
    target.reset()
    assert target.n_evals == 0


def test_eval_counter_returns_log_prob():
    """EvalCountingTarget passes through the log-prob values unchanged."""
    from eryn.flows.benchmark import EvalCountingTarget

    def lp(x):
        return -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)

    target = EvalCountingTarget(lp)
    pts = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_array_equal(target(pts), lp(pts))


# ---------------------------------------------------------------------------
# banana_periodic_target tests
# ---------------------------------------------------------------------------

def test_banana_periodic_target_returns_correct_ndim():
    """banana_periodic_target returns ndim = ndim_gauss + 1."""
    from eryn.flows.benchmark import banana_periodic_target

    _, _, ndim, periodic = banana_periodic_target(ndim_gauss=4)
    assert ndim == 5
    assert 4 in periodic


def test_banana_periodic_target_log_prob_shape():
    """log_prob returns shape (N,) for input (N, ndim)."""
    from eryn.flows.benchmark import banana_periodic_target

    lp, sampler, ndim, _ = banana_periodic_target(ndim_gauss=4)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, ndim))
    out = lp(x)
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))


def test_banana_periodic_target_sampler_shape():
    """sampler returns shape (n, ndim)."""
    from eryn.flows.benchmark import banana_periodic_target

    _, sampler, ndim, _ = banana_periodic_target(ndim_gauss=4)
    x = sampler(100, seed=0)
    assert x.shape == (100, ndim)


# ---------------------------------------------------------------------------
# chain_kde_target tests (requires scipy)
# ---------------------------------------------------------------------------

def test_chain_kde_target_basic():
    """chain_kde_target wraps scipy KDE and returns correct shapes."""
    scipy = pytest.importorskip("scipy")  # noqa: F841
    from eryn.flows.benchmark import chain_kde_target

    rng = np.random.default_rng(0)
    chain = rng.normal(size=(500, 3))
    lp, sampler, ndim, periodic = chain_kde_target(chain)
    assert ndim == 3
    assert periodic == {}
    x = rng.normal(size=(10, 3))
    out = lp(x)
    assert out.shape == (10,)
    draws = sampler(15, seed=1)
    assert draws.shape == (15, 3)


def test_chain_kde_target_with_periodic():
    """chain_kde_target propagates periodic dict."""
    pytest.importorskip("scipy")
    from eryn.flows.benchmark import chain_kde_target

    rng = np.random.default_rng(1)
    chain = rng.normal(size=(300, 2))
    periodic_in = {1: (0.0, 2 * np.pi)}
    _, _, _, periodic_out = chain_kde_target(chain, periodic=periodic_in)
    assert periodic_out == periodic_in


def test_chain_kde_target_sampler_oversample_falls_back_to_replacement():
    """sampler(n) with n > len(chain) draws with replacement instead of raising."""
    pytest.importorskip("scipy")
    from eryn.flows.benchmark import chain_kde_target

    rng = np.random.default_rng(2)
    chain = rng.normal(size=(10, 2))  # short chain
    _, sampler, _, _ = chain_kde_target(chain)
    # n > len(chain): would raise ValueError under replace=False
    draws = sampler(25, seed=0)
    assert draws.shape == (25, 2)
    # n <= len(chain): unique rows (no replacement)
    draws_small = sampler(10, seed=0)
    assert draws_small.shape == (10, 2)
    assert len(np.unique(draws_small, axis=0)) == 10


# ---------------------------------------------------------------------------
# GMMProposalDistribution tests (requires sklearn)
# ---------------------------------------------------------------------------

def test_gmm_proposal_logpdf_shape():
    """GMMProposalDistribution.logpdf returns shape (N,)."""
    pytest.importorskip("sklearn")
    from eryn.flows.benchmark import GMMProposalDistribution

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(200, 3))
    gmm = GMMProposalDistribution(samples, n_components=3, seed=0)
    x = rng.normal(size=(10, 3))
    out = gmm.logpdf(x)
    assert out.shape == (10,)
    assert np.all(np.isfinite(out))


def test_gmm_proposal_rvs_shape():
    """GMMProposalDistribution.rvs returns shape (n, ndim)."""
    pytest.importorskip("sklearn")
    from eryn.flows.benchmark import GMMProposalDistribution

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(200, 3))
    gmm = GMMProposalDistribution(samples, n_components=3, seed=0)
    draws = gmm.rvs(15)
    assert draws.shape == (15, 3)


def test_gmm_proposal_rvs_shuffle():
    """rvs draws from GMMProposalDistribution are shuffled.

    sklearn.GaussianMixture.sample returns draws grouped by component.
    Without the shuffle, walkers in an independent-MH move get pinned to one
    component each.  We verify that repeated rvs calls produce different
    orderings (if shuffle were absent the ordering would be deterministic and
    the two batches would be identical modulo the component assignments).
    This is a probabilistic test; with 100 draws and 5 components the
    probability that two shuffled batches have the same first-row value is
    negligible.
    """
    pytest.importorskip("sklearn")
    from eryn.flows.benchmark import GMMProposalDistribution

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 2))
    gmm = GMMProposalDistribution(samples, n_components=5, seed=42)
    a = gmm.rvs(100)
    b = gmm.rvs(100)
    # Two independent rvs calls should NOT be identical
    assert not np.array_equal(a, b)


def test_gmm_proposal_requires_sklearn(monkeypatch):
    """GMMProposalDistribution raises a clear ImportError without scikit-learn.

    The lazy ``from sklearn.mixture import GaussianMixture`` lives inside
    ``__init__``.  We block sklearn by setting its sys.modules entries to
    ``None`` (importing a ``None`` module raises ImportError), which exercises
    the production try/except even though sklearn is installed in the dev venv.
    """
    from eryn.flows.benchmark import GMMProposalDistribution

    # Setting a sys.modules entry to None makes ``import <name>`` raise
    # ImportError. monkeypatch restores both entries at teardown.
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.mixture", None)

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 2))
    with pytest.raises(ImportError, match="scikit-learn"):
        GMMProposalDistribution(samples, n_components=2)


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------

def test_benchmark_result_fields():
    """BenchmarkResult is a dataclass with the expected fields."""
    from eryn.flows.benchmark import BenchmarkResult

    r = BenchmarkResult(
        name="flow",
        ess_min=42.0,
        ess_per_eval=1e-3,
        n_evals=42000,
        acceptance=0.25,
    )
    assert r.name == "flow"
    assert r.ess_min == 42.0
    assert r.ess_per_eval == 1e-3
    assert r.n_evals == 42000
    assert r.acceptance == 0.25
    assert r.extra == {}


def test_benchmark_result_extra():
    """BenchmarkResult.extra defaults to empty dict and can be set."""
    from eryn.flows.benchmark import BenchmarkResult

    r = BenchmarkResult("x", 1.0, 1e-4, 100, 0.5, extra={"foo": "bar"})
    assert r.extra == {"foo": "bar"}
