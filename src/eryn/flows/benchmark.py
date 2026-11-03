# src/eryn/flows/benchmark.py
"""ESS/eval benchmark utilities for the flow-proposal P0 go/no-go gate.

This module provides:

- :func:`effective_sample_size` — ensemble-aware ESS (Sokal windowing).
- :class:`EvalCountingTarget` — wraps any log-density and counts evaluations.
- :func:`banana_periodic_target` — synthetic curved target with one periodic dim.
- :func:`chain_kde_target` — realistic non-Gaussian target from a saved chain.
- :class:`GMMProposalDistribution` — sklearn GMM wrapped as a logpdf/rvs proposal.
- :class:`BenchmarkResult` — lightweight result dataclass.

**Import API**

This module is *not* re-exported from ``eryn.flows.__all__``; import it
explicitly::

    from eryn.flows.benchmark import effective_sample_size, BenchmarkResult
    # or
    from eryn.flows import benchmark

**Optional dependencies**

``scipy`` (``gaussian_kde`` in :func:`chain_kde_target`) and
``scikit-learn`` (:class:`GMMProposalDistribution`) are imported lazily inside
the functions/classes that need them.  Importing this module never triggers
either import.  ``torch`` is never imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# ESS / autocorrelation (Sokal windowing, ensemble-aware)
# ---------------------------------------------------------------------------

def _autocorr_1d(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT convolution."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acf[0] == 0:
        return np.zeros_like(acf)
    return acf / acf[0]


def _tau_from_acf(acf: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time via Sokal automatic windowing."""
    taus = 2.0 * np.cumsum(acf) - 1.0
    m = np.arange(len(taus)) < c * taus  # Sokal automatic windowing condition
    window = int(np.argmin(m)) if np.any(~m) else len(taus) - 1
    return max(taus[window], 1.0)


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Per-dimension effective sample size, ensemble-aware.

    Parameters
    ----------
    samples : np.ndarray
        Either a single chain of shape ``(nsteps, ndim)`` or an ensemble of
        shape ``(nsteps, nwalkers, ndim)``.

    Returns
    -------
    ess : np.ndarray, shape (ndim,)
        Effective sample size per dimension.

    Notes
    -----
    For the ensemble case the autocorrelation is averaged across walkers
    *before* Sokal windowing (emcee-style).  Do **not** flatten walkers into
    the time axis — that destroys the temporal autocorrelation structure and
    biases ESS upward (and differently across proposals, which would corrupt
    the gate comparison).
    """
    samples = np.asarray(samples)
    if samples.ndim == 2:  # (nsteps, ndim) — single chain
        samples = samples[:, None, :]
    n, nwalkers, ndim = samples.shape
    ess = np.empty(ndim)
    for d in range(ndim):
        acf = np.mean(
            [_autocorr_1d(samples[:, w, d]) for w in range(nwalkers)], axis=0
        )
        ess[d] = n * nwalkers / _tau_from_acf(acf)
    return ess


# ---------------------------------------------------------------------------
# Eval-counting target wrapper
# ---------------------------------------------------------------------------

class EvalCountingTarget:
    """Wraps a vectorized log-density and counts point evaluations.

    Parameters
    ----------
    log_prob : callable
        Vectorized log-density.  Must accept ``(N, ndim)`` and return ``(N,)``.

    Attributes
    ----------
    n_evals : int
        Total number of individual point evaluations since construction or
        the last :meth:`reset` call.

    Notes
    -----
    ``n_evals`` is a plain attribute increment, not an atomic counter; it is
    not safe under concurrent calls (e.g. a multithreaded pool).

    Examples
    --------
    >>> import numpy as np
    >>> target = EvalCountingTarget(lambda x: -0.5 * np.sum(x**2, axis=1))
    >>> _ = target(np.zeros((10, 3)))
    >>> target.n_evals
    10
    """

    def __init__(self, log_prob: Callable[[np.ndarray], np.ndarray]):
        self._log_prob = log_prob
        self.n_evals: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate log-density and increment :attr:`n_evals` by row count."""
        x = np.atleast_2d(x)
        self.n_evals += x.shape[0]
        return self._log_prob(x)

    def reset(self) -> None:
        """Reset the evaluation counter to zero."""
        self.n_evals = 0


# ---------------------------------------------------------------------------
# Synthetic and KDE targets
# ---------------------------------------------------------------------------

def banana_periodic_target(ndim_gauss: int = 4, period: float = 2 * np.pi):
    """Non-Gaussian / curved (banana) target with a periodic dimension.

    A realistic posterior stand-in when no real chain file is available.
    The Gaussian part is a curved 'banana' distribution; the last dimension
    is periodic.

    Parameters
    ----------
    ndim_gauss : int, optional
        Number of Gaussian (curved) dimensions.  Default is 4.
    period : float, optional
        Period of the last dimension.  Default is 2π.

    Returns
    -------
    log_prob : callable
        Vectorized log-density accepting ``(N, ndim)`` and returning ``(N,)``.
    sampler : callable
        ``sampler(n, seed=0)`` draws ``n`` exact samples from the target.
    ndim : int
        Total number of dimensions (``ndim_gauss + 1``).
    periodic : dict
        Maps the periodic dimension index to ``(0.0, period)``.

    Examples
    --------
    >>> lp, sampler, ndim, periodic = banana_periodic_target(ndim_gauss=4)
    >>> ndim
    5
    >>> 4 in periodic
    True
    """
    b = 0.5

    def log_prob(x):
        x = np.atleast_2d(x)
        g = x[:, :ndim_gauss].copy()
        g[:, 1] = g[:, 1] + b * (g[:, 0] ** 2 - 1.0)  # un-bend
        lpg = -0.5 * np.sum(g ** 2, axis=1)
        d = (x[:, ndim_gauss] - 1.0 + np.pi) % period - np.pi
        lpp = -0.5 * (d / 0.4) ** 2
        return lpg + lpp

    def sampler(n, seed=0):
        rng = np.random.default_rng(seed)
        g = rng.normal(size=(n, ndim_gauss))
        g[:, 1] = g[:, 1] - b * (g[:, 0] ** 2 - 1.0)  # bend
        ang = (rng.normal(1.0, 0.4, size=n)) % period
        return np.column_stack([g, ang])

    return log_prob, sampler, ndim_gauss + 1, {ndim_gauss: (0.0, period)}


def chain_kde_target(
    chain: np.ndarray,
    periodic: dict | None = None,
    n_reference: int = 2000,
    bw: str | float = "scott",
    seed: int = 0,
):
    """Realistic non-Gaussian target built from a Gaussian KDE over a saved chain.

    Uses ``scipy.stats.gaussian_kde`` (imported lazily).  Because the KDE is
    non-parametric it does not trivially match the NSF flow or the GMM baseline,
    making the benchmark fair.

    Parameters
    ----------
    chain : np.ndarray, shape (N, ndim)
        Cold-chain samples for one leaf (or any single-mode distribution).
    periodic : dict or None, optional
        Maps dimension index to ``(lower, upper)`` period bounds.  Passed
        through unchanged.  Default is ``None`` (no periodic dimensions).
    n_reference : int, optional
        Number of chain samples used to build the KDE.  Default is 2000.
    bw : str or float, optional
        Bandwidth method passed to ``scipy.stats.gaussian_kde``.  Default is
        ``"scott"``.
    seed : int, optional
        Random seed used when sub-sampling ``n_reference`` points from
        ``chain``.  Default is 0.

    Returns
    -------
    log_prob : callable
        Vectorized log-density accepting ``(N, ndim)`` and returning ``(N,)``.
    sampler : callable
        ``sampler(n, seed=0)`` draws ``n`` samples by sub-sampling ``chain``.
        Drawing is without replacement when ``n <= len(chain)``; if ``n``
        exceeds the chain length the sampler falls back to drawing *with*
        replacement so oversampling never raises.
    ndim : int
        Number of dimensions (``chain.shape[1]``).
    periodic : dict
        The ``periodic`` argument (or ``{}`` if ``None``).

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    """
    from scipy.stats import gaussian_kde  # lazy — scipy is optional

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chain), size=min(n_reference, len(chain)), replace=False)
    ref = np.asarray(chain[idx], dtype=np.float64)
    kde = gaussian_kde(ref.T, bw_method=bw)

    def log_prob(x):
        return kde.logpdf(np.atleast_2d(x).T)

    def sampler(n, seed=0):
        r = np.random.default_rng(seed)
        # oversampling (n > len(chain)) can't draw without replacement, so
        # fall back to replacement in that case rather than raising.
        return chain[r.choice(len(chain), size=n, replace=n > len(chain))]

    ndim = chain.shape[1]
    return log_prob, sampler, ndim, (periodic or {})


# ---------------------------------------------------------------------------
# GMM baseline proposal
# ---------------------------------------------------------------------------

class GMMProposalDistribution:
    """sklearn GaussianMixture wrapped as an independent-proposal (logpdf/rvs).

    Compatible with :class:`eryn.moves.IndependentProposalMove` — exposes
    ``logpdf(x) -> (N,)`` and ``rvs(size) -> (N, ndim)``.

    Requires ``scikit-learn`` (imported lazily; ``pip install scikit-learn``).

    Parameters
    ----------
    samples : np.ndarray, shape (N, ndim)
        Training data used to fit the GMM.
    n_components : int, optional
        Number of mixture components.  Default is 10.
    seed : int, optional
        Random seed for reproducibility (passed to ``GaussianMixture`` and the
        internal NumPy RNG used for shuffling).  Default is 0.

    Raises
    ------
    ImportError
        If ``scikit-learn`` is not installed.
    """

    def __init__(self, samples: np.ndarray, n_components: int = 10, seed: int = 0):
        try:
            from sklearn.mixture import GaussianMixture  # lazy — sklearn is optional
        except ImportError as exc:
            raise ImportError(
                "GMMProposalDistribution requires scikit-learn; "
                "pip install scikit-learn"
            ) from exc

        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
        ).fit(np.asarray(samples, dtype=np.float64))
        self._rng = np.random.default_rng(seed)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-density at ``x``.

        Parameters
        ----------
        x : np.ndarray, shape (N, ndim)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        return self.gmm.score_samples(np.atleast_2d(x))

    def rvs(self, size) -> np.ndarray:
        """Draw samples from the mixture.

        Parameters
        ----------
        size : int or tuple of int
            Number of samples to draw.

        Returns
        -------
        np.ndarray, shape (n, ndim)
        """
        n = int(np.prod(size)) if isinstance(size, (tuple, list)) else int(size)
        x = self.gmm.sample(n)[0]
        # sklearn returns draws ordered by mixture component; shuffle so an
        # independent-MH move that binds draw i to walker i does not pin each
        # walker to a single component (which would cripple the GMM baseline's
        # ESS and let the flow pass the gate spuriously).
        self._rng.shuffle(x)
        return x


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Benchmark results for one proposal method.

    Parameters
    ----------
    name : str
        Human-readable name of the proposal (e.g. ``"flow"``, ``"gmm"``,
        ``"stretch"``).
    ess_min : float
        Minimum ESS across all dimensions.
    ess_per_eval : float
        ``ess_min / n_evals`` — the P0 gate metric.
    n_evals : int
        Total likelihood evaluations during sampling.
    acceptance : float
        Mean per-walker acceptance fraction.
    extra : dict, optional
        Additional metadata for logging.  Default is an empty dict.
    """

    name: str
    ess_min: float
    ess_per_eval: float
    n_evals: int
    acceptance: float
    extra: dict = field(default_factory=dict)
