# -*- coding: utf-8 -*-
"""Independent-Metropolis move backed by any logpdf/rvs distribution."""

from __future__ import annotations

import numpy as np

from .mh import MHMove

__all__ = ["IndependentProposalMove"]


class IndependentProposalMove(MHMove):
    """Independent-Metropolis move wrapping any object with ``logpdf`` / ``rvs``.

    Proposes new coordinates for a single named branch by drawing from an
    arbitrary proposal distribution.  Unlike :class:`eryn.moves.DistributionGenerate`,
    the distribution does not need to be an :class:`eryn.prior.ProbDistContainer`;
    any object that exposes ``logpdf(x) -> (N,)`` and ``rvs(size) -> (N, ndim)`` is
    accepted.  This covers distributions that cannot be expressed as
    ``ProbDistContainer`` objects (e.g. full-covariance Gaussian mixture models).

    The Hastings factor follows the independent-Metropolis convention::

        factors += +log q(x_old) - log q(x_new)

    Accumulation uses :func:`numpy.add.at` (not ``+=``) so that duplicate
    ``(temp, walker)`` indices from multi-leaf proposals sum instead of being
    silently overwritten by the last write.

    Parameters
    ----------
    dist : object
        Proposal distribution.  Must expose:

        - ``logpdf(x: np.ndarray) -> np.ndarray`` — coords-space log density,
          shape ``(N,)`` for input shape ``(N, ndim)``.
        - ``rvs(size: int) -> np.ndarray`` — draw ``size`` samples, returning
          shape ``(size, ndim)``.

    branch_name : str
        Name of the branch this move proposes for.  All other branches in
        ``branches_coords`` are copied unchanged.
    *args, **kwargs
        Passed through to :class:`eryn.moves.MHMove`.

    Examples
    --------
    >>> from scipy.stats import multivariate_normal as mvn
    >>> class GaussianDist:
    ...     def logpdf(self, x): return mvn.logpdf(x)
    ...     def rvs(self, size): return mvn.rvs(size=size)
    >>> move = IndependentProposalMove(GaussianDist(), branch_name="x")
    """

    def __init__(self, dist, branch_name: str, *args, **kwargs):
        self.dist = dist
        self.branch_name = branch_name
        super().__init__(*args, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate independent-Metropolis proposals for ``branch_name``.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` with current
            walker coordinates.
        random : object
            Current random state (unused by this move, passed for API compatibility).
        branches_inds : dict, optional
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max]`` booleans indicating
            which leaves are active.  Defaults to all-``True`` when ``None``.
        **kwargs
            Accepted and ignored for compatibility.

        Returns
        -------
        q : dict
            Proposed coordinates; same structure as ``branches_coords``.
        factors : np.ndarray, shape (ntemps, nwalkers)
            Hastings log-factors: ``+log q(x_old) - log q(x_new)`` accumulated
            with :func:`numpy.add.at` over all active leaves of ``branch_name``.
        """
        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        q = {}

        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords.items()
            }

        first = next(iter(branches_coords.values()))
        factors = np.zeros(first.shape[:2])

        for name, coords in branches_coords.items():
            q[name] = coords.copy()

            if name != self.branch_name:
                continue

            where = np.where(branches_inds[name])
            num = len(where[0])
            if num == 0:
                continue

            # + log q(old): np.add.at so multi-leaf duplicate (temp, walker) indices
            # accumulate instead of last-write-wins on repeated fancy-index +=.
            np.add.at(factors, where[:2], self.dist.logpdf(coords[where]))

            new = self.dist.rvs(num)

            # - log q(new)
            np.add.at(factors, where[:2], -self.dist.logpdf(new))

            q[name][where] = new

        return q, factors
