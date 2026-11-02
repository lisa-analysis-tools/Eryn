# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree

from .mh import MHMove
from ..utils.periodic import PeriodicContainer

__all__ = ["LeafPoolMove", "LeafPoolAMMove", "LeafPoolStretchMove"]


class LeafPoolMove(MHMove):
    """Base class for in-model moves of one (reversible-jump) branch built from a
    pool of past leaves.

    Every ``n_update`` calls the active leaves of each temperature are stored; the
    last ``n_snap`` snapshots form the pool, with one k-d tree per temperature on the
    ``metric`` columns (scaled by their spread in the pool). Each call moves
    ``n_move`` randomly chosen active leaves per walker (all of them by default),
    with the proposal implemented by subclasses in :func:`propose_leaves`.

    The proposals use the ``k`` pooled leaves nearest to the moved leaf. ``k`` is set
    at each pool update to a fraction ``k_frac`` of the pool size per source,
    ``k = max(k_frac * n_snapshots * nwalkers, 10 * ndim)``: assuming every walker
    holds each source, a pool of ``n_snapshots`` snapshots has
    ``n_snapshots * nwalkers`` samples of each source, so the neighbours cover a fixed
    fraction of the posterior of one source. The floor keeps the local covariance
    well determined.

    Leaves are paired by position rather than by leaf slot, so the moves stay
    well defined under RJ births and deaths and temperature swaps. With many
    sources in one branch, the nearest pooled leaves are usually the other walkers'
    samples of the same source, when the sources are well separated compared with
    their posterior widths, and the proposal then follows the local posterior of
    that source. This is an assumption about the problem, not a requirement: when
    sources are close together, the neighbours can include samples of nearby
    sources, the proposals become broader (lower acceptance, occasional jumps
    between sources), and only the efficiency is affected.

    Between two pool updates the pool is fixed and the move satisfies detailed
    balance, whatever the neighbours are. Updating the pool is not a proposal, so
    it needs no acceptance factor; this is the same scheme as the stationary group
    of :class:`eryn.moves.GroupMove`. Because the pool is built from the chain's own
    past states, the sampler as a whole is an adaptive MCMC. Convergence to the
    posterior is guaranteed when the adaptation diminishes: the interval between
    updates grows by ``update_growth`` after each update (``update_growth > 1``),
    or the pool is frozen after ``freeze_after`` calls. With ``update_growth = 1``
    and no freezing the pool keeps following the chain, as the group moves do. The
    recommended use is to update the pool during burn-in and freeze it afterwards,
    so that every post-burn-in step uses one fixed, exact kernel.

    These moves are good at large jumps but weaker at fine local refinement. In
    practice they work best combined with a small ``GaussianMove`` on the same
    branch.

    Args:
        branch (str): Name of the branch the move acts on.
        metric (list, optional): Columns used for the nearest-neighbour search. If
            ``None``, all columns are used. (default: ``None``)
        n_snap (int, optional): Number of snapshots kept in the pool. (default: 50)
        n_update (int, optional): Calls between the first two snapshots. (default: 20)
        n_move (int, optional): Number of active leaves moved per walker and call.
            If ``None``, all active leaves are moved. (default: ``None``)
        k_frac (float, optional): Number of neighbours used by the proposals, as a
            fraction of the pool size per source (see above). Too few neighbours make
            the proposals too narrow and the chain mix slowly. (default: 0.1)
        update_growth (float, optional): Factor by which the interval between two
            snapshots grows after each snapshot. Values above 1 make the adaptation
            diminish. (default: 1.0)
        freeze_after (int, optional): Stop updating the pool after this many calls.
            If ``None``, the pool is never frozen; setting it to the number of calls
            during burn-in is recommended. (default: ``None``)
        periodic (dict or :class:`eryn.utils.PeriodicContainer`, optional): Periodic
            parameters, as for :class:`eryn.ensemble.EnsembleSampler`:
            ``{branch: {column: period}}``. (default: ``None``)
        **kwargs (dict, optional): Kwargs for parent classes. ``gibbs_sampling_setup``
            is set to ``[branch]``.

    """

    def __init__(
        self,
        branch,
        metric=None,
        n_snap=50,
        n_update=20,
        n_move=None,
        k_frac=0.1,
        update_growth=1.0,
        freeze_after=None,
        periodic=None,
        **kwargs
    ):
        if isinstance(periodic, dict):
            periodic = PeriodicContainer(periodic)
        super(LeafPoolMove, self).__init__(
            gibbs_sampling_setup=[branch], periodic=periodic, **kwargs
        )
        self.branch = branch
        self.metric = None if metric is None else np.asarray(metric)
        self.n_snap, self.n_update, self.n_move = n_snap, n_update, n_move
        self.k_frac, self.k = float(k_frac), None
        if update_growth < 1:
            raise ValueError("update_growth must be greater than or equal to 1.")
        self.update_growth = float(update_growth)
        self.freeze_after = freeze_after
        self.calls, self.snaps, self.trees = 0, [], None
        self.next_update, self.interval = 0, float(n_update)

    @property
    def frozen(self):
        """Whether the pool is no longer updated."""
        return self.freeze_after is not None and self.calls >= self.freeze_after

    def diff(self, x1, x2):
        """``x2 - x1`` for arrays of leaves ``(..., ndim)``, wrapping periodic columns."""
        if self.periodic is None:
            return x2 - x1
        shape = np.broadcast_shapes(x1.shape, x2.shape)
        x1 = np.broadcast_to(x1, shape).reshape(-1, 1, shape[-1])
        x2 = np.broadcast_to(x2, shape).reshape(-1, 1, shape[-1])
        return self.periodic.distance({self.branch: x1}, {self.branch: x2})[
            self.branch
        ].reshape(shape)

    def wrap(self, x):
        """Wrap periodic columns of an array of leaves ``(..., ndim)`` in place."""
        if self.periodic is None:
            return x
        return self.periodic.wrap({self.branch: x.reshape(-1, 1, x.shape[-1])})[
            self.branch
        ].reshape(x.shape)

    def _rebuild(self):
        nt = len(self.snaps[0])
        self.pool, self.trees, self.mscale = [], [], []
        for t in range(nt):
            P = np.concatenate([s[t] for s in self.snaps], axis=0)
            sc = (
                np.std(P[:, self.metric], axis=0) + 1e-300
                if len(P) > 1
                else np.ones(self.metric.size)
            )
            self.pool.append(P)
            self.mscale.append(sc)
            self.trees.append(cKDTree(P[:, self.metric] / sc) if len(P) else None)

    def neighbours(self, t, X, k, exclude_self=False):
        """``k`` pooled leaves of temperature ``t`` nearest to each row of ``X``
        (metric columns). With ``exclude_self``, pool points identical to ``X`` (its
        own snapshot) are skipped, the same rule in both directions of a move.

        Returns:
            tuple: (neighbours, their indices in the pool) with shapes
            ``(len(X), k, ndim)`` and ``(len(X), k)``.

        """
        P, tree = self.pool[t], self.trees[t]
        kq = min(k + (1 if exclude_self else 0), len(P))
        _, j = tree.query(X[:, self.metric] / self.mscale[t], k=kq)
        j = np.asarray(j).reshape(len(X), kq)
        if exclude_self:
            same = np.all(P[j] == X[:, None, :], axis=-1)
            keep = np.argsort(same, axis=1, kind="stable")[:, : min(k, kq - 1)]
            j = np.take_along_axis(j, keep, axis=1)
        return P[j], j

    def propose_leaves(self, t, X, random, n_moved):
        """Propose new positions for the leaves ``X`` of temperature ``t``.

        Args:
            t (int): Temperature index.
            X (np.ndarray): Leaves to move with shape ``(n, ndim)``.
            random (object): Current random state object.
            n_moved (np.ndarray): Number of leaves moved together with each leaf in
                its walker (itself included) with shape ``(n,)``.

        Returns:
            tuple: (new leaves ``(n, ndim)``, log proposal factors ``(n,)``)

        """
        raise NotImplementedError

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from the leaf pool

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)

        """
        b = self.branch
        coords = branches_coords[b]
        nt, nw, L, d = coords.shape
        inds = (
            np.ones((nt, nw, L), dtype=bool)
            if branches_inds is None
            else branches_inds[b]
        )
        if self.metric is None:
            self.metric = np.arange(d)

        q = {n: c.copy() for n, c in branches_coords.items()}
        factors = np.zeros((nt, nw))
        if self.trees is None or (self.calls >= self.next_update and not self.frozen):
            self.snaps.append([coords[t][inds[t]].copy() for t in range(nt)])
            self.snaps = self.snaps[-self.n_snap :]
            self._rebuild()
            self.k = max(int(round(self.k_frac * len(self.snaps) * nw)), 10 * d)
            self.next_update = self.calls + max(int(round(self.interval)), 1)
            self.interval *= self.update_growth
        self.calls += 1

        # n_move random active leaves per walker
        key = np.where(inds, random.rand(nt, nw, L), np.inf)
        order = np.argsort(key, axis=-1)[..., : (L if self.n_move is None else self.n_move)]
        for t in range(nt):
            if self.trees[t] is None or len(self.pool[t]) < 3:
                continue
            ww, ll = np.nonzero(np.take_along_axis(inds[t], order[t], axis=-1))
            ll = order[t][ww, ll]
            if ww.size == 0:
                continue
            n_moved = np.bincount(ww, minlength=nw)[ww]
            Y, f = self.propose_leaves(t, coords[t, ww, ll], random, n_moved)
            q[b][t, ww, ll] = Y
            np.add.at(factors[t], ww, f)
        return q, factors


class LeafPoolAMMove(LeafPoolMove):
    """Adaptive-Metropolis Gaussian move from the leaf pool.

    ``x' = x + s (2.38/sqrt(m d)) L z`` with ``L L^T`` the covariance of the ``k``
    pooled leaves nearest to ``x`` (see :class:`LeafPoolMove`) (periodic columns as wrapped differences), ``s``
    drawn uniformly from ``scales`` and ``m`` the number of leaves of the walker
    moved together, so the step follows the adaptive-Metropolis scaling for the
    ``m d`` parameters moved at once. The covariance depends on the position, so the
    Hastings factor uses the scale-mixture densities ``q(x|x') / q(x'|x)``, with the
    covariance evaluated at ``x'`` for the reverse move.

    Args:
        branch (str): Name of the branch the move acts on.
        scales (tuple, optional): Step scales of the mixture. (default:
            ``(1.0, 0.3, 0.1, 0.03)``)
        **kwargs (dict, optional): Kwargs for :class:`LeafPoolMove`.

    """

    def __init__(self, branch, scales=(1.0, 0.3, 0.1, 0.03), **kwargs):
        super(LeafPoolAMMove, self).__init__(branch, **kwargs)
        self.scales = np.asarray(scales, float)

    def local_cov(self, t, X):
        """Neighbour covariance ``C = D V diag(lam) V^T D``. The columns are
        standardised by ``D`` (the neighbour spread) before the eigen-decomposition,
        so parameters of very different magnitudes are handled."""
        nb, _ = self.neighbours(t, X, self.k)
        d = self.diff(X[:, None, :], nb)
        d = d - d.mean(axis=1, keepdims=True)
        D = np.sqrt(np.mean(d**2, axis=1))
        D = np.where(D > 0, D, np.maximum(np.abs(X) * 1e-12, 1e-300))
        d = d / D[:, None, :]
        C = np.einsum("nki,nkj->nij", d, d) / max(d.shape[1] - 1, 1)
        lam, V = np.linalg.eigh(C)
        lam = np.maximum(lam, 1e-8 * np.maximum(lam[:, -1:], 1e-300))
        return lam, V, D

    def logq(self, y, x, lam, V, D, n_moved):
        """log of the scale mixture of N(y; x, (s c)^2 C), C = D V diag(lam) V^T D."""
        d = x.shape[1]
        c2 = 2.38**2 / (d * n_moved)
        r = np.einsum("nij,ni->nj", V, self.diff(x, y) / D) ** 2 / lam
        m = r.sum(axis=1)
        ld = np.sum(np.log(lam), axis=1) + 2 * np.sum(np.log(D), axis=1)
        terms = [
            -0.5 * m / (s * s * c2) - 0.5 * (ld + d * np.log(2 * np.pi * s * s * c2))
            for s in self.scales
        ]
        return np.logaddexp.reduce(np.stack(terms), axis=0) - np.log(len(self.scales))

    def propose_leaves(self, t, X, random, n_moved):
        n, d = X.shape
        lam, V, D = self.local_cov(t, X)
        s = self.scales[random.randint(len(self.scales), size=n)]
        z = random.randn(n, d) * np.sqrt(lam)
        Y = self.wrap(
            X
            + (s * 2.38 / np.sqrt(d * n_moved))[:, None]
            * D
            * np.einsum("nij,nj->ni", V, z)
        )
        lam2, V2, D2 = self.local_cov(t, Y)
        return Y, self.logq(X, Y, lam2, V2, D2, n_moved) - self.logq(
            Y, X, lam, V, D, n_moved
        )


class LeafPoolStretchMove(LeafPoolMove):
    """Group stretch move with friends from the leaf pool.

    A friend ``c`` is drawn uniformly from the ``k`` pooled leaves nearest to
    ``x`` and ``x' = c + z (x - c)``, with ``g(z) ~ 1/sqrt(z)`` on ``[1/a_m, a_m]`` and
    factor ``(d - 1) ln z`` for each moved leaf. All leaves of a walker are accepted
    or rejected together, so the scale shrinks with the number ``m`` of leaves moved
    in the walker, ``a_m = a ** (1 / sqrt(m))`` (the spread of ``ln z`` falls as
    ``1 / sqrt(m)``, as the step of :class:`LeafPoolAMMove`). The move is rejected unless ``c`` is
    also among the friends of ``x'``, so the friend is selected with the same
    probability in both directions.

    ``k`` is the same as for :class:`LeafPoolAMMove` (see :class:`LeafPoolMove`).
    Larger ``k`` puts the friends farther from ``x`` and gives larger steps.

    Args:
        branch (str): Name of the branch the move acts on.
        a (float, optional): Stretch scale for a single moved leaf. (default: 2.0)
        **kwargs (dict, optional): Kwargs for :class:`LeafPoolMove`.

    """

    def __init__(self, branch, a=2.0, **kwargs):
        super(LeafPoolStretchMove, self).__init__(branch, **kwargs)
        self.a = float(a)

    def propose_leaves(self, t, X, random, n_moved):
        n, d = X.shape
        nb, j = self.neighbours(t, X, self.k, exclude_self=True)
        pick = random.randint(nb.shape[1], size=n)
        C, jc = nb[np.arange(n), pick], j[np.arange(n), pick]
        am = self.a ** (1.0 / np.sqrt(n_moved))
        zz = ((am - 1.0) * random.rand(n) + 1.0) ** 2 / am
        Y = self.wrap(C + zz[:, None] * self.diff(C, X))
        _, j2 = self.neighbours(t, Y, self.k, exclude_self=True)
        back = np.any(j2 == jc[:, None], axis=1)
        return Y, np.where(back, (d - 1.0) * np.log(zz), -np.inf)
