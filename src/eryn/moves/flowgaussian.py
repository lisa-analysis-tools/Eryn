# -*- coding: utf-8 -*-
"""Metropolis-Hastings move: a conditional flow on a subset of the parameters, a Gaussian walk on the rest.

Torch-free at import time, like :mod:`eryn.moves.flow`.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from .flow import ConditionalFlowMove
from .gaussian import _make_proposal

__all__ = ["FlowGaussianMove"]
logger = logging.getLogger(__name__)


class FlowGaussianMove(ConditionalFlowMove):
    """Flow proposal on the columns in ``flow_params``, Gaussian random walk on the other columns.

    The flow columns ``A`` are drawn independently from the (leaf-conditioned) flow, as in
    :class:`ConditionalFlowMove`; the remaining columns ``B`` take a symmetric Gaussian
    step, as in :class:`eryn.moves.GaussianMove`.

    **Which columns move.** With ``flow_prob`` (default ``0.5``) each active row moves
    *one* half: the flow columns with probability ``flow_prob``, otherwise the Gaussian
    columns. The choice is drawn independently of the state, so this is a random-scan
    mixture of two valid MH kernels, and each half is accepted on its own (as a Gibbs
    split would be). With ``flow_prob=None`` both halves move together in one joint
    proposal; its acceptance is roughly the product of the two halves', so it suits
    weakly coupled ``A``/``B`` with a well-trained flow.

    Either way the Hastings factor is the flow's alone: the flow does not depend on
    ``B`` and the Gaussian step is symmetric, so for a flow (or joint) row::

        factors += +log q_flow(A_old) - log q_flow(A_new)

    and a Gaussian-only row has a zero factor. The split happens inside
    :meth:`get_proposal`, so callers see an ordinary full-vector MH move and no
    ``gibbs_sampling_setup`` is needed.

    Everything else of :class:`ConditionalFlowMove` is inherited: online training
    (``executor`` / harvest / hot reload), :attr:`active_condition` and
    :attr:`active_betas` (the base scaling applies to the flow columns only; the
    Gaussian step is not beta-scaled, which keeps it exact -- hot rows just explore
    less). Training data reaching the executor is sliced to the flow columns here,
    so :meth:`submit_by_leaf` takes full rows.

    Until the flow's data transform is fitted every row moves its Gaussian columns only
    (zero factor) -- still a valid MH step, unlike the full no-op of
    :class:`ConditionalFlowMove`; likewise every row moves its flow columns while the
    Gaussian columns have no covariance yet.

    **Covariance of the Gaussian columns.** ``cov`` is fixed by default. With
    ``adapt_cov=True`` the covariance of each leaf (keyed by
    :attr:`active_condition`) is re-estimated from the rows passed to
    :meth:`submit_by_leaf`, i.e. from the same chain history the flow is trained on.
    This is adaptive Metropolis: the kernel uses past states, the same kind of
    adaptation as the online-trained flow.

    The Gaussian columns are typically the multimodal ones (sky, orientation), with
    jumps between modes left to other moves. Their covariance across all walkers would
    include the distance between modes and size the steps to land between them, so when
    the submitted rows keep the walker axis, ``(n_snapshots, nwalkers, ndim)``, the
    estimate is the **within-mode covariance** over the last ``adapt_window`` submits:

    1. each walker's trajectory is cut wherever it jumps (a step more than ``adapt_trim``
       times the typical nonzero step in some column, e.g. a sky reflection or a
       temperature swap); the spread within the pieces gives a length scale (narrower
       than the mode, as consecutive snapshots are autocorrelated);
    2. at each snapshot the walkers are grouped into modes by single linkage, split where
       they are more than ``adapt_trim`` length scales apart, and the covariance is the
       spread about each group's mean, pooled over groups and snapshots. Walkers at one
       snapshot are close to independent, so it has no autocorrelation bias.

    Periodic columns use wrapped differences throughout. Flat ``(n_samples, ndim)`` rows
    give the plain covariance over all rows.

    Args:
        flow (eryn.flows.Flow): Conditional flow over the ``flow_params`` columns
            (``flow.dims == flow_params.sum()``).
        branch_name (str): Branch this move proposes for.
        flow_params (array-like of bool): ``(ndim,)`` mask of the flow columns. At least
            one column must be outside the mask (otherwise use
            :class:`ConditionalFlowMove`).
        cov (scalar, np.ndarray, dict or None, optional): Covariance of the Gaussian
            columns: a scalar, ``(nB,)`` or ``(nB, nB)`` as in
            :class:`eryn.moves.GaussianMove` (with ``mode="am"`` a scalar or vector is
            expanded to the diagonal matrix, the target covariance), or a dict ``{leaf: cov}`` for per-leaf
            covariances (a ``None`` key is the default for leaves without an entry).
            ``None`` is only allowed with ``adapt_cov=True``: the Gaussian columns are
            then kept fixed until the first adaptation. (default: ``None``)
        mode (str, optional): Gaussian proposal mode, see :class:`eryn.moves.GaussianMove`.
            (default: ``"am"``)
        factor (float, optional): See :class:`eryn.moves.GaussianMove`. (default: ``None``)
        single_component_prob (float, optional): See :class:`eryn.moves.GaussianMove`.
            (default: ``0.0``)
        adapt_cov (bool, optional): Re-estimate each leaf's covariance in
            :meth:`submit_by_leaf`. (default: ``False``)
        adapt_min_samples (int, optional): Fewest rows of a leaf used for an adaptation.
            (default: ``max(10 * nB, 2 * nB + 2)``)
        adapt_window (int, optional): Number of recent submits per leaf pooled for the
            within-mode covariance. (default: ``5``)
        adapt_trim (float, optional): How many typical scales apart counts as another mode:
            a walker step this many times the typical nonzero step cuts its trajectory, and
            walkers this many length scales apart fall in different groups. (default: ``10``)
        flow_prob (float or None, optional): Probability that a row moves its flow columns
            (otherwise its Gaussian columns); ``None`` moves both halves jointly.
            (default: ``0.5``)
        executor, harvest_every, harvest_temp_index, **kwargs: As in
            :class:`ConditionalFlowMove`.

    Raises:
        ValueError: If the mask, the flow dimension or the covariance arguments are
            inconsistent.

    """

    def __init__(
        self,
        flow,
        branch_name: str,
        flow_params,
        cov=None,
        mode: str = "am",
        factor=None,
        single_component_prob: float = 0.0,
        adapt_cov: bool = False,
        adapt_min_samples: int | None = None,
        adapt_window: int = 5,
        adapt_trim: float = 10.0,
        flow_prob: float | None = 0.5,
        executor=None,
        harvest_every: int = 1,
        harvest_temp_index: int = 0,
        **kwargs,
    ):
        flow_params = np.asarray(flow_params)
        if flow_params.dtype != bool or flow_params.ndim != 1:
            raise ValueError(f"flow_params must be a 1-D boolean mask, got {flow_params!r}.")
        if flow_params.all() or not flow_params.any():
            raise ValueError(
                "flow_params must select some but not all columns "
                "(all of them: use ConditionalFlowMove; none: use GaussianMove)."
            )
        dims = getattr(flow, "dims", None)
        if dims is not None and dims != flow_params.sum():
            raise ValueError(f"flow.dims = {dims} but flow_params selects {flow_params.sum()} columns.")
        if cov is None and not adapt_cov:
            raise ValueError("cov is required unless adapt_cov=True.")
        if mode not in ("vector", "am", "random", "sequential"):
            raise ValueError(f"'{mode}' is not a recognized Gaussian mode.")
        if flow_prob is not None and not 0.0 < flow_prob < 1.0:
            raise ValueError(f"flow_prob must be in (0, 1) or None, got {flow_prob}.")

        self.flow_params = flow_params
        self.ndim = len(flow_params)
        self.nB = int((~flow_params).sum())
        self.mode = mode
        self.factor = factor
        self.single_component_prob = single_component_prob
        self.flow_prob = None if flow_prob is None else float(flow_prob)
        self.adapt_cov = bool(adapt_cov)
        self.adapt_min_samples = (
            max(10 * self.nB, 2 * self.nB + 2) if adapt_min_samples is None else int(adapt_min_samples)
        )
        if int(adapt_window) < 1 or adapt_trim <= 1.0:
            raise ValueError(f"adapt_window must be >= 1 and adapt_trim > 1, got {adapt_window}, {adapt_trim}.")
        self.adapt_window = int(adapt_window)
        self.adapt_trim = float(adapt_trim)
        self._adapt_history = {}  # leaf -> deque of (n_snapshots, nwalkers, ndim) submits

        # Gaussian proposal objects keyed by leaf (None: default for every leaf)
        self.gaussian_proposals = {}
        if cov is not None:
            self.update_covariance(cov if isinstance(cov, dict) else {None: cov})

        super().__init__(
            flow,
            branch_name,
            executor=executor,
            harvest_every=harvest_every,
            harvest_temp_index=harvest_temp_index,
            **kwargs,
        )

    def update_covariance(self, cov_by_leaf: dict):
        """Set the covariance of the Gaussian columns.

        Args:
            cov_by_leaf (dict): Keys are leaf indices (``None``: default for leaves
                without an entry), values a covariance as accepted by
                :class:`eryn.moves.GaussianMove`.

        """
        for leaf, cov in cov_by_leaf.items():
            cov = np.asarray(cov, dtype=float)
            if cov.shape not in ((), (self.nB,), (self.nB, self.nB)):
                raise ValueError(
                    f"{type(self).__name__}: leaf {leaf} cov has shape {cov.shape}; the Gaussian"
                    f" columns take a scalar, ({self.nB},) or ({self.nB}, {self.nB})."
                )
            if self.mode == "am" and cov.ndim < 2:
                # the "am" proposal decomposes a full matrix: expand isotropic/diagonal input
                cov = np.diag(np.broadcast_to(cov, (self.nB,)))
            key = None if leaf is None else int(leaf)
            self.gaussian_proposals[key] = _make_proposal(
                cov, self.mode, self.factor, self.single_component_prob
            )

    def gaussian_proposal(self, leaf: int):
        """The Gaussian proposal used for ``leaf``, or ``None`` if there is none yet."""
        return self.gaussian_proposals.get(int(leaf), self.gaussian_proposals.get(None))

    def submit_by_leaf(self, batch: dict):
        """Adapt the Gaussian covariances (if enabled) and send the flow columns to the executor.

        Args:
            batch (dict): Keys are leaf indices; values are full rows, either
                ``(n_samples, ndim)`` or ``(n_snapshots, nwalkers, ndim)``. With the walker
                axis the adaptation uses the within-walker covariance (see the class docstring).

        Raises:
            ValueError: If a value does not have ``ndim`` columns or is not 2-D/3-D.

        """
        batch = {leaf: np.asarray(rows) for leaf, rows in batch.items()}
        for leaf, rows in batch.items():
            if rows.ndim not in (2, 3) or rows.shape[-1] != self.ndim:
                raise ValueError(
                    f"{type(self).__name__}: leaf {leaf} rows have shape {rows.shape}, expected"
                    f" (n_samples, {self.ndim}) or (n_snapshots, nwalkers, {self.ndim}) full rows."
                )

        if self.adapt_cov:
            adapted = {}
            for leaf, rows in batch.items():
                cov = self._within_walker_cov(leaf, rows) if rows.ndim == 3 else self._marginal_cov(rows)
                if cov is not None:
                    # a tiny per-column relative jitter: well posed for collapsed walkers,
                    # scale-free across columns of very different size
                    adapted[leaf] = cov + np.diag(1e-12 * np.diag(cov))
            if adapted:
                self.update_covariance(adapted)
                logger.debug(f"{type(self).__name__}: adapted the Gaussian covariance of leaves {sorted(adapted)}.")

        super().submit_by_leaf({leaf: rows[..., self.flow_params] for leaf, rows in batch.items()})

    def _marginal_cov(self, rows):
        """Covariance of the Gaussian columns over all ``(n_samples, ndim)`` rows, or None if too few."""
        if len(rows) < self.adapt_min_samples:
            return None
        return np.atleast_2d(np.cov(rows[:, ~self.flow_params], rowvar=False))

    def _within_walker_cov(self, leaf, snapshots):
        """Within-mode covariance of the Gaussian columns over the rolling window, or None.

        Args:
            leaf (int): Leaf the snapshots belong to.
            snapshots (np.ndarray): ``(n_snapshots, nwalkers, ndim)`` cold-chain snapshots.

        """
        history = self._adapt_history.get(leaf)
        if history is None or history[0].shape[1] != snapshots.shape[1]:  # first or new nwalkers
            history = self._adapt_history[leaf] = deque(maxlen=self.adapt_window)
        history.append(snapshots)
        x = np.concatenate(history, axis=0).astype(float)[..., ~self.flow_params]  # (n_snap, nwalkers, nB)
        n_snap, nwalkers = x.shape[:2]
        if n_snap < 2 or nwalkers < 2:
            return None
        wrap = self._wrap_gaussian_columns

        # 1. a length scale: the spread of each walker's trajectory between jumps. A jump between
        # modes (e.g. a SkyMove reflection, a temperature swap) is a step far larger than the
        # typical nonzero one in some column. Consecutive snapshots are autocorrelated, so this
        # is narrower than the mode; it only sets the unit for step 2.
        step = wrap(np.diff(x, axis=0))
        size = np.abs(step)
        typical = np.array([np.median(c[c > 0]) if np.any(c > 0) else np.inf for c in size.reshape(-1, self.nB).T])
        jump = np.any(size > self.adapt_trim * typical, axis=-1)  # (n_snap - 1, nwalkers)
        step[jump] = 0.0
        path = np.concatenate([np.zeros((1, nwalkers, self.nB)), np.cumsum(step, axis=0)]).reshape(-1, self.nB)
        segment = np.concatenate([np.zeros((1, nwalkers), dtype=int), np.cumsum(jump, axis=0)])
        label = (segment * nwalkers + np.arange(nwalkers)).ravel()
        counts = np.bincount(label)
        sums = np.zeros((len(counts), self.nB))
        np.add.at(sums, label, path)
        dev = path - (sums / np.maximum(counts, 1)[:, None])[label]
        dof = len(path) - np.count_nonzero(counts)
        if dof < 1:
            return None
        scale = np.sqrt(np.diag(dev.T @ dev) / dof)
        scale = np.where(scale > 0, scale, np.inf)  # a column that never moved does not separate modes

        # 2. the modes at each snapshot: single-linkage groups of walkers, split where they are more
        # than adapt_trim length scales apart. The spread about each group's mean, pooled over
        # groups and snapshots, is the within-mode covariance; walkers at one snapshot are close to
        # independent, so it has no autocorrelation bias.
        from scipy.cluster.hierarchy import fcluster, linkage

        upper = np.triu_indices(nwalkers, 1)
        devs, dof = [], 0
        for xs in x:
            dist = np.sqrt(np.sum((wrap(xs[:, None] - xs[None, :]) / scale) ** 2, axis=-1))[upper]
            groups = fcluster(linkage(dist, "single"), t=self.adapt_trim, criterion="distance")
            for g in np.unique(groups):
                members = xs[groups == g]
                rel = wrap(members - members[0])
                devs.append(rel - rel.mean(axis=0))
                dof += len(members) - 1
        if dof < self.adapt_min_samples:
            return None
        dev = np.concatenate(devs)
        return dev.T @ dev / dof

    def _wrap_gaussian_columns(self, d):
        """Map differences of the Gaussian columns (last axis) of periodic parameters to ``[-P/2, P/2)``."""
        periods = getattr(self.periodic, "periods", {}).get(self.branch_name)
        if periods is None or len(periods) == 0:
            return d
        cols = np.flatnonzero(~self.flow_params)
        for ind, period in zip(self.periodic.inds_periodic[self.branch_name], periods):
            j = np.searchsorted(cols, ind)
            if j < len(cols) and cols[j] == ind:
                d[..., j] = (d[..., j] + period / 2) % period - period / 2
        return d

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Propose the flow columns from the flow and/or the other columns from the Gaussian walk.

        Args:
            branches_coords (dict): Keys are branch names, values
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.
            random (object): Current random state (``np.random.RandomState``-like), used by
                the Gaussian step.
            branches_inds (dict, optional): Keys are branch names, values
                ``np.ndarray[ntemps, nwalkers, nleaves_max]`` booleans of the active
                leaves. (default: all active)
            **kwargs (ignored): For compatibility.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray). ``factors`` holds
            ``+log q(A_old) - log q(A_new)`` summed over the active leaves whose flow
            columns moved.

        """
        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        q = {name: coords.copy() for name, coords in branches_coords.items()}
        coords = branches_coords[self.branch_name]
        factors = np.zeros(coords.shape[:2])
        if coords.shape[-1] != self.ndim:
            raise ValueError(
                f"{type(self).__name__}: branch {self.branch_name!r} has {coords.shape[-1]}"
                f" columns, flow_params has {self.ndim}."
            )

        if branches_inds is None:
            where = np.where(np.ones(coords.shape[:-1], dtype=bool))
        else:
            where = np.where(branches_inds[self.branch_name])
        if len(where[0]) == 0:
            return q, factors

        old = coords[where]
        new = old.copy()
        m = self.flow_params

        # which half each row moves: a draw independent of the state (a random-scan choice
        # between two valid kernels), or both halves at once with flow_prob=None. A half that
        # cannot propose yet (unfitted flow, no covariance) hands every row to the other half.
        proposal = self.gaussian_proposal(self.active_condition)
        flow_ready = self.flow.data_transform.is_fitted
        if self.flow_prob is None or not flow_ready or proposal is None:
            use_flow = np.full(len(old), flow_ready)
            use_gauss = np.full(len(old), proposal is not None)
        else:
            use_flow = random.rand(len(old)) < self.flow_prob
            use_gauss = ~use_flow

        # Gaussian columns: symmetric step, no factor
        if use_gauss.any():
            new[np.ix_(use_gauss, ~m)], _ = proposal(old[np.ix_(use_gauss, ~m)], random)

        # flow columns: independent draw, +log q(old) - log q(new) on those rows only
        if use_flow.any():
            rows = tuple(w[use_flow] for w in where)
            new[np.ix_(use_flow, m)] = self._propose_from_flow(old[np.ix_(use_flow, m)], rows, factors)

        if self.periodic is not None:
            # wrap the Gaussian columns only: the flow values are kept as drawn, since
            # their log q is already in the factor (a value outside the prior is rejected)
            wrapped = self.periodic.wrap({self.branch_name: new.copy()}, xp=self.xp)[self.branch_name]
            new[:, ~m] = wrapped[:, ~m]

        q[self.branch_name][where] = new
        return q, factors
