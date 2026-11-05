# -*- coding: utf-8 -*-
"""NUTS proposal driven by a normalizing-flow surrogate.

This module is torch-free at import time (same discipline as
``eryn.moves.flow``): the torch-backed pieces are reached only through the
flow object's own methods and a lazy import inside the metric builder.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np

from .flow import ConditionalFlowMove
from .nuts import NUTSSampler

__all__ = ["FlowNUTSMove"]


def _unset_callable(x):
    """Placeholder installed on the internal NUTSSampler between proposals.

    Keeps the move deep-copyable (a module-level function is atomic under
    ``copy.deepcopy``; a per-propose closure holding the flow would not be) and
    fails loudly if the sampler is ever stepped outside ``get_proposal``.
    """
    raise RuntimeError(
        "FlowNUTSMove internal error: NUTS callables are only installed for "
        "the duration of get_proposal."
    )


class _FusedEvalCache:
    """Memoizing wrapper around ``flow.log_prob_and_grad`` for one proposal.

    ``NUTSSampler`` evaluates, per leapfrog step, ``grad(x)``, ``grad(x_new)``
    and then ``log_posterior(x_new)`` — and ``grad(x)`` repeats the previous
    step's ``grad(x_new)`` whenever the trajectory extends linearly.  One fused
    forward+backward pass returns both quantities, so memoizing ``(logq, grad)``
    pairs cuts the flow evaluations per leapfrog from 3 to ~2 without touching
    ``NUTSSampler``.  Keyed by the raw bytes of the (contiguous) input array;
    a fresh cache is built for every proposal, so hot-reloaded weights or a
    changed condition can never be served stale values.
    """

    def __init__(self, flow, condition: int, maxsize: int = 16):
        self.flow = flow
        self.condition = int(condition)
        self.maxsize = int(maxsize)
        self._store: OrderedDict = OrderedDict()

    def _fetch(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        key = (x.shape, x.tobytes())
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        val = self.flow.log_prob_and_grad(x, context=self.condition)
        self._store[key] = val
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return val

    def logq(self, x) -> np.ndarray:
        return self._fetch(x)[0]

    def grad(self, x) -> np.ndarray:
        return self._fetch(x)[1]


class FlowNUTSMove(ConditionalFlowMove):
    r"""NUTS proposal whose Hamiltonian dynamics run on the flow surrogate.

    Runs a No-U-Turn trajectory (:class:`eryn.moves.NUTSSampler`) on the
    **flow's** log density :math:`\log \tilde q(x)` — cheap, differentiable —
    and hands the endpoint to Eryn's Metropolis-Hastings acceptance against the
    **true** posterior.  The flow supplies both the gradient (via
    :meth:`eryn.flows.Flow.log_prob_and_grad`) and the correction factors.

    **Why this is exact.**  The NUTS transition kernel :math:`K` is reversible
    with respect to its target, here :math:`\tilde q`:
    :math:`\tilde q(x) K(x \to y) = \tilde q(y) K(y \to x)`.  Returning
    Hastings factors ``log q̃(x_old) − log q̃(x_new)`` (the exact
    :class:`~eryn.moves.ConditionalFlowMove` convention) therefore makes
    :meth:`~eryn.moves.MHMove.propose`'s acceptance

    .. math::

        \min\!\left(1,\;
        \frac{\pi_\beta(y)\,\tilde q(x)}{\pi_\beta(x)\,\tilde q(y)}\right),

    which is valid MH for the (tempered) true posterior at every temperature.
    A poor or stale flow only lowers acceptance — it never biases the chain,
    the same guarantee as :class:`~eryn.moves.ConditionalFlowMove`.

    **Why this inherits ConditionalFlowMove and not NUTSMove.**
    :meth:`NUTSMove.propose <eryn.moves.NUTSMove>` bypasses the MH acceptance
    (a walker counts as accepted whenever NUTS moved it), which is correct only
    when NUTS targets the true posterior.  With the dynamics running on the
    flow, skipping the correction would make the chain sample the *flow*, not
    the posterior.  Inheriting :class:`~eryn.moves.ConditionalFlowMove` instead keeps the
    entire online-training integration — executor harvest / poll / hot-reload
    in :meth:`~eryn.moves.ConditionalFlowMove.setup`, :attr:`active_condition`,
    :meth:`~eryn.moves.ConditionalFlowMove.submit_by_leaf`, identity pass-through while
    the data transform is unfitted — and ``isinstance(move, ConditionalFlowMove)`` stays
    true for downstream schedulers.  The NUTS integrator is composed
    (an internal :class:`~eryn.moves.NUTSSampler`), not inherited.

    **Mass matrix from the whitening transform** (``metric="whitening"``,
    default).  Raw coordinates are badly scaled, so identity-mass NUTS would
    crawl.  When the flow's data transform is a
    :class:`~eryn.flows.WhiteningTransform` (forward
    :math:`z = (\mathrm{wrap}(x) - \mu)\,M`), running x-space leapfrog with
    mass matrix :math:`\Sigma = M M^\top` is *exactly equivalent* to
    identity-mass leapfrog in the whitened latent space (for the non-periodic
    block :math:`\Sigma = \mathrm{cov}^{-1}` — the classical prescription), so
    ``step_size`` is expressed in whitened units and O(0.1–0.5) is sane.  The
    matrix is rebuilt automatically whenever the fitted transform object
    changes — an executor hot-reload installs a new transform with each
    snapshot, and switching :attr:`active_condition` selects a different
    per-condition map.  Pass ``metric=None`` for an identity mass matrix or an
    ``(ndim, ndim)`` array / callable to override (forwarded to
    :class:`~eryn.moves.NUTSSampler` untouched).

    **Periodic dimensions.**  The coords-space flow density is exactly
    periodic in the dims declared by the transform's ``periodic`` mapping and
    the dynamics is shift-equivariant there, so after the trajectory the
    proposal is wrapped back into ``[lo, hi)`` (``wrap_periodic=True``).  This
    preserves reversibility and is required in practice: an unwrapped angle
    would fall outside the bounded prior and be rejected every time.  The
    finite density jump at the periodic cut (the flow models the recentred
    angle as unbounded) only costs efficiency — trajectories crossing the cut
    are cut short by the slice/divergence checks — never correctness.

    **Adaptation and hot-reloads.**  Dual-averaging state persists across
    proposals and is deliberately *not* reset on weight hot-reloads: NUTS never
    runs before the transform is fitted, and reloads only perturb the whitened
    geometry mildly.  Call :meth:`reset_adaptation` to restart it manually.
    ``adapt_step_size=True`` is a diminishing-adaptation approximation (the
    step size freezes after ``n_adapt`` proposals); prefer enabling it during
    burn-in only.

    **Cost.**  Each proposal spends roughly ``2 x n_leapfrog`` fused
    forward+backward flow passes on the ``(n_active, ndim)`` batch
    (``max_tree_depth=6`` caps ``n_leapfrog`` at 64; a flow that resembles the
    posterior U-turns at depth 2–4).  This is substantially heavier than
    :class:`~eryn.moves.ConditionalFlowMove`'s single pass — the price of
    gradient-guided, position-local proposals.

    A per-temperature tempered-surrogate mode (dynamics on
    :math:`\beta \log \tilde q`) is deliberately deferred: the untempered
    kernel is valid at all temperatures, hot chains simply accept less.

    Parameters
    ----------
    flow : eryn.flows.Flow
        Conditional normalizing flow.  Must override
        :meth:`~eryn.flows.Flow.log_prob_and_grad` (the torch backend
        :class:`~eryn.flows.ZukoFlow` does); a flow without it raises
        :exc:`TypeError` at construction.
    branch_name : str
        Name of the branch this move proposes for.
    executor, harvest_every, harvest_temp_index
        Online-training plumbing, passed to :class:`~eryn.moves.ConditionalFlowMove`.
    metric : str, None, np.ndarray, or callable, optional
        ``"whitening"`` (default) derives the mass matrix from the flow's
        fitted :class:`~eryn.flows.WhiteningTransform` as described above;
        anything else is forwarded to :class:`~eryn.moves.NUTSSampler`.
    step_size : float, optional
        Leapfrog step size, in whitened units under the default metric.
        (default: ``0.3``)
    max_tree_depth : int, optional
        Maximum tree-doubling depth (at most ``2**depth`` leapfrogs).
        (default: ``6``)
    max_delta_energy : float, optional
        Divergence threshold.  (default: ``1000.0``)
    adapt_step_size, target_accept, n_adapt
        Dual-averaging step-size adaptation, see
        :class:`~eryn.moves.NUTSSampler`.  (defaults: ``False``, ``0.8``,
        ``100``)
    wrap_periodic : bool, optional
        Wrap periodic dims of the proposal back into their declared bounds.
        (default: ``True``)
    *args, **kwargs
        Passed through to :class:`~eryn.moves.ConditionalFlowMove` /
        :class:`~eryn.moves.MHMove`.

    Examples
    --------
    >>> move = FlowNUTSMove(my_flow, branch_name="x", step_size=0.3)
    >>> # online training, exactly as with ConditionalFlowMove:
    >>> move = FlowNUTSMove(my_flow, "x", executor=ex, harvest_every=10)
    """

    def __init__(
        self,
        flow,
        branch_name: str,
        executor=None,
        harvest_every: int = 1,
        harvest_temp_index: int = 0,
        metric="whitening",
        step_size: float = 0.3,
        max_tree_depth: int = 6,
        max_delta_energy: float = 1000.0,
        adapt_step_size: bool = False,
        target_accept: float = 0.8,
        n_adapt: int = 100,
        wrap_periodic: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            flow, branch_name, executor, harvest_every, harvest_temp_index,
            *args, **kwargs,
        )

        # Lazy import mirrors flow.py: eryn.flows.base is numpy-only today, but
        # keeping it out of module scope guarantees `import eryn.moves` stays
        # backend-free even if that ever changes.
        from eryn.flows.base import Flow

        if type(flow).log_prob_and_grad is Flow.log_prob_and_grad:
            raise TypeError(
                f"FlowNUTSMove requires a flow implementing log_prob_and_grad; "
                f"{type(flow).__name__} does not override the Flow ABC default."
            )

        self.metric_spec = metric
        self.wrap_periodic = bool(wrap_periodic)

        is_whitening = isinstance(metric, str) and metric == "whitening"
        self._nuts = NUTSSampler(
            grad_log_posterior_fn=_unset_callable,
            log_posterior_fn=_unset_callable,
            ndim=flow.dims,
            metric=None if is_whitening else metric,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            max_delta_energy=max_delta_energy,
            adapt_step_size=adapt_step_size,
            target_accept=target_accept,
            n_adapt=n_adapt,
        )
        # NUTSSampler defaults its RNG to the np.random *module*, which cannot
        # be deep-copied — and the move must survive deepcopy (Eryn settings
        # pipelines copy their inner moves).  The RNG is installed per proposal
        # from get_proposal's `random` argument, so hold None in between.
        self._nuts.random = None
        # Identity of the ComposeTransform the current mass matrix was built
        # from; a hot-reload or condition switch yields a different object and
        # triggers a rebuild.
        self._metric_source = None
        self._warned_no_whitening = False

    # ------------------------------------------------------------------
    # NUTS pass-throughs
    # ------------------------------------------------------------------

    @property
    def step_size(self) -> float:
        """Current leapfrog step size of the internal NUTS sampler."""
        return self._nuts.step_size

    @step_size.setter
    def step_size(self, value: float) -> None:
        self._nuts.step_size = float(value)

    @property
    def last_alpha(self):
        """Mean Metropolis-style acceptance of the last NUTS trajectory."""
        return self._nuts.last_alpha

    @property
    def last_tree_depth(self):
        """Tree depth reached on the last NUTS trajectory."""
        return self._nuts.last_tree_depth

    def reset_adaptation(self) -> None:
        """Restart dual-averaging step-size adaptation from the current step size."""
        nuts = self._nuts
        nuts._mu = np.log(10.0 * nuts.step_size)
        nuts._h_bar = 0.0
        nuts._log_eps_bar = 0.0
        nuts._adapt_count = 0

    # ------------------------------------------------------------------
    # Whitening-derived mass matrix
    # ------------------------------------------------------------------

    def _maybe_refresh_metric(self) -> None:
        """Rebuild the NUTS mass matrix from the fitted whitening transform.

        Only active for ``metric="whitening"``.  With forward
        ``z = (wrap(x) - mu) @ M``, x-space leapfrog with mass ``M @ M.T`` is
        equivalent to identity-mass leapfrog in the whitened latent space.
        Rebuilds are keyed on the identity of the per-condition
        ``ComposeTransform`` object, which changes exactly when a new fit is
        installed (executor hot-reload, direct ``flow.fit``) or when
        :attr:`active_condition` selects a different per-condition map.
        """
        if not (isinstance(self.metric_spec, str) and self.metric_spec == "whitening"):
            return

        transform_for = getattr(self.flow.data_transform, "_transform_for", None)
        if transform_for is None:
            if not self._warned_no_whitening:
                warnings.warn(
                    f"FlowNUTSMove(metric='whitening'): data transform "
                    f"{type(self.flow.data_transform).__name__} exposes no "
                    "per-condition transform; falling back to an identity "
                    "mass matrix.",
                    RuntimeWarning,
                )
                self._warned_no_whitening = True
            return

        ct = transform_for(self.active_condition)
        if ct is self._metric_source:
            return

        # Torch-backed transform guaranteed here; keep torch out of module scope.
        from eryn.flows.torch.transforms import LinearMatrixTransform

        matrix = None
        for part in getattr(ct, "parts", []):
            if isinstance(part, LinearMatrixTransform):
                matrix = part.matrix.detach().cpu().numpy().astype(np.float64)
                break

        if matrix is None:
            if not self._warned_no_whitening:
                warnings.warn(
                    "FlowNUTSMove(metric='whitening'): no LinearMatrixTransform "
                    "found in the fitted data transform; falling back to an "
                    "identity mass matrix.",
                    RuntimeWarning,
                )
                self._warned_no_whitening = True
            self._nuts._set_metric(None)
        else:
            self._nuts._set_metric(matrix @ matrix.T)
        self._metric_source = ct

    # ------------------------------------------------------------------
    # Periodic wrap
    # ------------------------------------------------------------------

    def _wrap_periodic(self, x: np.ndarray) -> np.ndarray:
        """Fold periodic dims of proposals back into their declared bounds (in place)."""
        if not self.wrap_periodic:
            return x
        periodic = getattr(self.flow.data_transform, "periodic", None) or {}
        for i, (lo, hi) in periodic.items():
            x[:, i] = (x[:, i] - lo) % (hi - lo) + lo
        return x

    # ------------------------------------------------------------------
    # MHMove interface
    # ------------------------------------------------------------------

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Propose via a NUTS trajectory on the flow surrogate.

        Same contract as :meth:`ConditionalFlowMove.get_proposal`: proposes only for
        :attr:`branch_name`, identity pass-through while the data transform is
        unfitted, factors ``+log q̃(old) − log q̃(new)`` accumulated with
        :func:`numpy.add.at` over active leaves.  Each active leaf-row is an
        independent NUTS walker; multiple leaves of one walker form a product
        kernel, which stays reversible w.r.t. the product of surrogate
        densities.  Walkers whose trajectory did not move return zero factors
        and unchanged coords (an identity proposal that always "accepts").
        """
        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        # Same startup contract as ConditionalFlowMove: until the trainer has fitted the
        # data transform the flow cannot be evaluated, so the move is a
        # harmless identity until the first snapshot is hot-reloaded.
        if not self.flow.data_transform.is_fitted:
            q = {name: coords.copy() for name, coords in branches_coords.items()}
            first = next(iter(branches_coords.values()))
            return q, np.zeros(first.shape[:2])

        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords.items()
            }

        first = next(iter(branches_coords.values()))
        factors = np.zeros(first.shape[:2])
        q = {}

        for name, coords in branches_coords.items():
            q[name] = coords.copy()

            if name != self.branch_name:
                continue

            where = np.where(branches_inds[name])
            old_points = coords[where]
            num = len(where[0])
            if num == 0:
                continue
            if coords.shape[-1] != self.flow.dims:
                raise ValueError(
                    f"{type(self).__name__}: branch {name!r} has ndim="
                    f"{coords.shape[-1]} but the flow has dims={self.flow.dims}."
                )

            self._maybe_refresh_metric()

            # Fresh cache per proposal: hot-reloaded weights or a changed
            # active_condition can never be served stale values.
            cache = _FusedEvalCache(self.flow, self.active_condition)
            self._nuts.grad_log_posterior_fn = cache.grad
            self._nuts.log_posterior_fn = cache.logq
            self._nuts.random = random
            try:
                x_new, _ = self._nuts.step(
                    np.ascontiguousarray(old_points, dtype=np.float64)
                )
            finally:
                # Detach per-proposal state so the move stays deep-copyable
                # (no closure over the flow, and no np.random module, survives
                # between proposals).
                self._nuts.grad_log_posterior_fn = _unset_callable
                self._nuts.log_posterior_fn = _unset_callable
                self._nuts.random = None

            x_new = self._wrap_periodic(x_new)

            # Surrogate-kernel MH correction (see class docstring).  logq at
            # the old points is a guaranteed cache hit (evaluated for H0).
            logq_old = cache.logq(old_points)
            logq_new = self.flow.log_prob(x_new, context=self.active_condition)
            np.add.at(factors, where[:2], logq_old - logq_new)

            q[name][where] = x_new

        return q, factors
