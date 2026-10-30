# -*- coding: utf-8 -*-
"""Independent-Metropolis move backed by an eryn.flows Flow.

This module is intentionally torch-free at import time: ``eryn.flows.base``
(where ``FlowProposalDistribution`` lives) is pure NumPy, but
``eryn.flows.torch`` (where the concrete backends live) may import torch.
``FlowProposalDistribution`` is therefore imported lazily inside
``get_proposal`` rather than at module level, ensuring that
``import eryn.moves`` never triggers a torch import.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .mh import MHMove

if TYPE_CHECKING:
    # Only for static type-checkers; never executed at runtime.
    from eryn.flows.base import Flow, FlowProposalDistribution  # noqa: F401

__all__ = ["ConditionalFlowMove"]
logger = logging.getLogger(__name__)

class ConditionalFlowMove(MHMove):
    """Independent-Metropolis proposal drawing from a conditional normalizing flow.

    Proposes new coordinates for a single named branch by sampling from a
    :class:`eryn.flows.Flow` conditioned on :attr:`active_condition`.  The
    Hastings factor convention matches :class:`eryn.moves.DistributionGenerate`::

        factors += +log q(x_old) - log q(x_new)

    A poor or stale flow only lowers acceptance — it never biases the posterior.
    This no-bias guarantee relies on IEEE-754: NaN logq values propagate into
    lnpdiff and NaN comparisons are False, so any proposal with a NaN log-prob
    is always rejected rather than silently accepted.
    Accumulation uses :func:`numpy.add.at` (not ``+=``) so that duplicate
    ``(temp, walker)`` indices from multi-leaf proposals sum instead of being
    silently overwritten by the last write.

    :meth:`Flow.sample_and_log_prob` is used for new points to avoid a second
    forward pass through the flow: the log-probability returned by sampling is
    reused directly as ``-log q(new)``.

    **Online training (optional)**

    When an ``executor`` (a :class:`eryn.flows.executors.TrainerExecutor`) is
    supplied, :meth:`setup` — called at the top of every :meth:`propose` — gains
    two non-blocking online-training hooks:

    1. **Harvest.** Every ``harvest_every``-th call, the cold-chain coordinates
       of this move's branch are flattened and submitted to the executor as
       training data.  The submit is non-blocking by contract; the executor
       trains a *clone* of the flow off the proposal's critical path.
    2. **Poll + hot-reload.** On *every* call, the newest trained weights are
       polled; if a strictly newer version is available it is loaded into this
       move's flow via ``flow.set_weights``.  The proposal therefore tracks an
       improving flow without ever blocking.

    Failures propagate loudly: a dead/failed trainer surfaces as
    :class:`~eryn.flows.executors.TrainerError` out of :meth:`setup` and is NOT
    swallowed.  A multi-day run silently sampling from a frozen flow is worse
    than a crash; graceful degradation is a future scheduler policy, not the
    move's job.

    **Temperature-scaled proposals (optional, via** :attr:`active_betas` **)**

    The flow is trained on cold-chain (``beta=1``) samples only, but this move
    proposes for walker rows at every temperature.  A hot row's target is
    ``propto L(theta)**beta * pi(theta)`` — roughly the cold posterior broadened
    by ``1/sqrt(beta)`` — so without correction, hot rows are proposed from deep
    in the cold flow's tails and are almost always rejected.  Setting
    :attr:`active_betas` (typically done by the outer move on every repeat for
    the current leaf) fixes this: :meth:`get_proposal` groups rows by temperature
    and re-proposes each group from the SAME trained flow but with its base
    distribution scaled by ``s = beta_t**-0.5`` using the ``base_scale`` keyword
    argument — an exact affine reparameterisation of the base, so the Hastings
    bookkeeping stays exact.  This requires a flow whose ``sample_and_log_prob``
    and ``log_prob`` methods accept the optional ``base_scale`` keyword (e.g.
    :class:`eryn.flows.ZukoFlow`); flows without this keyword will raise
    :exc:`TypeError`.  ``active_betas=None`` (the default) reproduces the
    original unscaled behaviour exactly.

    Parameters
    ----------
    flow : eryn.flows.Flow
        Fitted (or freshly initialised) conditional normalizing flow.  Must
        implement :meth:`~eryn.flows.Flow.log_prob`,
        :meth:`~eryn.flows.Flow.sample_and_log_prob`, and the rest of the
        :class:`eryn.flows.Flow` ABC.
    branch_name : str
        Name of the branch this move proposes for.  All other branches in
        ``branches_coords`` are copied unchanged.
    executor : eryn.flows.executors.TrainerExecutor or None, optional
        Online-training executor.  When ``None`` (the default), :meth:`setup`
        is a pure no-op and the move behaves exactly as a static flow proposal
        — zero overhead, no executor interaction.
    harvest_every : int, optional
        Submit cold-chain coordinates to the executor on every
        ``harvest_every``-th :meth:`setup` call.  Default is ``1`` (every call). If ``None``, harvesting is disabled. 

        Polling for new weights happens on every call regardless of this value.
    harvest_temp_index : int, optional
        Temperature index to harvest from.  Default is ``0`` (the cold chain).
        Harvesting flattens the ``nwalkers`` x ``nleaves`` coordinates of this
        temperature into ``(-1, ndim)`` training rows.
    *args, **kwargs
        Passed through to :class:`eryn.moves.MHMove`.

    Attributes
    ----------
    active_condition : int
        Condition id passed to the flow on every ``get_proposal`` call.
        Readable and writable; the setter coerces the value to ``int``.
    active_betas : np.ndarray or None
        Per-temperature inverse-temperature ladder, shape ``(ntemps,)``,
        matching the leading axis of the branch coords seen by
        :meth:`get_proposal`.  ``None`` (the default) reproduces today's
        behaviour exactly: every row is proposed from the flow's native
        (unscaled) base.  When set (typically by the outer move, re-assigned on
        every repeat for the current leaf, to that leaf's current temperature
        ladder), :meth:`get_proposal` groups rows by temperature ``t`` and
        proposes each group with ``base_scale = active_betas[t] ** -0.5`` via
        the ``base_scale`` keyword argument — broadening the flow's base for hot
        chains so they are no longer proposed from deep in the cold flow's tails
        (see the module/class docstring for the statistical motivation).  This
        requires a flow whose ``sample_and_log_prob`` and ``log_prob`` methods
        accept the optional ``base_scale`` keyword; flows without it will raise
        :exc:`TypeError`.  Plain attribute, not a property: set it directly.
    loaded_version : int
        Version of the most recently hot-loaded weights (``0`` if none have been
        loaded).  Read-only.

    Examples
    --------
    >>> move = ConditionalFlowMove(my_flow, branch_name="x")
    >>> move.active_condition = 2   # switch to condition 2 before sampling

    >>> # With online training:
    >>> from eryn.flows import InlineExecutor
    >>> ex = InlineExecutor(my_flow, min_train_samples=1000)
    >>> move = ConditionalFlowMove(my_flow, branch_name="x", executor=ex, harvest_every=10)
    """

    def __init__(
        self,
        flow,
        branch_name: str,
        executor=None,
        harvest_every: int = 1,
        harvest_temp_index: int = 0,
        *args,
        **kwargs,
    ):
        self.flow = flow
        self.branch_name = branch_name
        self._active_condition: int = 0
        # Per-temperature beta ladder for this move's branch, shape (ntemps,).
        # None (default): get_proposal is unscaled, exactly as before this
        # attribute existed.  Set externally, per leaf, by the outer move
        # (e.g. ResidualAddOneRemoveOneMove) — see the class docstring.
        self.active_betas = None
        self.harvest_every = int(harvest_every) if harvest_every is not None else None
        self.harvest_temp_index = int(harvest_temp_index)
        self._setup_calls = 0
        self._loaded_version = 0
        # Initialise through the property so the executor-swap bookkeeping
        # (version + harvest-counter reset) is defined in exactly one place.
        self._executor = None
        self.executor = executor
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # active_condition property
    # ------------------------------------------------------------------

    @property
    def active_condition(self) -> int:
        """Condition id used when calling the flow."""
        return self._active_condition

    @active_condition.setter
    def active_condition(self, c: int) -> None:
        self._active_condition = int(c)

    @property
    def loaded_version(self) -> int:
        """Version of the most recently hot-loaded flow weights (``0`` = none)."""
        return self._loaded_version

    # ------------------------------------------------------------------
    # executor property — resets hot-reload bookkeeping on a real swap
    # ------------------------------------------------------------------

    @property
    def executor(self):
        """Online-training executor (or ``None``).

        Assigning a **new** executor object resets this move's hot-reload
        memory (:attr:`loaded_version` back to ``0`` and the internal harvest
        counter back to ``0``).  This is essential because
        :attr:`~eryn.flows.executors.TrainerExecutor.version` is *per-executor-
        instance*: a replacement executor (e.g. a recreated
        :class:`~eryn.flows.executors.ProcessExecutor` after a worker died)
        restarts its version counter at ``1``.  Without the reset, :meth:`setup`
        would compare the new executor's version ``1`` against a remembered
        ``loaded_version`` of, say, ``7`` and silently ignore every hot-reload
        for the rest of the run.  Assigning the *same* object that is already
        installed is a no-op (identity comparison) and preserves the counters.
        """
        return self._executor

    @executor.setter
    def executor(self, new_executor) -> None:
        # Identity comparison: re-assigning the same object must NOT reset the
        # counters (that would discard legitimate hot-reload progress); only a
        # genuinely different executor instance restarts the bookkeeping.
        if new_executor is self._executor:
            return
        self._executor = new_executor
        self._loaded_version = 0
        self._setup_calls = 0

    # ------------------------------------------------------------------
    # Online-training hook (called at the top of every propose)
    # ------------------------------------------------------------------

    def setup(self, branches_coords):
        """Harvest cold-chain samples and hot-reload trained weights.

        Called by :meth:`eryn.moves.mh.MHMove.propose` at the top of every
        proposal.  With no executor configured this is a pure no-op.  With an
        executor it performs two non-blocking steps:

        - **Harvest** (every ``harvest_every``-th call): the coordinates of this
          move's branch at temperature ``harvest_temp_index`` (the cold chain by
          default) are reshaped to ``(-1, ndim)`` — flattening walkers x leaves —
          and submitted to the executor under :attr:`active_condition`.
        - **Poll + hot-reload** (every call): the newest trained weights are
          polled; a strictly newer version is loaded into :attr:`flow`.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` current
            coordinates.

        Raises
        ------
        KeyError
            If this move's ``branch_name`` is not present in ``branches_coords``
            (same loud guard as :meth:`get_proposal`).
        eryn.flows.executors.TrainerError
            If the executor's trainer has failed.  Propagated, never swallowed:
            silently sampling from a frozen flow for a long run is worse than a
            crash.
        """
        if self.executor is None:
            return

        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        self._setup_calls += 1

        # --- harvest (non-blocking submit) ---
        if self.harvest_every is not None and self._setup_calls % self.harvest_every == 0:
            self.submit(branches_coords)

        # --- poll + hot-reload (non-blocking; TrainerError propagates) ---
        lw = self.executor.latest_weights()
        if lw is not None:
            version, snapshot = lw
            if version > self._loaded_version:
                # ``snapshot`` is a self-contained {"net", "data_transform"}
                # payload: set_weights installs the matched transform + net
                # atomically, so this flow never disagrees with the trainer on
                # the coords-latent map.
                self.flow.set_weights(snapshot)
                logger.debug(
                    f"{type(self).__name__}: hot-loaded new flow weights version"
                    f" {version} (previously {self._loaded_version})."
                )
                self._loaded_version = version
            else:
                logger.debug(
                    f"{type(self).__name__}: no new flow weights (latest version"
                    f" {version}, previously loaded {self._loaded_version})."
                )
        else:
            logger.debug(
                f"{type(self).__name__}: executor has no trained weights yet"
                f" (previously loaded {self._loaded_version})."
            )

    def submit(self, branches_coords: dict):
        """
        Submit the cold-chain coordinates of this move's branch to the executor for training.

        Args:
            branches_coords (dict): Keys are branch names; values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` current
                coordinates, or ``np.ndarray[nsteps, ntemps, nwalkers, nleaves_max, ndim]`` if the move is called from a multi-step context.
        """
        coords = branches_coords[self.branch_name]
        
        batch = {
            leaf: coords[..., self.harvest_temp_index, :, leaf, :].reshape(-1, coords.shape[-1])
            for leaf in range(coords.shape[-2])
        }

        self.submit_by_leaf(batch)

    def submit_by_leaf(self, batch: dict):
        """
        Submit a batch of coordinates to the executor for training.

        Args:
            batch (dict): Keys are leaf indices; values are
                ``np.ndarray[n_samples, ndim]`` coordinates for that leaf.
        """
        if self.executor is None:
            return
        
        self.executor.submit(batch)

    # ------------------------------------------------------------------
    # MHMove interface
    # ------------------------------------------------------------------

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate independent-Metropolis proposals from the flow.

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

        Notes
        -----
        When :attr:`active_betas` is ``None`` (the default), every active row
        of ``branch_name`` is proposed from the flow's native base in a single
        ``sample_and_log_prob`` / ``logpdf`` pair — exactly as before this
        attribute existed.  When :attr:`active_betas` is set, rows are grouped
        by temperature (the leading axis of ``branches_coords[branch_name]``)
        and each temperature ``t`` is proposed with
        ``base_scale = active_betas[t] ** -0.5``, using its own
        ``sample_and_log_prob`` / ``log_prob`` pair (both draws and both logq
        terms of that row's Hastings factor use the same scale).
        """
        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        # Until the flow's data transform is fitted, the flow cannot be evaluated
        # (a WhiteningTransform raises before fit).  This is the normal startup
        # state when the move is mixed in from step 0 and the transform is fitted
        # lazily by the trainer: propose an identity move (old coords, zero
        # factors) so the move is a harmless pass-through until the first trained
        # snapshot is installed (via setup()'s hot-reload).  IdentityTransform is
        # always fitted, so a flow without whitening is never blocked here.
        if not self.flow.data_transform.is_fitted:
            q = {name: coords.copy() for name, coords in branches_coords.items()}
            first = next(iter(branches_coords.values()))
            return q, np.zeros(first.shape[:2])

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
                continue  # ConditionalFlowMove only proposes for its own branch

            where = np.where(branches_inds[name])
            if len(where[0]) == 0:
                continue

            q[name][where] = self._propose_from_flow(coords[where], where, factors)

        return q, factors

    def _propose_from_flow(self, old_points, where, factors):
        """Draw replacements for ``old_points`` from the flow and add their Hastings terms.

        Args:
            old_points (np.ndarray): ``(num, flow.dims)`` current points of the active rows.
            where (tuple): ``np.where`` of the active rows; ``where[:2]`` are the
                ``(temp, walker)`` indices the factors are accumulated on.
            factors (np.ndarray): ``(ntemps, nwalkers)`` log-factors, updated in place with
                ``+log q(old) - log q(new)``.

        Returns:
            np.ndarray: ``(num, flow.dims)`` new points, in the order of ``old_points``.

        """
        # Lazy import: FlowProposalDistribution is pure NumPy (no torch), but
        # importing it here guarantees that even if eryn.flows.base were ever
        # changed to pull in a backend, this module stays torch-free at the
        # top level.
        from eryn.flows.base import FlowProposalDistribution

        num = len(old_points)
        if self.active_betas is None:
            dist = FlowProposalDistribution(self.flow, condition=self.active_condition)

            # + log q(old): np.add.at so multi-leaf duplicate (temp, walker) indices
            # accumulate instead of last-write-wins on repeated fancy-index +=.
            # NaN logq values (e.g. from out-of-support old points) propagate into
            # lnpdiff.  NaN comparisons are always False, so such proposals are
            # unconditionally rejected — the "no-bias" claim in the class docstring
            # relies on this IEEE-754 property rather than on an explicit NaN guard.
            np.add.at(factors, where[:2], dist.logpdf(old_points))

            # Draw and - log q(new): sample_and_log_prob avoids a second forward
            # pass — the log-prob returned during sampling is reused directly.
            # Same NaN-rejection applies if the flow returns NaN for new points.
            new_points, logq_new = self.flow.sample_and_log_prob(
                num, context=self.active_condition
            )
            np.add.at(factors, where[:2], -logq_new)
        else:
            # Temperature-scaled proposals: group rows by temperature (the
            # first axis of the branch coords, per the class docstring) and, per
            # temperature t, propose from the SAME flow with
            # base_scale = active_betas[t]**-0.5 — one sample call + one
            # log_prob call per temperature (ntemps is small, <= a few).
            # `mask` sub-selects `where`/`old_points` while preserving row
            # order, so the returned points line up with `old_points` exactly
            # as in the active_betas=None path above.
            temp_of_row = where[0]
            new_points = np.empty_like(old_points)
            for t in range(factors.shape[0]):
                mask = temp_of_row == t
                num_t = int(mask.sum())
                if num_t == 0:
                    continue
                s = float(self.active_betas[t]) ** -0.5
                idx_t = tuple(w[mask] for w in where[:2])

                logq_old_t = self.flow.log_prob(
                    old_points[mask], context=self.active_condition, base_scale=s
                )
                np.add.at(factors, idx_t, logq_old_t)

                new_points_t, logq_new_t = self.flow.sample_and_log_prob(
                    num_t, context=self.active_condition, base_scale=s
                )
                np.add.at(factors, idx_t, -logq_new_t)
                new_points[mask] = new_points_t

        return new_points
