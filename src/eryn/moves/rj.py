# -*- coding: utf-8 -*-
"""Reversible-jump (trans-dimensional) proposal base class (:class:`ReversibleJumpMove`).

A reversible-jump move changes which leaves of a branch are alive (the ``inds`` mask) and
proposes coordinates for born leaves. :class:`ReversibleJumpMove` does the bookkeeping
around a subclass's ``get_proposal``: Gibbs splits over branches, the edge factors at
``nleaves_min`` / ``nleaves_max``, scoring, accept/reject and rung swaps without ladder
adaptation. The stock concrete move is :class:`~eryn.moves.DistributionGenerateRJ`.
"""

from copy import deepcopy
from multiprocessing.sharedctypes import Value

import numpy as np

from ..state import State
from .delayedrejection import DelayedRejection
from .distgen import DistributionGenerate
from .move import Move

__all__ = ["ReversibleJumpMove"]


class ReversibleJumpMove(Move):
    """Abstract reversible-jump move.

    Subclasses implement ``get_proposal`` (and usually ``get_model_change_proposal``).
    :meth:`propose` adds the edge factors that correct the birth/death choice asymmetry at
    the leaf-count limits and uses the move's (tempered) log posterior for acceptance.
    Gibbs splits may only name branches (``is_rj=True`` is passed to
    :class:`~eryn.moves.Move`).

    Args:
        nleaves_max (dict): ``{branch_name: int}`` maximum leaf count per branch. Keyword
            argument, but required.
        nleaves_min (dict): ``{branch_name: int}`` minimum leaf count per branch. Keyword
            argument, but required.
        dr (bool or Move, optional): Delayed rejection for rejected births. ``True`` wraps a
            :class:`~eryn.moves.DistributionGenerate` from ``self.generate_dist`` (which
            must already be set, as :class:`~eryn.moves.DistributionGenerateRJ` does);
            a move instance is wrapped as given; ``None`` / ``False`` disables it. The
            propose-time delayed-rejection branch is not implemented and raises
            ``NotImplementedError``. Default ``None``.
        dr_max_iter (int, optional): Maximum delayed-rejection iterations. Default ``5``.
        tune (bool, optional): Stored as ``self.tune``. Note that this instance attribute
            shadows the :meth:`~eryn.moves.Move.tune` method. Default ``False``.
        fix_change (int or None, optional): Force every walker to add (``+1``) or remove
            (``-1``) a leaf (still clamped at the limits); useful for some search
            functions. ``None`` picks the direction at random. Default ``None``.
        **kwargs: Forwarded to :class:`~eryn.moves.Move` (``temperature_control``,
            ``gibbs_sampling_setup``, ``prevent_swaps``, ...).

    Raises:
        ValueError: ``nleaves_min`` / ``nleaves_max`` missing or not dicts, or
            ``fix_change`` not in ``(None, +1, -1)``.

    Attributes:
        nleaves_max (dict): See ``nleaves_max``.
        nleaves_min (dict): See ``nleaves_min``.
        fix_change (int or None): See ``fix_change``.
        dr (:class:`~eryn.moves.delayedrejection.DelayedRejection`, None or False):
            Delayed-rejection wrapper when enabled, else the ``dr`` value given.
    """

    def __init__(
        self,
        nleaves_max=None,
        nleaves_min=None,
        dr=None,
        dr_max_iter=5,
        tune=False,
        fix_change=None,
        **kwargs
    ):
        # super().__init__(**kwargs)
        Move.__init__(self, is_rj=True, **kwargs)

        if nleaves_max is None or nleaves_min is None:
            raise ValueError(
                "Must provide nleaves_min and nleaves_max keyword arguments for RJ."
            )

        if not isinstance(nleaves_max, dict) or not isinstance(nleaves_min, dict):
            raise ValueError(
                "nleaves_min and nleaves_max must be provided as dictionaries with keys as branch names and values as the max or min leaf count."
            )
        # store info
        self.nleaves_max = nleaves_max
        self.nleaves_min = nleaves_min
        self.tune = tune
        self.dr = dr
        self.fix_change = fix_change
        if self.fix_change not in [None, +1, -1]:
            raise ValueError("fix_change must be None, +1, or -1.")

        # Decide if DR is desirable. TODO: Now it uses the prior generator, we need to
        # think carefully if we want to use the in-model sampling proposal
        if self.dr is not None and self.dr is not False:
            if self.dr is True:  # Check if it's a boolean, then we just generate
                # from prior (kills the purpose, but yields "healther" chains)
                dr_proposal = DistributionGenerate(
                    self.generate_dist, temperature_control=self.temperature_control
                )
            else:
                # Otherwise pass given input
                dr_proposal = self.dr

            self.dr = DelayedRejection(dr_proposal, max_iter=dr_max_iter)

    def setup(self, branches_coords):
        """Hook run at the start of every :meth:`propose` (no-op here).

        Args:
            branches_coords (dict): The state's ``{branch_name: Branch}`` (the caller passes
                ``state.branches``).
        """

    def fix_logp_gibbs(self, branch_names_run, inds_run, logp, inds):
        """Keep the prior of the proposed state as computed.

        :meth:`Move.fix_logp_gibbs` sets ``logp = -inf`` for walkers with no leaves in
        the branches being run while other branches have leaves: an in-model move has
        nothing to change there. A reversible-jump proposal changes the leaf count, and
        a death that empties the branches being run (allowed by ``nleaves_min = 0``) is
        a valid model: its prior is the one of the remaining leaves (0 for none).
        Rejecting it would make that model unreachable whenever another branch (e.g. a
        fixed-dimension branch) has leaves.
        """

    def get_proposal(
        self, all_coords, all_inds, nleaves_min_all, nleaves_max_all, random, **kwargs
    ):
        """Propose a change of model (implemented by subclasses).

        Args:
            all_coords (dict): ``{branch_name: coords}`` for the branches of this Gibbs
                split, shape ``(ntemps, nwalkers, nleaves_max, ndim)``.
            all_inds (dict): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)`` marking the alive leaves.
            nleaves_min_all (dict): ``{branch_name: int}`` minimum leaf counts.
            nleaves_max_all (dict): ``{branch_name: int}`` maximum leaf counts.
            random (object): Sampler random state (``model.random``).
            **kwargs: :meth:`propose` passes ``branch_supps`` (the split's branch
                supplementals) and ``supps`` (the state supplemental).

        Returns:
            tuple[dict, dict, np.ndarray]: ``(q, new_inds, factors)``: proposed coordinates
            (shape as ``all_coords``), proposed ``inds`` with leaves switched on (births) or
            off (deaths), and log proposal factors of shape ``(ntemps, nwalkers)`` (``+log``
            of numerator terms, ``-log`` of denominator terms of the acceptance ratio).

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def get_model_change_proposal(self, inds, random, nleaves_min, nleaves_max):
        """Choose, per walker, one leaf to add or remove (implemented by subclasses).

        For nested models where the leaf count changes by one at a time.

        Args:
            inds (np.ndarray): Boolean ``inds`` of one branch, shape
                ``(ntemps, nwalkers, nleaves_max)``.
            random (object): Sampler random state.
            nleaves_min (int): Minimum leaf count for this branch.
            nleaves_max (int): Maximum leaf count for this branch.

        Returns:
            dict: ``{"+1": idx, "-1": idx}`` for births and deaths; each ``idx`` is an
            integer array of shape ``(nchanging, 3)`` holding
            ``(temp, walker, leaf)`` indices.

        Raises:
            NotImplementedError: Always, in this base class.
        """

        raise NotImplementedError

    def propose(self, model, state):
        r"""Run one reversible-jump step per Gibbs split (branch group).

        For each split: call ``get_proposal``, restore the branches the split does not
        touch, and add edge factors for branches with ``nleaves_max - nleaves_min >= 2``:
        ``log(1/2)`` when the CURRENT count sits at a limit (the move there is forced) and
        ``-log(1/2)`` when the PROPOSED count does. Then score (``mt_lp`` / ``mt_ll`` from a
        multiple-try subclass are used when present), accept per walker with
        ``factors + logP_new - logP_old > log(U)`` and merge. Finally, unless
        ``prevent_swaps`` is set, run
        :meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` with
        ``adapt=False`` (swaps, no ladder adaptation).

        Args:
            model (:class:`eryn.model.Model`): Sampler callables and ``random``.
            state (:class:`eryn.state.State`): Current state; updated in place.

        Returns:
            tuple[:class:`eryn.state.State`, np.ndarray]: ``(state, accepted)``;
            ``accepted`` is the boolean ``(ntemps, nwalkers)`` result of the LAST split.

        Raises:
            ValueError: A split names no branch, or ``nleaves_min > nleaves_max``.
            NotImplementedError: Delayed rejection (``dr``) is enabled.
        """
        # this exposes anywhere in the proposal class to this information

        # Run any move-specific setup.
        self.setup(state.branches)

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        all_branch_names = list(state.branches.keys())

        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # gibbs sampling is only over branches so pick out that info
            coords_propose_in = {
                key: state.branches_coords[key] for key in branch_names_run
            }
            inds_propose_in = {
                key: state.branches_inds[key] for key in branch_names_run
            }
            branches_supp_propose_in = {
                key: state.branches_supplemental[key] for key in branch_names_run
            }

            if len(list(coords_propose_in.keys())) == 0:
                raise ValueError(
                    "Right now, no models are getting a reversible jump proposal. Check nleaves_min and nleaves_max or do not use rj proposal."
                )

            # get min and max leaf information
            nleaves_max_all = {brn: self.nleaves_max[brn] for brn in branch_names_run}
            nleaves_min_all = {brn: self.nleaves_min[brn] for brn in branch_names_run}

            self.current_model = model
            self.current_state = state
            # propose new sources and coordinates
            q, new_inds, factors = self.get_proposal(
                coords_propose_in,
                inds_propose_in,
                nleaves_min_all,
                nleaves_max_all,
                model.random,
                branch_supps=branches_supp_propose_in,
                supps=state.supplemental,
            )

            branches_supps_new = {
                key: item for key, item in branches_supp_propose_in.items()
            }
            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # put back any branches that were left out from Gibbs split
            for name, branch in state.branches.items():
                if name not in q:
                    q[name] = state.branches[name].coords[:].copy()
                if name not in new_inds:
                    new_inds[name] = state.branches[name].inds[:].copy()

                if name not in branches_supps_new:
                    branches_supps_new[name] = state.branches_supplemental[name]

            # fix any ordering issues
            q, new_inds, branches_supps_new = self.ensure_ordering(
                list(state.branches.keys()), q, new_inds, branches_supps_new
            )

            edge_factors = np.zeros((ntemps, nwalkers))
            # get factors for edges
            for name, branch in state.branches.items():
                nleaves_max = self.nleaves_max[name]
                nleaves_min = self.nleaves_min[name]

                if name not in branch_names_run:
                    continue

                # get old and new values
                old_nleaves = branch.nleaves
                new_nleaves = new_inds[name].sum(axis=-1)

                # do not work on sources with fixed source count
                if nleaves_min == nleaves_max or nleaves_min + 1 == nleaves_max:
                    # nleaves_min == nleaves_max --> no rj proposal
                    # nleaves_min + 1 == nleaves_max --> no edge factors because it is guaranteed to be nleaves_min or nleaves_max
                    continue

                elif nleaves_min > nleaves_max:
                    raise ValueError("nleaves_min cannot be greater than nleaves_max.")

                else:
                    # fix proposal asymmetry at bottom of k range (kmin -> kmin + 1)
                    inds_min = np.where(old_nleaves == nleaves_min)
                    # numerator term so +ln
                    edge_factors[inds_min] += np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range (kmax -> kmax - 1)
                    inds_max = np.where(old_nleaves == nleaves_max)
                    # numerator term so -ln
                    edge_factors[inds_max] += np.log(1 / 2.0)

                    # fix proposal asymmetry at bottom of k range (kmin + 1 -> kmin)
                    inds_min = np.where(new_nleaves == nleaves_min)
                    # numerator term so +ln
                    edge_factors[inds_min] -= np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range (kmax - 1 -> kmax)
                    inds_max = np.where(new_nleaves == nleaves_max)
                    # numerator term so -ln
                    edge_factors[inds_max] -= np.log(1 / 2.0)

            factors += edge_factors

            # setup supplemental information

            if state.supplemental is not None:
                # TODO: should there be a copy?
                new_supps = deepcopy(state.supplemental)

            else:
                new_supps = None

            # for_transfer information can be taken directly from custom proposal

            # supp info

            if hasattr(self, "mt_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_supps = self.mt_supps

            if hasattr(self, "mt_branch_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_branch_supps = self.mt_branch_supps

            # logp and logl

            # Compute prior of the proposed position
            if hasattr(self, "mt_lp"):
                logp = self.mt_lp.reshape(ntemps, nwalkers)

            else:
                logp = model.compute_log_prior_fn(q, inds=new_inds)

            self.fix_logp_gibbs(branch_names_run, inds_run, logp, new_inds)

            if hasattr(self, "mt_ll"):
                logl = self.mt_ll.reshape(ntemps, nwalkers)

            else:
                # Compute the ln like of the proposed position.
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=new_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=branches_supps_new,
                )

            # posterior and previous info

            logP = self.compute_log_posterior(logl, logp)

            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # acceptance fraction
            lnpdiff = factors + logP - prev_logP

            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # update with new state
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=None,
                inds=new_inds,
                supplemental=new_supps,
                branch_supplemental=branches_supps_new,
            )
            state = self.update(state, new_state, accepted)

            # apply delayed rejection to walkers that are +1
            # TODO: need to reexamine this a bit. I have a feeling that only applying
            # this to +1 may not be preserving detailed balance. You may need to
            # "simulate it" for -1 similar to what we do in multiple try
            if self.dr:
                raise NotImplementedError(
                    "Delayed Rejection will be implemented soon. Check for updated versions."
                )
                # for name, branch in state.branches.items():
                #     # We have to work with the binaries added only.
                #     # We need the a) rejected points, b) the model,
                #     # c) the current state, d) the indices where we had +1 (True),
                #     # and the e) factors.
                inds_for_change = {}
                for name in branch_names_run:
                    inds_for_change[name] = {
                        "+1": np.argwhere(new_inds[name] & (~state.branches[name].inds))
                    }

                state, accepted = self.dr.propose(
                    lnpdiff,
                    accepted,
                    model,
                    state,
                    new_state,
                    new_inds,
                    inds_for_change,
                    factors,
                )  # model, state

            # If RJ is true we control only on the in-model step, so no need to do it here as well
            # In most cases, RJ proposal is has small acceptance rate, so in the end we end up
            # switching back what was swapped in the previous in-model step.
            # TODO: MLK: I think we should allow for swapping but no adaptation.

        if self.temperature_control is not None and not self.prevent_swaps:
            state = self.temperature_control.temper_comps(state, adapt=False)

        # add to move-specific accepted information
        self.accepted += accepted
        self.num_proposals += 1

        return state, accepted
