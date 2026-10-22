# -*- coding: utf-8 -*-
"""Red-blue (split-ensemble) proposal base class.

:class:`RedBlueMove` splits the walkers of every temperature into ``nsplits`` sub-ensembles
and proposes each sub-ensemble in turn against the CURRENT positions of the others (the
complement), as in Foreman-Mackey et al. (2013). Subclasses supply ``get_proposal``; the
stock one is :class:`~eryn.moves.StretchMove`.
"""
import warnings
from abc import ABC
from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental, State
from .move import Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move, ABC):
    """Abstract red-blue ensemble move (Foreman-Mackey et al. 2013, arXiv:1202.3665).

    Heavily based on the original from ``emcee``. In :meth:`propose`, the walkers of every
    temperature row are assigned to ``nsplits`` equal sub-ensembles; each sub-ensemble in
    turn is proposed with ``get_proposal(s, c, model.random, gibbs_ndim=...)`` where ``s``
    holds its coordinates and ``c`` the current coordinates of the other sub-ensembles at
    the SAME temperature, then accepted or rejected per walker with the move's (tempered)
    log posterior. Supplementals are sliced to the sub-ensemble and merged back through
    :meth:`~eryn.moves.Move.update`.

    Args:
        nsplits (int, optional): Number of sub-ensembles. Each one is updated using the
            others as the complementary ensemble. Default ``2``.
        randomize_split (bool, optional): Shuffle the walker-to-sub-ensemble assignment
            (independently per temperature row) on every call, keeping the sub-ensemble
            sizes equal. Default ``True``.
        live_dangerously (bool, optional): By default :meth:`propose` raises
            ``RuntimeError`` when ``nwalkers < 2 * sum(nleaves_max * ndim)`` over branches,
            because the walkers would then be stuck on a low-dimensional subspace. Mixing in
            other moves (e.g. a Metropolis-Hastings step) avoids that; set ``True`` to
            suppress the error. (Thanks to @dstndstn for the terminology.) Default
            ``False``.
        **kwargs: Forwarded to :class:`~eryn.moves.Move`.

    """

    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, branches_coords):
        """Hook run at the start of every :meth:`propose` (no-op here).

        Args:
            branches_coords (dict): The state's branches, ``{branch_name: Branch}`` (the
                caller passes ``state.branches``); coordinates have shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
        """

    @classmethod
    def get_proposal(self, sample, complement, random, gibbs_ndim=None):
        """Propose new coordinates for one sub-ensemble (implemented by subclasses).

        Args:
            sample (dict): ``{branch_name: coords}`` of the sub-ensemble being moved, shape
                ``(ntemps, nsub, nleaves_max, ndim)``.
            complement (dict): ``{branch_name: list}``; the list holds one
                ``(ntemps, nsub_j, nleaves_max, ndim)`` coordinate array per OTHER
                sub-ensemble, used to build the proposal for ``sample``.
            random (object): Sampler random state (``model.random``).
            gibbs_ndim (int, optional): Number of parameters actually being moved in this
                Gibbs split; :meth:`propose` passes a scalar (the stock
                :meth:`~eryn.moves.StretchMove.adjust_factors` cannot take an array). See the
                tutorial. Default ``None``.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: proposed coordinates with the shape
            of ``sample``, and the log proposal factors of shape ``(ntemps, nsub)`` added to
            the log acceptance (``+log`` of numerator terms, ``-log`` of denominator terms).

        Raises:
            NotImplementedError: Always, in this base class.
        """

        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Run one red-blue sweep over every Gibbs split and sub-ensemble.

        For each Gibbs split and each sub-ensemble: propose with ``get_proposal``, restore
        the parameters outside the split, score the proposal with
        ``model.compute_log_prior_fn`` and ``model.compute_log_like_fn`` (NaN log-likelihoods
        become ``-1e300`` with a warning), accept walkers with
        ``factors + logP_new - logP_old > log(U)``, and merge them into ``state``. Ends with
        :meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` when a
        ``temperature_control`` is set.

        Args:
            model (:class:`eryn.model.Model`): Sampler callables and ``random``.
            state (:class:`eryn.state.State`): Current state; updated in place.

        Returns:
            tuple[:class:`eryn.state.State`, np.ndarray]: ``(state, accepted)``; ``accepted``
            is a boolean ``(ntemps, nwalkers)`` array, ``True`` where any proposal of the
            sweep was accepted.

        Raises:
            RuntimeError: Too few walkers for the dimension and ``live_dangerously`` is off.
        """
        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        if nwalkers < 2 * ndim_total and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions. If you would like to do this, please set live_dangerously"
                "to True."
            )

        # Run any move-specific setup.
        self.setup(state.branches)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        all_inds = np.tile(np.arange(nwalkers), (ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            [np.random.shuffle(x) for x in inds]

        all_branch_names = list(state.branches.keys())

        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # get gibbs sampling information
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # setup proposals based on Gibbs sampling
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            if not at_least_one_proposal:
                continue

            # prepare accepted fraction
            accepted_here = np.zeros((ntemps, nwalkers), dtype=bool)
            for split in range(self.nsplits):
                # get split information
                S1 = inds == split
                num_total_here = np.sum(inds == split)
                nwalkers_here = np.sum(S1[0])

                all_inds_shaped = all_inds[S1].reshape(ntemps, nwalkers_here)

                # inds including gibbs information
                new_inds = {
                    name: np.take_along_axis(
                        state.branches[name].inds,
                        all_inds_shaped[:, :, None],
                        axis=1,
                    )
                    for name in state.branches
                }

                # the actual inds for the subset
                real_inds_subset = {
                    name: new_inds[name] for name in inds_going_for_proposal
                }

                # actual coordinates of subset
                temp_coords = {
                    name: np.take_along_axis(
                        state.branches_coords[name],
                        all_inds_shaped[:, :, None, None],
                        axis=1,
                    )
                    for name in state.branches_coords
                }

                # prepare the sets for each model
                # goes into the proposal as (ntemps * (nwalkers / subset size), nleaves_max, ndim)
                sets = {
                    key: [
                        np.take_along_axis(
                            state.branches[key].coords,
                            all_inds[inds == j].reshape(ntemps, -1)[:, :, None, None],
                            axis=1,
                        )
                        for j in range(self.nsplits)
                    ]
                    for key in branch_names_run
                }

                # setup s and c based on splits
                s = {key: sets[key][split] for key in sets}
                c = {key: sets[key][:split] + sets[key][split + 1 :] for key in sets}

                # need to trick stretch proposal into using the dimenionality associated
                # with Gibbs sampling if it is being used
                # only the active leaves of the walkers in this split count,
                # shape (ntemps, nwalkers_here)
                gibbs_ndim = self.get_gibbs_ndim(
                    branch_names_run,
                    inds_run,
                    {
                        brn: np.take_along_axis(
                            inds_going_for_proposal[brn],
                            all_inds_shaped[:, :, None],
                            axis=1,
                        )
                        for brn in branch_names_run
                    },
                    {brn: state.branches[brn].shape[-1] for brn in branch_names_run},
                )

                # Get the move-specific proposal.
                q, factors = self.get_proposal(
                    s, c, model.random, gibbs_ndim=gibbs_ndim
                )

                # account for gibbs sampling
                self.cleanup_proposals_gibbs(branch_names_run, inds_run, q, temp_coords)

                # setup supplemental information
                if state.supplemental is not None:
                    # TODO: should there be a copy?
                    new_supps = BranchSupplemental(
                        state.supplemental.take_along_axis(all_inds_shaped, axis=1),
                        base_shape=(ntemps, nwalkers),
                        copy=False,
                    )

                else:
                    new_supps = None

                # default for removing inds info from supp
                if not np.all(
                    np.asarray(list(state.branches_supplemental.values())) == None
                ):
                    new_branch_supps_tmp = {
                        name: state.branches[name].branch_supplemental.take_along_axis(
                            all_inds_shaped[:, :, None], axis=1
                        )
                        for name in state.branches
                        if state.branches[name].branch_supplemental is not None
                    }

                    new_branch_supps = {
                        name: BranchSupplemental(
                            new_branch_supps_tmp[name],
                            base_shape=new_inds[name].shape,
                            copy=False,
                        )
                        for name in new_branch_supps_tmp
                    }

                else:
                    new_branch_supps = None

                # order everything properly
                q, new_inds, new_branch_supps = self.ensure_ordering(
                    list(state.branches.keys()), q, new_inds, new_branch_supps
                )

                # Compute prior of the proposed position
                # new_inds_prior is adjusted if product-space is used
                logp = model.compute_log_prior_fn(
                    q,
                    inds=new_inds,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

                self.fix_logp_gibbs(branch_names_run, inds_run, logp, real_inds_subset)

                # Compute the lnprobs of the proposed position.
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=new_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

                # catch and warn about nans
                if np.any(np.isnan(logl)):
                    logl[np.isnan(logl)] = -1e300
                    warnings.warn("Getting Nan in likelihood computation.")

                logP = self.compute_log_posterior(logl, logp)

                prev_logl = np.take_along_axis(state.log_like, all_inds_shaped, axis=1)

                prev_logp = np.take_along_axis(state.log_prior, all_inds_shaped, axis=1)

                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

                lnpdiff = factors + logP - prev_logP

                keep = lnpdiff > np.log(model.random.rand(ntemps, nwalkers_here))

                # if gibbs sampling, this will say it is accepted if
                # any of the gibbs proposals were accepted
                np.put_along_axis(
                    accepted_here,
                    all_inds_shaped,
                    keep,
                    axis=1,
                )

                # readout for total
                accepted = (accepted.astype(int) + accepted_here.astype(int)).astype(
                    bool
                )
                # new state
                new_state = State(
                    q,
                    log_like=logl,
                    log_prior=logp,
                    blobs=new_blobs,
                    inds=new_inds,
                    supplemental=new_supps,
                    branch_supplemental=new_branch_supps,
                )

                # update state
                state = self.update(
                    state, new_state, accepted_here, subset=all_inds_shaped
                )

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        # temp swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted
