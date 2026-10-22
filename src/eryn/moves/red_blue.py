# -*- coding: utf-8 -*-
import warnings
from abc import ABC
from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental, State
from .move import Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move, ABC):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.

    This class is heavily based on the original from ``emcee``.

    Args:
        nsplits (int, optional): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.
        randomize_split (bool, optional): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. (default: ``True``)
        live_dangerously (bool, optional): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology. (default: ``False``)
        **kwargs (dict, optional): Kwargs for parent classes.

    """

    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, branches_coords):
        """Any setup for the proposal.

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """

    @classmethod
    def get_proposal(self, sample, complement, random, gibbs_ndim=None):
        """Make a proposal

        Args:
            sample (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim].
            complement (dict): Keys are ``branch_names``. Values are lists of other
                other np.ndarray[ntemps, nwalkers - subset size, nleaves_max, ndim] from
                all other subsets. This is the compliment whose ``coords`` are
                used to form the proposal for the ``sample`` subset.
            random (object): Current random state of the sampler.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: Tuple contained proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as np.ndarray[ntemps, nwalkers, nleaves_max, ndim]
                of new coordinates. Second entry is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            tuple: (state, accepted)
                The first return is the state of the sampler after the move.
                The second return value is the accepted count array.

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
