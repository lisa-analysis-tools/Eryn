# -*- coding: utf-8 -*-
import warnings
from abc import ABC
from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental, State
from .move import Move

__all__ = ["GroupMove"]


class GroupMove(Move, ABC):
    """
    A "group" ensemble move based on the :class:`eryn.moves.RedBlueMove`.

    In moves like the :class:`eryn.moves.StretchMove`, the complimentary
    group for which the proposal is used is chosen from the current points in
    the ensemble. In "group" moves the complimentary group is a stationary group
    that is updated every `n_iter_update` iterations. This update is performed with the
    last set of coordinates to maintain detailed balance.

    Args:
        nfriends (int, optional): The number of friends to draw from as the complimentary
            ensemble. This group is determined from the stationary group. If ``None``, it will
            be set to the number of walkers. (default: ``None``)
        n_iter_update (int, optional): Number of iterations to run before updating the
            stationary distribution. (default: 100).
        live_dangerously (bool, optional): If ``True``, allow for ``n_iter_update == 1``.
            (deafault: ``False``)

    ``kwargs`` are passed to :class:`Move` class.

    """

    def __init__(
        self, nfriends=None, n_iter_update=100, live_dangerously=False, **kwargs
    ):

        Move.__init__(self, **kwargs)
        self.nfriends = None if nfriends is None else int(nfriends)
        self.n_iter_update = n_iter_update

        if self.n_iter_update <= 1 and not live_dangerously:
            raise ValueError("n_iter_update must be greather than or equal to 2.")

        self.iter = 0

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        """Function for finding friends.

        **Independent-samplers contract:** when the sampler runs with
        ``nsamplers > 1``, the arrays received here are FOLDED — axis 0 has size
        ``nsamplers * ntemps_per_sampler`` and row ``r`` belongs to independent
        sampler ``r // ntemps_per_sampler`` (see ``self.sampler_id_rows``,
        ``self.nsamplers``, ``self.ntemps_per_sampler``). Friends for a walker
        MUST be drawn only from rows belonging to the same sampler; anything
        else silently correlates the supposedly independent ensembles.
        Selecting friends row-by-row (per axis-0 row) is automatically safe;
        the hazard is pooling coordinates across rows (e.g. a global friends
        catalog built from all temperatures).

        Args:
            name (str): Branch name for proposal coordinates.
            s (np.ndarray): Coordinates array for the points to be moved.
            s_inds (np.ndarray, optional): ``inds`` arrays that represent which leaves are present.
                (default: ``None``)
            branch_supps (dict, optional): Keys are ``branch_names`` and values are
                :class:`BranchSupplemental` objects. For group proposals,
                ``branch_supps`` are the best device for passing and tracking useful
                information. (default: ``None``)

        Return:
            np.ndarray: Complimentary values.

        """
        raise NotImplementedError

    def choose_c_vals(self, name, s, s_inds=None, branch_supps=None):
        """Get the complimentary values."""
        return self.find_friends(name, s, s_inds=s_inds, branch_supps=branch_supps)

    def setup(self, branches):
        """Any setup necessary for the proposal"""
        pass

    def setup_friends(self, branches):
        """Setup anything for finding friends.

        **Independent-samplers contract:** with ``nsamplers > 1`` the branch
        arrays are folded (axis 0 = ``nsamplers * ntemps_per_sampler``) and any
        stationary friends catalog built here must be kept separate per sampler
        (use ``branch.coords_grouped`` / ``self.sampler_id_rows``). See
        :func:`find_friends` for the full contract.

        Args:
            branches (dict): Dictionary with all the current branches in the sampler.

        """
        raise NotImplementedError

    def fix_friends(self, branches):
        """Fix any friends that were born through RJ.

        This function is not required. If not implemented, it will just return immediately.

        Args:
            branches (dict): Dictionary with all the current branches in the sampler.

        """
        return

    @classmethod
    def get_proposal(self, s_all, random, gibbs_ndim=None, s_inds_all=None, **kwargs):
        """Generate group stretch proposal coordinates

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)
            s_inds_all (dict, optional): Keys are ``branch_names`` and values are
                ``inds`` arrays indicating which leaves are currently used. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

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

        if self.nfriends is None:
            self.nfriends = nwalkers

        # Run any move-specific setup.
        self.setup(state.branches)

        if self.iter == 0 or self.iter % self.n_iter_update == 0:
            self.setup_friends(state.branches)

        if self.iter != 0 and self.iter % self.n_iter_update == 0:
            # store old values to maintain detailed balance when updating
            old_branches = deepcopy(state.branches)

        # fix any friends that may have come through rj
        if self.iter != 0 and self.iter % self.n_iter_update != 0:
            self.fix_friends(state.branches)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        all_branch_names = list(state.branches.keys())

        # get gibbs sampling information
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):

            if not np.all(
                np.asarray(list(state.branches_supplemental.values())) == None
            ):
                new_branch_supps = deepcopy(state.branches_supplemental)
            else:
                new_branch_supps = None

            if state.supplemental is not None:
                new_supps = deepcopy(state.supplemental)
            else:
                new_supps = None

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

            # need to trick stretch proposal into using the dimenionality associated
            # with Gibbs sampling if it is being used
            # only the active leaves count, shape (ntemps, nwalkers)
            gibbs_ndim = self.get_gibbs_ndim(
                branch_names_run,
                inds_run,
                inds_going_for_proposal,
                {brn: state.branches[brn].shape[-1] for brn in branch_names_run},
            )

            self.current_model = model
            self.current_state = state
            # Get the move-specific proposal.
            q, factors = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                gibbs_ndim=gibbs_ndim,
                s_inds_all=inds_going_for_proposal,
                branch_supps=new_branch_supps,
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # order everything properly
            q, _, new_branch_supps = self.ensure_ordering(
                list(state.branches.keys()), q, state.branches_inds, new_branch_supps
            )

            # Compute prior of the proposed position
            # new_inds_prior is adjusted if product-space is used
            logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

            self.fix_logp_gibbs(branch_names_run, inds_run, logp, state.branches_inds)

            # Can adjust supplementals in place
            logl, new_blobs = model.compute_log_like_fn(
                q,
                inds=state.branches_inds,
                logp=logp,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # get log posterior
            logP = self.compute_log_posterior(logl, logp)

            # get previous information
            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # difference
            lnpdiff = factors + logP - prev_logP

            # draw against acceptance fraction
            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # Update the parameters
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=new_blobs,
                inds=state.branches_inds,
                supplemental=new_supps,
                branch_supplemental=new_branch_supps,
            )
            state = self.update(state, new_state, accepted)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        if self.iter != 0 and self.iter % self.n_iter_update == 0:
            # use old values to maintain detailed balance when updating
            # nfriends
            self.setup_friends(old_branches)

        self.iter += 1
        return state, accepted
