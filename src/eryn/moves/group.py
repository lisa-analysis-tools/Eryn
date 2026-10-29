# -*- coding: utf-8 -*-
"""Group ensemble moves: proposals built from a STATIONARY complement ("friends").

:class:`GroupMove` is the reversible-jump-friendly cousin of :class:`~eryn.moves.RedBlueMove`:
instead of the live positions of the other walkers, the complement is a frozen snapshot that
subclasses build in :meth:`GroupMove.setup_friends` and query in
:meth:`GroupMove.find_friends`, refreshed only every ``n_iter_update`` calls. The stock
concrete move is :class:`~eryn.moves.GroupStretchMove`.
"""
import warnings
from abc import ABC
from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental, State
from .move import Move

__all__ = ["GroupMove"]


class GroupMove(Move, ABC):
    """Abstract "group" ensemble move with a stationary complement.

    In moves like :class:`~eryn.moves.StretchMove` the complement used to build a proposal
    is the current position of the other walkers. In a group move it is a stationary group
    ("friends") that is rebuilt every ``n_iter_update`` calls to :meth:`propose`, from the
    coordinates as they were at the START of that call, and held fixed in between. This is
    the recommended in-model family under reversible jump (the sampler warns when a plain
    :class:`~eryn.moves.StretchMove` is combined with RJ moves).

    Subclasses implement :meth:`setup_friends` (build the stationary group from the
    branches), :meth:`find_friends` (return a complement point per proposed walker) and
    ``get_proposal``; :meth:`fix_friends` is an optional hook for leaves born through RJ.

    The acceptance of a group proposal assumes that the friend of a leaf is chosen
    with the same probability from the current and from the proposed position, so
    the friend choice must not depend on the current position of that leaf. Friends
    recomputed from the current coordinates at every iteration (for example the
    nearest points to each leaf, found in :meth:`setup`) break this assumption, and
    the posterior comes out biased, usually too wide. Friends kept per leaf and
    refreshed only in :meth:`setup_friends` are exact between refreshes, but each
    refresh adapts the friends to the current state, so the sampler is adaptive.
    The bias is small when the friends of a leaf cover most of the posterior samples
    of the same source (for example ``nfriends`` equal to the number of walkers, with
    one copy of the source per walker), and grows as the friends become more local.
    Setting ``freeze_after`` removes it after burn-in.

    Unlike :class:`~eryn.moves.RedBlueMove`, :meth:`propose` moves ALL walkers at once
    (no sub-ensemble split).

    Args:
        nfriends (int, optional): Number of friends a proposal draws from in the stationary
            group. If ``None``, it is set to the number of walkers at the first call to
            :meth:`propose`. Default ``None``.
        n_iter_update (int, optional): Calls between stationary-group rebuilds. Default
            ``100``. Each rebuild adapts the friends to the current state, so the move is
            not exact while rebuilds continue. Frequent rebuilds with friend lists that cover
            a small part of the posterior can bias the posterior (usually too wide). A longer
            period reduces this bias but does not remove it; see ``freeze_after``.
        freeze_after (int, optional): If given, the friends are no longer rebuilt after this
            call (counted over all calls to :meth:`propose`, including burn-in), and the move
            is exact from then on for a fixed number of leaves. Choose it at or before the end
            of burn-in, so that the friends have adapted to the posterior before they are fixed.
            With reversible jump, :meth:`fix_friends` still assigns friends to new leaves.
            If ``None``, the friends are rebuilt for the whole run. Default ``None``.
        live_dangerously (bool, optional): If ``True``, allow ``n_iter_update <= 1``.
            Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.moves.Move`.

    Raises:
        ValueError: ``n_iter_update <= 1`` without ``live_dangerously``.

    Attributes:
        nfriends (int): See ``nfriends``.
        n_iter_update (int): See ``n_iter_update``.
        freeze_after (int or None): See ``freeze_after``.
        iter (int): Number of :meth:`propose` calls so far.
        current_model (:class:`eryn.model.Model`): Model of the current call (set before
            ``get_proposal``, for subclasses).
        current_state (:class:`eryn.state.State`): State of the current call.

    Note:
        **Independent samplers:** with ``nsamplers > 1`` the arrays this move sees are
        FOLDED (leading axis ``nsamplers * ntemps_per_sampler``); friends must only be drawn
        from rows of the same sampler. See :meth:`find_friends`.
    """

    def __init__(
        self,
        nfriends=None,
        n_iter_update=100,
        freeze_after=None,
        live_dangerously=False,
        **kwargs
    ):

        Move.__init__(self, **kwargs)
        self.nfriends = None if nfriends is None else int(nfriends)
        self.n_iter_update = n_iter_update
        self.freeze_after = None if freeze_after is None else int(freeze_after)

        if self.n_iter_update <= 1 and not live_dangerously:
            raise ValueError("n_iter_update must be greather than or equal to 2.")

        self.iter = 0

    @property
    def update_friends_now(self):
        """``True`` if the friends are updated at the current iteration."""
        if self.iter == 0:
            return True
        if self.freeze_after is not None and self.iter > self.freeze_after:
            return False
        return self.iter % self.n_iter_update == 0

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        """Return the complement point for every proposed walker (implemented by subclasses).

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
            name (str): Branch name of the coordinates.
            s (xp.ndarray): Coordinates of the points to move, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            s_inds (xp.ndarray, optional): Boolean ``inds`` of those points, shape
                ``(ntemps, nwalkers, nleaves_max)``. Default ``None``.
            branch_supps (dict, optional): ``{branch_name: BranchSupplemental}`` (a deep
                copy of the state's); the natural place to carry per-leaf bookkeeping for
                the friend lookup. Default ``None``.

        Returns:
            xp.ndarray: Complement points, same shape as ``s``.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    def choose_c_vals(self, name, s, s_inds=None, branch_supps=None):
        """Return the complement points; delegates to :meth:`find_friends`.

        Overrides :meth:`~eryn.moves.StretchMove.choose_c_vals` (different signature) so
        :meth:`~eryn.moves.GroupStretchMove.get_proposal` reads the stationary group.
        """
        return self.find_friends(name, s, s_inds=s_inds, branch_supps=branch_supps)

    def setup(self, branches):
        """Hook run at the start of every :meth:`propose` (no-op here).

        Args:
            branches (dict): The state's ``{branch_name: Branch}``.
        """
        pass

    def setup_friends(self, branches):
        """Build the stationary friends group (implemented by subclasses).

        Called by :meth:`propose` on the first call and every ``n_iter_update`` calls, with
        the branches as they were at the start of the call.

        **Independent-samplers contract:** with ``nsamplers > 1`` the branch
        arrays are folded (axis 0 = ``nsamplers * ntemps_per_sampler``) and any
        stationary friends catalog built here must be kept separate per sampler
        (use ``branch.coords_grouped`` / ``self.sampler_id_rows``). See
        :func:`find_friends` for the full contract.

        Args:
            branches (dict): ``{branch_name: Branch}`` of the sampler state.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    def fix_friends(self, branches):
        """Optional hook to patch friends for leaves born through RJ (no-op here).

        Called by :meth:`propose` on every call that does not rebuild the group.

        Args:
            branches (dict): ``{branch_name: Branch}`` of the sampler state.
        """
        return

    @classmethod
    def get_proposal(self, s_all, random, gibbs_ndim=None, s_inds_all=None, **kwargs):
        """Generate proposal coordinates for all walkers (implemented by subclasses).

        Args:
            s_all (dict): ``{branch_name: coords}``, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (object): Sampler random state (``model.random``).
            gibbs_ndim (int, optional): Number of parameters actually moved in this Gibbs
                split; :meth:`propose` passes a scalar. Default ``None``.
            s_inds_all (dict, optional): ``{branch_name: inds}`` of the leaves taking part,
                shape ``(ntemps, nwalkers, nleaves_max)``. Default ``None``.
            **kwargs: Extra inputs; :meth:`propose` passes ``branch_supps``.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: proposed coordinates with the shapes
            of ``s_all`` and log proposal factors of shape ``(ntemps, nwalkers)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """

        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Propose all walkers against the stationary group, once per Gibbs split.

        Rebuilds the friends with :meth:`setup_friends` on the first call and every
        ``n_iter_update`` calls (calling :meth:`fix_friends` on the others). For each Gibbs
        split: deep-copy the supplementals, propose with ``get_proposal`` (passing the
        split's ``inds`` and the copied ``branch_supps``), restore parameters outside the
        split, score with ``model.compute_log_prior_fn`` / ``model.compute_log_like_fn``
        (which may adjust the copied supplementals in place), accept per walker with
        ``factors + logP_new - logP_old > log(U)`` and merge. Ends with
        :meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` when a
        ``temperature_control`` is set.

        Args:
            model (:class:`eryn.model.Model`): Sampler callables and ``random``.
            state (:class:`eryn.state.State`): Current state; updated in place.

        Returns:
            tuple[:class:`eryn.state.State`, np.ndarray]: ``(state, accepted)``;
            ``accepted`` is the boolean ``(ntemps, nwalkers)`` result of the LAST Gibbs split
            (all splits are added to :attr:`accepted`).
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

        update_friends = self.update_friends_now

        if update_friends:
            self.setup_friends(state.branches)

        if self.iter != 0 and update_friends:
            # store old values to maintain detailed balance when updating
            old_branches = deepcopy(state.branches)

        # fix any friends that may have come through rj
        if not update_friends:
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

        if self.iter != 0 and update_friends:
            # use old values to maintain detailed balance when updating nfriends
            self.setup_friends(old_branches)

        self.iter += 1
        return state, accepted
