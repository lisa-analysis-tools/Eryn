# -*- coding: utf-8 -*-
"""Base class shared by every Eryn proposal ("move").

:class:`Move` holds the machinery common to all proposals: the Gibbs-split bookkeeping
(:meth:`Move.gibbs_sampling_setup_iterator`, :meth:`Move.setup_proposals`,
:meth:`Move.cleanup_proposals_gibbs`, :meth:`Move.fix_logp_gibbs`), the switch between
the tempered and untempered log posterior driven by ``temperature_control``, the
per-move acceptance counters, and :meth:`Move.update`, which merges the accepted walkers of
a proposed :class:`~eryn.state.State` back into the current one.

Concrete moves (:class:`~eryn.moves.MHMove`, :class:`~eryn.moves.RedBlueMove`,
:class:`~eryn.moves.GroupMove`, :class:`~eryn.moves.ReversibleJumpMove`, ...) subclass it and
implement ``propose(model, state) -> (state, accepted)``, the call
:meth:`~eryn.ensemble.EnsembleSampler.sample` makes once per in-model (or RJ) repeat.
"""

from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

__all__ = ["Move"]


class Move(object):
    """Parent class for proposals ("moves").

    Subclasses implement ``propose(model, state) -> (state, accepted)``, which
    :meth:`~eryn.ensemble.EnsembleSampler.sample` calls once per in-model (or RJ) repeat:

    * ``model`` is the :class:`eryn.model.Model` namedtuple (``log_like_fn``,
      ``compute_log_like_fn``, ``compute_log_prior_fn``, ``temperature_control``, ``map_fn``,
      ``random``). Moves score points with ``model.compute_log_prior_fn(coords, inds=...)`` and
      ``model.compute_log_like_fn(coords, inds=..., logp=..., supps=..., branch_supps=...)``
      and draw random numbers from ``model.random``.
    * ``state`` is the current :class:`eryn.state.State`; each branch holds ``coords`` of
      shape ``(ntemps, nwalkers, nleaves_max, ndim)`` and boolean ``inds`` of shape
      ``(ntemps, nwalkers, nleaves_max)``; ``log_like`` and ``log_prior`` have shape
      ``(ntemps, nwalkers)``. With independent samplers (``nsamplers > 1``) the leading axis
      is FOLDED to ``nsamplers * ntemps``.
    * The returned ``accepted`` is a ``(ntemps, nwalkers)`` boolean (or count) array that
      the sampler adds to its per-iteration acceptance counter. A move also adds it to
      :attr:`accepted` and increments :attr:`num_proposals` itself.

    Args:
        temperature_control (:class:`~eryn.moves.tempering.TemperatureControl`, optional):
            Controls the tempering. Setting it (here or later through the property) makes
            :attr:`compute_log_posterior` the tempered
            :meth:`~eryn.moves.tempering.TemperatureControl.compute_log_posterior_tempered`
            and copies the ladder layout onto the move (``ntemps``, ``nsamplers``,
            ``ntemps_per_sampler``, ``sampler_id_rows``). ``None`` uses the untempered
            :meth:`compute_log_posterior_basic`. The
            :class:`~eryn.ensemble.EnsembleSampler` fills it in for moves passed with
            ``None`` when the sampler is tempered. Default ``None``.
        periodic (:class:`~eryn.utils.PeriodicContainer`, optional): Periodic-parameter
            information used to wrap proposed points and compute periodic distances. The
            sampler fills in its own for moves passed with ``None``. Default ``None``.
        gibbs_sampling_setup (str, tuple, dict, or list, optional): Gibbs splits to run within
            ONE call to ``propose``. All splits run in order, each as a separate
            propose/accept step.

            * ``None``: one split over all branches and all parameters.
            * ``str``: all parameters of that branch.
            * 2-tuple ``(branch_name, mask)``: ``mask`` is a 2D boolean array of shape
              ``(nleaves_max, ndim)`` selecting leaves and/or parameters, or ``None`` for all
              of them.
            * ``dict``: ``{branch_name: mask_or_None}``; all keys run together as one split.
            * ``list``: of the above; each entry is one split, run in list order.

            Default ``None``.
        prevent_swaps (bool, optional): If ``True``, skip the temperature swaps at the end of
            the proposal. Only :class:`~eryn.moves.ReversibleJumpMove` reads it. Default
            ``False``.
        skip_supp_names_update (list, optional): Supplemental keys (``str``, in the state's
            ``supplemental`` or any branch's :class:`~eryn.state.BranchSupplemental`) that
            :meth:`update` does not merge on accept/reject. Useful for large or frozen
            side-channel data. Default ``[]``.
        is_rj (bool, optional): ``True`` for reversible-jump moves. Then a Gibbs split may
            only name branches (no leaf/parameter masks); a tuple split raises
            ``ValueError``. Default ``False``.
        use_gpu (bool, optional): If ``True``, :attr:`xp` is ``cupy`` (or ``numpy`` when cupy
            is not installed). Default ``False``.
        random_seed (int, optional): If not ``None``, seed the global ``xp.random``
            generator with it. Default ``None``.
        **kwargs: Accepted for cooperative multiple inheritance and ignored (not stored).

    Raises:
        ValueError: ``gibbs_sampling_setup`` has the wrong type or a malformed mask, or a
            leaf/parameter mask is given with ``is_rj=True``.

    Attributes:
        num_proposals (int): Number of propose/accept steps run so far (one per Gibbs split
            for most moves). Divides :attr:`accepted` in :attr:`acceptance_fraction`.
        time (int): Initialised to ``0``; not advanced by the base class.
        gibbs_sampling_setup (list or None): Normalised Gibbs splits.
        branch_names_run_all (list): Per split, the branch names it runs (``[None]`` when
            there is no Gibbs setup, meaning all branches).
        inds_run_all (list): Per split, the matching masks (``None`` = all parameters).
        compute_log_posterior (callable): ``(logl, logp) -> logP``, tempered or basic.
        nsamplers (int): Number of independent samplers (class default ``1``).
        ntemps_per_sampler (int or None): Rungs per sampler.
        sampler_id_rows (np.ndarray or None): Sampler index of each folded leading-axis row.

    """

    def __init__(
        self,
        temperature_control=None,
        periodic=None,
        gibbs_sampling_setup=None,
        prevent_swaps=False,
        skip_supp_names_update=[],
        is_rj=False,
        use_gpu=False,
        random_seed=None,
        **kwargs
    ):
        # store all information
        self.temperature_control = temperature_control
        self.periodic = periodic
        self.skip_supp_names_update = skip_supp_names_update
        self.prevent_swaps = prevent_swaps

        self._initialize_branch_setup(gibbs_sampling_setup, is_rj=is_rj)

        # keep track of the number of proposals
        self.num_proposals = 0
        self.time = 0

        self.use_gpu = use_gpu

        # set the random seet of the library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

    @property
    def use_gpu(self):
        """bool: Whether :attr:`xp` is ``cupy`` (``True``) or ``numpy`` (``False``)."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu):
        self._use_gpu = use_gpu

    @property
    def xp(self):
        """module: ``cupy`` if :attr:`use_gpu` is truthy, else ``numpy``.

        Raises:
            ValueError: :attr:`use_gpu` is ``None``.
        """
        if self._use_gpu is None:
            raise ValueError("use_gpu has not been set.")
        xp = cp if self.use_gpu else np
        return xp

    def _initialize_branch_setup(self, gibbs_sampling_setup, is_rj=False):
        """Normalise ``gibbs_sampling_setup`` into per-split branch names and masks.

        Sets :attr:`gibbs_sampling_setup` (list), ``gibbs_sampling_setup_input`` (a copy of the
        input, only when a setup is given), :attr:`branch_names_run_all` and
        :attr:`inds_run_all`. See the class docstring for the accepted forms.

        Args:
            gibbs_sampling_setup (str, tuple, dict, list, or None): User Gibbs setup.
            is_rj (bool, optional): Reject leaf/parameter masks (RJ moves). Default ``False``.

        Raises:
            ValueError: Wrong type, malformed mask, or a mask given with ``is_rj=True``.
        """
        self.gibbs_sampling_setup = gibbs_sampling_setup

        message_rj = """inputting gibbs indexing at the leaf/parameter level is not allowed 
                                        with an RJ proposal. Only branch names."""

        message_non_rj = """When inputing gibbs indexing and using a 2-tuple, second item must be None or 2D np.ndarray of shape (nleaves_max, ndim)."""

        # setup proposal branches properly
        if self.gibbs_sampling_setup is not None:
            # string indicates one branch (all of it)
            if type(self.gibbs_sampling_setup) not in [str, tuple, list, dict]:
                raise ValueError(
                    "gibbs_sampling_setup must be string, dict, tuple, or list."
                )

            if not isinstance(self.gibbs_sampling_setup, list):
                self.gibbs_sampling_setup = [self.gibbs_sampling_setup]

            gibbs_sampling_setup_tmp = []
            for item in self.gibbs_sampling_setup:
                # all the arguments are treated

                # strings indicate single branch all parameters
                if isinstance(item, str):
                    gibbs_sampling_setup_tmp.append(item)

                # tuple is one branch with a split in the parameters
                elif isinstance(item, tuple):
                    # check inputs
                    assert len(item) == 2
                    if item is not None and is_rj:
                        raise ValueError(message_rj)

                    elif (
                        not isinstance(item[1], np.ndarray) and item[1] is not None
                    ) or (isinstance(item[1], np.ndarray) and item[1].ndim != 2):
                        breakpoint()
                        raise ValueError(message_non_rj)

                    gibbs_sampling_setup_tmp.append(item)

                # dict can include multiple models and parameter splits
                # these will all be in one iteration
                elif isinstance(item, dict):
                    tmp = []
                    for key, value in item.items():
                        # check inputs
                        if value is not None and is_rj:
                            raise ValueError(message_rj)

                        elif (
                            not isinstance(value, np.ndarray) and value is not None
                        ) or (isinstance(value, np.ndarray) and value.ndim != 2):
                            raise ValueError(message_non_rj)

                        tmp.append((key, value))

                    gibbs_sampling_setup_tmp.append(tmp)

                else:
                    raise ValueError(
                        "If providing a list for gibbs_sampling_setup, each item needs to be a string, tuple, or dict."
                    )

            # copy the original for information if needed
            self.gibbs_sampling_setup_input = deepcopy(self.gibbs_sampling_setup)

            # store as the setup that all proposals will follow
            self.gibbs_sampling_setup = gibbs_sampling_setup_tmp

            # now that we have everything out of the input
            # sort into branch names and indices to be run
            branch_names_run_all = []
            inds_run_all = []

            # for each split in the gibbs splits
            for prop_i, proposal_iteration in enumerate(self.gibbs_sampling_setup):
                # break out
                if isinstance(proposal_iteration, tuple):
                    # tuple is 1 entry loop
                    branch_names_run_all.append([proposal_iteration[0]])
                    inds_run_all.append([proposal_iteration[1]])
                elif isinstance(proposal_iteration, str):
                    # string is 1 entry loop
                    branch_names_run_all.append([proposal_iteration])
                    inds_run_all.append([None])

                elif isinstance(proposal_iteration, list):
                    # list allows more branches at the same time
                    branch_names_run_all.append([])
                    inds_run_all.append([])
                    for item in proposal_iteration:
                        if isinstance(item, str):
                            branch_names_run_all[prop_i].append(item)
                            inds_run_all[prop_i].append(None)
                        elif isinstance(item, tuple):
                            branch_names_run_all[prop_i].append(item[0])
                            inds_run_all[prop_i].append(item[1])

            # store information
            self.branch_names_run_all = branch_names_run_all
            self.inds_run_all = inds_run_all

        else:
            # no Gibbs sampling
            self.branch_names_run_all = [None]
            self.inds_run_all = [None]

    def gibbs_sampling_setup_iterator(self, all_branch_names):
        """Iterate over the Gibbs splits.

        Without a Gibbs setup this yields one split covering every branch with no masks.

        Args:
            all_branch_names (list): All branch names of the state, used for the
                "all branches" split.

        Yields:
            tuple[list, list]: ``(branch_names_run, inds_run)``: the branch names this split
            proposes and, per branch, a ``(nleaves_max, ndim)`` boolean mask or ``None``
            (all leaves/parameters).
        """
        for branch_names_run, inds_run in zip(
            self.branch_names_run_all, self.inds_run_all
        ):
            # adjust if branch_names_run is None
            if branch_names_run is None:
                branch_names_run = all_branch_names
                inds_run = [None for _ in branch_names_run]
            # yield to the iterator
            yield (branch_names_run, inds_run)

    def setup_proposals(
        self, branch_names_run, inds_run, branches_coords, branches_inds
    ):
        """Restrict the proposal inputs to one Gibbs split.

        For a branch with a mask, a leaf takes part when any of its parameters is selected
        AND the leaf is alive in ``branches_inds``; a branch with mask ``None`` keeps its
        ``inds`` unchanged. Coordinates are passed through (the same arrays, not copies).

        Args:
            branch_names_run (list): Branch names proposed together in this split.
            inds_run (list): Per branch, a ``(nleaves_max, ndim)`` boolean mask or ``None``.
            branches_coords (dict): ``{branch_name: coords}`` with coords of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            branches_inds (dict): ``{branch_name: inds}`` with boolean inds of shape
                ``(ntemps, nwalkers, nleaves_max)``.

        Returns:
            tuple[dict, dict, bool]: ``(coords, inds, at_least_one_proposal)`` restricted to
            ``branch_names_run``; ``at_least_one_proposal`` is ``True`` when any selected leaf is
            alive anywhere in the ensemble (callers skip the split otherwise).
        """
        inds_going_for_proposal = {}
        coords_going_for_proposal = {}

        at_least_one_proposal = False
        for bnr, ir in zip(branch_names_run, inds_run):
            if ir is not None:
                tmp = np.zeros_like(branches_inds[bnr], dtype=bool)

                # flatten coordinates to the leaves dimension
                ir_keep = ir.astype(int).sum(axis=-1).astype(bool)
                tmp[:, :, ir_keep] = True
                # make sure leavdes that are actually not there are not counted
                tmp[~branches_inds[bnr]] = False
                inds_going_for_proposal[bnr] = tmp
            else:
                inds_going_for_proposal[bnr] = branches_inds[bnr]

            if np.any(inds_going_for_proposal[bnr]):
                at_least_one_proposal = True

            coords_going_for_proposal[bnr] = branches_coords[bnr]

        return (
            coords_going_for_proposal,
            inds_going_for_proposal,
            at_least_one_proposal,
        )

    def get_gibbs_ndim(self, branch_names_run, inds_run, inds_going_for_proposal, ndims):
        """Number of parameters actually moved by an ensemble proposal, per walker.

        Only the active leaves (and, when Gibbs sampling, only the selected parameters
        of the selected leaves) count. Inactive leaves are carried along by the proposal
        but are not part of the target, so counting them (``nleaves_max * ndim``) gives
        the wrong dimension in the stretch factor ``(ndim - 1) ln z``.

        Args:
            branch_names_run (list): List of branch names to run concurrently.
            inds_run (list): List of ``inds`` arrays including Gibbs sampling information.
            inds_going_for_proposal (dict): ``inds`` arrays of the walkers being proposed,
                as returned by :func:`setup_proposals`. Values have shape
                ``(ntemps, nwalkers, nleaves_max)``.
            ndims (dict): Keys are ``branch_names`` and values are the number of
                parameters per leaf.

        Returns:
            np.ndarray: Moved dimension with shape ``(ntemps, nwalkers)``.

        """
        gibbs_ndim = 0
        for brn, ir in zip(branch_names_run, inds_run):
            inds_here = np.asarray(inds_going_for_proposal[brn])
            if ir is None:
                # all parameters of the active leaves
                gibbs_ndim = gibbs_ndim + inds_here.sum(axis=-1) * ndims[brn]
            else:
                # selected parameters of the active, selected leaves
                gibbs_ndim = gibbs_ndim + (inds_here * ir.sum(axis=-1)).sum(axis=-1)
        return gibbs_ndim

    def cleanup_proposals_gibbs(
        self,
        branch_names_run,
        inds_run,
        q,
        branches_coords,
        new_inds=None,
        branches_inds=None,
        new_branch_supps=None,
        branches_supplemental=None,
    ):
        """Restore everything a Gibbs split did not propose, in place.

        Parameters outside a branch's mask are copied back from ``branches_coords``; branches
        missing from ``q`` (and from ``new_inds`` / ``new_branch_supps`` when given) are filled
        with copies of the current values. ``q``, ``new_inds`` and ``new_branch_supps`` are
        modified in place.

        Args:
            branch_names_run (list): Branch names proposed together in this split.
            inds_run (list): Per branch, a ``(nleaves_max, ndim)`` boolean mask or ``None``.
            q (dict): Proposed coordinates ``{branch_name: (ntemps, nwalkers, nleaves_max,
                ndim)}``.
            branches_coords (dict): Current coordinates for all branches.
            new_inds (dict, optional): Proposed ``inds`` per branch. Default ``None``.
            branches_inds (dict, optional): Current ``inds``; required when ``new_inds`` is
                given. Default ``None``.
            new_branch_supps (dict, optional): Proposed branch supplementals. Default ``None``.
            branches_supplemental (dict, optional): Current branch supplementals; required
                when ``new_branch_supps`` is given. Default ``None``.
        """
        # add back any parameters that are fixed for this round
        for bnr, ir in zip(branch_names_run, inds_run):
            if ir is not None:
                q[bnr][:, :, ~ir] = branches_coords[bnr][:, :, ~ir]

        # add other models that were not included
        for key, value in branches_coords.items():
            if key not in q:
                q[key] = value.copy()
            if new_inds is not None and key not in new_inds:
                assert branches_inds is not None
                new_inds[key] = branches_inds[key].copy()

            if new_branch_supps is not None and key not in new_branch_supps:
                assert branches_supplemental is not None
                new_branch_supps[key] = branches_supplemental[key].copy()

    def ensure_ordering(self, correct_key_order, q, new_inds, new_branch_supps):
        """Reorder the proposal dictionaries to the state's branch order.

        Args:
            correct_key_order (list): Branch names in the state's order.
            q (dict): Proposed coordinates for all branches.
            new_inds (dict): Proposed ``inds`` for all branches.
            new_branch_supps (dict or None): Proposed branch supplementals. Missing branches
                become ``None``; the reordered dict is a deep copy.

        Returns:
            tuple[dict, dict, dict or None]: ``(q, new_inds, new_branch_supps)`` in
            ``correct_key_order`` (inputs returned unchanged when already ordered).
        """
        if list(q.keys()) != correct_key_order:
            q = {key: q[key] for key in correct_key_order}

        if list(new_inds.keys()) != correct_key_order:
            new_inds = {key: new_inds[key] for key in correct_key_order}

        if (
            new_branch_supps is not None
            and list(new_branch_supps.keys()) != correct_key_order
        ):
            temp = {key: None for key in correct_key_order}
            for key in new_branch_supps:
                temp[key] = new_branch_supps[key]
            new_branch_supps = deepcopy(temp)

        return q, new_inds, new_branch_supps

    def fix_logp_gibbs(self, branch_names_run, inds_run, logp, inds):
        """Fix the proposed log prior of walkers this Gibbs split cannot move, in place.

        A walker with leaves in the model but none selected by this split gets
        ``logp = -inf`` (so the proposal is rejected and nothing changes); a walker with no
        leaves at all gets ``logp = 0.0``.

        Args:
            branch_names_run (list): Branch names proposed together in this split.
            inds_run (list): Per branch, a ``(nleaves_max, ndim)`` boolean mask or ``None``.
            logp (np.ndarray): Proposed log prior, shape ``(ntemps, nwalkers)``. Modified in
                place.
            inds (dict): ``inds`` arrays ``(ntemps, nwalkers, nleaves_max)`` for all branches.
        """
        total_leaves = np.zeros_like(logp, dtype=int)
        total_leaves_here = np.zeros_like(logp, dtype=int)
        for bnr, ir in zip(branch_names_run, inds_run):
            if ir is not None:
                tmp = np.zeros_like(inds[bnr], dtype=bool)

                # flatten coordinates to the leaves dimension
                ir_keep = ir.astype(int).sum(axis=-1).astype(bool)
                tmp[:, :, ir_keep] = True
                # make sure leaves that are actually not there are not counted
                tmp[~inds[bnr]] = False

            else:
                tmp = inds[bnr]

            total_leaves += tmp.sum(axis=-1)
            total_leaves_here += tmp.sum(axis=-1)

        for name, inds_val in inds.items():
            if name not in branch_names_run:
                total_leaves += inds_val.sum(axis=-1)

        # adjust
        logp[(total_leaves != 0) & (total_leaves_here == 0)] = -np.inf  # no use in running because no change
        logp[(total_leaves == 0) & (total_leaves_here == 0)] = 0.0  # there is nothing in the model currently

    @property
    def accepted(self):
        """np.ndarray: Accepted counts for this move, shape ``(ntemps, nwalkers)``.

        Set by :class:`~eryn.ensemble.EnsembleSampler` (zeros) before sampling; the setter
        requires an ``np.ndarray``.
        """
        if self._accepted is None:
            raise ValueError(
                "accepted must be inititalized with the init_accepted function if you want to use it."
            )
        return self._accepted

    @accepted.setter
    def accepted(self, accepted):
        assert isinstance(accepted, np.ndarray)
        self._accepted = accepted

    @property
    def acceptance_fraction(self):
        """np.ndarray: ``accepted / num_proposals``, shape ``(ntemps, nwalkers)``."""
        return self.accepted / self.num_proposals

    # independent-samplers defaults (overwritten when a sampler-aware
    # TemperatureControl is attached, or set directly by EnsembleSampler)
    nsamplers = 1
    ntemps_per_sampler = None
    sampler_id_rows = None

    @property
    def temperature_control(self):
        """:class:`~eryn.moves.tempering.TemperatureControl` or None: Temperature controller.

        Setting it also selects :attr:`compute_log_posterior` (tempered when not ``None``) and
        copies the ladder layout onto the move: ``ntemps`` (the folded
        ``nsamplers * ntemps`` size), ``nsamplers``, ``ntemps_per_sampler`` and
        ``sampler_id_rows``.
        """
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        self._temperature_control = temperature_control

        # use the setting of the temperature control to determine which log posterior function to use
        # tempered or basic
        if temperature_control is None:
            self.compute_log_posterior = self.compute_log_posterior_basic
        else:
            self.compute_log_posterior = (
                self.temperature_control.compute_log_posterior_tempered
            )

            # self.ntemps is the folded axis-0 size (nsamplers * ntemps per sampler):
            # proposals treat independent samplers as extra temperatures
            self.ntemps = getattr(
                temperature_control, "ntemps_eff", temperature_control.ntemps
            )
            self.nsamplers = getattr(temperature_control, "nsamplers", 1)
            self.ntemps_per_sampler = temperature_control.ntemps
            self.sampler_id_rows = np.repeat(
                np.arange(self.nsamplers), self.ntemps_per_sampler
            )

    def compute_log_posterior_basic(self, logl, logp):
        r"""Compute the untempered log posterior :math:`\log{P} = \log{L} + \log{p}`.

        Used as :attr:`compute_log_posterior` when no ``temperature_control`` is set; same
        call signature as the tempered version.

        Args:
            logl (np.ndarray): Log-likelihood values, shape ``(ntemps, nwalkers)``.
            logp (np.ndarray): Log-prior values, same shape.

        Returns:
            np.ndarray: Log-posterior values, shape ``(ntemps, nwalkers)``.
        """
        return logl + logp

    def tune(self, state, accepted):
        """Tune the proposal after a ``propose`` call (no-op hook).

        :meth:`~eryn.ensemble.EnsembleSampler.sample` calls it after every ``propose`` when
        run with ``tune=True``. Override to adapt proposal settings.

        Args:
            state (:class:`eryn.state.State`): State after the proposal.
            accepted (np.ndarray): Accepted array returned by that ``propose`` call, shape
                ``(ntemps, nwalkers)``.
        """
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        """Merge the accepted walkers of ``new_state`` into ``old_state`` in place.

        Adapted from ``emcee`` for Eryn's branch structure. For every walker in ``subset``
        the accepted entries of ``new_state`` replace those of ``old_state`` for
        ``log_like``, ``log_prior``, each branch's ``inds`` and ``coords``, the branch
        supplementals and state supplemental (except keys in ``skip_supp_names_update``), and
        ``blobs``. ``log_like`` is merged with ``np.where`` so a rejected ``-inf`` proposal
        cannot leave a NaN behind; for ``log_prior`` the proposed ``-inf`` values are zeroed
        before the arithmetic blend for the same reason (a ``-inf`` prior is never accepted).

        Args:
            old_state (:class:`eryn.state.State`): Current state; modified in place.
            new_state (:class:`eryn.state.State`): Proposed state. Its arrays have the
                ``subset`` shape on the walker axis (``(ntemps, nsubset, ...)``).
            accepted (np.ndarray): Boolean accept flags, shape ``(ntemps, nwalkers)`` (full
                ensemble), indexed through ``subset``.
            subset (np.ndarray, optional): INTEGER walker indices of shape
                ``(ntemps, nsubset)``: row ``t`` lists the walkers of temperature ``t`` that
                ``new_state`` describes (as used by :class:`~eryn.moves.RedBlueMove` for one
                red/blue half). ``None`` means every walker. Default ``None``.

        Returns:
            :class:`eryn.state.State`: ``old_state``, updated.

        Raises:
            ValueError: ``new_state`` has blobs but ``old_state`` does not.
        """

        # TODO: update this to be use (tuples of inds) ??
        if subset is None:
            # subset of everything
            subset = np.tile(
                np.arange(old_state.log_like.shape[1]), (old_state.log_like.shape[0], 1)
            )

        # each computation is similar
        # 1. Take subset of values from old information (take_along_axis)
        # 2. Set new information
        # 3. Combine into a new temporary quantity based on accepted or not
        # 4. Put new combined subset back into full arrays (put_along_axis)

        # take_along_axis is necessary to do this all in higher dimensions
        accepted_temp = np.take_along_axis(accepted, subset, axis=1)

        # new log likelihood
        old_log_likes = np.take_along_axis(old_state.log_like, subset, axis=1)
        new_log_likes = new_state.log_like
        # np.where, NOT an arithmetic blend: a REJECTED proposal with
        # log_like = -inf gives (-inf) * False = NaN, which then poisons the
        # walker's stored log_like (coordinates unchanged) and makes any later
        # resume raise "The initial log_like was NaN". log_prior is already
        # guarded below; log_like was not. (CD1-L MBH runs, 2026-09-12.)
        temp_change_log_like = np.where(accepted_temp, new_log_likes, old_log_likes)

        np.put_along_axis(old_state.log_like, subset, temp_change_log_like, axis=1)

        # new log prior
        old_log_priors = np.take_along_axis(old_state.log_prior, subset, axis=1)
        new_log_priors = new_state.log_prior.copy()

        # deal with -infs
        new_log_priors[np.isinf(new_log_priors)] = 0.0

        temp_change_log_prior = new_log_priors * (accepted_temp) + old_log_priors * (
            ~accepted_temp
        )

        np.put_along_axis(old_state.log_prior, subset, temp_change_log_prior, axis=1)

        # inds
        old_inds = {
            name: np.take_along_axis(branch.inds, subset[:, :, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_inds = {name: branch.inds for name, branch in new_state.branches.items()}

        temp_change_inds = {
            name: new_inds[name] * (accepted_temp[:, :, None])
            + old_inds[name] * (~accepted_temp[:, :, None])
            for name in old_inds
        }

        [
            np.put_along_axis(
                old_state.branches[name].inds,
                subset[:, :, None],
                temp_change_inds[name],
                axis=1,
            )
            for name in new_inds
        ]

        # check for branches_supplemental
        run_branches_supplemental = False
        for name, value in old_state.branches_supplemental.items():
            if value is not None:
                run_branches_supplemental = True

        if run_branches_supplemental:
            # branch_supplemental
            temp_change_branch_supplemental = {}
            for name in old_state.branches:
                if old_state.branches[name].branch_supplemental is not None:
                    old_branch_supplemental = old_state.branches[
                        name
                    ].branch_supplemental.take_along_axis(
                        subset[:, :, None],
                        axis=1,
                        skip_names=self.skip_supp_names_update,
                    )
                    new_branch_supplemental = new_state.branches[
                        name
                    ].branch_supplemental[:]

                    tmp = {}
                    for key in old_branch_supplemental:
                        # need to check to see if we should skip anything
                        if key in self.skip_supp_names_update:
                            continue
                        accepted_temp_here = accepted_temp.copy()

                        # have adjust if it is an object array or a regular array
                        if new_branch_supplemental[key].dtype.name != "object":
                            for _ in range(
                                new_branch_supplemental[key].ndim
                                - accepted_temp_here.ndim
                            ):
                                accepted_temp_here = np.expand_dims(
                                    accepted_temp_here, (-1,)
                                )

                        # adjust for GPUs
                        try:
                            tmp[key] = new_branch_supplemental[key] * (
                                accepted_temp_here
                            ) + old_branch_supplemental[key] * (~accepted_temp_here)
                        except TypeError:
                            # for gpus
                            tmp[key] = new_branch_supplemental[key] * (
                                xp.asarray(accepted_temp_here)
                            ) + old_branch_supplemental[key] * (
                                xp.asarray(~accepted_temp_here)
                            )

                    temp_change_branch_supplemental[name] = BranchSupplemental(
                        tmp,
                        base_shape=new_state.branches_supplemental[name].base_shape,
                        copy=True,
                    )

                else:
                    temp_change_branch_supplemental[name] = None

            [
                old_state.branches[name].branch_supplemental.put_along_axis(
                    subset[:, :, None],
                    temp_change_branch_supplemental[name][:],
                    axis=1,
                )
                for name in new_inds
                if temp_change_branch_supplemental[name] is not None
            ]

        # sampler level supplemental
        if old_state.supplemental is not None:
            old_suppliment = old_state.supplemental.take_along_axis(subset, axis=1)
            new_suppliment = new_state.supplemental[:]

            accepted_temp_here = accepted_temp.copy()

            temp_change_suppliment = {}
            for name in old_suppliment:
                # make sure to get rid of specific supps if requested
                if name in self.skip_supp_names_update:
                    continue

                # adjust if it is not an object array
                if old_suppliment[name].dtype.name != "object":
                    for _ in range(old_suppliment[name].ndim - accepted_temp_here.ndim):
                        accepted_temp_here = np.expand_dims(accepted_temp_here, (-1,))
                try:
                    temp_change_suppliment[name] = new_suppliment[name] * (
                        accepted_temp_here
                    ) + old_suppliment[name] * (~accepted_temp_here)
                except TypeError:
                    temp_change_suppliment[name] = new_suppliment[name] * (
                        xp.asarray(accepted_temp_here)
                    ) + old_suppliment[name] * (xp.asarray(~accepted_temp_here))
            old_state.supplemental.put_along_axis(
                subset, temp_change_suppliment, axis=1
            )

        # coords
        old_coords = {
            name: np.take_along_axis(branch.coords, subset[:, :, None, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_coords = {
            name: branch.coords for name, branch in new_state.branches.items()
        }

        # change to copy then fill due to issue of adding Nans
        temp_change_coords = {name: old_coords[name].copy() for name in old_coords}

        for name in temp_change_coords:
            temp_change_coords[name][accepted_temp] = new_coords[name][accepted_temp]

        [
            np.put_along_axis(
                old_state.branches[name].coords,
                subset[:, :, None, None],
                temp_change_coords[name],
                axis=1,
            )
            for name in new_coords
        ]

        # take care of blobs
        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_like, "
                    "you also need to provide the current list of "
                    "blobs at that position."
                )

            old_blobs = np.take_along_axis(old_state.blobs, subset[:, :, None], axis=1)
            new_blobs = new_state.blobs
            temp_change_blobs = new_blobs * (accepted_temp[:, :, None]) + old_blobs * (
                ~accepted_temp[:, :, None]
            )

            np.put_along_axis(
                old_state.blobs, subset[:, :, None], temp_change_blobs, axis=1
            )

        return old_state
