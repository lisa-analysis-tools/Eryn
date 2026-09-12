# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

__all__ = ["Move"]


class Move(object):
    """Parent class for proposals or "moves"

    Args:
        temperature_control (:class:`tempering.TemperatureControl`, optional):
            This object controls the tempering. It is passed to the parent class
            to moves so that all proposals can share and use temperature settings.
            (default: ``None``)
        periodic (:class:`eryn.utils.PeriodicContainer, optional):
            This object holds periodic information and methods for periodic parameters. It is passed to the parent class
            to moves so that all proposals can share and use periodic information.
            (default: ``None``)
        gibbs_sampling_setup (str, tuple, dict, or list, optional): This sets the Gibbs Sampling setup if
            desired. The Gibbs sampling setup is completely customizable down to the leaf and parameters.
            All of the separate Gibbs sampling splits will be run within 1 call to this proposal.
            If ``None``, run all branches and all parameters. If ``str``, run all parameters within the
            branch given as the string. To enter a branch with a specific set of parameters, you can
            provide a 2-tuple with the first entry as the branch name and the second entry as a 2D
            boolean array of shape ``(nleaves_max, ndim)`` that indicates which leaves and/or parameters
            you want to run. ``None`` can also be entered in the second entry if all parameters are to be run.
            A dictionary is also possible with keys as branch names and values as the same 2D boolean array
            of shape ``(nleaves_max, ndim)`` that indicates which leaves and/or parameters
            you want to run. ``None`` can also be entered in the value of the dictionary
            if all parameters are to be run. If multiple keys are provided in the dictionary, those
            branches will be run simultaneously in the proposal as one iteration of the proposing loop.
            The final option is a list. This is how you make sure to run all the Gibbs splits. Each entry
            of the list can be a string, 2-tuple, or dictionary as described above. The list controls
            the order in which all of these splits are run. (default: ``None``)
        prevent_swaps (bool, optional): If ``True``, do not perform temperature swaps in this move.
        skip_supp_names_update (list, optional): List of names (`str`), that can be in any
            :class:`eryn.state.BranchSupplemental`,
            to skip when updating states (:func:`Move.update`). This is useful if a
            large amount of memory is stored in the branch supplementals.
        is_rj (bool, optional): If using RJ, this should be ``True``. (default: ``False``)
        use_gpu (bool, optional): If ``True``, use ``CuPy`` for computations.
            Use ``NumPy`` if ``use_gpu == False``. (default: ``False``)
        random_seed (int, optional): Set the random seed in ``CuPy/NumPy`` if not ``None``.
            (default: ``None``)

    Raises:
        ValueError: Incorrect inputs.

    Attributes:
        Note: All kwargs are stored as attributes.
        num_proposals (int): the number of times this move has been run. This is needed to
            compute the acceptance fraction.
        gibbs_sampling_setup (list): All of the Gibbs sampling splits as described above.
        xp (obj): ``NumPy`` or ``CuPy``.
        use_gpu (bool): Whether ``Cupy`` (``True``) is used or not (``False``).

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
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu):
        self._use_gpu = use_gpu

    @property
    def xp(self):
        if self._use_gpu is None:
            raise ValueError("use_gpu has not been set.")
        xp = cp if self.use_gpu else np
        return xp

    def _initialize_branch_setup(self, gibbs_sampling_setup, is_rj=False):
        """Initialize the gibbs setup properly."""
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
        """Iterate through the gibbs splits as a generator

        Args:
            all_branch_names (list): List of all branch names.

        Yields:
            2-tuple: Gibbs sampling split.
                        First entry is the branch names to run and the second entry is the index
                        into the leaves/parameters for this Gibbs split.

        Raises:
            ValueError: Incorrect inputs.

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
        """Setup proposals when gibbs sampling.

        Get inputs into the proposal including Gibbs split information.

        Args:
            branch_names_run (list): List of branch names to run concurrently.
            inds_run (list): List of ``inds`` arrays including Gibbs sampling information.
            branches_coords (dict): Dictionary of coordinate arrays for all branches.
            branches_inds (dict): Dictionary of ``inds`` arrays for all branches.

        Returns:
            tuple:  (coords, inds, at_least_one_proposal)
                        * Coords including Gibbs sampling info.
                        * ``inds`` including Gibbs sampling info.
                        * ``at_least_one_proposal`` is boolean. It is passed out to
                            indicate there is at least one leaf available for the requested branch names.

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
        """Set all not Gibbs-sampled parameters back

        Args:
            branch_names_run (list): List of branch names to run concurrently.
            inds_run (list): List of ``inds`` arrays including Gibbs sampling information.
            q (dict): Dictionary of new coordinate arrays for all proposal branches.
            branches_coords (dict): Dictionary of old coordinate arrays for all branches.
            new_inds (dict, optional): Dictionary of new inds arrays for all proposal branches.
            branches_inds (dict, optional): Dictionary of old inds arrays for all branches.
            new_branch_supps (dict, optional): Dictionary of new branches supplemental for all proposal branches.
            branches_supplemental (dict, optional): Dictionary of old branches supplemental for all branches.

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
        """Ensure proper order of key in dictionaries.

        Args:
            correct_key_order (list): Keys in correct order.
            q (dict): Dictionary of new coordinate arrays for all branches.
            new_inds (dict): Dictionary of new inds arrays for all branches.
            new_branch_supps (dict or None): Dictionary of new branches supplemental for all proposal branches.

        Returns:
            Tuple: (q, new_inds, new_branch_supps) in correct key order.

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
        """Set any walker with no leaves to have logp = -np.inf

        Args:
            branch_names_run (list): List of branch names to run concurrently.
            inds_run (list): List of ``inds`` arrays including Gibbs sampling information.
            logp (np.ndarray): Log of the prior going into final posterior computation.
            inds (dict): Dictionary of ``inds`` arrays for all branches.

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
        """Accepted counts for this move."""
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
        """Acceptance fraction for this move."""
        return self.accepted / self.num_proposals

    # independent-samplers defaults (overwritten when a sampler-aware
    # TemperatureControl is attached, or set directly by EnsembleSampler)
    nsamplers = 1
    ntemps_per_sampler = None
    sampler_id_rows = None

    @property
    def temperature_control(self):
        """Temperature controller"""
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
        r"""Compute the log of posterior

        :math:`\log{P} = \log{L} + \log{p}`

        This method is to mesh with the tempered log posterior computation.

        Args:
            logl (np.ndarray[ntemps, nwalkers]): Log-likelihood values.
            logp (np.ndarray[ntemps, nwalkers]): Log-prior values.

        Returns:
            np.ndarray[ntemps, nwalkers]: Log-Posterior values.
        """
        return logl + logp

    def tune(self, state, accepted):
        """Tune a proposal

        This is a place holder for tuning.

        Args:
            state (:class:`eryn.state.State`): Current state of sampler.
            accepted (np.ndarray[ntemps, nwalkers]): Accepted values for last pass
                through proposal.

        """
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        This class was updated from ``emcee`` to handle the added structure
        of Eryn.

        Args:
            old_state (:class:`eryn.state.State`): State with current information.
                New information is added to this state.
            new_state (:class:`eryn.state.State`): State with information from proposed
                points.
            accepted (np.ndarray[ntemps, nwalkers]): A vector of booleans indicating
                which walkers were accepted.
            subset (np.ndarray[ntemps, nwalkers], optional): A boolean mask
                indicating which walkers were included in the subset.
                This can be used, for example, when updating only the
                primary ensemble in a :class:`RedBlueMove`.
                (default: ``None``)

        Returns:
            :class:`eryn.state.State`: ``old_state`` with accepted points added from ``new_state``.

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
