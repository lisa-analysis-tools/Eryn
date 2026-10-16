# -*- coding: utf-8 -*-

import warnings

import numpy as np

from ..state import ParaState
from ..utils.utility import (
    psrf,
    stepping_stone_log_evidence,
    thermodynamic_integration_log_evidence,
)

__all__ = ["ParaBackend"]


class ParaBackend(object):
    """A simple default backend that stores the chain in memory

    Args:
        store_missing_leaves (double, optional): Number to store for leaves that are not
            used in a specific step. (default: ``np.nan``)
        dtype (dtype, optional): Dtype to use for data storage. If None,
            program uses np.float64. (default: ``None``)

    Attributes:
        accepted (2D int np.ndarray): Number of accepted steps for within-model moves.
            The shape is (ntemps, nwalkers).
        betas (2D double np.ndarray): Inverse temperature latter at each step.
            Shape is (nsteps, ntemps). This keeps track of adjustable temperatures.
        blobs (4D double np.ndarray): Stores extra blob information returned from
            likelihood function. Shape is (nsteps, ntemps, nwalkers, nblobs).
        branch_names (list of str): List of branch names.
        chain (dict): Dictionary with branch_names as keys. The values are
            5D double np.ndarray arrays with shape (nsteps, ntemps, nwalkers, nleaves_max, ndim).
            These are the locations of walkers over the MCMC run.
        dtype (dtype): Dtype to use for data storage.
        inds (dict): Keys are branch_names. Values are 4D bool np.ndarray
            of shape (nsteps, ntemps, nwalkers, nleaves_max). This array details which
            leaves are used in the current step. This is really only
            relevant for reversible jump.
        initiailized (bool): If ``True``, backend object has been initialized.
        iteration (int): Current index within the data storage arrays.
        log_prior (3D double np.ndarray): Log of the prior values. Shape is
            (nsteps, nwalkers, ntemps).
        log_like (3D double np.ndarray): Log of the likelihood values. Shape is
            (nsteps, nwalkers, ntemps).
        move_info (dict): Dictionary containing move info.
        move_keys (list): List of keys for ``move_info``.
        nbranches (int): Number of branches.
        ndims (dict): Dimensionality of each branch.
        nleaves_max (dict): Maximum allowable leaves for each branch.
        nwalkers (int): The size of the ensemble (per temperature).
        ntemps (int): Number of rungs in the temperature ladder.
        reset_args (tuple): Arguments to reset backend.
        reset_kwargs (dict): Keyword arguments to reset backend.
        rj (bool): If True, reversible-jump techniques are used.
        rj_accepted (2D int np.ndarray): Number of accepted steps for between-model moves.
            The shape is (ntemps, nwalkers).
        store_missing_leaves (double): Number to store for leaves that are not
            used in a specific step.

    """

    def __init__(self, store_missing_leaves=np.nan, dtype=None):
        warnings.warn(
            "ParaBackend is deprecated; use eryn.backends.Backend with "
            "reset(..., nsamplers=...) instead (storage always carries the "
            "sampler axis). It will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    def reset_base(self):
        """Allows for simple reset based on previous inputs"""
        self.reset(*self.reset_args, **self.reset_kwargs)

    def reset(
        self,
        ndim: int,
        nwalkers: int,
        ngroups: int,
        ntemps: int = 1,
        branch_name: str = "model_0",
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble (per temperature).
            ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
            nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
                It should have the same length as the number of branches.
                If ``dict``, keys should be the branch names and values the associated maximal leaf value.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            nbranches (int, optional): Number of branches. This is only used if ``branch_names is None``.
                (default: ``1``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            moves (list, optional): List of all of the move classes input into the sampler.
                (default: ``None``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend.

        """
        # store inputs for later resets
        self.reset_args = (ndim, nwalkers, ngroups)
        self.reset_kwargs = dict(
            ntemps=ntemps,
            branch_name=branch_name,
            **info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.ngroups = int(ngroups)

        self.branch_name = branch_name
        self.ndim = ndim

        self.iteration = 0

        # setup all the holder arrays
        self.accepted = np.zeros((self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.swaps_accepted = np.zeros(
            (
                self.ngroups,
                self.ntemps - 1,
            ),
            dtype=self.dtype,
        )

        # chains are stored in dictionaries
        self.chain = {
            self.branch_name: np.empty(
                (0, self.ngroups, self.ntemps, self.nwalkers, self.ndim),
                dtype=self.dtype,
            )
        }

        # inds correspond to leaves used or not
        self.groups_running = {self.branch_name: np.empty((0, self.ngroups), dtype=bool)}

        # log likelihood and prior
        self.log_like = np.empty((0, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)

        # temperature ladder
        self.betas = np.empty((0, self.ngroups, self.ntemps), dtype=self.dtype)

        self.random_state = None
        self.initialized = True

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): Ignored for non-HDFBackend.

        Returns:
            dict or np.ndarray: Values requested.

        """
        if slice_vals is not None:
            raise ValueError("slice_vals can only be used with an HDF Backend.")

        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )

        # prepare chain for output
        if name == "chain":
            v_all = self.chain[self.branch_name][discard + thin - 1 : self.iteration : thin]

            return v_all

        # prepare inds for output
        if name == "groups_running":
            v_all = self.groups_running[self.branch_name][
                discard + thin - 1 : self.iteration : thin
            ]

            return v_all

        # all other requests can filter through array output
        # rather than the dictionary output used above
        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        return v

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            dict: MCMC samples
                The dictionary contains np.ndarrays of samples
                across the branches.

        """
        return self.get_value("chain", **kwargs)

    def get_groups_running(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            dict: The ``inds`` associated with the MCMC samples.
                  The dictionary contains np.ndarrays of ``inds``
                  across the branches indicated which leaves were used at each step.

        """
        return self.get_value("groups_running", **kwargs)

    def get_log_like(self, **kwargs):
        """Get the chain of log Likelihood values evaluated at the MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log likelihood values.

        """
        return self.get_value("log_like", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the chain of log Prior evaluated at the MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log prior values.

        """
        return self.get_value("log_prior", **kwargs)

    def get_log_posterior(self, temper: bool = False, **kwargs):
        """Get the chain of log posterior values evaluated at the MCMC samples

        Args:
            temper (bool, optional): Apply tempering to the posterior values.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log prior values.

        """
        if temper:
            betas = self.get_betas(**kwargs)

        else:
            betas = np.ones_like(self.get_betas(**kwargs))

        log_like = self.get_log_like(**kwargs)
        log_prior = self.get_log_prior(**kwargs)

        return betas[:, :, None] * log_like + log_prior

    def get_betas(self, **kwargs):
        """Get the chain of inverse temperatures

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps]: The chain of temperatures.


        """
        return self.get_value("betas", **kwargs)

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of ParaState to return.

        Returns:
            ParaState: :class:`eryn.state.ParaState` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        # fill a ParaState with quantities from sample ``it`` in the chain;
        # each getter returns one step, so take index 0 to drop the step axis
        sample = ParaState(
            {self.branch_name: self.get_chain(discard=discard, thin=thin)[0]},
            log_like=self.get_log_like(discard=discard, thin=thin)[0],
            log_prior=self.get_log_prior(discard=discard, thin=thin)[0],
            groups_running=self.get_groups_running(discard=discard, thin=thin)[0],
            betas=self.get_betas(discard=discard, thin=thin)[0],
            random_state=self.random_state,
        )
        return sample

    def get_last_sample(self):
        """Access the most recent sample in the chain

        Returns:
            ParaState: :class:`eryn.state.ParaState` object containing the last sample from the chain.

        """
        it = self.iteration - 1

        # get the state from the last iteration
        last_sample = self.get_a_sample(it)
        return last_sample

    def get_evidence_estimate(
        self, discard=0, thin=1, return_error=True, method="thermodynamic", **ss_kwargs
    ):
        """Get an estimate of the evidence

        This function gets the sample information and uses
        :func:`thermodynamic_integration_log_evidence` or
        :func:`stepping_stone_log_evidence` to compute the evidence estimate.

        Args:
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            return_error (bool, optional): If True, return the error associated
                with the log evidence estimate. (default: ``True``)
            method (string, optional): Method to compute the evidence. Available
                methods are the 'thermodynamic' and 'stepping-stone' (default: ``thermodynamic``)

        Returns:
            double or tuple: Evidence estimate
                If requesting the error on the estimate, will receive a tuple:
                ``(logZ, dlogZ)``. Otherwise, just a double value of logZ.

        """
        # get all the likelihood and temperature values
        logls_all = self.get_log_like(discard=discard, thin=thin)
        betas_all = self.get_betas(discard=discard, thin=thin)

        # make sure that the betas were fixed during sampling (after burn in)
        if not (betas_all == betas_all[0]).all():
            raise ValueError(
                "Cannot compute evidence estimation if betas are allowed to vary. "
                "Use stop_adaptation kwarg in temperature settings."
            )

        # setup information
        betas = betas_all[0]

        # get log evidence and error
        if method.lower() in [
            "thermodynamic",
            "thermodynamic integration",
            "thermo",
            "ti",
        ]:
            logls = np.mean(logls_all, axis=(0, -1))
            logZ, dlogZ = thermodynamic_integration_log_evidence(betas, logls)
        elif method.lower() in [
            "stepping stone",
            "ss",
            "step",
            "stone",
            "stepping-stone",
        ]:
            logZ, dlogZ = stepping_stone_log_evidence(betas, logls_all, **ss_kwargs)
        else:
            raise ValueError(
                """Please choose only between 'thermodynamic' and 'stepping-stone' methods."""
            )

        if return_error:
            return (logZ, dlogZ)
        else:
            return logZ

    def get_gelman_rubin_convergence_diagnostic(
        self, discard=0, thin=1, doprint=True, **psrf_kwargs
    ):
        """
        The Gelman - Rubin convergence diagnostic.
        A general approach to monitoring convergence of MCMC output of multiple walkers.
        The function makes a comparison of within-chain and between-chain variances.
        A large deviation between these two variances indicates non-convergence, and
        the output [Rhat] deviates from unity.

        Based on
        a. Brooks, SP. and Gelman, A. (1998) General methods for monitoring convergence
        of iterative simulations. Journal of Computational and Graphical Statistics, 7, 434-455
        b. Gelman, A and Rubin, DB (1992) Inference from iterative simulation using multiple sequences,
        Statistical Science, 7, 457-511.

        Args:
            C (np.ndarray[nwalkers, nsamples, ndim]): The parameter traces. The MCMC chains.
            doprint (bool, optional): Flag to print the results on screen.
        discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
        doprint (bool, optional): Flag to print a table with the results, per temperature.

        Returns
            dict:   ``Rhat_all_branches``:
                Returns an estimate of the Gelman-Rubin convergence diagnostic ``Rhat``,
                per temperature, stored in a dictionary, per branch name.

        """
        Rhat_all_branches = dict()
        # Loop over the different models
        for branch in self.branch_names:
            Rhat = dict()  # Initialize
            # Loop over the temperatures
            for temp in range(self.ntemps):
                # Get all the chains per branch
                chains = self.get_chain(discard=discard, thin=thin)[branch][:, temp]

                # Handle the cases of multiple leaves on a given branch
                if chains.shape[2] == 1:
                    # If no multiple leaves, we squeeze and transpose to the
                    # right shape to pass to the psrf function, which is  (nwalkers, nsamples, ndim)
                    chains_in = chains.squeeze().transpose((1, 0, 2))
                else:
                    # Project onto the model dim all chains [in case of RJ and multiple leaves per branch]
                    inds = self.get_inds(discard=discard, thin=thin)[branch][
                        :, temp
                    ]  # [t, w, nleavesmax, dim]
                    min_leaves = inds.sum(axis=(0, 2)).min()
                    tmp = [inds[:, w].flatten() for w in range(self.nwalkers)]
                    keep = [np.where(tmp[w])[0][:min_leaves] for w in range(len(tmp))]
                    chains_in = np.asarray(
                        [
                            chains[:, w].reshape(-1, self.ndims[branch])[keep[w]]
                            for w in range(self.nwalkers)
                        ]
                    )

                Rhat[temp] = psrf(chains_in, self.ndims[branch], **psrf_kwargs)
            Rhat_all_branches[branch] = Rhat  # Store the Rhat per branch

        if doprint:  # Print table of results
            print("  Gelman-Rubin diagnostic \n  <R̂>: Mean value for all parameters\n")
            print("  --------------")
            for branch in self.branch_names:
                print(f" Model: {branch}")
                print("   T \t <R̂>")
                print("  --------------")
                for temp in range(self.ntemps):
                    print(f"   {temp:01d}\t{np.mean(Rhat_all_branches[branch][temp]):3.2f}")
                print("\n")

        return Rhat_all_branches

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            tuple: ``(ngroups, ntemps, nwalkers, ndim)``.

        """
        return (self.ngroups, self.ntemps, self.nwalkers, self.ndim)

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """

        # determine the number of entries in the chains
        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)

        # temperary addition to chains
        a = {
            self.branch_name: np.empty(
                (i, self.ngroups, self.ntemps, self.nwalkers, self.ndim),
                dtype=self.dtype,
            )
        }
        # combine with original chain
        self.chain = {
            self.branch_name: np.concatenate(
                (self.chain[self.branch_name], a[self.branch_name]), axis=0
            )
        }

        # temperorary addition to groups_running
        a = {self.branch_name: np.empty((i, self.ngroups), dtype=bool)}
        # combine with original groups_running
        self.groups_running = {
            self.branch_name: np.concatenate(
                (self.groups_running[self.branch_name], a[self.branch_name]), axis=0
            )
        }

        # temperorary addition for log likelihood
        a = np.empty((i, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log likelihood
        self.log_like = np.concatenate((self.log_like, a), axis=0)

        # temperorary addition for log prior
        a = np.empty((i, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log prior
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)

        # temperorary addition for betas
        a = np.empty((i, self.ngroups, self.ntemps), dtype=self.dtype)
        # combine with original betas
        self.betas = np.concatenate((self.betas, a), axis=0)

    def _check(
        self,
        state,
        accepted,
        swaps_accepted=None,
    ):
        """Check all the information going in is okay."""

        shape = self.shape

        ngroups, ntemps, nwalkers = self.ngroups, self.ntemps, self.nwalkers

        # make sure all of the coordinate and inds dimensions are okay
        ngroup1, ntemp1, nwalker1, ndim1 = state.branches[self.branch_name].shape
        ngroup2, ntemp2, nwalker2, ndim2 = shape

        if (ngroup1, ntemp1, nwalker1, ndim1) != (
            ngroup2,
            ntemp2,
            nwalker2,
            ndim2,
        ):
            raise ValueError(
                f"invalid coordinate dimensions for model {self.branch_name} with shape {state.branches[self.branch_name].shape}; expected {shape}"
            )

        if (ngroup1,) != state.groups_running.shape:
            raise ValueError(
                f"invalid inds dimensions for model {self.branch_name} with shape {state.groups_running.shape}; expected {(ngroup1,)}"
            )

        # make sure log likelihood, log prior, blobs, accepted, rj_accepted, betas are okay
        if state.log_like.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(
                f"invalid log probability size; expected {(ngroups, ntemps, nwalkers)}"
            )
        if state.log_prior.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(f"invalid log prior size; expected {(ngroups, ntemps, nwalkers)}")

        if accepted.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(f"invalid acceptance size; expected {(ngroups, ntemps, nwalkers)}")

        if swaps_accepted is not None and swaps_accepted.shape != (
            ngroups,
            ntemps - 1,
        ):
            raise ValueError(f"invalid swaps_accepted size; expected {(ngroups, ntemps - 1)}")

        if state.betas is not None and state.betas.shape != (
            ngroups,
            ntemps,
        ):
            raise ValueError(f"invalid beta size; expected {(ngroups, ntemps)}")

    def save_step(
        self,
        state,
        accepted,
        swaps_accepted=None,
    ):
        """Save a step to the backend

        Args:
            state (ParaState): The :class:`ParaState` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            rj_accepted (ndarray, optional): An array of the number of accepted steps
                for the reversible jump proposal for each walker.
                If :code:`self.rj` is True, then rj_accepted must be an array with
                :code:`rj_accepted.shape == accepted.shape`. If :code:`self.rj`
                is False, then rj_accepted must be None, which is the default.
            swaps_accepted (ndarray, optional): 1D array with number of swaps accepted
                for the in-model step. (default: ``None``)
            moves_accepted_fraction (dict, optional): Dict of acceptance fraction arrays for all of the
                moves in the sampler. This dict must have the same keys as ``self.move_keys``.
                (default: ``None``)

        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            swaps_accepted=swaps_accepted,
        )

        try:
            # save the coordinates and groups_running
            self.groups_running[self.branch_name][self.iteration] = state.groups_running.get()
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = (
                state.branches[self.branch_name].coords * state.groups_running[:, None, None, None]
            ).get()

            coords_in[~state.groups_running.get()] = self.store_missing_leaves
            self.chain[self.branch_name][self.iteration] = coords_in

            # save higher level quantities
            self.log_like[self.iteration, :, :] = state.log_like.get()
            self.log_prior[self.iteration, :, :] = state.log_prior.get()

            if state.betas is not None:
                self.betas[self.iteration, :] = state.betas.get()

            self.accepted += accepted.get()

            if swaps_accepted is not None:
                self.swaps_accepted += swaps_accepted.get()

        except AttributeError:
            # save the coordinates and groups_running
            self.groups_running[self.branch_name][self.iteration] = state.groups_running
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = (
                state.branches[self.branch_name].coords * state.groups_running[:, None, None, None]
            )

            coords_in[~state.groups_running] = self.store_missing_leaves
            self.chain[self.branch_name][self.iteration] = coords_in

            # save higher level quantities
            self.log_like[self.iteration, :, :] = state.log_like
            self.log_prior[self.iteration, :, :] = state.log_prior

            if state.betas is not None:
                self.betas[self.iteration, :] = state.betas

            self.accepted += accepted

            if swaps_accepted is not None:
                self.swaps_accepted += swaps_accepted

        self.random_state = state.random_state
        self.iteration += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
