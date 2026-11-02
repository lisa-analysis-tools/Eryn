# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HDFBackend", "TempHDFBackend", "does_hdf5_support_longdouble"]

import os
import time
from tempfile import NamedTemporaryFile

import numpy as np

from .. import __version__
from .backend import Backend

try:
    import h5py
except ImportError:
    h5py = None


def does_hdf5_support_longdouble():
    if h5py is None:
        return False
    with NamedTemporaryFile(
        prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
    ) as f:
        f.close()

        with h5py.File(f.name, "w") as hf:
            g = hf.create_group("group")
            g.create_dataset("data", data=np.ones(1, dtype=np.longdouble))
            if g["data"].dtype != np.longdouble:
                return False
        with h5py.File(f.name, "r") as hf:
            if hf["group"]["data"].dtype != np.longdouble:
                return False
    return True


class HDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str, optional): The name of the group where the chain will
            be saved. (default: ``"mcmc"``)
        read_only (bool, optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.
            (default: ``False``)
        dtype (dtype, optional): Dtype to use for data storage. If None,
            program uses np.float64. (default: ``None``)
        compression (str, optional): Compression type for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        compression_opts (int, optional): Compression level for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        store_missing_leaves (double, optional): Number to store for leaves that are not
            used in a specific step. (default: ``np.nan``)


    """

    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
        store_missing_leaves=np.nan,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")

        # store all necessary quantities
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    @property
    def initialized(self):
        """Check if backend file has been initialized properly."""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        """Opens the h5 file in the proper mode.

        Args:
            mode (str, optional): Mode to open h5 file.

        Returns:
            H5 file object: Opened file.

        Raises:
            RuntimeError: If backend is opened for writing when it is read-only.

        """

        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )

        # open the file
        file_opened = False

        try_num = 0
        max_tries = 100
        while not file_opened:
            try:
                f = h5py.File(self.filename, mode)
                file_opened = True
                
            except (BlockingIOError, OSError) as e:
                try_num += 1
                # Only retry on a true lock contention; bail on file-not-found,
                # bad-mode, etc. so we surface the real error instead of looping
                # for ~16 min and swallowing it.
                msg = str(e)
                lock_like = isinstance(e, BlockingIOError) or (
                    "unable to lock" in msg.lower()
                    or "resource temporarily unavailable" in msg.lower()
                )
                if not lock_like or try_num >= max_tries:
                    raise
                print(
                    f"h5 open(mode={mode!r}) lock contention (try {try_num}/{max_tries}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(1.0)

        # get the data type and store it if it is not previously set
        if not self.dtype_set and self.name in f:
            # get the group from the file
            g = f[self.name]
            if "chain" in g:
                # get the model names in chain
                keys = list(g["chain"])

                # they all have the same dtype so use the first one
                try:
                    self.dtype = g["chain"][keys[0]].dtype

                    # we now have it
                    self.dtype_set = True
                # catch error if the chain has not been initialized yet
                except IndexError:
                    pass

        return f

    def reset(
        self,
        nwalkers,
        ndims,
        nleaves_max=1,
        ntemps=1,
        branch_names=None,
        nbranches=1,
        rj=False,
        moves=None,
        key_order=None,
        nsamplers=1,
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble
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
            key_order (dict, optional): Keys are ``branch_names`` and values are lists of key ordering for each
                branch. For example, ``{"model_0": ["x1", "x2", "x3"]}``.
                (default: ``None``)
            nsamplers (int, optional): Number of independent samplers run simultaneously.
                New files always store datasets with a sampler axis directly after the
                step axis; getters squeeze it when ``nsamplers == 1``. Files written
                before this axis existed (no ``nsamplers`` attr) are read and written
                in the legacy layout. (default: ``1``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend. These are also added to the HDF5 file.

        """

        # open file in append mode
        with self.open("a") as f:
            # we are resetting so if self.name in the file we need to delete it
            if self.name in f:
                del f[self.name]

            # turn things into lists/dicts if needed
            if branch_names is not None:
                if isinstance(branch_names, str):
                    branch_names = [branch_names]

                elif not isinstance(branch_names, list):
                    raise ValueError("branch_names must be string or list of strings.")

            else:
                branch_names = [f"model_{i}" for i in range(nbranches)]

            nbranches = len(branch_names)

            if isinstance(ndims, int):
                assert len(branch_names) == 1
                ndims = {branch_names[0]: ndims}

            elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
                assert len(branch_names) == len(ndims)
                ndims = {bn: nd for bn, nd in zip(branch_names, ndims)}

            elif isinstance(ndims, dict):
                assert len(list(ndims.keys())) == len(branch_names)
                for key in ndims:
                    if key not in branch_names:
                        raise ValueError(
                            f"{key} is in ndims but does not appear in branch_names: {branch_names}."
                        )
            else:
                raise ValueError("ndims is to be a scalar int, list or dict.")

            if isinstance(nleaves_max, int):
                assert len(branch_names) == 1
                nleaves_max = {branch_names[0]: nleaves_max}

            elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
                assert len(branch_names) == len(nleaves_max)
                nleaves_max = {bn: nl for bn, nl in zip(branch_names, nleaves_max)}

            elif isinstance(nleaves_max, dict):
                assert len(list(nleaves_max.keys())) == len(branch_names)
                for key in nleaves_max:
                    if key not in branch_names:
                        raise ValueError(
                            f"{key} is in nleaves_max but does not appear in branch_names: {branch_names}."
                        )
            else:
                raise ValueError("nleaves_max is to be a scalar int, list, or dict.")

            # store all the info needed in memory and in the file

            g = f.create_group(self.name)

            g.attrs["version"] = __version__
            g.attrs["nbranches"] = len(branch_names)
            g.attrs["branch_names"] = branch_names
            g.attrs["ntemps"] = ntemps
            g.attrs["nwalkers"] = nwalkers
            g.attrs["nsamplers"] = nsamplers
            g.attrs["has_blobs"] = False
            g.attrs["rj"] = rj
            g.attrs["iteration"] = 0

            # create info group
            g.create_group("info")
            # load info into class and into file
            for key, value in info.items():
                setattr(self, key, value)
                if isinstance(value, np.ndarray) and value.size > 1000:
                    g["info"].create_dataset(key, data=value)
                else:
                    g["info"].attrs[key] = value

            # store nleaves max and ndims dicts
            g.create_group("ndims")
            for key, value in ndims.items():
                g["ndims"].attrs[key] = value

            g.create_group("nleaves_max")
            for key, value in nleaves_max.items():
                g["nleaves_max"].attrs[key] = value

            # prepare all the data sets
            # all step-indexed datasets carry the sampler axis directly after
            # the step axis; cumulative counters carry it as the leading axis

            g.create_dataset(
                "accepted",
                data=np.zeros((nsamplers, ntemps, nwalkers)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "swaps_accepted",
                data=np.zeros((nsamplers, ntemps - 1)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            if self.rj:
                g.create_dataset(
                    "rj_accepted",
                    data=np.zeros((nsamplers, ntemps, nwalkers)),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            g.create_dataset(
                "log_like",
                (0, nsamplers, ntemps, nwalkers),
                maxshape=(None, nsamplers, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "log_prior",
                (0, nsamplers, ntemps, nwalkers),
                maxshape=(None, nsamplers, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "betas",
                (0, nsamplers, ntemps),
                maxshape=(None, nsamplers, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "samplers_running",
                (0, nsamplers),
                maxshape=(None, nsamplers),
                dtype=bool,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # setup data sets for branch-specific items

            chain = g.create_group("chain")
            inds = g.create_group("inds")
            k_o_g = g.create_group("key_order")

            for name in branch_names:
                nleaves = self.nleaves_max[name]
                ndim = self.ndims[name]
                chain.create_dataset(
                    name,
                    (0, nsamplers, ntemps, nwalkers, nleaves, ndim),
                    maxshape=(None, nsamplers, ntemps, nwalkers, nleaves, ndim),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                inds.create_dataset(
                    name,
                    (0, nsamplers, ntemps, nwalkers, nleaves),
                    maxshape=(None, nsamplers, ntemps, nwalkers, nleaves),
                    dtype=bool,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                if key_order is not None:
                    k_o_g.attrs[name] = key_order[name]

            # store move specific information
            if moves is not None:
                move_group = g.create_group("moves")
                # setup info and keys
                for full_move_name in moves:

                    single_move = move_group.create_group(full_move_name)

                    # prepare information dictionary
                    # stored FOLDED (nsamplers * ntemps, nwalkers): identical to
                    # the legacy layout when nsamplers == 1
                    single_move.create_dataset(
                        "acceptance_fraction",
                        (nsamplers * ntemps, nwalkers),
                        maxshape=(nsamplers * ntemps, nwalkers),
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )

            else:
                self.move_info = None

            self.blobs = None

    @property
    def key_order(self):
        """Key order of parameters for each model."""
        with self.open() as f:
            return {key: list(value) for key, value in f[self.name]["key_order"].attrs.items()}

    @property
    def nwalkers(self):
        """Get nwalkers from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["nwalkers"]

    @property
    def ntemps(self):
        """Get ntemps from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["ntemps"]

    @property
    def nsamplers(self):
        """Get nsamplers from h5 file (1 for legacy files without the attr)."""
        with self.open() as f:
            return int(f[self.name].attrs.get("nsamplers", 1))

    @property
    def _has_sampler_axis(self):
        """Whether the file's datasets carry the sampler axis (False for legacy files)."""
        with self.open() as f:
            return "nsamplers" in f[self.name].attrs

    @property
    def rj(self):
        """Get rj from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["rj"]

    @property
    def nleaves_max(self):
        """Get nleaves_max from h5 file."""
        with self.open() as f:
            return {
                key: f[self.name]["nleaves_max"].attrs[key]
                for key in f[self.name]["nleaves_max"].attrs
            }

    @property
    def ndims(self):
        """Get ndims from h5 file."""
        with self.open() as f:
            return {
                key: f[self.name]["ndims"].attrs[key]
                for key in f[self.name]["ndims"].attrs
            }

    @property
    def move_keys(self):
        """Get move_keys from h5 file."""
        with self.open() as f:
            return list(f[self.name]["moves"])

    @property
    def branch_names(self):
        """Get branch names from h5 file."""
        with self.open() as f:
            return list(f[self.name].attrs["branch_names"])

    @property
    def nbranches(self):
        """Get number of branches from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["nbranches"]

    @property
    def reset_args(self):
        """Get reset_args from h5 file."""
        return [self.nwalkers, self.ndims]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            nleaves_max=self.nleaves_max,
            ntemps=self.ntemps,
            branch_names=self.branch_names,
            rj=self.rj,
            moves=self.moves if hasattr(self, 'moves') else None,
            key_order=self.key_order,
            nsamplers=self.nsamplers,
        )

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, thin=1, discard=0, slice_vals=None, temp_index=None, sampler_index=None, branch_names=None):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): If provided, slice the array directly
                from the HDF5 file with slice = ``slice_vals``. ``thin`` and ``discard`` will be
                ignored if slice_vals is not ``None``. This is particularly useful if files are
                very large and the user only wants a small subset of the overall array.
                (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)
            sampler_index (int, optional): Integer for the desired independent-sampler
                index. If ``None``, returns all samplers (the sampler axis is squeezed
                away entirely when ``nsamplers == 1``). (default: ``None``)
            branch_names (str or list, optional): Specific branch names requested. (default: ``None``)

        Returns:
            dict or np.ndarray: Values requested.

        """
        # check if initialized
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results."
                "When using the HDF backend, make sure you have the file"
                "path correctly set. This is the error that"
                "is given if the backend cannot find the file."
            )

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        # make sure branch_names input is a list
        if branch_names is not None:
            if isinstance(branch_names, str):
                branches_names = [branch_names]

        branch_names_in = self.branch_names if branch_names is None else branch_names

        # legacy files (no sampler axis in the datasets) keep the original
        # indexing path; new files go through the sampler-aware indexer
        has_sampler_axis = self._has_sampler_axis

        successful = False
        num_try = 0
        while not successful and num_try < 100:
            try:
                # open the file wrapped in a "with" statement
                with self.open() as f:
                    # get the group that everything is stored in
                    g = f[self.name]
                    iteration = g.attrs["iteration"]
                    if iteration <= 0:
                        raise AttributeError(
                            "You must run the sampler with "
                            "'store == True' before accessing the "
                            "results"
                        )

                    if has_sampler_axis:
                        indexer = self._step_indexer(
                            slice_vals, temp_index, sampler_index
                        )

                        if name == "chain":
                            v_all = {key: g["chain"][key][indexer] for key in branch_names_in}

                        elif name == "inds":
                            v_all = {key: g["inds"][key][indexer] for key in branch_names_in}

                        elif name == "blobs" and not g.attrs["has_blobs"]:
                            v_all = None

                        elif name == "samplers_running":
                            indexer = self._step_indexer(
                                slice_vals, temp_index, sampler_index, has_temp_axis=False
                            )
                            v_all = g["samplers_running"][indexer]

                        else:
                            v_all = g[name][indexer]

                    else:
                        if sampler_index is not None:
                            raise ValueError(
                                "sampler_index is not available for legacy files without the sampler axis."
                            )
                        if name == "samplers_running":
                            raise KeyError(
                                "samplers_running is not stored in legacy files."
                            )

                        if temp_index is None:
                            temp_index = np.arange(self.ntemps)
                        else:
                            assert isinstance(temp_index, int)

                        if name == "chain":
                            v_all = {key: g["chain"][key][slice_vals, temp_index] for key in branch_names_in}

                        elif name == "inds":
                            v_all = {key: g["inds"][key][slice_vals, temp_index] for key in branch_names_in}

                        elif name == "blobs" and not g.attrs["has_blobs"]:
                            v_all = None

                        else:
                            v_all = g[name][slice_vals, temp_index]

                    successful = True

            except OSError:
                num_try += 1
                print(f"Unable to read h5 file {num_try} times.")
                time.sleep(20.0)


        if not successful:
            raise OSError("Attempted to open file max try number of times. Likely cannot read data.")

        return v_all

    def get_move_info(self):
        """Get move information.

        Returns:
            dict: Keys are move names and values are dictionaries with information on the moves.

        """
        # setup output dictionary
        move_info_out = {}
        with self.open() as f:
            g = f[self.name]

            # iterate through everything and produce a dictionary
            for move_name in g["moves"]:
                move_info_out[move_name] = {}
                for info_name in g["moves"][move_name]:
                    move_info_out[move_name][info_name] = g["moves"][move_name][
                        info_name
                    ][:]

        return move_info_out

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            dict: Shape of samples
                Keys are ``branch_names`` and values are tuples with
                shapes of individual branches: (ntemps, nwalkers, nleaves_max, ndim),
                or (nsamplers, ntemps, nwalkers, nleaves_max, ndim) when ``nsamplers > 1``.

        """
        # open file wrapped in with
        with self.open() as f:
            g = f[self.name]
            nsamplers = int(g.attrs.get("nsamplers", 1))
            lead = (nsamplers,) if nsamplers > 1 else ()
            return {
                key: lead
                + (
                    g.attrs["ntemps"],
                    g.attrs["nwalkers"],
                    self.nleaves_max[key],
                    self.ndims[key],
                )
                for key in g.attrs["branch_names"]
            }

    @property
    def iteration(self):
        """Number of iterations stored in the hdf backend so far."""
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        """Number of accepted moves per walker."""
        with self.open() as f:
            out = f[self.name]["accepted"][...]
            if "nsamplers" in f[self.name].attrs and f[self.name].attrs["nsamplers"] == 1:
                out = out[0]
            return out

    @property
    def rj_accepted(self):
        """Number of accepted rj moves per walker."""
        with self.open() as f:
            out = f[self.name]["rj_accepted"][...]
            if "nsamplers" in f[self.name].attrs and f[self.name].attrs["nsamplers"] == 1:
                out = out[0]
            return out

    @property
    def swaps_accepted(self):
        """Number of accepted swaps."""
        with self.open() as f:
            out = f[self.name]["swaps_accepted"][...]
            if "nsamplers" in f[self.name].attrs and f[self.name].attrs["nsamplers"] == 1:
                out = out[0]
            return out

    @property
    def random_state(self):
        """Get the random state"""
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]

            has_sampler_axis = "nsamplers" in g.attrs
            nsamplers = int(g.attrs.get("nsamplers", 1))

            # resize all the arrays accordingly

            ntot = g.attrs["iteration"] + ngrow
            for key in g["chain"]:
                g["chain"][key].resize(ntot, axis=0)
                g["inds"][key].resize(ntot, axis=0)

            g["log_like"].resize(ntot, axis=0)
            g["log_prior"].resize(ntot, axis=0)
            g["betas"].resize(ntot, axis=0)
            if has_sampler_axis:
                g["samplers_running"].resize(ntot, axis=0)

            # deal with blobs
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                nwalkers = g.attrs["nwalkers"]
                ntemps = g.attrs["ntemps"]
                # if blobs have not been added yet
                if not has_blobs:
                    # incoming blobs are folded: (nsamplers * ntemps, nwalkers, nblobs)
                    blobs_shape = (
                        (ntot, nsamplers, ntemps, nwalkers, blobs.shape[-1])
                        if has_sampler_axis
                        else (ntot, ntemps, nwalkers, blobs.shape[-1])
                    )
                    g.create_dataset(
                        "blobs",
                        blobs_shape,
                        maxshape=(None,) + blobs_shape[1:],
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                else:
                    # resize the blobs if they have been there
                    g["blobs"].resize(ntot, axis=0)
                    blobs_shape_check = (
                        (nsamplers, ntemps) + blobs.shape[1:]
                        if has_sampler_axis
                        else blobs.shape
                    )
                    if g["blobs"].shape[1:] != blobs_shape_check:
                        raise ValueError(
                            "Existing blobs have shape {} but new blobs "
                            "requested with shape {}".format(
                                g["blobs"].shape[1:], blobs_shape_check
                            )
                        )
                g.attrs["has_blobs"] = True

    def save_step(
        self,
        state,
        accepted,
        rj_accepted=None,
        swaps_accepted=None,
        moves_accepted_fraction=None,
    ):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
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
        file_opened = False
        max_tries = 100
        try_num = 0
        while not file_opened:
            try:

                # open for appending in with statement
                with self.open("a") as f:
                    g = f[self.name]
                    # get the iteration left off on
                    iteration = g.attrs["iteration"]

                    has_sampler_axis = "nsamplers" in g.attrs
                    nsamplers = int(g.attrs.get("nsamplers", 1))
                    ntemps = int(g.attrs["ntemps"])

                    # make sure the backend has all the information needed to store everything
                    for key in [
                        "rj",
                        "ntemps",
                        "nwalkers",
                        "nbranches",
                        "branch_names",
                        "ndims",
                    ]:
                        if not hasattr(self, key):
                            setattr(self, key, g.attrs[key])

                    # check the inputs are okay
                    self._check(
                        state,
                        accepted,
                        rj_accepted=rj_accepted,
                        swaps_accepted=swaps_accepted,
                    )

                    # incoming state arrays are FOLDED; new-layout files store
                    # with the sampler axis (unfold via reshape at this write
                    # boundary), legacy files keep the original layout
                    def unfold(arr):
                        if not has_sampler_axis:
                            return arr
                        return np.reshape(
                            arr, (nsamplers, ntemps) + np.shape(arr)[1:]
                        )

                    # branch-specific
                    for name, model in state.branches.items():
                        g["inds"][name][iteration] = unfold(model.inds)
                        # use self.store_missing_leaves to set value for missing leaves
                        # state retains old coordinates
                        coords_in = model.coords * model.inds[:, :, :, None]
                        inds_all = np.repeat(
                            model.inds, coords_in.shape[-1], axis=-1
                        ).reshape(model.inds.shape + (coords_in.shape[-1],))
                        coords_in[~inds_all] = self.store_missing_leaves
                        g["chain"][name][self.iteration] = unfold(coords_in)

                    # store everything else in the file
                    g["log_like"][iteration] = unfold(state.log_like)
                    g["log_prior"][iteration] = unfold(state.log_prior)
                    if state.blobs is not None:
                        g["blobs"][iteration] = unfold(state.blobs)
                    if state.betas is not None:
                        if has_sampler_axis:
                            g["betas"][self.iteration] = np.reshape(
                                state.betas, (nsamplers, ntemps)
                            )
                        else:
                            g["betas"][self.iteration, :] = state.betas
                    if has_sampler_axis:
                        samplers_running = getattr(state, "samplers_running", None)
                        g["samplers_running"][iteration] = (
                            samplers_running
                            if samplers_running is not None
                            else np.ones(nsamplers, dtype=bool)
                        )
                    g["accepted"][:] += np.reshape(accepted, g["accepted"].shape)
                    if swaps_accepted is not None:
                        g["swaps_accepted"][:] += np.reshape(
                            swaps_accepted, g["swaps_accepted"].shape
                        )
                    if self.rj:
                        g["rj_accepted"][:] += np.reshape(
                            rj_accepted, g["rj_accepted"].shape
                        )

                    for i, v in enumerate(state.random_state):
                        g.attrs[f"random_state_{i}"] = v

                    g.attrs["iteration"] = iteration + 1

                    # moves
                    if moves_accepted_fraction is not None:
                        if "moves" not in g:
                            raise ValueError(
                                """moves_accepted_fraction was passed, but moves_info was not initialized. Use the moves kwarg 
                                in the reset function."""
                            )

                        # update acceptance fractions
                        for move_key in self.move_keys:
                            g["moves"][move_key]["acceptance_fraction"][:] = (
                                moves_accepted_fraction[move_key]
                            )
                file_opened = True
                
            except (BlockingIOError, OSError) as e:
                try_num += 1
                # Only retry on a true lock contention; bail on file-not-found,
                # bad-mode, etc. so we surface the real error instead of looping
                # for ~16 min and swallowing it.
                msg = str(e)
                lock_like = isinstance(e, BlockingIOError) or (
                    "unable to lock" in msg.lower()
                    or "resource temporarily unavailable" in msg.lower()
                )
                if not lock_like or try_num >= max_tries:
                    raise
                print(
                    f"h5 open(mode={mode!r}) lock contention (try {try_num}/{max_tries}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(1.0)


class TempHDFBackend(object):
    """Check if HDF5 is working and available."""

    def __init__(self, dtype=None, compression=None, compression_opts=None):
        self.dtype = dtype
        self.filename = None
        self.compression = compression
        self.compression_opts = compression_opts

    def __enter__(self):
        f = NamedTemporaryFile(
            prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
        )
        f.close()
        self.filename = f.name
        return HDFBackend(
            f.name,
            "test",
            dtype=self.dtype,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
