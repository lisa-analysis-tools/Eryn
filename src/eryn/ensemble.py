# -*- coding: utf-8 -*-

import warnings
from copy import deepcopy
from itertools import count

import numpy as np

from .backends import Backend, HDFBackend
from .model import Model
from .moves import (
    DistributionGenerateRJ,
    GaussianMove,
    MultipleTryMove,
    StretchMove,
    TemperatureControl,
)
from .pbar import get_progress_bar
from .prior import ProbDistContainer
from .state import State
from .utils import PeriodicContainer, PlotContainer
from .utils.utility import groups_from_inds

__all__ = ["EnsembleSampler", "walkers_independent"]


from collections.abc import Iterable


class EnsembleSampler(object):
    """An ensemble MCMC sampler

    The class controls the entire sampling run. It can handle
    everything from a basic non-tempered MCMC to a parallel-tempered,
    global fit containing multiple branches (models) and a variable
    number of leaves (sources) per branch.
    See `here <https://lisa-analysis-tools.github.io/Eryn/Eryn_tutorial.html#The-Tree-Metaphor>`_
    for a basic explainer.

    Parameters related to parallelization can be controlled via the ``pool`` argument.

    Args:
        nwalkers (int): The number of walkers in the ensemble per temperature.
        ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
        log_like_fn (callable): A function that returns the natural logarithm of the
            likelihood for that position. The inputs to ``log_like_fn`` depend on whether
            the function is vectorized (kwarg ``vectorize`` below), if you are using reversible jump,
            and how many branches you have.

                In the simplest case where ``vectorize == False``, no reversible jump, and only one
                type of model, the inputs are just the array of parameters for one walker, so shape is ``(ndim,)``.

                If ``vectorize == True``, no reversible jimp, and only one type of model, the inputs will
                be a 2D array of parameters of all the walkers going in. Shape: ``(num positions, ndim)``.

                If using reversible jump, the leaves that go together in the same Likelihood will be grouped
                together into a single function call. If ``vectorize == False``, then each group is sent as
                an individual computation. With ``N`` different branches (``N > 1``), inputs would be a list
                of 2D arrays of the coordinates for all leaves within each branch: ``([x0, x1,...,xN])``
                where ``xi`` is 2D with shape ``(number of leaves in this branch, ndim)``. If ``N == 1``, then
                a list is not provided, just x0, the 2D array of coordinates for the one branch considered.

                If using reversible jump and ``vectorize == True``, then the arrays of parameters will be output
                with information as regards the grouping of branch and leaf set. Inputs will be
                ``([X0, X1,..XN], [group0, group1,...,groupN])`` where ``Xi`` is a 2D array of all
                leaves in the sampler for branch ``i``. ``groupi`` is an index indicating which unique group
                that sources belongs. For example, if we have 3 walkers with (1, 2, 1) leaves for model ``i``,
                respectively, we wil have an ``Xi = array([params0, params1, params2, params3])`` and
                ``groupsi = array([0, 1, 1, 2])``.
                If ``N == 1``, then the lists are removed and the inputs become ``(X0, group0)``.

                Extra ``args`` and ``kwargs`` for the Likelihood function can be added with the kwargs
                ``args`` and ``kwargs`` below.

                Please see the
                `tutorial <https://lisa-analysis-tools.github.io/Eryn/Eryn_tutorial.html#>`_
                for more information.

        priors (dict): The prior dictionary can take four forms.
            1) A dictionary with keys as int or tuple containing the int or tuple of int
            that describe the parameter number over which to assess the prior, and values that
            are prior probability distributions that must have a ``logpdf`` class method.
            2) A :class:`eryn.prior.ProbDistContainer` object.
            3) A dictionary with keys that are ``branch_names`` and values that are dictionaries for
            each branch as described for (1).
            4) A dictionary with keys that are ``branch_names`` and values are
            :class:`eryn.prior.ProbDistContainer` objects.
            If the priors dictionary has the specific key ``"all_models_together"`` in it, then a special
            prior must be input by the user that can produce the prior ``logpdf`` information that can depend on all of the models
            together rather than handling them separately as is the default. In this case, the user must input
            as the item attached to this special key a class object with a ``logpdf`` method that takes as input
            two arguments: ``(coords, inds)``, which are the coordinate and index dictionaries across all models with
            shapes of ``(ntemps, nwalkers, nleaves_max, ndim)`` and ``(ntemps, nwalkers, nleaves_max)`` for each
            individual model, respectively. This function must then return the numpy array of logpdf values for the
            prior value with shape ``(ntemps, nwalkers)``.
        provide_groups (bool, optional): If ``True``, provide groups as described in ``log_like_fn`` above.
            A group parameter is added for each branch. (default: ``False``)
        provide_supplemental (bool, optional): If ``True``, it will provide keyword arguments to
            the Likelihood function: ``supps`` and ``branch_supps``. Please see the `Tutorial <https://lisa-analysis-tools.github.io/Eryn/Eryn_tutorial.html#>`_
            and :class:`eryn.state.BranchSupplemental` for more information.
        tempering_kwargs (dict, optional): Keyword arguments for initialization of the
            tempering class: :class:`eryn.moves.tempering.TemperatureControl`.  (default: ``{}``)
        branch_names (list, optional): List of branch names. If ``None``, models will be assigned
            names as ``f"model_{index}"`` based on ``nbranches``. (default: ``None``)
        nbranches (int, optional): Number of branches (models) tested.
            Only used if ``branch_names is None``.
            (default: ``1``)
        nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
            It should have the same length as the number of branches.
            If ``dict``, keys should be the branch names and values the associated maximal leaf value.
            (default: ``1``)
        nleaves_min (int, list of ints, or dict, optional): Minimum allowable leaf count for each branch.
            It should have the same length as the number of branches. Only used with Reversible Jump.
            If ``dict``, keys should be the branch names and values the associated maximal leaf value.
            If ``None`` and using Reversible Jump, will fill all branches with zero.
            (default: ``0``)
        pool (object, optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        moves (list or object, optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(eryn.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``None``, the default will be :class:`StretchMove`.
            (default: ``None``)
        rj_moves (list or object or bool or str, optional): If ``None`` or ``False``, reversible jump will not be included in the run.
            This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(eryn.moves.DistributionGenerateRJ(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``True``, it defaults to :class:`DistributionGenerateRJ`. When running just the :class:`DistributionGenerateRJ`
            with multiple branches, it will propose changes to all branhces simultaneously.
            When running with more than one branch, useful options for ``rj_moves`` are ``"iterate_branches"``,
            ``"separate_branches"``, or ``"together"``. If ``rj_moves == "iterate_branches"``, sample one branch by one branch in order of
            the branch names. This occurs within one RJ proposal, for each RJ proposal. If ``rj_moves == "separate_branches"``,
            there will be one RJ move per branch. During each individual RJ move, one of these proposals is chosen at random with
            equal propability. This is generally recommended when using multiple branches.
            If ``rj_moves == "together"``, this is equivalent to ``rj_moves == True``.
            (default: ``None``)
        dr_moves (bool, optional): If ``None`` ot ``False``, delayed rejection when proposing "birth"
            of new components/models will be switched off for this run. Requires ``rj_moves`` set to ``True``.
            Not implemented yet. Working on it.
            (default: ``None``)
        dr_max_iter (int, optional): Maximum number of iterations used with delayed rejection. (default: 5)
        args (optional): A list of extra positional arguments for
            ``log_like_fn``. ``log_like_fn`` will be called as
            ``log_like_fn(sampler added args, *args, sampler added kwargs, **kwargs)``.
        kwargs (optional): A dict of extra keyword arguments for
            ``log_like_fn``. ``log_like_fn`` will be called as
            ``log_like_fn(sampler added args, *args, sampler added kwargs, **kwargs)``.
        backend (optional): Either a :class:`backends.Backend` or a subclass
            (like :class:`backends.HDFBackend`) that is used to store and
            serialize the state of the chain. By default, the chain is stored
            as a set of numpy arrays in memory, but new backends can be
            written to support other mediums.
        vectorize (bool, optional): If ``True``, ``log_like_fn`` is expected
            to accept an array of position vectors instead of just one. Note
            that ``pool`` will be ignored if this is ``True``. See ``log_like_fn`` information
            above to understand the arguments of ``log_like_fn`` based on whether
            ``vectorize`` is ``True``.
            (default: ``False``)
        periodic (dict, optional): Keys are ``branch_names``. Values are dictionaries
            that have (key: value) pairs as (index to parameter: period). Periodic
            parameters are treated as having periodic boundary conditions in proposals.
        update_fn (callable, optional): :class:`eryn.utils.updates.AdjustStretchProposalScale`
            object that allows the user to update the sampler in any preferred way
            every ``update_iterations`` sampler iterations. The callable must have signature:
            ``(sampler iteration, last sample state object, EnsembleSampler object)``.
        update_iterations (int, optional): Number of iterations between sampler
            updates using ``update_fn``. Updates are only performed at the thinning rate.
            If ``thin_by>1`` when :func:`EnsembleSampler.run_mcmc` is used, the sampler
            is updated every ``thin_by * update_iterations`` iterations.
        stopping_fn (callable, optional): :class:`eryn.utils.stopping.Stopping` object that
            allows the user to end the sampler if specified criteria are met.
            The callable must have signature:
            ``(sampler iteration, last sample state object, EnsembleSampler object)``.
        stopping_iterations (int, optional): Number of iterations between sampler
            attempts to evaluate the ``stopping_fn``. Stopping checks are only performed at the thinning rate.
            If ``thin_by>1`` when :func:`EnsembleSampler.run_mcmc` is used, the sampler
            is checked for the stopping criterion every ``thin_by * stopping_iterations`` iterations.
        fill_zero_leaves_val (double, optional): When there are zero leaves in a
            given walker (across all branches), fill the likelihood value with
            ``fill_zero_leaves_val``. If wanting to keep zero leaves as a possible
            model, this should be set to the value of the contribution to the Likelihood
            from the data. (Default: ``-1e300``).
        num_repeats_in_model (int, optional): Number of times to repeat the in-model step
            within in one sampler iteration. When analyzing the acceptance fraction, you must
            include the value of ``num_repeats_in_model`` to get the proper denominator.
        num_repeats_rj (int, optional): Number of time to repeat the reversible jump step
            within in one sampler iteration. When analyzing the acceptance fraction, you must
            include the value of ``num_repeats_rj`` to get the proper denominator.
        track_moves (bool, optional): If ``True``, track acceptance fraction of each move
            in the backend. If ``False``, no tracking is done. If ``True`` and run is interrupted, it will check
            that the move configuration has not changed. It will not allow the run to go on
            if it is changed. In this case, the user should declare a new backend and use the last
            state from the previous backend. **Warning**: If the order of moves of the same move class
            is changed, the check may not catch it, so the tracking may mix move acceptance fractions together.
            (default: ``True'')
        nsamplers (int, optional): Number of fully independent samplers (ensembles) to run
            simultaneously. All samplers' likelihood evaluations are batched into single
            calls (useful for GPU likelihoods). Each sampler carries its own temperature
            ladder; temperature swaps and ladder adaptation are per sampler. Initial
            coordinates then take shape ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)``
            and stored/returned quantities gain a leading ``nsamplers`` axis (after the
            step axis). With the default of 1, nothing changes anywhere user-facing.
            Internally the sampler axis is folded into the temperature axis, so proposals
            simply see ``nsamplers * ntemps`` "temperatures". Use
            ``State.samplers_running`` (bool array of shape ``(nsamplers,)``) to freeze
            converged samplers: frozen samplers are skipped in likelihood/prior evaluation
            and tempering. (default: ``1``)
        prior_transform_fn (object or dict, optional): Per-sampler prior-basis transform,
            replacing ``ParaEnsembleSampler.prior_transform_fn``. A dict maps branch names
            to objects implementing ``transform_to_prior_basis(coords, samplers_running_idx)``
            (in-place map of ``(num_running, ntemps, nwalkers, ndim)`` coordinates to the
            basis in which ``priors`` is defined) and ``adjust_logp(logp, samplers_running_idx)``
            (in-place Jacobian adjustment). Useful for per-sampler prior bounds (e.g.
            per-band frequency limits). Requires ``nleaves_max == 1`` for the transformed
            branch and is incompatible with reversible jump, multiple-try moves,
            ``provide_groups``, and ``all_models_together`` priors. (default: ``None``)
        info (dict, optional): Key and value pairs reprenting any information
            the user wants to add to the backend if the user is not inputing
            their own backend.

    Raises:
        ValueError: Any startup issues.

    """

    def __init__(
        self,
        nwalkers,
        ndims,  # assumes ndim_max
        log_like_fn,
        priors,
        provide_groups=False,
        provide_supplemental=False,
        tempering_kwargs={},
        branch_names=None,
        nbranches=1,
        nleaves_max=1,
        nleaves_min=0,
        pool=None,
        moves=None,
        rj_moves=None,
        dr_moves=None,
        dr_max_iter=5,
        args=None,
        kwargs=None,
        backend=None,
        vectorize=False,
        blobs_dtype=None,  # TODO check this
        plot_iterations=-1,  # TODO: do plot stuff?
        plot_generator=None,
        plot_folder=None,
        periodic=None,
        update_fn=None,
        update_iterations=-1,
        stopping_fn=None,
        stopping_iterations=-1,
        fill_zero_leaves_val=-1e300,
        num_repeats_in_model=1,
        num_repeats_rj=1,
        track_moves=True,
        nsamplers=1,
        prior_transform_fn=None,
        info={},
    ):
        # store priors
        self.priors = priors

        # number of independent samplers run simultaneously; the sampler axis
        # is FOLDED into the temperature axis everywhere internally
        # (leading axis size nsamplers * ntemps), so proposals treat the
        # independent samplers as extra temperatures
        self.nsamplers = int(nsamplers)
        if self.nsamplers < 1:
            raise ValueError(f"nsamplers must be >= 1; got {nsamplers}.")

        # store some kwargs
        self.provide_groups = provide_groups
        self.provide_supplemental = provide_supplemental
        self.fill_zero_leaves_val = fill_zero_leaves_val
        self.num_repeats_in_model = num_repeats_in_model
        self.num_repeats_rj = num_repeats_rj
        self.track_moves = track_moves

        # setup emcee-like basics
        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

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

        self.nbranches = len(branch_names)

        self.branch_names = branch_names
        self.ndims = ndims
        self.nleaves_max = nleaves_max

        # setup temperaing information
        # default is no temperatures
        if tempering_kwargs == {}:
            self.ntemps = 1
            self.temperature_control = None
        else:
            # get effective total dimension
            total_ndim = 0
            for key in self.branch_names:
                total_ndim += self.nleaves_max[key] * self.ndims[key]
            self.temperature_control = TemperatureControl(
                total_ndim, nwalkers, nsamplers=self.nsamplers, **tempering_kwargs
            )
            self.ntemps = self.temperature_control.ntemps

        # folded leading-axis size: independent samplers x temperatures
        self.ntemps_eff = self.nsamplers * self.ntemps

        # set basic variables for sampling settings
        self.nwalkers = nwalkers
        self.nbranches = nbranches

        # eryn wraps periodic parameters
        if periodic is not None:
            if not isinstance(periodic, PeriodicContainer) and not isinstance(
                periodic, dict
            ):
                raise ValueError(
                    "periodic must be PeriodicContainer or dict if not None."
                )
            elif isinstance(periodic, dict):
                periodic = PeriodicContainer(periodic, key_order=self.key_order)

        # Parse the move schedule
        if moves is None:
            if rj_moves is not None:
                raise ValueError(
                    "If providing rj_moves, must provide moves kwarg as well."
                )

            # defaults to stretch move
            self.moves = [
                StretchMove(
                    temperature_control=self.temperature_control,
                    periodic=periodic,
                    a=2.0,
                )
            ]
            self.weights = [1.0]

        elif isinstance(moves, Iterable):
            try:
                self.moves, self.weights = [list(tmp) for tmp in zip(*moves)]

            except TypeError:
                self.moves = moves
                self.weights = np.ones(len(moves))
        else:
            self.moves = [moves]
            self.weights = [1.0]

        self.weights = np.atleast_1d(self.weights).astype(float)
        self.weights /= np.sum(self.weights)

        # parse the reversible jump move schedule
        if rj_moves is None:
            self.has_reversible_jump = False
        elif (isinstance(rj_moves, bool) and rj_moves) or isinstance(rj_moves, str):
            self.has_reversible_jump = True

            if self.has_reversible_jump:
                if nleaves_min is None:
                    nleaves_min = {bn: 0 for bn in branch_names}
                elif isinstance(nleaves_min, int):
                    assert len(branch_names) == 1
                    nleaves_min = {branch_names[0]: nleaves_min}

                elif isinstance(nleaves_min, list) or isinstance(
                    nleaves_min, np.ndarray
                ):
                    assert len(branch_names) == len(nleaves_min)
                    nleaves_min = {bn: nl for bn, nl in zip(branch_names, nleaves_min)}

                elif isinstance(nleaves_min, dict):
                    assert len(list(nleaves_min.keys())) == len(branch_names)
                    for key in nleaves_min:
                        if key not in branch_names:
                            raise ValueError(
                                f"{key} is in nleaves_min but does not appear in branch_names: {branch_names}."
                            )
                else:
                    raise ValueError(
                        "If providing nleaves_min, nleaves_min is to be a scalar int, list, or dict."
                    )

                self.nleaves_min = nleaves_min

                if (isinstance(rj_moves, bool) and rj_moves) or (
                    isinstance(rj_moves, str) and rj_moves == "together"
                ):
                    # default to DistributionGenerateRJ

                    # gibbs sampling setup here means run all of them together
                    gibbs_sampling_setup = None

                    rj_move = DistributionGenerateRJ(
                        self.priors,
                        nleaves_max=self.nleaves_max,
                        nleaves_min=self.nleaves_min,
                        dr=dr_moves,
                        dr_max_iter=dr_max_iter,
                        tune=False,
                        temperature_control=self.temperature_control,
                        gibbs_sampling_setup=gibbs_sampling_setup,
                    )
                    self.rj_moves = [rj_move]
                    self.rj_weights = [1.0]

                elif isinstance(rj_moves, str) and rj_moves == "iterate_branches":
                    # will iterate through all branches within one RJ proposal
                    gibbs_sampling_setup = deepcopy(branch_names)

                    # default to DistributionGenerateRJ
                    rj_move = DistributionGenerateRJ(
                        self.priors,
                        nleaves_max=self.nleaves_max,
                        nleaves_min=self.nleaves_min,
                        dr=dr_moves,
                        dr_max_iter=dr_max_iter,
                        tune=False,
                        temperature_control=self.temperature_control,
                        gibbs_sampling_setup=gibbs_sampling_setup,
                    )
                    self.rj_moves = [rj_move]
                    self.rj_weights = [1.0]

                elif isinstance(rj_moves, str) and rj_moves == "separate_branches":
                    # will iterate through all branches within one RJ proposal
                    rj_moves = []
                    rj_weights = []
                    for branch_name in branch_names:

                        # only do one branch per move
                        gibbs_sampling_setup = [branch_name]

                        # default to DistributionGenerateRJ
                        rj_move_tmp = DistributionGenerateRJ(
                            self.priors,
                            nleaves_max=self.nleaves_max,
                            nleaves_min=self.nleaves_min,
                            dr=dr_moves,
                            dr_max_iter=dr_max_iter,
                            tune=False,
                            temperature_control=self.temperature_control,
                            gibbs_sampling_setup=gibbs_sampling_setup,
                        )
                        rj_moves.append(rj_move_tmp)
                        # will renormalize after
                        rj_weights.append(1.0)
                    self.rj_moves = rj_moves
                    self.rj_weights = rj_weights

                elif isinstance(rj_moves, str):
                    raise ValueError(
                        f"When providing a str for rj_moves, must be 'together', 'iterate_branches', or 'separate_branches'. Input is {rj_moves}"
                    )

        # same as above for moves
        elif isinstance(rj_moves, Iterable):
            self.has_reversible_jump = True

            try:
                self.rj_moves, self.rj_weights = zip(*rj_moves)
            except TypeError:
                self.rj_moves = rj_moves
                self.rj_weights = np.ones(len(rj_moves))

        elif isinstance(rj_moves, bool) and not rj_moves:
            self.has_reversible_jump = False
            self.rj_moves = None
            self.rj_weights = None

        elif not isinstance(rj_moves, bool):
            self.has_reversible_jump = True

            self.rj_moves = [rj_moves]
            self.rj_weights = [1.0]

        # adjust rj weights properly
        if self.has_reversible_jump:
            self.rj_weights = np.atleast_1d(self.rj_weights).astype(float)
            self.rj_weights /= np.sum(self.rj_weights)

            # warn if base stretch is used
            for move in self.moves:
                if type(move) == StretchMove:
                    warnings.warn(
                        "If using revisible jump, using the Stretch Move for in-model proposals is not advised. It will run and work, but it will not be using the correct complientary group of parameters meaning it will most likely be very inefficient."
                    )

        # make sure moves have temperature module
        if self.temperature_control is not None:
            for move in self.moves:
                if move.temperature_control is None:
                    move.temperature_control = self.temperature_control

            if self.has_reversible_jump:
                for move in self.rj_moves:
                    if move.temperature_control is None:
                        move.temperature_control = self.temperature_control

        # make sure moves have temperature module
        if periodic is not None:
            for move in self.moves:
                if move.periodic is None:
                    move.periodic = periodic

            if self.has_reversible_jump:
                for move in self.rj_moves:
                    if move.periodic is None:
                        move.periodic = periodic

        # store
        self.periodic = periodic

        # give every move the independent-samplers structure (also set by the
        # temperature_control setter, but needed for untempered runs and
        # user-constructed moves); guard against mismatched user-provided
        # temperature controls
        all_moves_for_setup = list(self.moves) + (
            list(self.rj_moves) if self.has_reversible_jump else []
        )
        for move in all_moves_for_setup:
            if (
                move.temperature_control is not None
                and getattr(move.temperature_control, "nsamplers", 1) != self.nsamplers
            ):
                raise ValueError(
                    f"Move {move.__class__.__name__} carries a TemperatureControl with "
                    f"nsamplers={getattr(move.temperature_control, 'nsamplers', 1)}, but the "
                    f"sampler was built with nsamplers={self.nsamplers}."
                )
            move.nsamplers = self.nsamplers
            move.ntemps_per_sampler = self.ntemps
            move.sampler_id_rows = np.repeat(np.arange(self.nsamplers), self.ntemps)

        # prepare the per proposal accepted values that are held as attributes in the specific classes
        for move in self.moves:
            move.accepted = np.zeros((self.ntemps_eff, self.nwalkers))

        if self.has_reversible_jump:
            for move in self.rj_moves:
                move.accepted = np.zeros((self.ntemps_eff, self.nwalkers))

        # setup backend if not provided or initialized
        if backend is None:
            self.backend = Backend()
        elif isinstance(backend, str):
            self.backend = HDFBackend(backend)
        else:
            self.backend = backend

        self.info = info

        all_moves_tmp = list(
            tuple(self.moves)
            if not self.has_reversible_jump
            else tuple(self.moves) + tuple(self.rj_moves)
        )

        self.all_moves = {}
        if self.track_moves:
            current_indices_move_keys = {}
            for move in all_moves_tmp:
                # get out of tuple if weights are given
                if isinstance(move, tuple):
                    move = move[0]

                # get the name of the class instance as a string
                move_name = move.__class__.__name__

                # need to keep track how many times each type of move class has been used
                if move_name not in current_indices_move_keys:
                    current_indices_move_keys[move_name] = 0

                else:
                    current_indices_move_keys[move_name] += 1

                # get the full name including the index
                full_move_name = move_name + f"_{current_indices_move_keys[move_name]}"
                self.all_moves[full_move_name] = move

            # get move keys out
            move_keys = list(self.all_moves.keys())

        else:
            move_keys = None

        self.move_keys = move_keys

        # Deal with re-used backends
        if not self.backend.initialized:
            self._previous_state = None
            self.reset(
                branch_names=branch_names,
                ntemps=self.ntemps,
                nleaves_max=nleaves_max,
                rj=self.has_reversible_jump,
                moves=move_keys,
                key_order=self.key_order,
                nsamplers=self.nsamplers,
                **info,
            )
            state = np.random.get_state()
        else:
            if self.track_moves:
                moves_okay = True
                if len(self.move_keys) != len(self.backend.move_keys):
                    moves_okay = False

                for key in self.move_keys:
                    if key not in self.backend.move_keys:
                        moves_okay = False

                if not moves_okay:
                    raise ValueError(
                        "Configuration of moves has changed. Cannot use the same backend. Declare a new backend and start from the previous state. If you would prefer not to track move acceptance fraction, set track_moves to False in the EnsembleSampler."
                    )

            def _check_key_orders(ko1, ko2):
                if isinstance(ko1, dict) and isinstance(ko2, dict):
                    if ko1.keys() != ko2.keys():
                        return False
                    return all(_check_key_orders(ko1[k], ko2[k]) for k in ko1)
                elif isinstance(ko1, np.ndarray) or isinstance(ko2, np.ndarray):
                    return np.array_equal(ko1, ko2)
                elif isinstance(ko1, (list, tuple)) and isinstance(ko2, (list, tuple)):
                    if len(ko1) != len(ko2):
                        return False
                    return all(_check_key_orders(v1, v2) for v1, v2 in zip(ko1, ko2))
                else:
                    return ko1 == ko2

            if not _check_key_orders(self.key_order, self.backend.key_order):
                if self.backend.key_order == {}:
                    reset_args = self.backend.reset_args
                    reset_kwargs = self.backend.reset_kwargs
                    reset_kwargs['key_order'] = self.key_order

                    self.backend.reset(*reset_args, **reset_kwargs)

                    warnings.warn(
                        "The backend did not have a key order but the sampler does. The backend key order has been set to match the sampler. If this is not what you wanted, please declare a new backend and start from the previous state."
                    )                
                else:
                    raise ValueError("Input key order from priors does not match backend.")

            # legacy files / backends have no sampler axis and only support nsamplers == 1
            if getattr(self.backend, "nsamplers", 1) != self.nsamplers:
                raise ValueError(
                    f"Backend was initialized with nsamplers={getattr(self.backend, 'nsamplers', 1)}, "
                    f"but the sampler was built with nsamplers={self.nsamplers}. "
                    "Restarting with a different nsamplers (including from a legacy file "
                    "without the sampler axis) is not supported."
                )

            # Check the backend shape
            for i, (name, shape) in enumerate(self.backend.shape.items()):
                test_shape = ((self.nsamplers,) if self.nsamplers > 1 else ()) + (
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[name],
                    self.ndims[name],
                )
                if shape != test_shape:
                    raise ValueError(
                        (
                            f"the shape of the backend ({shape}) is incompatible with the "
                            f"shape of the sampler ({test_shape} for model {name})"
                        )
                    )

            # Get the last random state
            state = self.backend.random_state
            if state is None:
                state = np.random.get_state()

            # Grab the last step so that we can restart
            it = self.backend.iteration
            if it > 0:
                self._previous_state = self.get_last_sample()

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()
        self._random.set_state(state)

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.log_like_fn = _FunctionWrapper(log_like_fn, args, kwargs)

        self.all_walkers = self.nwalkers * self.ntemps_eff

        # prepare plotting
        # TODO: adjust plotting maybe?
        self.plot_iterations = plot_iterations

        if self.nsamplers > 1 and plot_iterations > 0:
            raise ValueError(
                "Runtime plotting (plot_iterations > 0) is not supported for nsamplers > 1."
            )

        # per-sampler prior-basis transform (para parity):
        # dict of branch_name -> object with in-place
        # transform_to_prior_basis(coords, samplers_running_idx) and
        # adjust_logp(logp, samplers_running_idx) methods
        if prior_transform_fn is not None:
            if not isinstance(prior_transform_fn, dict):
                if len(self.branch_names) != 1:
                    raise ValueError(
                        "With multiple branches, prior_transform_fn must be a dict keyed by branch name."
                    )
                prior_transform_fn = {self.branch_names[0]: prior_transform_fn}

            if self.has_reversible_jump:
                raise ValueError(
                    "prior_transform_fn is not compatible with reversible jump: RJ moves "
                    "evaluate the priors directly on sampling-basis coordinates."
                )
            if self.provide_groups:
                raise ValueError(
                    "prior_transform_fn is not compatible with provide_groups=True."
                )
            if "all_models_together" in self.priors:
                raise ValueError(
                    "prior_transform_fn is not compatible with 'all_models_together' priors."
                )
            for name in prior_transform_fn:
                if name not in self.branch_names:
                    raise ValueError(
                        f"prior_transform_fn key {name} is not a branch name: {self.branch_names}."
                    )
                if self.nleaves_max[name] > 1:
                    raise ValueError(
                        "prior_transform_fn requires nleaves_max == 1 for the transformed branch."
                    )
            for move in all_moves_for_setup:
                if isinstance(move, MultipleTryMove):
                    raise ValueError(
                        "prior_transform_fn is not compatible with multiple-try moves: they "
                        "evaluate the priors directly on sampling-basis coordinates."
                    )
        self.prior_transform_fn = prior_transform_fn

        if plot_generator is None and self.plot_iterations > 0:
            # set to default if not provided
            if plot_folder is  None:
                plot_folder = "./runtime_plots"
            self.plot_generator = PlotContainer(
                backend=self.backend, plots=['base', 'rj'], parent_folder=plot_folder, discard=0.2
            )
        elif self.plot_iterations > 0:
            self.plot_generator = plot_generator
            self.plot_generator.backend = self.backend # make sure backend is correctly set

        # prepare stopping functions
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations

        # prepare update functions
        self.update_fn = update_fn
        self.update_iterations = update_iterations

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.

        """
        return self._random.get_state()
    
    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def priors(self):
        """
        Return the priors in the sampler.

        """
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Set priors information.

        This performs checks to make sure the inputs are okay.

        """
        if isinstance(priors, dict):
            self._priors = {}

            for key in priors.keys():
                test = priors[key]
                if isinstance(test, dict):
                    # check all dists
                    for ind, dist in test.items():
                        if not hasattr(dist, "logpdf"):
                            raise ValueError(
                                f"Distribution for model {key} and index {ind} does not have logpdf method."
                            )

                    self._priors[key] = ProbDistContainer(test)

                elif isinstance(test, ProbDistContainer):
                    self._priors[key] = test

                elif hasattr(test, "logpdf"):
                    self._priors[key] = test

                else:
                    raise ValueError(
                        "priors dictionary items must be dictionaries with prior information or instances of the ProbDistContainer class."
                    )

        elif isinstance(priors, ProbDistContainer):
            self._priors = {"model_0": priors}

        else:
            raise ValueError("Priors must be a dictionary.")

        self.key_order = {key: value.key_order for key, value in self._priors.items()}
        return

    @property
    def iteration(self):
        return self.backend.iteration

    def reset(self, **kwargs):
        """
        Reset the backend.

        Args:
            **info (dict, optional): information to pass to backend reset method.

        """
        kwargs.setdefault("nsamplers", self.nsamplers)
        self.backend.reset(self.nwalkers, self.ndims, **kwargs)

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["pool"] = None
        return d

    def get_model(self):
        """Get ``Model`` object from sampler

        The model object is used to pass necessary information to the
        proposals. This method can be used to retrieve the ``model`` used
        in the sampler from outside the sampler.

        Returns:
            :class:`Model`: ``Model`` object used by sampler.

        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = Model(
            self.log_like_fn,
            self.compute_log_like,
            self.compute_log_prior,
            self.temperature_control,
            map_fn,
            self._random,
        )
        return model

    def sample(
        self,
        initial_state,
        iterations=1,
        tune=False,
        skip_initial_state_check=True,
        thin_by=1,
        store=True,
        progress=False,
    ):
        """Advance the chain as a generator

        Args:
            initial_state (State or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            iterations (int or None, optional): The number of steps to generate.
                ``None`` generates an infinite stream (requires ``store=False``).
                (default: 1)
            tune (bool, optional): If ``True``, the parameters of some moves
                will be automatically tuned. (default: ``False``)
            thin_by (int, optional): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made. (default: 1)
            store (bool, optional): By default, the sampler stores in the backend
                the positions (and other information) of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``. (default: ``True``)
            progress (bool or str, optional): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown. (default: ``False``)
            skip_initial_state_check (bool, optional): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. If using reversible jump, the user needs to ensure this on their own
                (``skip_initial_state_check``is set to ``False`` in this case.
                (default: ``True``)

        Returns:
            State: Every ``thin_by`` steps, this generator yields the :class:`State` of the ensemble.

        Raises:
            ValueError: Improper initialization.

        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")

        # Interpret the input as a walker state and check the dimensions.

        # convenience for nsamplers > 1: accept para-style 4D array input
        # (nsamplers, ntemps, nwalkers, ndim) by inserting the leaf axis; only
        # applied when the shape matches the sampler configuration exactly and
        # cannot be confused with the folded 4D layout
        if self.nsamplers > 1 and not hasattr(initial_state, "branches"):
            initial_state_in = initial_state
            if isinstance(initial_state_in, np.ndarray):
                initial_state_in = {self.branch_names[0]: initial_state_in}
            if isinstance(initial_state_in, dict):
                new_coords = {}
                for name, arr in initial_state_in.items():
                    if (
                        isinstance(arr, np.ndarray)
                        and arr.ndim == 4
                        and self.nleaves_max[name] == 1
                        and arr.shape
                        == (
                            self.nsamplers,
                            self.ntemps,
                            self.nwalkers,
                            self.ndims[name],
                        )
                        and arr.shape
                        != (
                            self.ntemps_eff,
                            self.nwalkers,
                            self.nleaves_max[name],
                            self.ndims[name],
                        )
                    ):
                        arr = arr[:, :, :, None, :]
                    new_coords[name] = arr
                initial_state = new_coords

        # initial_state.__class__ rather than State in case it is a subclass
        # of State
        if (
            hasattr(initial_state, "__class__")
            and issubclass(initial_state.__class__, State)
            and not isinstance(initial_state.__class__, State)
        ):
            state = initial_state.__class__(initial_state, copy=True)
            if getattr(state, "nsamplers", 1) != self.nsamplers:
                raise ValueError(
                    f"Input state has nsamplers={getattr(state, 'nsamplers', 1)}, but the "
                    f"sampler was built with nsamplers={self.nsamplers}. Build the State "
                    "with the nsamplers kwarg or pass 5D coordinates "
                    "(nsamplers, ntemps, nwalkers, nleaves_max, ndim)."
                )
        else:
            state = State(initial_state, copy=True, nsamplers=self.nsamplers)

        # source of truth for which independent samplers are running; the
        # array object is shared by reference through the whole iteration, so
        # in-place edits (e.g. from update_fn) propagate
        self._samplers_running = getattr(state, "samplers_running", None)

        # Check the backend shape
        for i, (name, branch) in enumerate(state.branches.items()):
            ntemps_, nwalkers_, nleaves_, ndim_ = branch.shape
            if (ntemps_, nwalkers_, nleaves_, ndim_) != (
                self.ntemps_eff,
                self.nwalkers,
                self.nleaves_max[name],
                self.ndims[name],
            ):
                raise ValueError(
                    f"incompatible input dimensions for branch {name}: folded shape "
                    f"{branch.shape} vs expected "
                    f"{(self.ntemps_eff, self.nwalkers, self.nleaves_max[name], self.ndims[name])}"
                    + (
                        " (for nsamplers > 1, provide coordinates with shape "
                        f"({self.nsamplers}, {self.ntemps}, {self.nwalkers}, nleaves_max, ndim))"
                        if self.nsamplers > 1
                        else ""
                    )
                )

        # do an initial state check if is requested and we are not using reversible jump
        if (not skip_initial_state_check) and (
            not walkers_independent(state.coords) and not self.has_reversible_jump
        ):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_prior = self.compute_log_prior(coords, inds=inds)

        if state.log_like is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_like, state.blobs = self.compute_log_like(
                coords,
                inds=inds,
                logp=state.log_prior,
                supps=state.supplemental,  # only used if self.provide_supplemental is True
                branch_supps=state.branches_supplemental,  # only used if self.provide_supplemental is True
            )

        # get betas out of state object if they are there
        if state.betas is not None:
            if state.betas.shape[0] != self.ntemps_eff:
                raise ValueError(
                    "Input state has inverse temperatures (betas), but not the correct number of temperatures according to sampler inputs."
                )

            self.temperature_control.betas = state.betas.copy()

        else:
            if hasattr(self, "temperature_control") and hasattr(
                self.temperature_control, "betas"
            ):
                state.betas = self.temperature_control.betas.copy()

        if np.shape(state.log_like) != (self.ntemps_eff, self.nwalkers):
            raise ValueError("incompatible input dimensions")
        if np.shape(state.log_prior) != (self.ntemps_eff, self.nwalkers):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``. Rows of non-running samplers hold fill values, so only
        # check the running rows.
        if self._samplers_running is None:
            check_rows = slice(None)
        else:
            check_rows = np.repeat(
                np.asarray(self._samplers_running, dtype=bool), self.ntemps
            )

        if np.any(np.isnan(state.log_like[check_rows])):
            raise ValueError("The initial log_like was NaN")

        if np.any(np.isinf(state.log_like[check_rows])):
            raise ValueError("The initial log_like was +/- infinite")

        if np.any(np.isnan(state.log_prior[check_rows])):
            raise ValueError("The initial log_prior was NaN")

        if np.any(np.isinf(state.log_prior[check_rows])):
            raise ValueError("The initial log_prior was +/- infinite")

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

        if hasattr(self, "override_thin_by"):
            thin_by = self.override_thin_by
        self.yield_step = thin_by
        self.checkpoint_step = thin_by
        if store:
            self.backend.grow(iterations, state.blobs)

        # get the model object
        model = self.get_model()

        # Inject the progress bar
        total = None if iterations is None else iterations * self.yield_step
        with get_progress_bar(progress, total) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(self.yield_step):
                    # re-sync the running mask (update_fn may replace the array
                    # on the state between iterations)
                    self._samplers_running = getattr(state, "samplers_running", None)

                    # in model moves
                    accepted = np.zeros((self.ntemps_eff, self.nwalkers))
                    for repeat in range(self.num_repeats_in_model):
                        # Choose a random move
                        move = self._random.choice(self.moves, p=self.weights)
                        
                        # if i == 0: # To make sure that we first select the Fstat move
                        #     move = self.moves[0]
                        #     # breakpoint()
                        # try:
                        #     print("Current move is", move.name)
                        # except AttributeError:
                        #     pass
                            
                        # Propose (in model)
                        state, accepted_out = move.propose(model, state)
                        accepted += accepted_out
                        if self.ntemps > 1:
                            try:
                                in_model_swaps = move.temperature_control.swaps_accepted
                            except AttributeError:
                                in_model_swaps = None
                        else:
                            in_model_swaps = None

                        state.random_state = self.random_state

                        if tune:
                            move.tune(state, accepted_out)

                    if self.has_reversible_jump:
                        rj_accepted = np.zeros((self.ntemps_eff, self.nwalkers))
                        for repeat in range(self.num_repeats_rj):
                            rj_move = self._random.choice(
                                self.rj_moves, p=self.rj_weights
                            )

                            # Propose (Between models)
                            state, rj_accepted_out = rj_move.propose(model, state)
                            rj_accepted += rj_accepted_out
                            # Again commenting out this section: We do not control temperature on RJ moves
                            # if self.ntemps > 1:
                            #     rj_swaps = rj_move.temperature_control.swaps_accepted
                            # else:
                            #     rj_swaps = None
                            rj_swaps = None

                            state.random_state = self.random_state

                            if tune:
                                rj_move.tune(state, rj_accepted_out)

                    else:
                        rj_accepted = None
                        rj_swaps = None

                    # Save the new step
                    if store and (i + 1) % self.checkpoint_step == 0:
                        if self.track_moves:
                            moves_accepted_fraction = {
                                key: move_tmp.acceptance_fraction
                                for key, move_tmp in self.all_moves.items()
                            }
                        else:
                            moves_accepted_fraction = None

                        self.backend.save_step(
                            state,
                            accepted,
                            rj_accepted=rj_accepted,
                            swaps_accepted=in_model_swaps,
                            moves_accepted_fraction=moves_accepted_fraction,
                        )

                    # update after diagnostic and stopping check
                    # if updating and using burn_in, need to make sure it does not use
                    # previous chain samples since they are not stored.
                    if (
                        self.update_iterations > 0
                        and self.update_fn is not None
                        and (i + 1) % (self.update_iterations) == 0
                    ):
                        self.update_fn(i, state, self)

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state

    def run_mcmc(
        self, initial_state, nsteps, burn=None, post_burn_update=False, **kwargs
    ):
        """
        Iterate :func:`sample` for ``nsteps`` iterations and return the result.

        Args:
            initial_state (State or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            nsteps (int): The number of steps to generate. The total number of proposals is ``nsteps * thin_by``.
            burn (int, optional): Number of burn steps to run before storing information. The ``thin_by`` kwarg is ignored when counting burn steps since there is no storage (equivalent to ``thin_by=1``).
            post_burn_update (bool, optional): If ``True``, run ``update_fn`` after burn in.

        Other parameters are directly passed to :func:`sample`.

        Returns:
            State: This method returns the most recent result from :func:`sample`.

        Raises:
            ValueError: ``If initial_state`` is None and ``run_mcmc`` has never been called.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        # run burn in
        if burn is not None and burn != 0:
            # prepare kwargs that relate to burn
            burn_kwargs = deepcopy(kwargs)
            burn_kwargs["store"] = False
            burn_kwargs["thin_by"] = 1
            i = 0
            for results in self.sample(initial_state, iterations=burn, **burn_kwargs):
                i += 1

            # run post-burn update
            if post_burn_update and self.update_fn is not None:
                self.update_fn(i, results, self)

            initial_state = results

        if nsteps == 0:
            return initial_state

        results = None

        i = 0
        for results in self.sample(initial_state, iterations=nsteps, **kwargs):
            # diagnostic plots
            # TODO: adjust diagnostic plots
            if self.plot_iterations > 0 and (i + 1) % (self.plot_iterations) == 0:
                self.plot_generator.produce_plots(sampler=self)  # TODO: remove defaults

            # check for stopping before updating
            if (
                self.stopping_iterations > 0
                and self.stopping_fn is not None
                and (i + 1) % (self.stopping_iterations) == 0
            ):
                stop = self.stopping_fn(i, results, self)

                if stop:
                    break

            i += 1
            
        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results

    def compute_log_prior(self, coords, inds=None, supps=None, branch_supps=None):
        """Calculate the vector of log-prior for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the ``inds`` np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)

        Returns:
            np.ndarray[ntemps, nwalkers]: Prior Values

        """

        # get number of temperature and walkers
        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        # independent-samplers row mask on the folded leading axis; only
        # applied when the input carries the full folded layout (some moves,
        # e.g. multiple-try internals, call this wrapper with other shapes)
        samplers_running = getattr(self, "_samplers_running", None)
        row_mask = None
        if samplers_running is not None and ntemps == self.ntemps_eff:
            running = np.asarray(samplers_running, dtype=bool)
            if not running.all():
                row_mask = np.repeat(running, self.ntemps)

        if inds is None:
            # default use all sources
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # take information out of dict and spread to x1..xn
        x_in = {}

        # for completely customizable priors
        if "all_models_together" in self.priors:
            prior_out = self.priors["all_models_together"].logpdf(
                coords, inds, supps=supps, branch_supps=branch_supps
            )
            assert prior_out.shape == (ntemps, nwalkers)

        elif self.provide_groups:
            # get leaf-group information from the inds dict
            # ("leaf groups" = leaf-groupings per likelihood call; unrelated to
            # the independent-samplers axis)
            leaf_groups = groups_from_inds(inds)

            # get the coordinates that are used
            for i, (name, coords_i) in enumerate(coords.items()):
                x_in[name] = coords_i[inds[name]]

            prior_out = np.zeros((ntemps * nwalkers))
            for name in x_in:
                # get prior for individual binaries
                prior_out_temp = self.priors[name].logpdf(x_in[name])

                # arrange prior values by leaf groups
                # TODO: vectorize this?
                for i in np.unique(leaf_groups[name]):
                    # which members are in the leaf group i
                    inds_temp = np.where(leaf_groups[name] == i)[0]
                    # num_in_group = len(inds_temp)

                    # add to the prior for this leaf group
                    prior_out[i] += prior_out_temp[inds_temp].sum()

            # reshape
            prior_out = prior_out.reshape(ntemps, nwalkers)

        else:
            prior_out = np.zeros((ntemps, nwalkers))
            for name, coords_i in coords.items():
                ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

                if (
                    self.prior_transform_fn is not None
                    and name in self.prior_transform_fn
                    and ntemps == self.ntemps_eff
                ):
                    # per-sampler prior-basis transform (para parity):
                    # slice to the running samplers, transform in place,
                    # evaluate, Jacobian-adjust in place, scatter back with
                    # -inf on non-running samplers (nleaves_max == 1 enforced
                    # at initialization)
                    running = (
                        np.full(self.nsamplers, True)
                        if samplers_running is None
                        else np.asarray(samplers_running, dtype=bool)
                    )
                    running_idx = np.arange(self.nsamplers)[running]

                    coords_buffer = coords_i.reshape(
                        self.nsamplers, self.ntemps, nwalkers, ndim
                    )[running_idx].copy()
                    self.prior_transform_fn[name].transform_to_prior_basis(
                        coords_buffer, running_idx
                    )

                    logp_tmp = (
                        self.priors[name]
                        .logpdf(coords_buffer.reshape(-1, ndim))
                        .reshape(len(running_idx), self.ntemps, nwalkers)
                    )
                    self.prior_transform_fn[name].adjust_logp(logp_tmp, running_idx)

                    prior_out_branch = np.full(
                        (self.nsamplers, self.ntemps, nwalkers), -np.inf
                    )
                    prior_out_branch[running_idx] = logp_tmp
                    prior_out += prior_out_branch.reshape(ntemps, nwalkers)

                else:
                    # flatten coordinate arrays
                    x_in[name] = coords_i.reshape(-1, ndim)
                    prior_out_temp = (
                        self.priors[name]
                        .logpdf(x_in[name])
                        .reshape(ntemps, nwalkers, nleaves_max)
                    )

                    # fix any infs / nans from binaries that are not being used (inds == False)
                    prior_out_temp[~inds[name]] = 0.0

                    # vectorized because everything is rectangular (no groups to indicate model difference)
                    prior_out += prior_out_temp.sum(axis=-1)

        # rows of non-running samplers get -inf: proposals there always reject
        # and the likelihood wrapper skips them entirely
        if row_mask is not None:
            prior_out[~row_mask] = -np.inf

        if np.any(np.isnan(prior_out)):
            raise ValueError("The prior function is returning Nan.")

        return prior_out

    def compute_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        """Calculate the vector of log-likelihood for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the inds np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)
            logp (np.ndarray[ntemps, nwalkers], optional): Log prior values associated
                with all walkers. If not provided, it will be calculated because
                if a walker has logp = -inf, its likelihood is not calculated.
                This prevents evaluting likelihood outside the prior.
                (default: ``None``)

        Returns:
            tuple: Carries log-likelihood and blob information.
                First entry is np.ndarray[ntemps, nwalkers] with values corresponding
                to the log likelihood of each walker. Second entry is ``blobs``.

         Raises:
            ValueError: Infinite or NaN values in parameters.

        """

        # if inds not provided, use all
        if inds is None:
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # Check that the parameters are in physical ranges.
        for name, ptemp in coords.items():
            if np.any(np.isinf(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was infinite")
            if np.any(np.isnan(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was NaN")

        # if no prior values are added, compute_prior
        # this is necessary to ensure Likelihood is not evaluated outside of the prior
        if logp is None:
            logp = self.compute_log_prior(
                coords, inds=inds, supps=supps, branch_supps=branch_supps
            )

        # if all points are outside the prior
        if np.all(np.isinf(logp)):
            warnings.warn(
                "All points input for the Likelihood have a log prior of -inf."
            )
            return np.full_like(logp, -1e300), None

        # do not run log likelihood where logp = -inf
        inds_copy = deepcopy(inds)
        inds_bad = np.where(np.isinf(logp))
        for key in inds_copy:
            inds_copy[key][inds_bad] = False

            # if inds_keep in branch supps, indicate which to not keep
            if (
                branch_supps is not None
                and key in branch_supps
                and branch_supps[key] is not None
                and "inds_keep" in branch_supps[key]
            ):
                # TODO: indicate specialty of inds_keep in branch_supp
                branch_supps[key][inds_bad] = {"inds_keep": False}

        # take information out of dict and spread to x1..xn
        x_in = {}
        if self.provide_supplemental:
            if supps is None and branch_supps is None:
                raise ValueError(
                    """supps and branch_supps are both None. If self.provide_supplemental
                       is True, must provide some supplemental information."""
                )
            if branch_supps is not None:
                branch_supps_in = {}

        # determine leaf groupings from inds
        # ("leaf groups" = which leaves feed each individual likelihood call;
        # unrelated to the independent-samplers (nsamplers) axis)
        leaf_groups = groups_from_inds(inds_copy)

        # need to map group inds properly
        # this is the unique group indexes
        unique_groups = np.unique(
            np.concatenate([groups_i for groups_i in leaf_groups.values()])
        )

        # this is the map to those indexes that are used in the likelihood
        groups_map = np.arange(len(unique_groups))

        # get the indices with groups_map for the Likelihood
        ll_leaf_groups = {}
        for key, group in leaf_groups.items():
            # get unique groups in this sub-group (or branch)
            temp_unique_groups, inverse = np.unique(group, return_inverse=True)

            # use groups_map by finding where temp_unique_groups overlaps with unique_groups
            keep_groups = groups_map[np.isin(unique_groups, temp_unique_groups)]

            # fill group information for Likelihood
            ll_leaf_groups[key] = keep_groups[inverse]

        for i, (name, coords_i) in enumerate(coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords_i.shape
            nwalkers_all = ntemps * nwalkers

            # fill x_values properly into dictionary
            x_in[name] = coords_i[inds_copy[name]]

            # prepare branch supplementals for each branch
            if self.provide_supplemental:
                if branch_supps is not None:  #  and
                    if branch_supps[name] is not None:
                        # index the branch supps
                        # it will carry in a dictionary of information
                        branch_supps_in[name] = branch_supps[name][inds_copy[name]]
                    else:
                        # fill with None if this branch does not have a supplemental
                        branch_supps_in[name] = None

        # deal with overall supplemental not specific to the branches
        if self.provide_supplemental:
            if supps is not None:
                # get the flattened supplemental
                # this will produce the shape (ntemps * nwalkers,...)
                temp = supps.flat

                # unique_groups will properly index the flattened array
                supps_in = {
                    name: values[unique_groups] for name, values in temp.items()
                }

        # prepare group information
        # this gets the group_map indexing into a list
        groups_in = list(ll_leaf_groups.values())

        # if only one branch, take the group array out of the list
        if len(groups_in) == 1:
            groups_in = groups_in[0]

        # list of paramter arrays
        params_in = list(x_in.values())

        # Likelihoods are vectorized across groups
        if self.vectorize:
            # prepare args list
            args_in = []

            # when vectorizing, if params_in has one entry, take out of list
            if len(params_in) == 1:
                params_in = params_in[0]

            # add parameters to args
            args_in.append(params_in)

            # if providing groups, add to args
            if self.provide_groups:
                args_in.append(groups_in)

            # prepare supplementals as kwargs to the Likelihood
            kwargs_in = {}
            if self.provide_supplemental:
                if supps is not None:
                    kwargs_in["supps"] = supps_in
                if branch_supps is not None:
                    # get list of branch_supps values
                    branch_supps_in_2 = list(branch_supps_in.values())

                    # if only one entry, take out of list
                    if len(branch_supps_in_2) == 1:
                        kwargs_in["branch_supps"] = branch_supps_in_2[0]

                    else:
                        kwargs_in["branch_supps"] = branch_supps_in_2

            # provide args, kwargs as a tuple
            args_and_kwargs = (args_in, kwargs_in)

            # get vectorized results
            results = self.log_like_fn(args_and_kwargs)

        # each Likelihood is computed individually
        else:
            # if groups in is an array, need to put it in a list.
            if isinstance(groups_in, np.ndarray):
                groups_in = [groups_in]

            # prepare input args for all Likelihood calls
            # to be spread out with map functions below
            args_in = []

            # each individual group in the groups_map
            for group_i in groups_map:
                # args and kwargs for the individual Likelihood
                arg_i = [None for _ in self.branch_names]
                kwarg_i = {}

                # iterate over the group information from the branches
                for branch_i, groups_in_set in enumerate(groups_in):
                    # which entries in this branch are in the overall group tested
                    # this accounts for multiple leaves (or model counts)
                    inds_keep = np.where(groups_in_set == group_i)[0]

                    branch_name_i = self.branch_names[branch_i]

                    if inds_keep.shape[0] > 0:
                        # get parameters

                        params = params_in[branch_i][inds_keep]

                        # if leaf count is constant and leaf count is 1
                        # just give 1D parameters
                        if not self.has_reversible_jump and params.shape[0] == 1:
                            params = params[0]

                        # add them to the specific args for this Likelihood
                        arg_i[branch_i] = params
                        if self.provide_supplemental:
                            if supps is not None:
                                # supps are specific to each group
                                kwarg_i["supps"] = {
                                    key: supps_in[key][group_i] for key in supps_in
                                }
                            if branch_supps is not None:
                                # make sure there is a dictionary ready in this kwarg dictionary
                                if "branch_supps" not in kwarg_i:
                                    kwarg_i["branch_supps"] = {}

                                # fill these branch supplementals for the specific group
                                if branch_supps_in[branch_name_i] is not None:
                                    # get list of branch_supps values
                                    kwarg_i["branch_supps"][branch_name_i] = (
                                        branch_supps_in[branch_name_i][inds_keep]
                                    )
                                else:
                                    kwarg_i["branch_supps"][branch_name_i] = None

                # if only one model type, will take out of groups
                add_term = arg_i[0] if len(groups_in) == 1 else arg_i

                # based on how this is dealth with in the _FunctionWrapper
                # add_term is wrapped in a list
                args_in.append([[add_term], kwarg_i])

            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map

            else:
                map_func = map

            # get results and turn into an array
            results = np.asarray(list(map_func(self.log_like_fn, args_in)))

        assert isinstance(results, np.ndarray)

        # -1e300 because -np.inf screws up state acceptance transfer in proposals
        ll = np.full(nwalkers_all, -1e300)
        inds_fix_zeros = np.delete(np.arange(nwalkers_all), unique_groups)

        # make sure second dimension is not 1
        if results.ndim == 2 and results.shape[1] == 1:
            results = np.squeeze(results)

        # parse the results if it has blobs
        if results.ndim == 2:
            # get the results and put into groups that were analyzed
            ll[unique_groups] = results[:, 0]

            # fix groups that were not analyzed
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            # deal with blobs
            _blobs_out = np.zeros((nwalkers_all, results.shape[1] - 1))
            _blobs_out[unique_groups] = results[:, 1:]
            blobs_out = _blobs_out.reshape(ntemps, nwalkers)

        elif results.dtype == "object":
            # TODO: check blobs and add this capability
            raise NotImplementedError

        else:
            # no blobs
            ll[unique_groups] = results
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            blobs_out = None

        if False:  # self.provide_supplemental:
            # TODO: need to think about how to return information, we may need to add a function to do that
            if branch_supps is not None:
                for name_i, name in enumerate(branch_supps):
                    if branch_supps[name] is not None:
                        # TODO: better way to do this? limit to
                        if "inds_keep" in branch_supps[name]:
                            inds_back = branch_supps[name][:]["inds_keep"]
                            inds_back2 = branch_supps_in[name]["inds_keep"]
                        else:
                            inds_back = inds_copy[name]
                            inds_back2 = slice(None)
                        try:
                            branch_supps[name][inds_back] = {
                                key: branch_supps_in_2[name_i][key][inds_back2]
                                for key in branch_supps_in_2[name_i]
                            }
                        except ValueError:
                            breakpoint()
                            branch_supps[name][inds_back] = {
                                key: branch_supps_in_2[name_i][key][inds_back2]
                                for key in branch_supps_in_2[name_i]
                            }

        if np.any(np.isnan(ll)):
            raise ValueError("The likelihood function is returning Nan.")
        
        # return Likelihood and blobs
        return ll.reshape(ntemps, nwalkers), blobs_out

    @property
    def acceptance_fraction(self):
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend.iteration)

    @property
    def rj_acceptance_fraction(self):
        """The fraction of proposed reversible jump steps that were accepted"""
        if self.has_reversible_jump:
            return self.backend.rj_accepted / float(self.backend.iteration)
        else:
            return None

    @property
    def swap_acceptance_fraction(self):
        """The fraction of proposed temperature swaps that were accepted"""
        return self.backend.swaps_accepted / float(
            self.backend.iteration * self.nwalkers
        )

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_like(self, **kwargs):
        return self.backend.get_log_like(**kwargs)

    get_log_like.__doc__ = Backend.get_log_prior.__doc__

    def get_log_prior(self, **kwargs):
        return self.backend.get_log_prior(**kwargs)

    get_log_prior.__doc__ = Backend.get_log_prior.__doc__

    def get_log_posterior(self, **kwargs):
        return self.backend.get_log_posterior(**kwargs)

    get_log_posterior.__doc__ = Backend.get_log_posterior.__doc__

    def get_inds(self, **kwargs):
        return self.get_value("inds", **kwargs)

    get_inds.__doc__ = Backend.get_inds.__doc__

    def get_nleaves(self, **kwargs):
        return self.backend.get_nleaves(**kwargs)

    get_nleaves.__doc__ = Backend.get_nleaves.__doc__

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    get_last_sample.__doc__ = Backend.get_last_sample.__doc__

    def get_betas(self, **kwargs):
        return self.backend.get_betas(**kwargs)

    get_betas.__doc__ = Backend.get_betas.__doc__

    def get_value(self, name, **kwargs):
        """Get a specific value"""
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        """Compute autocorrelation time through backend."""
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(
        self,
        f,
        args,
        kwargs,
    ):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, args_and_kwargs):
        """
        Internal function that takes a tuple (args, kwargs) for entrance into the Likelihood.

        ``self.args`` and ``self.kwargs`` are added to these inputs.

        """

        args_in_add, kwargs_in_add = args_and_kwargs

        try:
            args_in = args_in_add + type(args_in_add)(self.args)
            kwargs_in = {**kwargs_in_add, **self.kwargs}

            out = self.f(*args_in, **kwargs_in)
            return out

        except:  # pragma: no cover
            import traceback

            print("eryn: Exception while calling your likelihood function:")
            print("  args added:", args_in_add)
            print("  args:", self.args)
            print("  kwargs added:", kwargs_in_add)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def walkers_independent(coords_in):
    """Determine if walkers are independent

    Orginall from ``emcee``.

    Args:
        coords_in (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): Coordinates of the walkers.

    Returns:
        bool: If walkers are independent.

    """
    # make sure it is 4-dimensional and reshape
    # so it groups by temperature and walker
    assert coords_in.ndim == 4
    ntemps, nwalkers, nleaves_max, ndim = coords_in.shape
    coords = coords_in.reshape(ntemps * nwalkers, nleaves_max * ndim)

    # make sure all coordinates are finite
    if not np.all(np.isfinite(coords)):
        return False

    # roughly determine covariance information
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C**2, axis=0))
    C /= C_colsum
    return np.linalg.cond(C.astype(float)) <= 1e8
