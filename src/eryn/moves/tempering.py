# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from ..state import BranchSupplemental, State

__all__ = ["TemperatureControl"]


def make_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.

    This function is originally from ``ptemcee`` `github.com/willvousden/ptemcee <https://github.com/willvousden/ptemcee>`_.

    Temperatures are chosen according to the following algorithm:
    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
    information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
    posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:
    * If ``Tmax = inf``, raise an exception (insufficient information).
    * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
    * If ``Tmax`` and ``ntemps`` are specified:
    * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
    * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.`

    Args:
        ndim (int): The number of dimensions in the parameter space.
        ntemps (int, optional): If set, the number of temperatures to generate.
        Tmax (float, optional): If set, the maximum temperature for the ladder.

    Returns:
        np.ndarray[ntemps]: Output inverse temperature (beta) array.

    Raises:
        ValueError: Improper inputs.

    """

    # make sure all inputs are okay
    if type(ndim) != int or ndim < 1:
        raise ValueError("Invalid number of dimensions specified.")
    if ntemps is None and Tmax is None:
        raise ValueError("Must specify one of ``ntemps`` and ``Tmax``.")
    if Tmax is not None and Tmax <= 1:
        raise ValueError("``Tmax`` must be greater than 1.")
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError("Invalid number of temperatures specified.")

    # step size in temperature based on ndim
    tstep = np.array(
        [
            25.2741,
            7.0,
            4.47502,
            3.5236,
            3.0232,
            2.71225,
            2.49879,
            2.34226,
            2.22198,
            2.12628,
            2.04807,
            1.98276,
            1.92728,
            1.87946,
            1.83774,
            1.80096,
            1.76826,
            1.73895,
            1.7125,
            1.68849,
            1.66657,
            1.64647,
            1.62795,
            1.61083,
            1.59494,
            1.58014,
            1.56632,
            1.55338,
            1.54123,
            1.5298,
            1.51901,
            1.50881,
            1.49916,
            1.49,
            1.4813,
            1.47302,
            1.46512,
            1.45759,
            1.45039,
            1.4435,
            1.4369,
            1.43056,
            1.42448,
            1.41864,
            1.41302,
            1.40761,
            1.40239,
            1.39736,
            1.3925,
            1.38781,
            1.38327,
            1.37888,
            1.37463,
            1.37051,
            1.36652,
            1.36265,
            1.35889,
            1.35524,
            1.3517,
            1.34825,
            1.3449,
            1.34164,
            1.33847,
            1.33538,
            1.33236,
            1.32943,
            1.32656,
            1.32377,
            1.32104,
            1.31838,
            1.31578,
            1.31325,
            1.31076,
            1.30834,
            1.30596,
            1.30364,
            1.30137,
            1.29915,
            1.29697,
            1.29484,
            1.29275,
            1.29071,
            1.2887,
            1.28673,
            1.2848,
            1.28291,
            1.28106,
            1.27923,
            1.27745,
            1.27569,
            1.27397,
            1.27227,
            1.27061,
            1.26898,
            1.26737,
            1.26579,
            1.26424,
            1.26271,
            1.26121,
            1.25973,
        ]
    )

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
    else:
        # get correct step for dimension
        tstep = tstep[ndim - 1]

    # wheter to add the infinite temperature to the end
    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        # non-infinite temperatures will now have 1 less
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError(
                "Must specify at least one of ``ntemps" " and " "finite ``Tmax``."
            )

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with
        # infinity.
        betas = np.concatenate((betas, [0]))

    return betas


class TemperatureControl(object):
    """Controls the temperature ladder and operations in the sampler.

    All of the tempering features within Eryn are controlled from this class.
    This includes the evaluation of the tempered posterior, swapping between temperatures, and
    the adaptation of the temperatures over time. The adaptive tempering model can be
    found in the Eryn paper as well as the paper for `ptemcee`, which acted
    as a basis for the code below.

    Args:
        effective_ndim (int): Effective dimension used to determine temperatures if betas not given.
        nwalkers (int): Number of walkers in the sampler. Must maintain proper order of branches.
        ntemps (int, optional): Number of temperatures. If this is provided rather than ``betas``,
            :func:`make_ladder` will be used to generate the temperature ladder. (default: 1)
        betas (np.ndarray[ntemps], optional): If provided, will use as the array of inverse temperatures.
            (default: ``None``).
        Tmax (float, optional): If provided and ``betas`` is not provided, this will be included with
            ``ntemps`` when determing the temperature ladder with :func:`make_ladder`.
            See that functions docs for more information. (default: ``None``)
        adaptive (bool, optional): If ``True``, adapt the temperature ladder during sampling.
            (default: ``True``).
        adaptation_lag (int, optional): lag parameter from
            `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_. ``adaptation_lag`` must be
            much greater than ``adapation_time``. (default: 10000)
        adaptation_time (int, optional): initial amplitude of adjustments from
            `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_. ``adaptation_lag`` must be
            much greater than ``adapation_time``. (default: 100)
        stop_adaptation (int, optional): If ``stop_adaptation > 0``, the adapating will stop after
            ``stop_adaption`` steps. The number of steps is counted as the number times adaptation
            has happened which is generally once per sampler iteration. For example,
            if you only want to adapt temperatures during burn-in, you set ``stop_adaption = burn``.
            This can become complicated when using the repeating proposal options, so the
            user must be careful and verify constant temperatures in the backend.
            (default: -1)
        permute (bool, optional): If ``True``, permute the walkers in each temperature during
            swaps. (default: ``True``)
        skip_swap_supp_names (list, optional): List of strings that indicate supplemental keys that are not to be swapped.
            (default: ``[]``)
        nsamplers (int, optional): Number of independent samplers run simultaneously.
            Each sampler carries its own temperature ladder; the ladders are FOLDED
            into ``self.betas``, a flat array of length ``nsamplers * ntemps``
            (see ``betas_grouped``). Swaps only occur within a sampler's own ladder
            and each ladder adapts independently. (default: 1)

    Attributes:
        betas (np.ndarray[ntemps * nsamplers]): Flat (folded) inverse-temperature array.
            Row ``s * ntemps + t`` of the folded sampler arrays is temperature ``t``
            of independent sampler ``s``. For ``nsamplers == 1`` this is the usual
            ``(ntemps,)`` ladder.

    """

    def __init__(
        self,
        effective_ndim,
        nwalkers,
        ntemps=1,
        betas=None,
        Tmax=None,
        adaptive=True,
        adaptation_lag=10000,
        adaptation_time=100,
        stop_adaptation=-1,
        permute=True,
        skip_swap_supp_names=[],
        skip_swap_branches=[],
        nsamplers=1,
    ):

        if betas is None:
            if ntemps == 1:
                betas = np.array([1.0])
            else:
                # A compromise for building a temperature ladder for the case of rj.
                # We start by assuming that the dimensionality will be defined by the number of
                # components. We take that maximum divided by two, and multiply it with the higher
                # dimensional component.
                betas = make_ladder(effective_ndim, ntemps=ntemps, Tmax=Tmax)

        # store information
        self.nwalkers = nwalkers
        self.nsamplers = int(nsamplers)

        # normalize betas input into the flat, folded representation:
        # 1D input is one per-sampler ladder (tiled across samplers);
        # 2D input (nsamplers, ntemps) provides per-sampler ladders directly
        betas = np.asarray(betas)
        if betas.ndim == 2:
            if betas.shape[0] != self.nsamplers:
                raise ValueError(
                    f"2D betas must have shape (nsamplers, ntemps); got {betas.shape} with nsamplers={self.nsamplers}."
                )
            self.ntemps = ntemps = betas.shape[1]
            self.betas = betas.reshape(-1).copy()
        else:
            self.ntemps = ntemps = len(betas)
            self.betas = (
                np.tile(betas, self.nsamplers) if self.nsamplers > 1 else betas
            )

        self.permute = permute
        self.skip_swap_supp_names = skip_swap_supp_names
        self.skip_swap_branches = skip_swap_branches


        # number of times adapted
        self.time = 0

        # store adapting inf
        self.adaptive = adaptive
        self.adaptation_time, self.adaptation_lag = adaptation_time, adaptation_lag
        self.stop_adaptation = stop_adaptation

        if self.nsamplers == 1:
            self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)
        else:
            self.swaps_proposed = np.full(
                (self.nsamplers, self.ntemps - 1), self.nwalkers
            )

    @property
    def ntemps_eff(self):
        """Size of the folded leading axis: ``nsamplers * ntemps``."""
        return self.nsamplers * self.ntemps

    @property
    def betas_grouped(self):
        """View of ``betas`` with the sampler axis unfolded: shape ``(nsamplers, ntemps)``.

        This is a reshape view: in-place writes update the flat ``betas`` array.
        """
        return self.betas.reshape(self.nsamplers, self.ntemps)

    def compute_log_posterior_tempered(self, logl, logp, betas=None):
        """Compute the log of the tempered posterior

        Args:
            logl (np.ndarray): Log of the Likelihood. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            logp (np.ndarray): Log of the Prior. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            betas (np.ndarray[ntemps]): If provided, inverse temperatures as 1D array.
                If not provided, it will use ``self.betas``. (default: ``None``)

        Returns:
            np.ndarray: Log of the temperated posterior.

        Raises:
            AssertionError: Inputs are incorrectly shaped.

        """
        assert logl.shape == logp.shape
        tempered_logl = self.tempered_likelihood(logl, betas=betas)
        return tempered_logl + logp

    def tempered_likelihood(self, logl, betas=None):
        """Compute the log of the tempered Likelihood

        From `ptemcee`: "This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.
        Here, we find a singularity that demands more careful attention; we allow the
        likelihood to dominate the temperature, since wandering outside the
        likelihood support causes a discontinuity."

        Args:
            logl (np.ndarray): Log of the Likelihood. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            betas (np.ndarray[ntemps]): If provided, inverse temperatures as 1D array.
                If not provided, it will use ``self.betas``. (default: ``None``)

        Returns:
            np.ndarray: Log of the temperated Likelihood.

        Raises:
            ValueError: betas not provided if needed.

        """
        # perform calculation on 1D likelihoods.
        if logl.ndim == 1:
            if betas is None:
                raise ValueError(
                    "If inputing a 1D logl array, need to provide 1D betas array of the same length."
                )
            loglT = logl * betas

        else:
            if betas is None:
                betas = self.betas

            with np.errstate(invalid="ignore"):
                loglT = logl * betas[:, None]

        # anywhere the likelihood is nan, turn into -infinity
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def do_swaps_indexing(
        self,
        i,
        iperm_sel,
        i1perm_sel,
        dbeta,
        x,
        logP,
        logl,
        logp,
        inds=None,
        blobs=None,
        supps=None,
        branch_supps=None,
    ):

        # for x and inds, just do full copy
        x_temp = {name: np.copy(x[name]) for name in x}
        if inds is not None:
            inds_temp = {name: np.copy(inds[name]) for name in inds}
        if branch_supps is not None:
            branch_supps_temp = {
                name: deepcopy(branch_supps[name]) for name in branch_supps
            }

        logl_temp = np.copy(logl[i, iperm_sel])
        logp_temp = np.copy(logp[i, iperm_sel])
        logP_temp = np.copy(logP[i, iperm_sel])
        if blobs is not None:
            blobs_temp = np.copy(blobs[i, iperm_sel])
        if supps is not None:
            supps_temp = deepcopy(supps[i, iperm_sel])

        # swap from i1 to i
        for name in x:
            if name in self.skip_swap_branches:
                continue
            # coords first
            x[name][i, iperm_sel, :, :] = x[name][i - 1, i1perm_sel, :, :]

            # then inds
            if inds is not None:
                inds[name][i, iperm_sel, :] = inds[name][i - 1, i1perm_sel, :]

            # do something special for branch_supps in case in contains a large amount of data
            # that is heavy to copy
            if branch_supps[name] is not None:
                tmp = branch_supps[name][i - 1, i1perm_sel, :]

                for key in (self.skip_swap_supp_names):
                    if key in tmp:
                        tmp.pop(key)

                branch_supps[name][i, iperm_sel, :] = tmp
                """# where the inds are alive in the current permutation
                # need inds_temp because that is the original
                inds_i = np.where(inds_temp[name][i][iperm_sel])

                # gives the associated walker for each spot in the permuted array
                walker_inds_i = iperm_sel[inds_i[0]]

                # represents which permuted leaves are alive
                leaf_inds_i = inds_i[1]

                # all of these are at the same temperature
                temp_inds_i = np.full_like(leaf_inds_i, i)

                # repeat all for the i1 permutated temperature
                # need inds_temp because that is the original
                inds_i1 = np.where(inds_temp[name][i - 1][i1perm_sel])
                walker_inds_i1 = i1perm_sel[inds_i1[0]]
                leaf_inds_i1 = inds_i1[1]
                temp_inds_i1 = np.full_like(leaf_inds_i1, i - 1)

                # go through the values within each branch supplemental holder
                # do direct movement of things that need to change
                # rather than copying the whole thing
                for name2 in branch_supps[name].holder:
                    # store temperarily
                    bring_back_branch_supps = (
                        branch_supps[name]
                        .holder[name2][(temp_inds_i, walker_inds_i, leaf_inds_i)]
                        .copy()
                    )

                    # make switch from i1 to i
                    branch_supps[name].holder[name2][
                        (temp_inds_i, walker_inds_i, leaf_inds_i)
                    ] = branch_supps[name].holder[name2][
                        (temp_inds_i1, walker_inds_i1, leaf_inds_i1)
                    ]

                    # make switch from i to i1
                    branch_supps[name].holder[name2][
                        (temp_inds_i1, walker_inds_i1, leaf_inds_i1)
                    ] = bring_back_branch_supps"""

        # switch everythin else from i1 to i
        logl[i, iperm_sel] = logl[i - 1, i1perm_sel]
        logp[i, iperm_sel] = logp[i - 1, i1perm_sel]
        logP[i, iperm_sel] = logP[i - 1, i1perm_sel] - dbeta * logl[i - 1, i1perm_sel]
        if blobs is not None:
            blobs[i, iperm_sel] = blobs[i - 1, i1perm_sel]
        if supps is not None:
            tmp_supps = supps[i - 1, i1perm_sel]
            for key in (self.skip_swap_supp_names):
                if key in tmp_supps:
                    tmp_supps.pop(key)
            supps[i, iperm_sel] = tmp_supps

        # switch x from i to i1
        for name in x:
            if name in self.skip_swap_branches:
                continue
            x[name][i - 1, i1perm_sel, :, :] = x_temp[name][i, iperm_sel, :, :]
            if inds is not None:
                inds[name][i - 1, i1perm_sel, :] = inds_temp[name][i, iperm_sel, :]
            if branch_supps[name] is not None:
                tmp = branch_supps_temp[name][i, iperm_sel, :]

                for key in (self.skip_swap_supp_names):
                    if key in tmp:
                        tmp.pop(key)
                branch_supps[name][i - 1, i1perm_sel, :] = tmp

        # switch the rest from i to i1
        logl[i - 1, i1perm_sel] = logl_temp
        logp[i - 1, i1perm_sel] = logp_temp
        logP[i - 1, i1perm_sel] = logP_temp + dbeta * logl_temp

        if blobs is not None:
            blobs[i - 1, i1perm_sel] = blobs_temp
        if supps is not None:
            tmp_supps = supps_temp
            for key in (self.skip_swap_supp_names):
                if key in tmp_supps:
                    tmp_supps.pop(key)
            supps[i - 1, i1perm_sel] = tmp_supps

        return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

    def _apply_swaps_vectorized(
        self,
        idx_i,
        idx_i1,
        dbeta_sel,
        x,
        logP,
        logl,
        logp,
        inds=None,
        blobs=None,
        supps=None,
        branch_supps=None,
    ):
        """Swap accepted ``(row, walker)`` pairs between two rungs, vectorized.

        ``idx_i`` / ``idx_i1`` are matching ``(rows, walkers)`` fancy-index tuples
        for the higher / lower rung of each accepted pair. Because the two rungs
        occupy different rows of the folded axis, the two index sets are disjoint.
        ``dbeta_sel`` is the per-pair inverse-temperature difference.
        """
        # coordinates, inds, and branch supplementals
        for name in x:
            if name in self.skip_swap_branches:
                continue

            x_tmp = x[name][idx_i].copy()
            x[name][idx_i] = x[name][idx_i1]
            x[name][idx_i1] = x_tmp

            if inds is not None and inds[name] is not None:
                inds_tmp = inds[name][idx_i].copy()
                inds[name][idx_i] = inds[name][idx_i1]
                inds[name][idx_i1] = inds_tmp

            if branch_supps is not None and branch_supps[name] is not None:
                # fancy indexing returns copies, so these are safe to cross-assign
                bs_i = branch_supps[name][idx_i]
                bs_i1 = branch_supps[name][idx_i1]
                for key in self.skip_swap_supp_names:
                    bs_i.pop(key, None)
                    bs_i1.pop(key, None)
                branch_supps[name][idx_i] = bs_i1
                branch_supps[name][idx_i1] = bs_i

        # scalar per-walker quantities
        logl_tmp = logl[idx_i].copy()
        logp_tmp = logp[idx_i].copy()
        logP_tmp = logP[idx_i].copy()

        logl[idx_i] = logl[idx_i1]
        logp[idx_i] = logp[idx_i1]
        # idx_i1 rows are still unmodified here (disjoint from idx_i)
        logP[idx_i] = logP[idx_i1] - dbeta_sel * logl[idx_i1]

        logl[idx_i1] = logl_tmp
        logp[idx_i1] = logp_tmp
        logP[idx_i1] = logP_tmp + dbeta_sel * logl_tmp

        if blobs is not None:
            blobs_tmp = blobs[idx_i].copy()
            blobs[idx_i] = blobs[idx_i1]
            blobs[idx_i1] = blobs_tmp

        if supps is not None:
            s_i = supps[idx_i]
            s_i1 = supps[idx_i1]
            for key in self.skip_swap_supp_names:
                s_i.pop(key, None)
                s_i1.pop(key, None)
            supps[idx_i] = s_i1
            supps[idx_i1] = s_i

        return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

    def temperature_swaps(
        self, x, logP, logl, logp, inds=None, blobs=None, supps=None, branch_supps=None, compute_log_like=None, compute_log_prior=None, fancy_swap=False, permute_here=None, samplers_running=None
    ):
        """Perform parallel-tempering temperature swaps

        This function performs the swapping between neighboring temperatures. It cascades from
        high temperature down to low temperature. When ``nsamplers > 1``, swaps only occur
        within each independent sampler's own ladder (rows ``s * ntemps + t`` of the folded
        leading axis) and are vectorized across samplers.

        Args:
            x (dict): Dictionary with keys as branch names and values as coordinate arrays.
            logP (np.ndarray[ntemps, nwalkers]): Log of the posterior probability.
            logl (np.ndarray[ntemps, nwalkers]): Log of the Likelihood.
            logp (np.ndarray[ntemps, nwalkers]): Log of the prior probability.
            inds (dict, optional): Dictionary with keys as branch names and values as the index arrays
                indicating which leaves are used. (default: ``None``)
            blobs (object, optional): Blobs associated with each walker. (default: ``None``)
            supps (object, optional): :class:`eryn.state.BranchSupplemental` object. (default: ``None``)
            branch_supps (dict, optional): Dictionary with keys as branch names and values as
                :class:`eryn.state.BranchSupplemental` objects for each branch (can be ``None`` for some branches). (default: ``None``)
            samplers_running (np.ndarray[nsamplers], optional): Boolean mask of actively
                running samplers. Swaps are only proposed within running samplers. This
                masking is required for correctness: frozen samplers carry fill values
                (``-1e300``) in ``logl`` on every rung, so an unmasked swap would always
                accept (``paccept = 0``) and permute frozen coordinates.
                (default: ``None``)

        Returns:
            tuple: All of the information that was input now swapped (output in the same order as input).

        """
        if permute_here is None:
            permute_here = self.permute
        else:
            assert isinstance(permute_here, bool)

        ntemps, nwalkers, nsamplers = self.ntemps, self.nwalkers, self.nsamplers

        if fancy_swap and nsamplers > 1:
            raise NotImplementedError(
                "fancy_swap is not implemented for nsamplers > 1."
            )

        # active sampler indices
        if samplers_running is None:
            act = np.arange(nsamplers)
        else:
            act = np.arange(nsamplers)[np.asarray(samplers_running, dtype=bool)]
        nact = len(act)

        # prepare information on how many swaps are accepted this time
        if nsamplers == 1:
            self.swaps_accepted = np.zeros(ntemps - 1)
        else:
            self.swaps_accepted = np.zeros((nsamplers, ntemps - 1))

        if nact == 0:
            return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

        betas_grouped = self.betas_grouped

        # iterate from highest to lowest temperatures
        for i in range(ntemps - 1, 0, -1):

            # per-sampler difference in inverse temps at this rung
            dbeta = betas_grouped[act, i - 1] - betas_grouped[act, i]  # (nact,)

            # folded row indices of the two rungs for each active sampler
            rows_i = act * ntemps + i
            rows_i1 = act * ntemps + (i - 1)

            # permute the indices for the walkers in each temperature to randomize swap positions
            # NOTE: the per-sampler loop preserves the historical RNG stream for nsamplers == 1
            if permute_here:
                iperm = np.array([np.random.permutation(nwalkers) for _ in range(nact)])
                i1perm = np.array([np.random.permutation(nwalkers) for _ in range(nact)])

            # do not permute if desired
            else:
                iperm = np.tile(np.arange(nwalkers), (nact, 1))
                i1perm = np.tile(np.arange(nwalkers), (nact, 1))

            # random draw that produces log of the acceptance fraction
            raccept = np.log(np.random.uniform(size=(nact, nwalkers)))

            # log of the detailed balance fraction
            if not fancy_swap:
                paccept = dbeta[:, None] * (
                    logl[rows_i[:, None], iperm] - logl[rows_i1[:, None], i1perm]
                )
            else:
                # nsamplers == 1 only (guarded above)
                paccept = self.perform_fancy_swap_acceptance_fraction(
                    dbeta[0], i, iperm[0], i1perm[0],
                    x, logl, inds=inds, blobs=blobs, supps=supps, branch_supps=branch_supps, compute_log_like=compute_log_like
                ).reshape(1, -1)

            # How many swaps were accepted
            sel = paccept > raccept  # (nact, nwalkers)
            if nsamplers == 1:
                self.swaps_accepted[i - 1] = np.sum(sel)
            else:
                self.swaps_accepted[act, i - 1] = np.sum(sel, axis=-1)

            # build matching (row, walker) fancy-index tuples for the accepted pairs
            sel_flat = sel.reshape(-1)
            rows_i_mat = np.repeat(rows_i[:, None], nwalkers, axis=1).reshape(-1)
            rows_i1_mat = np.repeat(rows_i1[:, None], nwalkers, axis=1).reshape(-1)
            dbeta_mat = np.repeat(dbeta[:, None], nwalkers, axis=1).reshape(-1)

            idx_i = (rows_i_mat[sel_flat], iperm.reshape(-1)[sel_flat])
            idx_i1 = (rows_i1_mat[sel_flat], i1perm.reshape(-1)[sel_flat])
            dbeta_sel = dbeta_mat[sel_flat]

            (x, logP, logl, logp, inds, blobs, supps, branch_supps) = (
                self._apply_swaps_vectorized(
                    idx_i,
                    idx_i1,
                    dbeta_sel,
                    x,
                    logP,
                    logl,
                    logp,
                    inds=inds,
                    blobs=blobs,
                    supps=supps,
                    branch_supps=branch_supps,
                )
            )

            if fancy_swap:
                # need to recalculate likelihood and priors just in case
                if compute_log_prior is not None:
                    logp = compute_log_prior(x, inds=inds, supps=supps, branch_supps=branch_supps)

                assert compute_log_like is not None
                logl = compute_log_like(x, inds=inds, supps=supps, branch_supps=branch_supps, logp=logp)[0]

                logP = self.compute_log_posterior_tempered(logl, logp)

        return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

    def perform_fancy_swap_acceptance_fraction(self, dbeta, i, iperm, i1perm, x, logl, inds=None, blobs=None, supps=None, branch_supps=None, compute_log_like=None):
        
        old_like = logl[i-1 :i + 1]
        if inds is not None:
            raise NotImplementedError("Not Implemented for RJ.")
        
        if branch_supps is not None:
            branch_supps_temp = {}
            branch_supps_here = {}
            for name in branch_supps:
                if branch_supps[name] is not None:
                    branch_supps_temp[name] = deepcopy(branch_supps[name][i-1:i + 1])
                    branch_supps_here[name] = deepcopy(branch_supps[name][i-1:i + 1])
                else:
                    branch_supps_temp[name] = None
                    branch_supps_here[name] = None

        supps_here = None
        if supps is not None:
            supps_here = BranchSupplemental(supps[i-1:i + 1], base_shape=logl.shape)

        assert compute_log_like is not None
        x_here = {name: np.copy(x[name][i-1:i + 1]) for name in x}
        x_temp = {name: np.copy(x[name][i-1:i + 1]) for name in x}
        
        # swap from i1 to i
        for name in x_here:
            if name in self.skip_swap_branches:
                continue
            # coords first
            x_here[name][1, iperm, :, :] = x_here[name][0, i1perm, :, :]

            # do something special for branch_supps in case in contains a large amount of data
            # that is heavy to copy
            if branch_supps_here[name] is not None:
                tmp = branch_supps_here[name][0, i1perm, :]

                for key in (self.skip_swap_supp_names):
                    if key in tmp:
                        tmp.pop(key)

                branch_supps_here[name][1, iperm, :] = tmp

        for name in x_here:
            if name in self.skip_swap_branches:
                continue
            x_here[name][0, i1perm, :, :] = x_temp[name][1, iperm, :, :]
            
            if branch_supps_here[name] is not None:
                tmp = branch_supps_temp[name][1, iperm, :]

                for key in (self.skip_swap_supp_names):
                    if key in tmp:
                        tmp.pop(key)
                branch_supps_here[name][0, i1perm, :] = tmp

        new_like = compute_log_like(x_here, inds=inds, supps=supps_here, branch_supps=branch_supps_here)[0]
        # check_like = compute_log_like(x_temp, inds=inds, supps=supps, branch_supps=branch_supps_temp)

        # fancy paccept calculation. Slot k proposes the PAIR
        # (i, iperm[k]) <-> (i-1, i1perm[k]) -- the same pair the caller
        # applies on acceptance (temperature_swaps builds idx_i/idx_i1
        # from iperm/i1perm) -- so each rung's delta must be read at that
        # pair's own walker. Indexing by slot position instead (the
        # pre-2026-09-14 form, new_like[.] - old_like[.] unpermuted)
        # mixed the temp-i delta of one proposed pair with the temp-(i-1)
        # delta of a different one; it was exact only for the identity
        # permutation, i.e. never on the permuting path -- a detailed-
        # balance violation. Evaluating every pair's swap in ONE
        # compute_log_like call stays valid because each (rung, walker)
        # likelihood depends only on its own coordinates and data.
        beta_i = self.betas[i]
        beta_i1 = self.betas[i - 1]
        paccept = beta_i * (new_like[1][iperm] - old_like[1][iperm]) + beta_i1 * (
            new_like[0][i1perm] - old_like[0][i1perm]
        )

        return paccept

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        Axis-aware: accepts a 1D ladder ``(ntemps,)`` or stacked per-sampler
        ladders ``(nsamplers, ntemps)``; the temperature axis is always last.
        """
        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[..., :-1] - ratios[..., 1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[..., :-1], axis=-1)
        deltaTs *= np.exp(dSs)
        betas[..., 1:-1] = 1 / (
            np.cumsum(deltaTs, axis=-1) + 1 / betas[..., 0][..., None]
        )

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def adapt_temps(self, samplers_running=None):
        """Adapt the temperature ladder(s) from the last round of swaps.

        Args:
            samplers_running (np.ndarray[nsamplers], optional): Boolean mask of
                actively running samplers. Only running samplers' ladders adapt.
                (default: ``None``)

        """
        # determine ratios of swaps accepted to swaps proposed (the ladder is fixed)
        ratios = self.swaps_accepted / self.swaps_proposed

        # adapt if desired
        if self.adaptive and self.ntemps > 1:
            if self.stop_adaptation < 0 or self.time < self.stop_adaptation:
                if self.nsamplers == 1:
                    dbetas = self._get_ladder_adjustment(self.time, self.betas, ratios)
                    self.betas = self.betas + dbetas
                else:
                    ratios_grouped = ratios.reshape(self.nsamplers, self.ntemps - 1)
                    dbetas = self._get_ladder_adjustment(
                        self.time, self.betas_grouped, ratios_grouped
                    )
                    if samplers_running is not None:
                        dbetas = dbetas * np.asarray(samplers_running, dtype=bool)[
                            :, None
                        ]
                    self.betas = self.betas + dbetas.reshape(-1)

            # only increase time if it is adaptive.
            self.time += 1

    def temper_comps(self, state, adapt=True):
        """Perfrom temperature-related operations on a state.

        This includes making swaps and then adapting the temperatures for the next round.

        Args:
            state (object): Filled ``State`` object.
            adapt (bool, optional): If True, swaps are to be performed, but no
                adaptation is made. In this case, ``self.time`` does not increase by 1.
                (default: ``True``)

        Returns:
            :class:`eryn.state.State`: State object after swaps.

        """
        # get initial values
        logl = state.log_like
        logp = state.log_prior

        # independent-samplers information carried on the state
        nsamplers = getattr(state, "nsamplers", 1)
        samplers_running = getattr(state, "samplers_running", None)

        # do posterior just for the hell of it
        logP = self.compute_log_posterior_tempered(logl, logp)

        # make swaps
        x, logP, logl, logp, inds, blobs, supps, branch_supps = self.temperature_swaps(
            state.branches_coords,
            logP.copy(),
            logl.copy(),
            logp.copy(),
            inds=state.branches_inds,
            blobs=state.blobs,
            supps=state.supplemental,
            branch_supps=state.branches_supplemental,
            samplers_running=samplers_running,
        )

        if adapt and self.adaptive and self.ntemps > 1:
            self.adapt_temps(samplers_running=samplers_running)

        # create a new state out of the swapped information
        # TODO: make this more memory efficient?
        new_state = State(
            x,
            log_like=logl,
            log_prior=logp,
            blobs=blobs,
            inds=inds,
            betas=self.betas,
            supplemental=supps,
            branch_supplemental=branch_supps,
            random_state=state.random_state,
            nsamplers=nsamplers,
            samplers_running=samplers_running,
        )

        return new_state
