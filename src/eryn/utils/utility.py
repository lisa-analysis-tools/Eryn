# -*- coding: utf-8 -*-

import warnings

import numpy as np
from scipy.special import logsumexp


def groups_from_inds(inds):
    """Convert inds to group information

    Args:
        inds (dict): Keys are ``branch_names`` and values are inds
            np.ndarrays[ntemps, nwalkers, nleaves_max] that specify
            which leaves are used in this step.

    Returns:
        dict: Dictionary with group information.
            Keys are ``branch_names`` and values are
            np.ndarray[total number of used leaves]. The array is flat.

    """
    # prepare output
    groups = {}
    for name, inds_temp in inds.items():

        # shape information
        ntemps, nwalkers, nleaves_max = inds_temp.shape
        num_groups = ntemps * nwalkers

        # place which group each active leaf belongs to along flattened array
        group_id = np.repeat(
            np.arange(num_groups).reshape(ntemps, nwalkers)[:, :, None],
            nleaves_max,
            axis=-1,
        )

        # fill new information
        groups[name] = group_id[inds_temp]

    return groups


def get_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
    """

    x = np.atleast_1d(x)
    m = [
        slice(None),
    ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2 ** np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]


def get_integrated_act(x, axis=0, window=50, fast=False, average=True):
    """
    Estimate the integrated autocorrelation time of a time series.
    See `Sokal's notes on
    MCMC and sample estimators for autocorrelation times.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param window: (optional)
        The size of the window to use. (default: 50)
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
    """

    if axis != 0:
        # TODO: need to check this
        raise NotImplementedError

    # Compute the autocorrelation function.
    if isinstance(x, dict):
        is_dict = True
        ndim_total = 0
        values_out = []
        ind_breaks = []
        for name, values in x.items():
            nsteps, ntemps, nwalkers, nleaves_max, ndim = values.shape
            ndim_total += ndim
            ind_breaks.append(ndim_total)
            values_out.append(values.reshape(nsteps, ntemps, nwalkers, -1))

        x_in = np.concatenate(values_out, axis=-1)

    elif isinstance(x, np.ndarray):
        is_dict = False
        x_in = x
    else:
        raise ValueError("x must be dictionary of np.ndarrays or an np.ndarray.")

    f = get_acf(x_in, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2 * np.sum(f[1:window])

    # N-dimensional case.
    m = [
        slice(None),
    ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2 * np.sum(f[tuple(m)], axis=axis)

    if average:
        tau = np.average(tau, axis=1)

    if is_dict:
        splits = np.split(tau, ind_breaks, axis=-1)
        out = {name: split for name, split in zip(x.keys(), splits)}

    else:
        out = tau

    return out


def thermodynamic_integration_log_evidence(betas, logls):
    """
    Thermodynamic integration estimate of the evidence.

    This function origindated in ``ptemcee``.

    Args:
        betas (np.ndarray[ntemps]): The inverse temperatures to use for the quadrature.
        logls (np.ndarray[ntemps]): The mean log-Likelihoods corresponding to ``betas`` to use for
            computing the thermodynamic evidence.
    Returns:
        tuple:   ``(logZ, dlogZ)``:
                Returns an estimate of the
                log-evidence and the error associated with the finite
                number of temperatures at which the posterior has been
                sampled.

    The evidence is the integral of the un-normalized posterior
    over all of parameter space:
    .. math::
        Z \\equiv \\int d\\theta \\, l(\\theta) p(\\theta)
    Thermodymanic integration is a technique for estimating the
    evidence integral using information from the chains at various
    temperatures.  Let
    .. math::
        Z(\\beta) = \\int d\\theta \\, l^\\beta(\\theta) p(\\theta)
    Then
    .. math::
        \\frac{d \\log Z}{d \\beta}
        = \\frac{1}{Z(\\beta)} \\int d\\theta l^\\beta p \\log l
        = \\left \\langle \\log l \\right \\rangle_\\beta
    so
    .. math::
        \\log Z(1) - \\log Z(0)
        = \\int_0^1 d\\beta \\left \\langle \\log l \\right\\rangle_\\beta
    By computing the average of the log-likelihood at the
    difference temperatures, the sampler can approximate the above
    integral.

    """

    # make sure they are the same length
    if len(betas) != len(logls):
        raise ValueError("Need the same number of log(L) values as temperatures.")

    # make sure they are in order
    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0.0:
        betas = np.concatenate((betas0, [0.0]))
        betas2 = np.concatenate((betas0[::2], [0.0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0.0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    # integrate by trapezoid rule (np.trapz was removed in numpy 2.0)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    logZ = -trapezoid(logls, betas)
    logZ2 = -trapezoid(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)


def stepping_stone_log_evidence(betas, logls, block_len=50, repeats=100):
    """
    Stepping stone approximation for the evidence calculation.

    Based on
    a. https://arxiv.org/abs/1810.04488 and
    b. doi: 10.1093/sysbio/syq085

    Args:
        betas (np.ndarray[ntemps]): The inverse temperatures to use for the quadrature.
        logls (np.ndarray[ntemps]): The mean log-Likelihoods corresponding to ``betas`` to use for
            computing the thermodynamic evidence.
        block_len (int): The length of each chain block to compute the evidence from. Useful for computing the error-bars.
        repeats (int): The number of repeats to compute the evidence (using the block above).

    Returns
        tuple:   ``(logZ, dlogZ)``:
            Returns an estimate of the
            log-evidence and the error associated with the finite
            number of temperatures at which the posterior has been
            sampled.
    """

    def calculate_stepping_stone(betas, logls):
        n = logls.shape[0]
        n_T = betas.shape[0]
        delta_betas = betas[1:] - betas[:-1]
        throwaways = np.any(~np.isfinite(logls), axis=1) # a safeguard against non-finite entries
        # each ratio is a mean over the retained samples: normalise each by log(n) once
        log_ratio = logsumexp(delta_betas * logls[~throwaways, :-1], axis=0) - np.log(n - np.sum(throwaways))
        return np.sum(log_ratio), log_ratio
    
    # make sure they are the same length
    if len(betas) != logls.shape[1]:
        raise ValueError(
            "Need the log(L).shape[1] to be the same as the number of temperatures."
        )

    # make sure they are in order
    order = np.argsort(betas)
    betas = betas[order]
    logls = logls[:, order, :]
    # Get all samples per temperature: (steps, ntemps, nwalkers) -> (steps * nwalkers, ntemps)
    logls = np.moveaxis(logls, 1, -1).reshape(-1, betas.shape[0])
    steps = logls.shape[0]  # Get number of samples

    logZ, _ = calculate_stepping_stone(betas, logls)

    # Estimate the evidence uncertainty (Maturana-Russel et. al. (2019))
    logZ_i = np.zeros(repeats)
    try:
        for i in range(repeats):
            idxs = [
                np.random.randint(i, i + block_len) for i in range(steps - block_len)
            ]
            logZ_i[i] = calculate_stepping_stone(betas, logls[idxs, :])[0]
        dlogZ = np.std(logZ_i)
    except ValueError:
        warnings.warn(
            "Warning: Failed to compute evidence uncertainty via Stepping Stone algorithm"
        )
        dlogZ = np.nan

    return logZ, dlogZ


def psrf(C, ndims, per_walker=False):
    """
    The Gelman - Rubin convergence diagnostic.
    A general approach to monitoring convergence of MCMC output of multiple walkers.
    The function makes a comparison of within-chain and between-chain variances.
    A large deviation between these two variances indicates non-convergence, and
    the output [Rhat] deviates from unity.

    By default, it combines the MCMC chains for all walkers, and then computes the
    Rhat for the first and last 1/3 parts of the traces. This can be tuned with the
    ``per_walker`` flag.

    Based on
    a. Brooks, SP. and Gelman, A. (1998) General methods for monitoring convergence
       of iterative simulations. Journal of Computational and Graphical Statistics, 7, 434-455
    b. Gelman, A and Rubin, DB (1992) Inference from iterative simulation using multiple sequences,
       Statistical Science, 7, 457-511.

    Args:
        C (np.ndarray[nwalkers, ndim]): The parameter traces. The MCMC chains.
        ndims (int): The dimensions
        per_walker (bool, optional): Do the test on the combined chains, or using
        each if the walkers separatelly.

    Returns
        tuple:   ``(Rhat, neff)``:
            Returns an estimate of the Gelman-Rubin convergence diagnostic ``Rhat``,
            and the effective number od samples ``neff``.

    Code taken from https://joergdietrich.github.io/emcee-convergence.html
    """
    if not per_walker:
        # Split the complete chains into three parts and perform the
        # diagnostic on the forst and last 1/3 of the chains.
        C = C.reshape(-1, ndims)
        n = int(np.floor(C[:, 0].shape[0] / 3))
        c1 = C[0:n, :]
        c2 = C[-n:, :]
        C = np.zeros((2, c1.shape[0], c1.shape[1]))
        C = np.array([c1, c2])

    ssq = np.var(C, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    θb = np.mean(C, axis=1)
    θbb = np.mean(θb, axis=0)
    m = C.shape[0]
    nn = C.shape[1]
    B = nn / (m - 1) * np.sum((θbb - θb) ** 2, axis=0)

    var_θ = (nn - 1) / nn * W + 1 / nn * B
    R̂ = np.sqrt(var_θ / W)
    return R̂

from typing import TYPE_CHECKING, TypeAlias, Union
if TYPE_CHECKING:
    # Strict typing context: Pylance will use this branch.
    try:
        import cupy as cp # type: ignore[import]
        NDArrayLike: TypeAlias = Union[np.ndarray, cp.ndarray]
    except ImportError:
        NDArrayLike: TypeAlias = np.ndarray
else:
    # Runtime context: Python executes this branch without loading cupy.
    # Set to np.ndarray so it evaluates as a valid Type at runtime without crashing.
    NDArrayLike: TypeAlias = np.ndarray  