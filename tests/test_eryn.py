#!/usr/bin/env python
# coding: utf-8

# In[14]:


from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer, PeriodicContainer
from eryn.moves import (
    GaussianMove,
    StretchMove,
    CombineMove,
    DistributionGenerateRJ,
    GroupStretchMove,
)
from eryn.utils.utility import groups_from_inds
from eryn.backends import HDFBackend

import os
import unittest
import matplotlib.pyplot as plt
import numpy as np

# set random seed
np.random.seed(42)

import corner


# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()


def gaussian_pulse(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c**2))
    return f_x


def combine_gaussians(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template


def log_like_fn_gauss_pulse(params, t, data, sigma):
    template = combine_gaussians(t, params)

    ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll


def gaussian_pulse(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c**2))
    return f_x


def combine_gaussians(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template


def sine(x, a, b, c):
    f_x = a * np.sin(2 * np.pi * b * x + c)
    return f_x


def combine_sine(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += sine(t, *param)  # *params -> a, b, c
    return template


def log_like_fn_gauss_and_sine(params_both, t, data, sigma):
    params_gauss, params_sine = params_both
    template = np.zeros_like(t)

    if params_gauss is not None:
        template += combine_gaussians(t, params_gauss)

    if params_sine is not None:
        template += combine_sine(t, params_sine)

    ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll


class ErynTest(unittest.TestCase):
    def test_base(self):
        ndim = 5
        nwalkers = 99

        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        lims = 5.0
        priors_in = {
            f"x{i}": uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, invcov],
        )

        coords = priors.rvs(size=(nwalkers,))

        # check log_like
        log_like = np.asarray(
            [log_like_fn(coords[i], means, invcov) for i in range(nwalkers)]
        )

        # check log_prior
        log_prior = np.asarray([priors.logpdf(coords[i]) for i in range(nwalkers)])

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

        samples = ensemble.get_chain()["model_0"].reshape(-1, ndim)

        ll = ensemble.backend.get_log_like()
        lp = ensemble.backend.get_log_prior()

        samples = ensemble.get_chain()

        ensemble.backend.shape

        last_state = ensemble.backend.get_last_sample()

        last_state.branches

        last_state.branches["model_0"].coords

    def test_pt(self):
        # set up problem
        ndim = 5
        nwalkers = 100
        ntemps = 10

        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        lims = 5.0
        priors_in = {
             f"x{i}": uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        # fill kwargs dictionary
        tempering_kwargs = dict(ntemps=ntemps)

        # randomize throughout prior
        coords = priors.rvs(
            size=(
                ntemps,
                nwalkers,
            )
        )

        backend_test_file = "_hdf_backend_test_file.h5"
        # initialize sampler
        ensemble_pt = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, cov],
            backend=backend_test_file,
            tempering_kwargs=tempering_kwargs,
        )

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        ensemble_pt.run_mcmc(coords, nsteps, burn=burn, progress=False, thin_by=thin_by)

        for temp in range(ntemps):
            samples = ensemble_pt.get_chain()["model_0"][:, temp].reshape(-1, ndim)

        ll = ensemble_pt.backend.get_log_like()

        # check temperature index and branch_names
        cold_chain = ensemble_pt.backend.get_chain(discard=10, thin=2, temp_index=0, branch_names=["model_0"])
        os.remove(backend_test_file)

    def test_rj(self):
        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = {"gauss": 1}
        nleaves_min = {"gauss": 0}

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            # [3.3, -0.2, 0.1],
            # [2.6, -0.1, 0.1],
            # [3.4, 0.0, 0.1],
            # [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 0.00001

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"], ndim))}

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        # for nn in range(nleaves_max["gauss"]):
        #     if nn >= len(gauss_inj_params):
        #         # not going to add parameters for these unused leaves
        #         continue

        #     coords["gauss"][:, :, nn] = np.random.multivariate_normal(
        #         gauss_inj_params[nn],
        #         np.diag(np.ones(3) * sig1),
        #         size=(ntemps, nwalkers),
        #     )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"]), dtype=bool)}

        # turn False -> True for any binary in the sampler
        # inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                "amp": uniform_dist(2.5, 3.5),  # amplitude
                "mean": uniform_dist(t.min(), t.max()),  # mean
                "std": uniform_dist(0.01, 0.21),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        priors_gen = {"gauss": ProbDistContainer(priors["gauss"])}
        moves = GaussianMove(cov)
        rj_moves = [
            DistributionGenerateRJ(
                priors_gen, nleaves_min=nleaves_min, nleaves_max=nleaves_max
            ),
            DistributionGenerateRJ(
                priors_gen, nleaves_min=nleaves_min, nleaves_max=nleaves_max
            ),
        ]

        base_like = log_like_fn_gauss_pulse(np.asarray([]), t, y, sigma)
        
        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn_gauss_pulse,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            fill_zero_leaves_val=base_like,
            rj_moves=rj_moves,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 1000
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=1000, progress=False, thin_by=1
        )

        last_sample.branches["gauss"].nleaves

        nleaves = ensemble.get_nleaves()["gauss"]
        bns = (
            np.arange(1, nleaves_max["gauss"] + 2) - 0.5
        )  # Just to make it pretty and center the bins

        samples = ensemble.get_chain()["gauss"][:, 0].reshape(-1, ndim)

        # same as ensemble.get_chain()['gauss'][ensemble.get_inds()['gauss']]
        samples = samples[~np.isnan(samples[:, 0])]

        # check temperature index and branch_names
        cold_chain = ensemble.backend.get_chain(discard=10, thin=2, temp_index=1, branch_names=["gauss"])
        
        #means = np.asarray(gauss_inj_params)[:, 1]
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.hist(nleaves[:, 0].flatten(), np.arange(0, 3) - 0.5)
        # ax2.plot(t, y)
        # ax2.plot(t, injection)
        # plt.show()
        # breakpoint()

    def test_rj_multiple_branches(self):
        nwalkers = 20
        ntemps = 8
        ndims = {"gauss": 3, "sine": 3}
        nleaves_max = {"gauss": 8, "sine": 4}
        nleaves_min = {"gauss": 0, "sine": 0}

        branch_names = ["gauss", "sine"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        sine_inj_params = [
            [1.3, 10.1, 1.0],
            [0.8, 4.6, 1.2],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))
        injection += combine_sine(t, np.asarray(sine_inj_params))

        # set noise level
        sigma = 200.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"], ndims["gauss"])),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max["sine"], ndims["sine"])),
        }

        # make sure to start near the proper setup
        inds = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"]), dtype=bool),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max["sine"]), dtype=bool),
        }

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        # start with gaussians
        for nn in range(nleaves_max["gauss"]):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )
            inds["gauss"][:, :, nn] = True

        # next do sine waves
        for nn in range(nleaves_max["sine"]):
            if nn >= len(sine_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["sine"][:, :, nn] = np.random.multivariate_normal(
                sine_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)
            )
            inds["sine"][:, :, nn] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
            "sine": {
                0: uniform_dist(0.5, 1.5),  # amplitude
                1: uniform_dist(1.0, 20.0),  # mean
                2: uniform_dist(0.0, 2 * np.pi),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {
            "gauss": np.diag(np.ones(ndims["gauss"])) * factor,
            "sine": np.diag(np.ones(ndims["sine"])) * factor,
        }

        moves = GaussianMove(cov)

        fp = "_test_backend.h5"

        # just to test iterate_branches
        tmp = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn_gauss_and_sine,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves="iterate_branches",  # basic generation of new leaves from the prior
            backend=None,
        )
        del tmp

        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn_gauss_and_sine,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves="separate_branches",  # basic generation of new leaves from the prior
            backend=fp,
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        state.branches

        nsteps = 50
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        np.array(
            [
                last_sample.branches["gauss"].nleaves[0],
                last_sample.branches["sine"].nleaves[0],
            ]
        ).T

        nleaves_gauss = ensemble.get_nleaves()["gauss"]
        nleaves_sine = ensemble.get_nleaves()["sine"]

        samples = ensemble.get_chain()["gauss"][:, 0].reshape(-1, ndims["gauss"])

        # same as ensemble.get_chain()['gauss'][ensemble.get_inds()['gauss']]
        samples = samples[~np.isnan(samples[:, 0])]

        means = np.asarray(gauss_inj_params)[:, 1]

        os.remove(fp)

    def test_gibbs_sampling(self):
        nwalkers = 20
        ntemps = 8
        ndims = {"gauss": 3, "sine": 3}
        nleaves_max = {"gauss": 8, "sine": 2}  # same min and max means no changing
        nleaves_min = {"gauss": 0, "sine": 2}

        branch_names = ["gauss", "sine"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        sine_inj_params = [
            [1.3, 10.1, 1.0],
            [0.8, 4.6, 1.2],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))
        injection += combine_sine(t, np.asarray(sine_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"], ndims["gauss"])),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max["sine"], ndims["sine"])),
        }

        # make sure to start near the proper setup
        inds = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max["gauss"]), dtype=bool),
            "sine": np.ones((ntemps, nwalkers, nleaves_max["sine"]), dtype=bool),
        }

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        # start with gaussians
        for nn in range(nleaves_max["gauss"]):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )
            inds["gauss"][:, :, nn] = True

        # next do sine waves
        for nn in range(nleaves_max["sine"]):
            if nn >= len(sine_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["sine"][:, :, nn] = np.random.multivariate_normal(
                sine_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)
            )
            # inds["sine"][:, :, nn] = True  # already True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
            "sine": {
                0: uniform_dist(0.5, 1.5),  # amplitude
                1: uniform_dist(1.0, 20.0),  # mean
                2: uniform_dist(0.0, 2 * np.pi),  # sigma
            },
        }

        # for the Gaussian Move
        factor = 0.00001
        cov = {
            "gauss": np.diag(np.ones(ndims["gauss"])) * factor,
        }

        # pass boolean array of shape (nleaves_max["gauss"], ndims["gauss"])
        gibbs_sampling_gauss = [
            ("gauss", np.zeros((nleaves_max["gauss"], ndims["gauss"]), dtype=bool))
            for _ in range(nleaves_max["gauss"])
        ]

        for i in range(nleaves_max["gauss"]):
            gibbs_sampling_gauss[i][-1][i] = True

        gauss_move = GaussianMove(cov, gibbs_sampling_setup=gibbs_sampling_gauss)

        gibbs_sampling_sine = [
            ("sine", np.zeros((nleaves_max["sine"], ndims["sine"]), dtype=bool))
            for _ in range(2 * nleaves_max["sine"])
        ]
        for i in range(nleaves_max["sine"]):
            for j in range(2):
                if j == 0:
                    gibbs_sampling_sine[2 * i + j][-1][i, :2] = True
                else:
                    gibbs_sampling_sine[2 * i + j][-1][i, 2:] = True

        sine_move = StretchMove(
            live_dangerously=True, gibbs_sampling_setup=gibbs_sampling_sine
        )

        move = CombineMove([gauss_move, sine_move])

        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn_gauss_and_sine,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=move,
            rj_moves=True,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 50
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

    def test_utilities(self):
        def transform1(x, y):
            return x * y, y / x

        # this will do transform lambda x, y: (x**2, y**2) before transform1
        parameter_transforms = {
            0: lambda x: np.log(x),
            (1, 2): lambda x, y: (x**2, y**2),
            (0, 2): transform1,
        }

        fill_dict = {2: 0.0, 3: 1.0, 5: -1.0}
        input_basis = [0, 1, 4]
        output_basis = [0, 1, 2, 3, 4, 5]
        tc = TransformContainer(
            input_basis, output_basis, parameter_transforms=parameter_transforms, fill_dict=fill_dict
        )

        x = np.random.uniform(0.1, 4.0, size=(40, 3))

        # can copy and transpose values if needed
        out = tc.transform_base_parameters(x, copy=True, return_transpose=False)

        def lnprob(x1, group1, x2, group2, transform_containers):
            x = [x1, x2]
            for i, (x_i, transform) in enumerate(zip([x1, x2], transform_containers)):
                temp = transform.transform_base_parameters(
                    x_i, copy=True, return_transpose=False
                )
                x[i] = transform.fill_values(temp)

            ## do more in the likelihood here with transformed information

        # setup transforms for x1
        parameter_transforms1 = {0: lambda x: np.log(x)}

        # setup transforms for x2
        parameter_transforms2 = {(1, 2): lambda x, y: (x**2, y**2)}

        # fill dict for x1
        fill_dict1 = {2: 0.0, 3: 1.0, 5: -1.0}

        # fill dict for x2
        fill_dict2 = {1: -1.0}

        input_basis1 = [0, 1, 4]
        output_basis1 = [0, 1, 2, 3, 4, 5]

        input_basis2 = [0, 2, 3, 4]
        output_basis2 = [0, 1, 2, 3, 4]
        
        tcs = [
            TransformContainer(
                input_basis1, output_basis1, parameter_transforms=parameter_transforms1, fill_dict=fill_dict1
            ),
            TransformContainer(
                input_basis2, output_basis2, parameter_transforms=parameter_transforms2, fill_dict=fill_dict2
            ),
        ]

        num = 40
        x1 = np.random.uniform(0.1, 4.0, size=(num, 3))
        x2 = np.random.uniform(0.1, 4.0, size=(num, 4))

        group1 = np.arange(num)
        group2 = np.arange(num)

        # it can be added via args or kwargs in the ensemble sampler
        lnprob(x1, group1, x2, group2, tcs)

        # ### Periodic Container

        from eryn.utils import PeriodicContainer

        periodic = PeriodicContainer({"sine": {"phi0": 2 * np.pi}}, key_order={"sine": ["amp", "f", "phi0"]})
        ntemps, nwalkers, nleaves_max, ndim = (10, 100, 2, 3)

        params_before_1 = {
            "sine": np.random.uniform(
                0, 7.0, size=(ntemps * nwalkers, nleaves_max, ndim)
            )
        }
        params_before_2 = {
            "sine": np.random.uniform(
                0, 7.0, size=(ntemps * nwalkers, nleaves_max, ndim)
            )
        }

        distance = periodic.distance(params_before_1, params_before_2)

        # the max distance should be near half the period

        params_after_1 = periodic.wrap(params_before_1)

        # max after wrapping should be near the period

        # ### Stopping & Update Functions
        from eryn.utils import SearchConvergeStopping

        ndim = 5
        nwalkers = 100

        # mean
        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        # set prior limits
        lims = 50.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        stop = SearchConvergeStopping(n_iters=5, diff=0.01, verbose=False)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, invcov],
            stopping_fn=stop,
            stopping_iterations=5,
        )

        # starting positions
        # randomize throughout prior
        coords = priors.rvs(size=(nwalkers,))

        # check log_like
        log_like = np.asarray(
            [log_like_fn(coords[i], means, invcov) for i in range(nwalkers)]
        )

        # check log_prior
        log_prior = np.asarray([priors.logpdf(coords[i]) for i in range(nwalkers)])

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

    def test_group_stretch(self):

        from eryn.moves import GroupStretchMove

        class MeanGaussianGroupMove(GroupStretchMove):
            def __init__(self, **kwargs):
                # make sure kwargs get sent into group stretch parent class
                GroupStretchMove.__init__(self, **kwargs)

            def setup_friends(self, branches):

                # store cold-chain information
                friends = branches["gauss"].coords[0, branches["gauss"].inds[0]]
                means = friends[:, 1].copy()  # need the copy

                # take unique to avoid errors at the start of sampling
                self.means, uni_inds = np.unique(means, return_index=True)
                self.friends = friends[uni_inds]

                # sort
                inds_sort = np.argsort(self.means)
                self.friends[:] = self.friends[inds_sort]
                self.means[:] = self.means[inds_sort]

                # get all current means from all temperatures
                current_means = branches["gauss"].coords[branches["gauss"].inds, 1]

                # calculate their distances to each stored friend
                dist = np.abs(current_means[:, None] - self.means[None, :])

                # get closest friends
                inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends]

                # store in branch supplemental
                branches["gauss"].branch_supplemental[branches["gauss"].inds] = {
                    "inds_closest": inds_closest
                }

                # make sure to "turn off" leaves that are deactivated by setting their
                # index to -1.
                branches["gauss"].branch_supplemental[~branches["gauss"].inds] = {
                    "inds_closest": -np.ones(
                        (ntemps, nwalkers, nleaves_max, self.nfriends), dtype=int
                    )[~branches["gauss"].inds]
                }

            def fix_friends(self, branches):
                # when RJMCMC activates a new leaf, when it gets to this proposal, its inds_closest
                # will need to be updated

                # activated & does not have an assigned index
                fix = branches["gauss"].inds & (
                    np.all(
                        branches["gauss"].branch_supplemental[:]["inds_closest"] == -1,
                        axis=-1,
                    )
                )

                if not np.any(fix):
                    return

                # same process as above, only for fix
                current_means = branches["gauss"].coords[fix, 1]

                dist = np.abs(current_means[:, None] - self.means[None, :])
                inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends]

                branches["gauss"].branch_supplemental[fix] = {
                    "inds_closest": inds_closest
                }

                # verify everything worked
                fix_check = branches["gauss"].inds & (
                    np.all(
                        branches["gauss"].branch_supplemental[:]["inds_closest"] == -1,
                        axis=-1,
                    )
                )
                assert not np.any(fix_check)

            def find_friends(self, name, s, s_inds=None, branch_supps=None):

                # prepare buffer array
                friends = np.zeros_like(s)

                # determine the closest friends for s_inds == True
                inds_closest_here = branch_supps[name][s_inds]["inds_closest"]

                # take one at random
                random_inds = inds_closest_here[
                    np.arange(inds_closest_here.shape[0]),
                    np.random.randint(
                        self.nfriends, size=(inds_closest_here.shape[0],)
                    ),
                ]

                # store in buffer array
                friends[s_inds] = self.friends[random_inds]
                return friends

        # set random seed
        np.random.seed(42)

        def gaussian_pulse(x, a, b, c):
            f_x = a * np.exp(-((x - b) ** 2) / (2 * c**2))
            return f_x

        def combine_gaussians(t, params):
            template = np.zeros_like(t)
            for param in params:
                template += gaussian_pulse(t, *param)  # *params -> a, b, c
            return template

        def log_like_fn_gauss_pulse(params, t, data, sigma):
            template = combine_gaussians(t, params)

            ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
            return ll

        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = 8
        nleaves_min = 0

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max, ndim))}

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        for nn in range(nleaves_max):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue

            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        # factor = 0.00001
        # cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        # moves = GaussianMove(cov)
        nfriends = nwalkers
        moves = MeanGaussianGroupMove(nfriends=nfriends)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn_gauss_pulse,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves=True,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        from eryn.state import BranchSupplemental

        branch_supps = {
            "gauss": BranchSupplemental(
                {
                    "inds_closest": np.zeros(
                        inds["gauss"].shape + (nfriends,), dtype=int
                    )
                },
                base_shape=(ntemps, nwalkers, nleaves_max),
            )
        }
        state = State(
            coords,
            log_like=log_like,
            log_prior=log_prior,
            inds=inds,
            branch_supplemental=branch_supps,
        )

        nsteps = 2000
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        nleaves = ensemble.get_nleaves()["gauss"][:, 0].flatten()

    def test_mt(self):
        # Gaussian likelihood
        def log_like_fn(x, mu, invcov):
            diff = x - mu
            return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()

        ndim = 5
        nwalkers = 100

        # mean
        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        # set prior limits
        lims = 5.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        nwalkers = 20
        ntemps = 10
        nleaves_max = 1

        from eryn.moves import MTDistGenMove

        mt_prior = MTDistGenMove(priors, num_try=25, independent=True)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, invcov],
            moves=mt_prior,
            tempering_kwargs={"ntemps": ntemps},
        )

        # starting positions
        # randomize throughout prior
        coords = priors.rvs(size=(ntemps, nwalkers, 1))

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1

        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

        samples_out = ensemble.get_chain()["model_0"][:, 0].reshape(-1, ndim)

    def test_mt_rj(self):
        def gaussian_pulse(x, a, b, c):
            f_x = a * np.exp(-((x - b) ** 2) / (2 * c**2))
            return f_x

        def combine_gaussians(t, params):
            template = np.zeros_like(t)
            for param in params:
                template += gaussian_pulse(t, *param)  # *params -> a, b, c
            return template

        def log_like_fn_gauss_pulse(params, t, data, sigma):
            template = combine_gaussians(t, params)

            ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
            return ll

        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = 8
        nleaves_min = 0

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max, ndim))}

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        for nn in range(nleaves_max):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue

            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": ProbDistContainer(
                {
                    0: uniform_dist(2.5, 3.5),  # amplitude
                    1: uniform_dist(t.min(), t.max()),  # mean
                    2: uniform_dist(0.01, 0.21),  # sigma
                }
            )
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        moves = GaussianMove(cov)

        from eryn.moves import MTDistGenMoveRJ

        mt_rj_prior = MTDistGenMoveRJ(
            priors,
            nleaves_max={"gauss": nleaves_max},
            nleaves_min={"gauss": nleaves_min},
            num_try=25,
            rj=True,
        )

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn_gauss_pulse,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves=mt_rj_prior,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 20
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        nleaves = ensemble.get_nleaves()["gauss"]
        bns = (
            np.arange(1, nleaves_max + 2) - 0.5
        )  # Just to make it pretty and center the bins

    def test_2d_prior(self):
        cov = np.array([[0.8, -0.2], [-0.2, 0.4]])
        from scipy.stats import multivariate_normal

        priors_in = {(0, 1): multivariate_normal(cov=cov)}
        priors = ProbDistContainer(priors_in)
        prior_vals = priors.rvs(size=12)

    def test_periodic_stretch(self):
        from scipy import stats

        # 1. Verify that calling get_new_points on periodic branch without _lift raises ValueError
        P = 2 * np.pi
        periodic = PeriodicContainer({"p": {"theta": P}}, key_order={"p": ("theta",)})
        move = StretchMove(periodic=periodic)
        s = np.zeros((1, 10, 1, 1))
        c_temp = np.zeros((1, 10, 1, 1))
        with self.assertRaises(ValueError) as ctx:
            move.get_new_points("p", s, c_temp, 10, (1, 10, 1, 1), 0, np.random)
        self.assertIn("requires lift setup in get_proposal", str(ctx.exception))

        # 2. Detailed balance invariance test on periodic torus
        rng = np.random.RandomState(42)
        n_samples = 20000
        x = rng.uniform(0, P, size=(1, n_samples, 1, 1))
        c = rng.uniform(0, P, size=(1, n_samples, 1, 1))
        newpos, factors = move.get_proposal({"p": x}, {"p": [c]}, rng)
        y = newpos["p"]
        self.assertTrue(np.all(y >= 0.0) and np.all(y < P))

        # Flat prior -> alpha = min(1, exp(factors))
        log_alpha = factors[0]
        alpha = np.exp(np.clip(log_alpha, -500.0, 0.0))
        u = rng.uniform(0, 1, size=n_samples)
        accepted = u < alpha
        final_pos = np.where(accepted[:, None, None], y[0], x[0]).squeeze()
        ks_res = stats.kstest(final_pos, stats.uniform(loc=0.0, scale=P).cdf)
        self.assertGreater(ks_res.pvalue, 0.01)

        # 3. End-to-end MCMC sampling with EnsembleSampler on periodic parameters
        def log_like(x):
            return np.zeros(x.shape[0])

        priors = {"model_0": {0: uniform_dist(0.0, P)}}
        periodic_mcmc = PeriodicContainer({"model_0": {0: P}})
        coords = {"model_0": rng.uniform(0, P, size=(1, 12, 1, 1))}
        sampler = EnsembleSampler(
            12, 1, log_like, priors, periodic=periodic_mcmc, moves=StretchMove(periodic=periodic_mcmc)
        )
        sampler.run_mcmc(coords, 20, progress=False)
        chain = sampler.get_chain()["model_0"]
        self.assertTrue(np.all(chain >= 0.0) and np.all(chain < P))

    def test_periodic_group_stretch(self):
        from scipy import stats

        P = 4.0
        periodic = PeriodicContainer({"gb": {"phi": P}}, key_order={"gb": ("phi",)})
        rng = np.random.RandomState(123)
        n_pool = 100
        friends_pool = rng.uniform(0, P, size=(n_pool, 1))

        class ConcreteGroupStretch(GroupStretchMove):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def find_friends(self, name, s, s_inds=None, branch_supps=None):
                ntemps, nwalkers, nleaves, ndim = s.shape
                idx = rng.randint(0, n_pool, size=(ntemps, nwalkers, nleaves))
                res = friends_pool[idx]
                if s_inds is not None:
                    res = np.where(s_inds[:, :, :, None], res, -999.0)
                return res

        move = ConcreteGroupStretch(
            periodic=periodic, nfriends=20, lift_scale=1.5, lift_kmax=6
        )

        # Verify proposal properties
        s = rng.uniform(0, P, size=(1, 20, 1, 1))
        newpos, factors = move.get_proposal({"gb": s}, rng)
        self.assertIn("gb", newpos)
        self.assertTrue(np.all(newpos["gb"] >= 0.0) and np.all(newpos["gb"] < P))
        self.assertTrue(np.all(np.isfinite(factors)))
        self.assertIsNone(move._lift)
        self.assertIsNone(move._last_log_norm)

        # Verify RJ inactive leaf masking
        ntemps, nwalkers, nleaves = 2, 10, 4
        s_rj = rng.uniform(0, P, size=(ntemps, nwalkers, nleaves, 1))
        s_inds = rng.rand(ntemps, nwalkers, nleaves) > 0.5
        s_inds[:, :, 0] = True  # Ensure at least one active leaf
        newpos_rj, factors_rj = move.get_proposal(
            {"gb": s_rj}, rng, s_inds_all={"gb": s_inds}
        )
        self.assertTrue(np.all(np.isfinite(factors_rj)))

        # Invariance under uniform distribution on periodic torus
        n_samples = 20000
        x = rng.uniform(0, P, size=(1, n_samples, 1, 1))
        newpos_inv, factors_inv = move.get_proposal({"gb": x}, rng)
        y = newpos_inv["gb"]
        log_alpha = factors_inv[0]
        alpha = np.exp(np.clip(log_alpha, -500.0, 0.0))
        u = rng.uniform(0, 1, size=n_samples)
        accepted = u < alpha
        final_pos = np.where(accepted[:, None, None], y[0], x[0]).squeeze()
        ks_res = stats.kstest(final_pos, stats.uniform(loc=0.0, scale=P).cdf)
        self.assertGreater(ks_res.pvalue, 0.01)


if __name__ == "__main__":
    unittest.main()

