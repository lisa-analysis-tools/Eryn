import warnings
from abc import ABC
from copy import deepcopy
from multiprocessing.sharedctypes import Value

import numpy as np

# from scipy.special import logsumexp

try:
    import cupy as cp

    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    gpu_available = False

from ..priors import ProbDistContainer
from ..utils.utility import groups_from_inds
from .rj import ReversibleJumpMove

___ = ["MultipleTryMove"]


def logsumexp(a, axis=None, xp=None):
    if xp is None:
        xp = np

    max = xp.max(a, axis=axis)
    ds = a - max[:, None]

    sum_of_exp = xp.exp(ds).sum(axis=axis)
    return max + xp.log(sum_of_exp)

def get_mt_computations(logP, log_proposal_pdf, symmetric=False, xp=None):

    if xp is None:
        xp = np

        # set weights based on if symmetric
    if symmetric:
        log_importance_weights = logP
    else:
        log_importance_weights = logP - log_proposal_pdf

    # get the sum of weights
    log_sum_weights = logsumexp(log_importance_weights, axis=-1, xp=xp)

    # probs = wi / sum(wi)
    log_of_probs = log_importance_weights - log_sum_weights[:, None]

    # probabilities to choose try
    probs = xp.exp(log_of_probs)

    # draw based on likelihood
    inds_keep = (
        probs.cumsum(1) > xp.random.rand(probs.shape[0])[:, None]
    ).argmax(1)

    return log_importance_weights, log_sum_weights, inds_keep


def get_mt_computations(logP, log_proposal_pdf, symmetric=False, xp=None):

    if xp is None:
        xp = np

        # set weights based on if symmetric
    if symmetric:
        log_importance_weights = logP
    else:
        log_importance_weights = logP - log_proposal_pdf

    # get the sum of weights
    log_sum_weights = logsumexp(log_importance_weights, axis=-1, xp=xp)

    # probs = wi / sum(wi)
    log_of_probs = log_importance_weights - log_sum_weights[:, None]

    # probabilities to choose try
    probs = xp.exp(log_of_probs)

    # draw based on likelihood
    inds_keep = (probs.cumsum(1) > xp.random.rand(probs.shape[0])[:, None]).argmax(1)

    return log_importance_weights, log_sum_weights, inds_keep


class MultipleTryMove(ABC):
    """Generate multiple proposal tries.

    This class should be inherited by another proposal class
    with the ``@classmethods`` overwritten. See :class:`eryn.moves.MTDistGenMove`
    and :class:`MTDistGenRJ` for examples.

    Args:
        num_try (int, optional): Number of tries. (default: 1)
        independent (bool, optional): Set to ``True`` if the proposal is independent of the current points.
            (default: ``False``).
        symmetric (bool, optional): Set to ``True`` if the proposal is symmetric.
            (default: ``False``).
        rj (bool, optional): Set to ``True`` if this is a nested reversible jump proposal.
            (default: ``False``).
        **kwargs (dict, optional): for compatibility with other proposals.

        Raises:
            ValueError: Input issues.

    """

    def __init__(
        self,
        num_try=1,
        independent=False,
        symmetric=False,
        rj=False,
        use_gpu=None,
        **kwargs
    ):
        self.num_try = num_try

        self.independent = independent
        self.symmetric = symmetric
        self.rj = rj

        if self.rj:
            if self.symmetric or self.independent:
                raise ValueError(
                    "If rj==True, symmetric and independt must both be False."
                )

        self.use_gpu = use_gpu

    @property
    def xp(self):
        xp = cp if self.use_gpu else np
        return xp

    @classmethod
    def special_like_func(self, generated_coords, *args, inds_leaves_rj=None, **kwargs):
        """Calculate the Likelihood for sampled points.

        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            *args (tuple, optional): additional arguments passed by overwriting the
                ``get_proposal`` function and passing ``args_like`` keyword argument.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the
                ``get_proposal`` function and passing ``kwargs_like`` keyword argument.

        Returns:
            np.ndarray: Likelihood values with shape ``(generated_coords.shape[0], num_try).``

        Raises:
            NotImplementedError: Function not included.

        """
        raise NotImplementedError

    @classmethod
    def special_prior_func(self, generated_coords, *args, **kwargs):
        """Calculate the Prior for sampled points.

        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            *args (tuple, optional): additional arguments passed by overwriting the
                ``get_proposal`` function and passing ``args_prior`` keyword argument.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the
                ``get_proposal`` function and passing ``kwargs_prior`` keyword argument.

        Returns:
            np.ndarray: Prior values with shape ``(generated_coords.shape[0], num_try).``

        Raises:
            NotImplementedError: Function not included.

        """
        raise NotImplementedError

    @classmethod
    def special_generate_func(
        coords, random, size=1, *args, fill_tuple=None, fill_values=None, **kwargs
    ):
        """Generate samples and calculate the logpdf of their proposal function.

        Args:
            coords (np.ndarray): Current coordinates of walkers.
            random (obj): Random generator.
            *args (tuple, optional): additional arguments passed by overwriting the
                ``get_proposal`` function and passing ``args_generate`` keyword argument.
            size (int, optional): Number of tries to generate.
            fill_tuple (tuple, optional): Length 2 tuple with the indexing of which values to fill
                when generating. Can be used for auxillary proposals or reverse RJ proposals. First index is the index into walkers and the second index is
                the index into the number of tries. (default: ``None``)
            fill_values (np.ndarray): values to fill associated with ``fill_tuple``. Should
                have size ``(len(fill_tuple[0]), ndim)``. (default: ``None``).
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the
                ``get_proposal`` function and passing ``kwargs_generate`` keyword argument.

        Returns:
            tuple: (generated points, logpdf of generated points).

        Raises:
            NotImplementedError: Function not included.

        """
        raise NotImplementedError

    @classmethod
    def special_generate_logpdf(self, coords):
        """Get logpdf of generated coordinates.

        Args:
            coords (np.ndarray): Current coordinates of walkers.

        Returns:
            np.ndarray: logpdf of generated points.

        Raises:
            NotImplementedError: Function not included.
        """
        raise NotImplementedError

    def get_mt_log_posterior(self, ll, lp, betas=None):
        """Calculate the log of the posterior for all tries.

        Args:
            ll (np.ndarray): Log Likelihood values with shape ``(nwalkers, num_tries)``.
            lp (np.ndarray): Log Prior values with shape ``(nwalkers, num_tries)``.
            betas (np.ndarray, optional): Inverse temperatures to include in log Posterior computation.
                (default: ``None``)

        Returns:
            np.ndarray: Log of the Posterior with shape ``(nwalkers, num_tries)``.

        """
        if betas is None:
            ll_temp = ll.copy()
        else:
            assert isinstance(betas, self.xp.ndarray)
            if ll.ndim > 1:
                betas_tmp = self.xp.expand_dims(betas, ll.ndim - 1)
            else:
                betas_tmp = betas
            ll_temp = betas_tmp * ll

        return ll_temp + lp

    def readout_adjustment(self, out_vals, all_vals_prop, aux_all_vals):
        """Read out values from the proposal.

        Allows the user to read out any values from the proposal that may be needed elsewhere. This function must be overwritten.

        Args:
            out_vals (list): ``[logP_out, ll_out, lp_out, log_proposal_pdf_out, log_sum_weights]``.
            all_vals_prop (list): ``[logP, ll, lp, log_proposal_pdf, log_sum_weights]``.
            aux_all_vals (list): ``[aux_logP, aux_ll, aux_lp, aux_log_proposal_pdf, aux_log_sum_weights]``.

        """
        pass

    def get_mt_proposal(
        self,
        coords,
        random,
        args_generate=(),
        kwargs_generate={},
        args_like=(),
        kwargs_like={},
        args_prior=(),
        kwargs_prior={},
        betas=None,
        ll_in=None,
        lp_in=None,
        inds_leaves_rj=None,
        inds_reverse_rj=None,
    ):
        """Make a multiple-try proposal

        Here, ``nwalkers`` refers to all independent walkers which generally
        will mean ``nwalkers * ntemps`` in terms of the rest of the sampler.

        Args:
            coords (np.ndarray): Current coordinates of walkers.
            random (obj): Random generator.
            args_generate (tuple, optional): Additional ``*args`` to pass to generate function.
                Must overwrite ``get_proposal`` function to use these.
                (default: ``()``)
            kwargs_generate (dict, optional): Additional ``**kwargs`` to pass to generate function.
                (default: ``{}``)
                Must overwrite ``get_proposal`` function to use these.
            args_like (tuple, optional): Additional ``*args`` to pass to Likelihood function.
                Must overwrite ``get_proposal`` function to use these.
                (default: ``()``)
            kwargs_like (dict, optional): Additional ``**kwargs`` to pass to Likelihood function.
                Must overwrite ``get_proposal`` function to use these.
                (default: ``{}``)
            args_prior (tuple, optional): Additional ``*args`` to pass to Prior function.
                Must overwrite ``get_proposal`` function to use these.
                (default: ``()``)
            kwargs_prior (dict, optional): Additional ``**kwargs`` to pass to Prior function.
                Must overwrite ``get_proposal`` function to use these.
                (default: ``{}``)
            betas (np.ndarray, optional): Inverse temperatures passes to the proposal with shape ``(nwalkers,)``.
            ll_in (np.ndarray, optional): Log Likelihood values coming in for current coordinates. Must be provided
                if ``self.rj is True``. If ``self.rj is True``, must be nested.
                Also, for all proposed removals, this value must be the Likelihood with the binary
                removed so all proposals are pretending to add a binary.
                Useful if ``self.independent is True``. (default: ``None``)
            lp_in (np.ndarray, optional): Log Prior values coming in for current coordinates. Must be provided
                if ``self.rj is True``. If ``self.rj is True``, must be nested.
                Also, for all proposed removals, this value must be the Likelihood with the binary
                removed so all proposals are pretending to add a binary.
                Useful if ``self.independent is True``. (default: ``None``)
            inds_leaves_rj (np.ndarray, optional): Array giving the leaf index of each incoming walker.
                Must be provided if ``self.rj is True``. (default: ``None``)
            inds_reverse_rj (np.ndarray, optional): Array giving the walker index for which proposals are
                reverse proposal removing a leaf.
                Must be provided if ``self.rj is True``. (default: ``None``)

        Returns:
            tuple: (generated points, factors).

        Raises:
            ValueError: Inputs are incorrect.

        """

        # check if rj and make sure we have all the information in that case
        if self.rj:
            try:
                assert ll_in is not None and lp_in is not None
                assert inds_leaves_rj is not None and inds_reverse_rj is not None
            except AssertionError:
                raise ValueError(
                    "If using rj, must provide ll_in, lp_in, inds_leaves_rj, and inds_reverse_rj."
                )

            # if using reversible jump, fill first spot with values that are proposed to remove
            fill_tuple = (inds_reverse_rj, np.zeros_like(inds_reverse_rj))
            fill_values = coords[inds_reverse_rj]
        else:
            fill_tuple = None
            fill_values = None

        # generate new points and get log of the proposal probability
        generated_points, log_proposal_pdf = self.special_generate_func(
            coords,
            random,
            *args_generate,
            size=self.num_try,
            fill_values=fill_values,
            fill_tuple=fill_tuple,
            **kwargs_generate
        )

        # compute the Likelihood functions
        ll = self.special_like_func(
            generated_points, *args_like, inds_leaves_rj=inds_leaves_rj, **kwargs_like
        )

        # check for nans
        if self.xp.any(self.xp.isnan(ll)):
            warnings.warn("Getting nans for ll in multiple try.")
            ll[self.xp.isnan(ll)] = -1e300

        # compute the Prior functions
        lp = self.special_prior_func(
            generated_points, *args_prior, inds_leaves_rj=inds_leaves_rj, **kwargs_prior
        )

        # if rj, make proposal distribution for all other leaves the prior value
        # this will properly cancel the prior with the proposal for leaves that already exists
        if self.rj:
            log_proposal_pdf += lp_in[:, None]

        # get posterior distribution including tempering
        logP = self.get_mt_log_posterior(ll, lp, betas=betas)

        log_importance_weights, log_sum_weights, inds_keep = get_mt_computations(
            logP, log_proposal_pdf, symmetric=self.symmetric, xp=self.xp
        )

        # tuple of index arrays of which try chosen per walker
        inds_tuple = (self.xp.arange(len(inds_keep)), inds_keep)

        if self.rj:
            # this just ensures the cancellation of logP and aux_logP outside of proposal
            inds_tuple[1][inds_reverse_rj] = 0

        # get chosen prior, Likelihood, posterior information
        lp_out = lp[inds_tuple]
        ll_out = ll[inds_tuple]
        logP_out = logP[inds_tuple]

        # store this information for access outside of multiple try part
        self.mt_lp = lp_out
        self.mt_ll = ll_out

        # choose points and get the log of the proposal for storage
        generated_points_out = generated_points[inds_tuple].copy()  # theta^j
        log_proposal_pdf_out = log_proposal_pdf[inds_tuple]

        # prepare auxillary information based on if it is nested rj, independent, or not
        if self.independent:
            # if independent, all the tries can be repeated for the auxillary draws
            aux_ll = ll.copy()
            aux_lp = lp.copy()

            # sub in the generation pdf for the current coordinates
            aux_log_proposal_pdf_sub = self.special_generate_logpdf(coords)

            # set sub ll based on if it is provided
            if ll_in is None:
                aux_ll_sub = self.special_generate_like(coords)

            else:
                assert ll_in.shape[0] == coords.shape[0]
                aux_ll_sub = ll_in

            # set sub lp based on if it is provided
            if lp_in is None:
                aux_lp_sub = self.special_generate_prior(coords)

            else:
                assert lp_in.shape[0] == coords.shape[0]
                aux_lp_sub = lp_in

            # sub in this information from the current coordinates
            aux_ll[inds_tuple] = aux_ll_sub
            aux_lp[inds_tuple] = aux_lp_sub

            # get auxillary posterior
            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # get aux_log_proposal_pdf information
            aux_log_proposal_pdf = log_proposal_pdf.copy()
            aux_log_proposal_pdf[inds_tuple] = aux_log_proposal_pdf_sub

            # set auxillary weights
            aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

        elif self.rj:
            # in rj, set aux_ll and aux_lp to be repeats of the model with one less leaf
            aux_ll = np.repeat(ll_in[:, None], self.num_try, axis=-1)
            aux_lp = np.repeat(lp_in[:, None], self.num_try, axis=-1)

            # probability is the prior for the existing points
            aux_log_proposal_pdf = aux_lp.copy()

            # get log posterior
            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # get importance weights
            aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

        else:
            # generate auxillary points based on chosen new points
            aux_generated_points, aux_log_proposal_pdf = self.special_generate_func(
                generated_points_out,
                random,
                *args_generate,
                size=self.num_try,
                fill_tuple=inds_tuple,
                fill_values=generated_points_out,
                **kwargs_generate
            )

            # get ll, lp, and lP
            aux_ll = self.special_like_func(
                aux_generated_points, *args_like, **kwargs_like
            )

            aux_lp = self.special_prior_func(aux_generated_points)

            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # set auxillary weights
            if not self.symmetric:
                aux_log_importance_weights = aux_logP - aux_log_proposal_pdf_sub
            else:
                aux_log_importance_weights = aux_logP

        # chosen output old Posteriors
        aux_logP_out = aux_logP[inds_tuple]
        # get sum of log weights
        aux_log_sum_weights = logsumexp(aux_log_importance_weights, axis=-1, xp=self.xp)

        aux_log_proposal_pdf_out = aux_log_proposal_pdf[inds_tuple]
        # this is setup to make clear with the math.
        # setting up factors properly means the
        # final lnpdiff will be effectively be the ratio of the sums
        # of the weights

        # IMPORTANT: logP_out must be subtracted against log_sum_weights before anything else due to -1e300s.
        factors = (
            (aux_logP_out - aux_log_sum_weights)
            - aux_log_proposal_pdf_out
            + aux_log_proposal_pdf_out
        ) - ((logP_out - log_sum_weights) - log_proposal_pdf_out + log_proposal_pdf_out)

        if self.rj:
            # adjust all information for reverese rj proposals
            factors[inds_reverse_rj] *= -1
            self.mt_ll[inds_reverse_rj] = ll_in[inds_reverse_rj]
            self.mt_lp[inds_reverse_rj] = lp_in[inds_reverse_rj]

        # store output information
        self.aux_logP_out = aux_logP_out
        self.logP_out = logP_out
        self.aux_ll = aux_ll
        self.aux_lp = aux_lp

        self.log_sum_weights = log_sum_weights
        self.aux_log_sum_weights = aux_log_sum_weights

        if self.rj:
            self.inds_reverse_rj = inds_reverse_rj
            self.inds_forward_rj = np.delete(
                np.arange(coords.shape[0]), inds_reverse_rj
            )

        # prepare to readout any information the user would like in readout_adjustment
        out_vals = [logP_out, ll_out, lp_out, log_proposal_pdf_out, log_sum_weights]
        all_vals_prop = [logP, ll, lp, log_proposal_pdf, log_sum_weights]
        aux_all_vals = [
            aux_logP,
            aux_ll,
            aux_lp,
            aux_log_proposal_pdf,
            aux_log_sum_weights,
        ]
        self.readout_adjustment(out_vals, all_vals_prop, aux_all_vals)

        return (
            generated_points_out,
            factors,
        )

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)

        Raises:
            ValueError: Input issues.

        """

        # mutliple try is only made for one branch here
        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        # get main key
        key_in = list(branches_coords.keys())[0]
        self.key_in = key_in

        # get inds information
        if branches_inds is None:
            branches_inds = {}
            branches_inds[key_in] = np.ones(
                branches_coords[key_in].shape[:-1], dtype=bool
            )

        # Make sure for base proposals that there is only one leaf
        if np.any(branches_inds[key_in].sum(axis=-1) > 1):
            raise ValueError

        ntemps, nwalkers, _, _ = branches_coords[key_in].shape

        # get temperature information
        betas_here = np.repeat(
            self.temperature_control.betas[:, None],
            np.prod(branches_coords[key_in].shape[1:-1]),
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # previous Likelihoods in case proposal is independent
        ll_here = np.repeat(
            self.current_state.log_like[:, :, None],
            branches_coords[key_in].shape[2],
            axis=-1,
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # previous Priors in case proposal is independent
        lp_here = np.repeat(
            self.current_state.log_prior[:, :, None],
            branches_coords[key_in].shape[2],
            axis=-1,
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # get mt proposal
        generated_points, factors = self.get_mt_proposal(
            branches_coords[key_in][branches_inds[key_in]],
            random,
            betas=betas_here,
            ll_in=ll_here,
            lp_in=lp_here,
        )

        # store this information for access outside
        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)

        return (
            {key_in: generated_points.reshape(ntemps, nwalkers, 1, -1)},
            factors.reshape(ntemps, nwalkers),
        )


class MultipleTryMoveRJ(MultipleTryMove):
    def get_proposal(
        self,
        branches_coords,
        branches_inds,
        nleaves_min_all,
        nleaves_max_all,
        random,
        **kwargs
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (list): Minimum values of leaf ount for each model. Must have same order as ``all_cords``.
            nleaves_max_all (list): Maximum values of leaf ount for each model. Must have same order as ``all_cords``.
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        # get main key
        key_in = list(branches_coords.keys())[0]
        self.key_in = key_in

        if branches_inds is None:
            raise ValueError("In MT RJ proposal, branches_inds cannot be None.")

        ntemps, nwalkers, nleaves_max, ndim = branches_coords[key_in].shape

        # get temperature information
        betas_here = np.repeat(
            self.temperature_control.betas[:, None], nwalkers, axis=-1
        ).flatten()

        # current Likelihood and prior information
        ll_here = self.current_state.log_like.flatten()
        lp_here = self.current_state.log_prior.flatten()

        # do rj setup
        assert len(nleaves_min_all) == 1 and len(nleaves_max_all) == 1
        nleaves_min = nleaves_min_all[key_in]
        nleaves_max = nleaves_max_all[key_in]

        if nleaves_min == nleaves_max:
            raise ValueError("MT RJ proposal requires that nleaves_min != nleaves_max.")
        elif nleaves_min > nleaves_max:
            raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")

        # get the inds adjustment information
        all_inds_for_change = self.get_model_change_proposal(
            branches_inds[key_in], random, nleaves_min, nleaves_max
        )

        # preparing leaf information for going into the proposal
        inds_leaves_rj = np.zeros(ntemps * nwalkers, dtype=int)
        coords_in = np.zeros((ntemps * nwalkers, ndim))
        inds_reverse_rj = None

        # prepare proposal dictionaries
        new_inds = deepcopy(branches_inds)
        q = deepcopy(branches_coords)
        for change in all_inds_for_change.keys():
            if change not in ["+1", "-1"]:
                raise ValueError("MT RJ is only implemented for +1/-1 moves.")

            # get indicies of changing leaves
            temp_inds = all_inds_for_change[change][:, 0]
            walker_inds = all_inds_for_change[change][:, 1]
            leaf_inds = all_inds_for_change[change][:, 2]

            # leaf index to change
            inds_leaves_rj[temp_inds * nwalkers + walker_inds] = leaf_inds
            coords_in[temp_inds * nwalkers + walker_inds] = branches_coords[key_in][
                (temp_inds, walker_inds, leaf_inds)
            ]

            # adjustment of indices
            new_val = {"+1": True, "-1": False}[change]

            # adjust indices
            new_inds[key_in][(temp_inds, walker_inds, leaf_inds)] = new_val

            if change == "-1":
                # which walkers are removing
                inds_reverse_rj = temp_inds * nwalkers + walker_inds

        if inds_reverse_rj is not None:
            # setup reversal coords and inds
            # need to determine Likelihood and prior of removed binaries.
            # this goes into the multiple try proposal as previous ll and lp
            temp_reverse_coords = {}
            temp_reverse_inds = {}

            for key in self.current_state.branches:
                (
                    ntemps_tmp,
                    nwalkers_tmp,
                    nleaves_max_tmp,
                    ndim_tmp,
                ) = self.current_state.branches[key].shape

                # coords from reversal
                temp_reverse_coords[key] = self.current_state.branches[
                    key
                ].coords.reshape(ntemps_tmp * nwalkers_tmp, nleaves_max_tmp, ndim_tmp)[
                    inds_reverse_rj
                ][
                    None, :
                ]

                # which inds array to use
                inds_tmp_here = (
                    new_inds[key]
                    if key == key_in
                    else self.current_state.branches[key].inds
                )
                temp_reverse_inds[key] = inds_tmp_here.reshape(
                    ntemps * nwalkers, nleaves_max_tmp
                )[inds_reverse_rj][None, :]

            # calculate information for the reverse
            lp_reverse_here = self.current_model.compute_log_prior_fn(
                temp_reverse_coords, inds=temp_reverse_inds
            )[0]
            ll_reverse_here = self.current_model.compute_log_like_fn(
                temp_reverse_coords, inds=temp_reverse_inds, logp=lp_here
            )[0]

            # fill the here values
            ll_here[inds_reverse_rj] = ll_reverse_here
            lp_here[inds_reverse_rj] = lp_reverse_here

        # get mt proposal
        generated_points, factors = self.get_mt_proposal(
            coords_in,
            random,
            betas=betas_here,
            ll_in=ll_here,
            lp_in=lp_here,
            inds_leaves_rj=inds_leaves_rj,
            inds_reverse_rj=inds_reverse_rj,
        )

        # for reading outside
        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)

        # which walkers have information added
        inds_forward_rj = np.delete(np.arange(coords_in.shape[0]), inds_reverse_rj)

        # updated the coordinates
        temp_inds = all_inds_for_change["+1"][:, 0]
        walker_inds = all_inds_for_change["+1"][:, 1]
        leaf_inds = all_inds_for_change["+1"][:, 2]
        q[key_in][(temp_inds, walker_inds, leaf_inds)] = generated_points[
            inds_forward_rj
        ]

        return q, new_inds, factors.reshape(ntemps, nwalkers)
