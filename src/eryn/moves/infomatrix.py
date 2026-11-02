# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Callable

from .mh import MHMove
from ..utils.transform import TransformContainer

__all__ = ["InfoMatrixMove"]

class InfoMatrixMove(MHMove):
    """
    Metropolis-Hastings move using the Information Matrix as proposal.

    In a parallel tempering scheme, the proposal covariance can be automatically
    scaled by the inverse temperature :math:`\beta` of each chain. This is
    theoretically motivated: the tempered Information Matrix at inverse
    temperature :math:`\beta` is :math:`F_\beta = \beta \, F`, so the
    corresponding covariance is :math:`\Sigma_\beta = \Sigma / \beta`.
    Hot chains (small :math:`\beta`) therefore explore more broadly, while the
    cold chain (:math:`\beta=1`) retains the original proposal width.

    The implementation avoids creating separate distribution objects per
    temperature. Samples are drawn from the base (:math:`\beta=1`) proposal
    and rescaled:

    .. math::

        \theta_{\rm new} = \mu + \frac{z - \mu}{\sqrt{\beta}},
        \qquad z \sim \mathcal{N}(\mu,\,\Sigma)

    The :math:`\tfrac{d}{2}\log\beta` normalisation terms cancel exactly in
    the Metropolis-Hastings ratio, so the acceptance factors are computed by
    evaluating the base pdf at rescaled arguments with no additional cost.

    Args:
        means (Dict[str, np.ndarray]): Dictionary with mean parameter values for each branch.
        covariances (Dict[str, np.ndarray], optional): Dictionary with covariance matrices for each branch. If not provided, it will be computed using the Information Matrix. Defaults to None.
        multiplying_factor (float, optional): Factor to multiply the covariance matrices. Defaults to 1.0.
        info_matrix_generator (callable, optional): Function to generate the Information Matrix given mean parameters. Defaults to None.
        info_matrix_kwargs (dict, optional): Additional keyword arguments to pass to the info_matrix_generator. Defaults to {}.
        transform_to_physical (TransformContainer, optional): Container with transformation functions to physical space. Defaults to None.
        transform_to_sampling (TransformContainer, optional): Container with transformation functions to sampling space. Defaults to None.
        log_jacobian_fn (Callable, optional): Function to compute the log-determinant of the transformation Jacobian. Defaults to None.
        temperature_scaling (bool, optional): If ``True``, scale the proposal
            covariance by :math:`1/\beta` for each temperature level.
            Only takes effect when a ``temperature_control`` is attached by
            the sampler. Defaults to ``True``.
        **kwargs: Additional keyword arguments for the MHMove base class.
    """

    def __init__(self, 
                 means: Dict[str, np.ndarray], 
                 covariances: Dict[str, np.ndarray] = None,
                 multiplying_factor: float = 1.0,
                 info_matrix_generator: callable = None,
                 info_matrix_kwargs: dict = {},
                 transform_to_physical: TransformContainer = None,
                 transform_to_sampling: TransformContainer = None,
                 log_jacobian_fn: Callable = None,
                 temperature_scaling: bool = True,
                 **kwargs
                ):
        
        self.all_param_names = kwargs.pop('all_param_names', None)

        super().__init__(**kwargs)

        self.temperature_scaling = temperature_scaling
        self.means = means
        self.multiplying_factor = multiplying_factor
        self.info_matrix_generator = info_matrix_generator
        self.info_matrix_kwargs = info_matrix_kwargs

        if covariances is not None:
            self.covariances = covariances
        else:
            self.covariances = self.compute_all_covariances(means)

        # use the setter to set all_proposal
        self.all_proposal = (self.means, self.covariances)
        
        # store transform container. use it to go to and from physical to sampling space if needed
        if transform_to_physical is None:
            transform_to_physical = TransformContainer()
        self.transform_to_physical = transform_to_physical

        if transform_to_sampling is None:
            transform_to_sampling = TransformContainer()
        self.transform_to_sampling = transform_to_sampling

        if log_jacobian_fn is None:
            if (transform_to_physical is None or transform_to_sampling is None):
                self.log_jacobian_fn = lambda x: 0.0
            else:
                raise ValueError("log_jacobian_fn must be provided if transform containers are provided.")
        else:
            self.log_jacobian_fn = log_jacobian_fn

    def compute_single_covariance(self, mean):
        """
        Compute the covariance matrix for a single set of mean parameters using the Fisher Information Matrix.

        Args:
            mean (np.ndarray): Mean parameters for which to compute the covariance.

        Returns:
            np.ndarray: Covariance matrix.
        """
        info_mat = self.info_matrix_generator(mean, **self.info_matrix_kwargs)
        covariance = np.linalg.inv(info_mat)
        return covariance
    
    def compute_all_covariances(self, means):
        """
        Compute the covariance matrices for all branches using the Information Matrix.
        
        Args:
            means (Dict[str, np.ndarray]): Dictionary with mean parameter values for each branch.
        Returns:
            Dict[str, np.ndarray]: Dictionary with covariance matrices for each branch.
        """

        covariances = {
            name: self.compute_single_covariance(means[name])
            for name in means.keys()
        }
        return covariances

    def setup_single_distribution(self, mean, covariance):
        """
        Setup a multivariate normal distribution for the proposal.
        Args:
            mean (np.ndarray): Mean parameters for the distribution.
            covariance (np.ndarray): Covariance matrix for the distribution.
        Returns:
            multivariate_normal: Multivariate normal distribution object.
        """

        proposal_distribution = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
        return proposal_distribution

    @property
    def all_proposal(self):
        return self._all_proposal
    @all_proposal.setter
    def all_proposal(self, value):
        means, covariances = value
        self._all_proposal = {
            name: self.setup_single_distribution(means[name], self.multiplying_factor * covariances[name])
            for name in means.keys()
        }

    def _get_betas_per_point(self, inds_here):
        """Map each active leaf to its chain's inverse temperature.

        Args:
            inds_here (tuple of np.ndarray): Output of ``np.where(inds)``.

        Returns:
            np.ndarray: Inverse temperatures with shape ``(n_active,)``.
        """
        betas = self.temperature_control.betas  # (ntemps,)
        return betas[inds_here[0]]

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """
        Get proposal from Gaussian distribution using Information Matrix as covariance.

        When ``temperature_scaling`` is enabled and a
        :class:`~eryn.moves.tempering.TemperatureControl` is attached, the
        proposal covariance is broadened for hot chains by a factor
        :math:`1/\beta`.  Samples are drawn from the base distribution and
        rescaled, so only one distribution object per branch is needed.

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
        """

        # initialize ouput
        q = {}
        ntemps, nwalkers = branches_coords[list(branches_coords.keys())[0]].shape[:2]
        factors = np.zeros(ntemps * nwalkers)

        # Determine whether to apply per-temperature covariance scaling
        use_temp_scaling = (
            self.temperature_scaling
            and getattr(self, 'temperature_control', None) is not None
        )

        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            mean = self.means[name]
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # draw from the base (β=1) proposal distribution
            n_points = len(coords[inds_here])
            base_draw = proposal_fn.rvs(size=n_points, random_state=random)

            if use_temp_scaling:
                # ----- temperature-scaled proposal --------------------------
                # Effective covariance at inverse temperature β is Σ/β.
                # Rescale the base draw: x_new = μ + (z - μ) / √β
                betas_per_point = self._get_betas_per_point(inds_here)
                scale = (1.0 / np.sqrt(betas_per_point))[:, np.newaxis]
                new_coords_physical = mean + (base_draw - mean) * scale

                new_coords = self.transform_to_sampling.both_transforms(
                    new_coords_physical
                )

                # log q_β(x) = log q_1(μ + √β(x-μ)) + d/2·log β
                # The d/2·log β terms cancel in the ratio old-new, so we
                # only need to evaluate the base pdf at rescaled arguments.
                #   For x_new:  μ + √β(x_new - μ) = base_draw  (by construction)
                new_logp = (
                    proposal_fn.logpdf(base_draw)
                    + self.log_jacobian_fn(new_coords)
                )

                old_physical = self.transform_to_physical.both_transforms(
                    coords[inds_here]
                )
                old_rescaled = (
                    mean
                    + np.sqrt(betas_per_point)[:, np.newaxis] * (old_physical - mean)
                )
                old_logp = (
                    proposal_fn.logpdf(old_rescaled)
                    + self.log_jacobian_fn(coords[inds_here])
                )
            else:
                # ----- standard (no temperature scaling) --------------------
                new_coords_physical = base_draw
                new_coords = self.transform_to_sampling.both_transforms(
                    new_coords_physical
                )

                new_logp = (
                    proposal_fn.logpdf(new_coords_physical)
                    + self.log_jacobian_fn(new_coords)
                )
                old_logp = (
                    proposal_fn.logpdf(
                        self.transform_to_physical.both_transforms(coords[inds_here])
                    )
                    + self.log_jacobian_fn(coords[inds_here])
                )

            factors += old_logp - new_logp

            # put into coords in proper location
            q[name][inds_here] = new_coords

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, factors.reshape((ntemps, nwalkers))
