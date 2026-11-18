# -*- coding: utf-8 -*-

import numpy as np

from .mh import MHMove

__all__ = ["GaussianMove"]


class GaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    This class is heavily based on the same class in ``emcee``.

    Args:
        cov (dict): The covariance of the proposal function. The keys are branch names and the
            values are covariance information. This information can be provided as a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general, respectively.
        mode (str, optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"am"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"am"`` proposes in the eigenbasis of the supplied covariance
            (optionally one component at a time, see ``single_component_prob``).
            The covariance is held fixed but can be refreshed during sampling
            via :func:`update_covariance` (e.g. from an ``update_fn``),
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn. (default: ``"vector"``)
        factor (float, optional): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode. (default: ``None``)
        single_component_prob (float, optional): Probability of using the single
            component (SCAM) proposal, i.e. jumping along a single eigen-direction
            at each step instead of all of them. Only valid for the ``"am"`` mode.
            (default: ``0.0``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """
    allowed_modes = ["vector", "am", "random", "sequential"]

    def __init__(self, cov_all, mode="vector", factor=None, single_component_prob=0.0, **kwargs):
        assert mode in self.allowed_modes, (
            "'{0}' is not a recognized mode. "
            "Please select from: {1}".format(mode, self.allowed_modes)
        )
        
        self.all_proposal = {}
        for name, cov in cov_all.items():
            # Parse the proposal type.
            try:
                float(cov)

            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    # A diagonal proposal was given.
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)

                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                    ndim = cov.shape[0]
                    if mode == "am":
                        proposal = _adaptive_metropolis_proposal(cov, factor, "vector", prob_single_component=single_component_prob)
                    else:
                        proposal = _proposal(cov, factor, mode)

                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                if mode == "am":
                    proposal = _adaptive_metropolis_proposal(cov, factor, "vector", prob_single_component=single_component_prob)
                else:
                    proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)

            self.all_proposal[name] = proposal

        super(GaussianMove, self).__init__(**kwargs)

    def update_covariance(self, cov_all):
        """Update the proposal covariance(s) in place.

        Intended for the ``"am"`` mode: the proposal covariance is otherwise
        fixed, so this lets it be refreshed externally during sampling (e.g.
        from an ``update_fn``). Assigning the new covariance recomputes the
        underlying decomposition automatically, so it never goes stale.

        Args:
            cov_all (dict): Keys are ``branch_names`` and values are the new
                covariance matrices.

        """
        for name, cov in cov_all.items():
            self.all_proposal[name].scale = cov

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution

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
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, _ = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

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

        return q, np.zeros((ntemps, nwalkers))


class _isotropic_proposal(object):
    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        
        if isinstance(scale, float):
            self.invscale = 1. / scale
        else:
            self.invscale = np.linalg.inv(np.linalg.cholesky(scale))

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                ("'{0}' is not a recognized mode. " "Please select from: {1}").format(
                    mode, self.allowed_modes
                )
            )
        self.mode = mode

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)

# code credits: Lorenzo Speri
class _diagonal_proposal(_isotropic_proposal):
    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))


class _proposal(_isotropic_proposal):
    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale, size=len(x0)
        )


class _adaptive_metropolis_proposal(_isotropic_proposal):
    
    allowed_modes = ["vector"]

    def __init__(self, scale, factor, mode, prob_single_component=0.):
        super().__init__(scale, factor, mode)
        self.prob_single_component = prob_single_component

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        # Store the covariance and (re)compute its decomposition. Exposing this
        # as a setter means the covariance can be updated externally (e.g. from
        # an ``update_fn``) and the cached SVD never goes stale.
        self._scale = value
        self.svd = np.linalg.svd(np.atleast_2d(value))
    
    def get_updated_vector(self, rng, x0):
        """
        Get updated vector using the svd decomposition of the covariance matrix and optionally moving along a single component of the eigenbasis.
        """
        scale = self.get_factor(rng)
        
        new_pos = x0.copy()
        nw, nd = new_pos.shape
        U, S, v = self.svd
        
        # go in eigen basis
        y = np.dot(U.T,x0.T).T # np.asarray([np.dot(U.T, x0[i]) for i in range(nw)])
        # choose a random parameter in the uncorrelated basis
        ind_vec = np.arange(nd)
        
        if rng.random() < self.prob_single_component:
            # move along only one uncorrelated direction SCAM
            rng.shuffle(ind_vec)
            rand_j = ind_vec[:1]
        else:
            # move along all of them AM
            rand_j = ind_vec

        # 2.38/sqrt(k) is the optimal RWM scaling for the k directions actually
        # jumped: k = nd for full AM, k = 1 for single-component (SCAM).
        y[:,rand_j] += scale * rng.normal(size=(nw, rand_j.size)) * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(rand_j.size)
        
        # go back to the basis
        new_pos = np.dot(U,y.T).T # np.asarray([np.dot(U, y[i]) for i in range(nw)]) 

        return new_pos