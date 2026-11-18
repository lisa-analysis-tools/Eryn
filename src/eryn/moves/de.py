# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from .red_blue import RedBlueMove

__all__ = ["DEMove", "DESnookerMove"]

class DEMove(RedBlueMove):
    r"""A proposal using differential evolution. This proposal is directly based on the `emcee` implementation.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[dict|float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.

    """

    def __init__(self, sigma=1.0e-5, gamma0=None, return_gpu=False, random_seed=None, target_acceptance=0.25, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        self.target_acceptance = target_acceptance

        RedBlueMove.__init__(self, **kwargs)

        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

    def setup(self, branch_coords):
        if isinstance(self.gamma0, float):
            tmp = {}
            for key in branch_coords.keys():
                tmp[key] = self.gamma0
            self.gamma0 = tmp

        self.g0 = self.gamma0

        if self.g0 is None:
            self.g0 = {}
            for key, coords in branch_coords.items():
                ndim = coords.shape[-1]
                self.g0[key] = 2.38 / np.sqrt(2 * ndim)

    
    def get_new_points(
        self, name, s, c, Ns, g0, branch_shape, branch_i, random_number_generator
    ):
        """Get new points in DE move.

        Takes complement and uses pairs to get new points for those being proposed.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c (np.ndarray): Full complement array with shape ``(ntemps, Nc, nleaves_max, ndim)``.
            Ns (int): Number to generate.
            g0 (float): Gamma0 parameter for this branch.
            branch_shape (tuple): Full branch shape.
            branch_i (int): Which branch in the order is being run now. This ensures that the
                randomly generated quantity per walker remains the same over branches.
            random_number_generator (object): Random state object.

        Returns:
            np.ndarray: New proposed points with shape ``(ntemps, Ns, nleaves_max, ndim)``.


        """

        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape
        nc = c.shape[1]

        # Only generate pairs and random factors for the first branch to ensure consistency
        if branch_i == 0:
            # Get the pair indices
            pairs = _get_nondiagonal_pairs(nc)

            # Sample from the pairs
            indices = random_number_generator.choice(pairs.shape[0], size=Ns, replace=True)
            self.pairs = pairs[indices]  # (Ns, 2)

            # Sample the random factor (like zz in StretchMove)
            self.gamma_factor = 1 + self.sigma * random_number_generator.randn(ntemps, Ns, 1, 1)
        
        # Apply g0 to the gamma factor (can be different per branch)
        gamma = g0 * self.gamma_factor

        # Get the two complement points for each walker
        c1 = self.xp.take_along_axis(c, self.pairs[:, 0][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)
        c2 = self.xp.take_along_axis(c, self.pairs[:, 1][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)

        # Compute diff vectors
        if self.periodic is not None:
            diff = self.periodic.distance(
                {name: c1.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                {name: c2.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here)
        else:
            diff = c2 - c1

        # Apply DE formula
        temp = s + gamma * diff

        # wrap periodic values
        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here)

        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
        return temp

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate DE proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """

        # needs to be set before we reach the end
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            # get compliment possibilities
            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            # need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # Get g0 for this branch
            g0 = self.g0[name]
            
            # Use DE to get new proposals
            newpos[name] = self.get_new_points(
                name, s, c, Ns, g0, s.shape, i, random_number_generator
            )
        # proper factors
        factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        return newpos, factors
    
    def tune(self, **kwargs):
        """Tune the proposal parameters.

        This method tunes the proposal parameters based on the acceptance fraction of the chain."""
        
        current_acceptance = np.mean(self.acceptance_fraction[0])
        print(f"Current acceptance fraction: {current_acceptance}")
        self.gamma0 = self.g0.copy()
        for key in self.gamma0.keys():
            if current_acceptance > 0.31:
                self.gamma0[key] *= 1.1
            elif current_acceptance < 0.2:
                self.gamma0[key] *= 0.9
            else:
                self.gamma0[key] *= np.sqrt(current_acceptance / self.target_acceptance)



class DESnookerMove(RedBlueMove):
    r"""A snooker proposal using differential evolution.

    Based on `Ter Braak & Vrugt (2008)
    <http://link.springer.com/article/10.1007/s11222-008-9104-9>`_.

    This class was originally implemented in ``emcee``.

    Args:
        gammas (float, optional): The mean stretch factor for the proposal
            vector. By default, it is ``1.7`` as recommended by the reference.
        return_gpu (bool, optional): If ``use_gpu == True and return_gpu == True``,
            the returned arrays will be returned as ``CuPy`` arrays. (default: ``False``)
        random_seed (int, optional): Random seed for reproducibility. (default: ``None``)

    """

    def __init__(self, gammas=1.7, return_gpu=False, random_seed=None, **kwargs):
        self.gammas = gammas
        kwargs["nsplits"] = 4
        RedBlueMove.__init__(self, **kwargs)

        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

    def get_new_points(
        self, name, s, c_list, Ns, branch_shape, branch_i, random_number_generator
    ):
        """Get new points using snooker DE move.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c_list (list): List of 3 complement arrays, each with shape ``(ntemps, Nc_j, nleaves_max, ndim)``.
            Ns (int): Number to generate.
            branch_shape (tuple): Full branch shape.
            branch_i (int): Which branch in the order is being run now.
            random_number_generator (object): Random state object.

        Returns:
            tuple: (New proposed points, Metropolis factors) each with appropriate shapes.

        """
        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape
        ndim_flat = nleaves_max * ndim_here

        # Only generate indices and permutations for the first branch
        if branch_i == 0:
            # Get sizes of complement sets
            Nc = [c.shape[1] for c in c_list]

            # Sample one walker from each of the 3 complement sets
            self.idx_0 = random_number_generator.randint(Nc[0], size=(ntemps, Ns))
            self.idx_1 = random_number_generator.randint(Nc[1], size=(ntemps, Ns))
            self.idx_2 = random_number_generator.randint(Nc[2], size=(ntemps, Ns))

            # Generate permutations to shuffle which complement goes to z, z1, z2
            # For each (temp, walker), generate a permutation of [0, 1, 2]
            self.shuffle_perms = self.xp.zeros((ntemps, Ns, 3), dtype=self.xp.int32)
            for t in range(ntemps):
                for w in range(Ns):
                    perm = self.xp.arange(3)
                    random_number_generator.shuffle(perm)
                    self.shuffle_perms[t, w] = perm

        # Get the 3 sampled walkers from each complement set
        w0 = self.xp.take_along_axis(c_list[0], self.idx_0[:, :, None, None], axis=1)
        w1 = self.xp.take_along_axis(c_list[1], self.idx_1[:, :, None, None], axis=1)
        w2 = self.xp.take_along_axis(c_list[2], self.idx_2[:, :, None, None], axis=1)

        # Stack and apply shuffle permutation
        w_stack = self.xp.stack([w0, w1, w2], axis=2)  # (ntemps, Ns, 3, nleaves_max, ndim)

        # Apply permutation to get z, z1, z2
        perm_idx_0 = self.shuffle_perms[:, :, 0, None, None, None]
        perm_idx_1 = self.shuffle_perms[:, :, 1, None, None, None]
        perm_idx_2 = self.shuffle_perms[:, :, 2, None, None, None]

        z = self.xp.take_along_axis(w_stack, perm_idx_0, axis=2).squeeze(axis=2)
        z1 = self.xp.take_along_axis(w_stack, perm_idx_1, axis=2).squeeze(axis=2)
        z2 = self.xp.take_along_axis(w_stack, perm_idx_2, axis=2).squeeze(axis=2)

        # Compute delta = s - z
        if self.periodic is not None:
            delta = self.periodic.distance(
                {name: s.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                {name: z.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here)
        else:
            delta = s - z

        # Compute norm and unit vector
        delta_flat = delta.reshape(ntemps, Ns, ndim_flat)
        norm_sz = self.xp.linalg.norm(delta_flat, axis=-1)  # (ntemps, Ns)
        u = delta_flat / norm_sz[:, :, None]  # (ntemps, Ns, ndim_flat)

        # Compute dot products for z1 and z2 projections
        z1_flat = z1.reshape(ntemps, Ns, ndim_flat)
        z2_flat = z2.reshape(ntemps, Ns, ndim_flat)

        dot_u_z1 = self.xp.sum(u * z1_flat, axis=-1)  # (ntemps, Ns)
        dot_u_z2 = self.xp.sum(u * z2_flat, axis=-1)  # (ntemps, Ns)

        # Apply snooker formula: q = s + u * gammas * (dot(u, z1) - dot(u, z2))
        displacement = (u * self.gammas * (dot_u_z1 - dot_u_z2)[:, :, None]).reshape(
            ntemps, Ns, nleaves_max, ndim_here
        )
        q = s + displacement

        # Wrap periodic values
        if self.periodic is not None:
            q = self.periodic.wrap(
                {name: q.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here)

        # Compute Metropolis factor: log(||q - z||) - log(||s - z||)
        if self.periodic is not None:
            qz_dist = self.periodic.distance(
                {name: q.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                {name: z.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, ndim_flat)
            norm_qz = self.xp.linalg.norm(qz_dist, axis=-1)
        else:
            norm_qz = self.xp.linalg.norm((q - z).reshape(ntemps, Ns, ndim_flat), axis=-1)

        factors = self.xp.log(norm_qz) - self.xp.log(norm_sz)

        # Get from GPU if needed
        if self.use_gpu and not self.return_gpu:
            q = q.get()
            factors = factors.get()

        return q, factors

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate DE Snooker proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values (should have 3 elements).
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality or complement sets.

        """
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}
        all_factors = []

        # Iterate over branches
        for i, name in enumerate(s_all):
            # Get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            if len(c_all[name]) != 3:
                raise ValueError(
                    f"DESnookerMove requires exactly 3 complement sets (got {len(c_all[name])}). "
                    "Make sure nsplits=4."
                )

            # Get complement possibilities (keep as list of 3)
            c_list = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            Ns = s.shape[1]
            ndim_temp = nleaves_max * ndim_here

            # Need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # Use snooker DE to get new proposals and metropolis factors
            q, factors = self.get_new_points(
                name, s, c_list, Ns, s.shape, i, random_number_generator
            )
            newpos[name] = q
            all_factors.append(factors)

        # Combine metropolis factors from all branches
        # Sum metropolis contributions from all branches
        total_factors = all_factors[0]
        for factor in all_factors[1:]:
            total_factors = total_factors + factor

        # Apply dimension factor
        factors = (ndim - 1.0) * total_factors

        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        # Adjust for Gibbs sampling if needed
        if gibbs_ndim is not None:
            # Adjust factors in place
            self.adjust_factors(factors, ndim, gibbs_ndim)

        return newpos, factors

    def adjust_factors(self, factors, ndims_old, ndims_new):
        """Adjust the ``factors`` based on changing dimensions.

        ``factors`` is adjusted in place.

        Args:
            factors (xp.ndarray): Array of ``factors`` values. It is adjusted in place.
            ndims_old (int or xp.ndarray): Old dimension. If given as an ``xp.ndarray``,
                must be broadcastable with ``factors``.
            ndims_new (int or xp.ndarray): New dimension. If given as an ``xp.ndarray``,
                must be broadcastable with ``factors``.

        """
        # adjusts in place
        if ndims_old == ndims_new:
            return
        # factors = (ndim - 1) * factors
        # So: factors_normalized = factors / (ndim - 1)
        # new_factors = (new_ndim - 1) * factors_normalized = (new_ndim - 1) * factors / (old_ndim - 1)
        factors_normalized = factors / (ndims_old - 1.0)
        factors[:] = factors_normalized * (ndims_new - 1.0)


@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal."""
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return pairs
