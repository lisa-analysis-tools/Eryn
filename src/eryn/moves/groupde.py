# -*- coding: utf-8 -*-

import numpy as np

from .group import GroupMove
from .de import DEMove, DESnookerMove

__all__ = ["GroupDEMove", "GroupDESnookerMove"]

class GroupDEMove(GroupMove, DEMove):
    """Proposal like DE with stationary compliment.

    This move uses the differential evolution proposal method and math, but the compliment
    of walkers used to propose a new point is chosen from a stationary group
    rather than the current walkers in the ensemble.

    This move allows for "de"-like proposal to be used in Reversible Jump MCMC.

    Args:
        **kwargs (dict, optional): Keyword arguments passed to :class:`GroupMove` and
            :class:`DEMove`.
    """
    
    def __init__(self, **kwargs):
        GroupMove.__init__(self, **kwargs)
        DEMove.__init__(self, **kwargs)

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

    def get_proposal(
        self,
        s_all,
        random,
        gibbs_ndim=None,
        s_inds_all=None,
        branch_supps=None,
        **kwargs
    ):
        """Generate group DE proposal coordinates

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)
            s_inds_all (dict, optional): Keys are ``branch_names`` and values are
                ``inds`` arrays indicating which leaves are currently used. (default: ``None``)
            branch_supps (dict, optional): Keys are ``branch_names`` and values are
                :class:`BranchSupplemental` objects. For the group DE,
                ``branch_supps`` are the best device for passing and tracking useful
                information. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            Ns = s.shape[1]

            if s_inds_all is not None:
                s_inds = self.xp.asarray(s_inds_all[name])
            else:
                s_inds = None

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape

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

            # get actual compliment values from stationary group
            c = self.choose_c_vals(
                name, s, s_inds=s_inds, branch_supps=branch_supps
            )

            # Get g0 for this branch
            g0 = self.g0[name]
            
            # Use DE to get new proposals
            # Note: c has shape (ntemps, Nc, nleaves_max, ndim)
            newpos[name] = self.get_new_points(
                name, s, c, Ns, g0, s.shape, i, random_number_generator
            )

        # proper factors (DE returns zero factors)
        factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        return newpos, factors


class GroupDESnookerMove(GroupMove, DESnookerMove):
    """Proposal like DE Snooker with stationary compliment.

    This move uses the differential evolution snooker proposal method and math,
    but the compliment of walkers used to propose a new point is chosen from a
    stationary group rather than the current walkers in the ensemble.

    This move allows for "snooker"-like proposal to be used in Reversible Jump MCMC.

    Args:
        **kwargs (dict, optional): Keyword arguments passed to :class:`GroupMove` and
            :class:`DESnookerMove`.
    """

    def __init__(self, **kwargs):
        GroupMove.__init__(self, **kwargs)
        # DESnookerMove will set nsplits=4, but this is not used in GroupMove
        # since we override get_proposal and don't split the current ensemble
        DESnookerMove.__init__(self, **kwargs)

    def get_proposal(
        self,
        s_all,
        random,
        gibbs_ndim=None,
        s_inds_all=None,
        branch_supps=None,
        **kwargs
    ):
        """Generate group DE Snooker proposal coordinates

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)
            s_inds_all (dict, optional): Keys are ``branch_names`` and values are
                ``inds`` arrays indicating which leaves are currently used. (default: ``None``)
            branch_supps (dict, optional): Keys are ``branch_names`` and values are
                :class:`BranchSupplemental` objects. For the group DE Snooker,
                ``branch_supps`` are the best device for passing and tracking useful
                information. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}
        all_metropolis = []

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            Ns = s.shape[1]

            if s_inds_all is not None:
                s_inds = self.xp.asarray(s_inds_all[name])
            else:
                s_inds = None

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape

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

            # Get 3 independent samples from the stationary group
            # These will be used as z, z1, z2 after shuffling
            c_list = [
                self.choose_c_vals(name, s, s_inds=s_inds, branch_supps=branch_supps)
                for _ in range(3)
            ]

            # Use snooker DE to get new proposals and metropolis factors
            q, metropolis = self.get_new_points(
                name, s, c_list, Ns, s.shape, i, random_number_generator
            )
            newpos[name] = q
            all_metropolis.append(metropolis)

        # Combine metropolis factors from all branches
        total_metropolis = all_metropolis[0]
        for metro in all_metropolis[1:]:
            total_metropolis = total_metropolis + metro

        # Apply dimension factor
        factors = (ndim - 1.0) * total_metropolis

        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        # Adjust for Gibbs sampling if needed
        if gibbs_ndim is not None:
            # adjust factors in place
            self.adjust_factors(factors, ndim, gibbs_ndim)

        return newpos, factors