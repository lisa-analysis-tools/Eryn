# -*- coding: utf-8 -*-
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .group import GroupMove
from .stretch import StretchMove

__all__ = ["GroupStretchMove"]


class GroupStretchMove(GroupMove, StretchMove):
    """Proposal like stretch with stationary compliment.

    This move uses the stretch proposal method and math, but the compliment
    of walkers used to propose a new point is chosen from a stationary group
    rather than the current walkers in the ensemble.

    This move allows for "stretch"-like proposal to be used in Reversible Jump MCMC.
    For periodic parameters, this move uses the randomized lift proposal from
    :class:`StretchMove`, preserving detailed balance across periodic boundaries.

    Args:
        lift_scale (double, optional): Width of the lift weights in units of the complement's
            circular standard deviation. (default: ``2.0``)
        lift_kmax (int, optional): Most extra turns a lift may take. (default: ``8``)
        **kwargs (dict, optional): Keyword arguments passed to :class:`GroupMove` and
            :class:`StretchMove`.

    """

    def __init__(self, **kwargs):
        GroupMove.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

    def get_proposal(
        self,
        s_all,
        random,
        gibbs_ndim=None,
        s_inds_all=None,
        branch_supps=None,
        **kwargs
    ):
        """Generate group stretch proposal coordinates

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
                :class:`BranchSupplemental` objects. For the group stretch,
                ``branch_supps`` are the best device for passing and tracking useful
                information. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors,
                including the stretch dimension factor :math:`(D - 1) \\log z` and the
                periodic lift correction ratio :math:`\\sum \\log [w(z r) / w(r)]`.

        Raises:
            ValueError: Issues with dimensionality.

        """
        # needs to be set before we reach the end
        self.zz = None
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # lift widths per periodic branch, and log w(z r) - log w(r) summed by get_new_points
        self._lift = {"log_ratio": 0.0}

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

            Ns = nwalkers

            # get actual compliment values
            c_temp = self.xp.asarray(
                self.choose_c_vals(
                    name, s, s_inds=s_inds, branch_supps=branch_supps
                )
            )

            if (
                self.periodic is not None
                and name in self.periodic.periods
                and len(self.periodic.periods[name]) > 0
            ):
                c_eval = c_temp
                if s_inds is not None:
                    c_eval = self.xp.where(s_inds[:, :, :, None], c_temp, self.xp.nan)
                self._lift[name] = self.lift_widths(name, c_eval)

            # use stretch to get new proposals
            newpos[name] = self.get_new_points(
                name, s, c_temp, Ns, s.shape, i, random_number_generator
            )

        # proper factors
        factors = (ndim - 1.0) * self.xp.log(self.zz)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        if gibbs_ndim is not None:
            # adjust factors in place
            self.adjust_factors(factors, ndim, gibbs_ndim)

        # the lift weights' ratio (after adjust_factors, which assumes (ndim - 1) log z)
        lift_log_ratio = self._lift["log_ratio"]
        self._lift = None
        self._last_log_norm = None
        if self.use_gpu and not self.return_gpu and hasattr(lift_log_ratio, "get"):
            lift_log_ratio = lift_log_ratio.get()

        return newpos, factors + lift_log_ratio
