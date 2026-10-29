# -*- coding: utf-8 -*-
"""Stretch move with a stationary complement (:class:`GroupStretchMove`)."""
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .group import GroupMove
from .stretch import StretchMove

__all__ = ["GroupStretchMove"]


class GroupStretchMove(GroupMove, StretchMove):
    """Stretch proposal whose complement comes from a stationary group.

    Uses the stretch math of :class:`~eryn.moves.StretchMove` (``y = c + z (x - c)``,
    log factor ``(d - 1) log z``), but the complement point ``c`` of each walker comes from
    :meth:`~eryn.moves.GroupMove.find_friends` (a stationary group refreshed every
    ``n_iter_update`` calls) rather than from the live ensemble, and all walkers move at
    once. This makes a stretch-like proposal usable alongside reversible jump.

    The class is still abstract: a subclass implements
    :meth:`~eryn.moves.GroupMove.setup_friends` and
    :meth:`~eryn.moves.GroupMove.find_friends`.

    The stretch factor :math:`z^{N-1}` is the full acceptance factor only if the friend
    of each leaf is chosen independently of the current position of that leaf. See
    :class:`GroupMove` for how the friends should be set up, how often updating them
    (``n_iter_update``) can bias the posterior, and ``freeze_after``, which makes the
    move exact after burn-in.

    Args:
        **kwargs: Passed whole to BOTH the :class:`~eryn.moves.GroupMove` constructor
            (``nfriends``, ``n_iter_update``, ``live_dangerously``) and the
            :class:`~eryn.moves.StretchMove` constructor (``a``, ``return_gpu``,
            ``random_seed``, plus the :class:`~eryn.moves.RedBlueMove` keywords); each
            ignores what it does not use, and both reach :class:`~eryn.moves.Move`
            (so ``Move.__init__`` runs twice).

    Example:
        .. code-block:: python

            import numpy as np
            from eryn.moves import GroupStretchMove

            class RandomFriends(GroupStretchMove):
                def setup_friends(self, branches):
                    # snapshot of the current coordinates
                    self.friends = {n: b.coords.copy() for n, b in branches.items()}

                def find_friends(self, name, s, s_inds=None, branch_supps=None):
                    ntemps, nwalkers = s.shape[:2]
                    pick = np.random.randint(nwalkers, size=(ntemps, nwalkers))
                    return np.take_along_axis(
                        self.friends[name], pick[:, :, None, None], axis=1
                    )

            move = RandomFriends(nfriends=16, n_iter_update=10, a=2.0)
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
        """Generate a group-stretch proposal for all walkers.

        For each branch, the complement comes from :meth:`choose_c_vals` (that is,
        :meth:`~eryn.moves.GroupMove.find_friends`) and the new points from
        :meth:`~eryn.moves.StretchMove.get_new_points`, so every branch of a walker shares
        one stretch factor ``z``.

        Args:
            s_all (dict): ``{branch_name: coords}``, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (object): Host random state (``model.random``); ``xp.random`` is used
                when ``use_gpu`` is set.
            gibbs_ndim (int, optional): Number of parameters actually moved; when given the
                factors are rescaled to it with
                :meth:`~eryn.moves.StretchMove.adjust_factors` (scalar only). Default
                ``None``.
            s_inds_all (dict, optional): ``{branch_name: inds}`` of the leaves taking part,
                shape ``(ntemps, nwalkers, nleaves_max)``; forwarded to ``find_friends``.
                Default ``None``.
            branch_supps (dict, optional): ``{branch_name: BranchSupplemental}`` forwarded to
                ``find_friends``; the natural place to carry friend-lookup bookkeeping.
                Default ``None``.
            **kwargs: Ignored.

        Returns:
            tuple[dict, np.ndarray]: ``(newpos, factors)``: proposed coordinates with the
            shapes of ``s_all`` and log factors ``(d - 1) log z`` of shape
            ``(ntemps, nwalkers)``, with ``d = sum(nleaves_max * ndim)`` over the branches
            (or ``gibbs_ndim`` when given).

        Raises:
            ValueError: Branches have different walker counts.
        """
        # needs to be set before we reach the end
        self.zz = None
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

            Ns = nwalkers

            # get actual compliment values
            c_temp = self.choose_c_vals(
                name, s, s_inds=s_inds, branch_supps=branch_supps
            )

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

        return newpos, factors
