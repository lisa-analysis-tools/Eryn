# -*- coding: utf-8 -*-
"""Affine-invariant stretch move (Goodman & Weare 2010).

:class:`StretchMove` is the :class:`~eryn.moves.RedBlueMove` with the stretch proposal; it
is the default in-model move of :class:`~eryn.ensemble.EnsembleSampler`. Its
``choose_c_vals`` / ``get_new_points`` / ``adjust_factors`` pieces are reused by
:class:`~eryn.moves.GroupStretchMove` and by downstream moves.
"""
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    r"""Affine-invariant stretch proposal.

    A `Goodman & Weare (2010) <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move"
    with the split-ensemble parallelization of `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_; originally implemented in ``emcee``.

    Each walker ``x`` is moved along the line through a randomly chosen complement walker
    ``c`` (drawn with replacement from the other sub-ensembles at the same temperature):

    .. math::

        y = c + z\,(x - c), \qquad g(z) \propto z^{-1/2} \text{ on } [1/a, a],

    with one ``z`` per walker shared by all branches, and log proposal factor
    ``(d - 1) log z`` where ``d`` is the number of parameters moved (``nleaves_max * ndim``
    summed over branches, rescaled to the Gibbs dimension when a Gibbs split is active).
    Differences and the result are wrapped when ``periodic`` is set.

    Args:
        a (float, optional): Stretch scale parameter, ``> 1``. Default ``2.0``.
        return_gpu (bool, optional): With ``use_gpu=True``, return proposals as ``cupy``
            arrays instead of copying them to the host. Default ``False``.
        random_seed (int, optional): If not ``None``, seed the global ``xp.random``
            generator. Default ``None``.
        **kwargs: Forwarded to :class:`~eryn.moves.RedBlueMove` (``nsplits``,
            ``randomize_split``, ``live_dangerously``) and on to
            :class:`~eryn.moves.Move` (``temperature_control``, ``periodic``,
            ``gibbs_sampling_setup``, ``use_gpu``, ...).

    Attributes:
        a (float): The stretch scale parameter.
        return_gpu (bool): Whether proposals are returned as ``cupy`` arrays.
        zz (xp.ndarray or None): Stretch factors ``z`` of the last proposal, shape
            ``(ntemps, nsub)``.

    Example:
        >>> from eryn.moves import StretchMove
        >>> move = StretchMove(a=2.0, nsplits=2)
        >>> move.a, move.nsplits
        (2.0, 2)

        Passed to the sampler as its in-model move:

        .. code-block:: python

            from eryn.ensemble import EnsembleSampler

            sampler = EnsembleSampler(nwalkers, ndim, log_like_fn, priors,
                                      moves=StretchMove(a=2.0))
    """

    def __init__(self, a=2.0, return_gpu=False, random_seed=None, **kwargs):
        # store scale factor
        self.a = a

        # pass kwargs up
        RedBlueMove.__init__(self, **kwargs)

        # change array library based on GPU usage

        # set the random seet of the library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

        # how it was formerly
        # super().__init__(**kwargs)

    def adjust_factors(self, factors, ndims_old, ndims_new):
        """Rescale stretch factors ``(ndims_old - 1) log z`` to ``(ndims_new - 1) log z``.

        Used when a Gibbs split moves fewer parameters than the full coordinate block.
        ``factors`` is adjusted in place; nothing happens when the dimensions are equal.
        Both dimensions must be scalars in practice: the early-return test
        ``if ndims_old == ndims_new`` raises ``ValueError`` for an array with more than one
        element (every Eryn caller passes scalars).

        Args:
            factors (xp.ndarray): Stretch log factors, shape ``(ntemps, nsub)``. Modified in
                place.
            ndims_old (int): Dimension the factors were computed with.
            ndims_new (int): Dimension to rescale to.
        """
        # adjusts in place
        if not isinstance(factors, np.ndarray):
            # factors on the GPU
            ndims_new = self.xp.asarray(ndims_new)
        if np.all(ndims_old == ndims_new):
            return
        logzz = factors / (ndims_old - 1.0)
        factors[:] = logzz * (ndims_new - 1.0)

    def choose_c_vals(self, c, Nc, Ns, ntemps, random_number_generator, **kwargs):
        """Draw one complement point per moving walker, uniformly with replacement.

        The draw stays within each temperature row.

        Args:
            c (xp.ndarray): Candidate complement points, shape
                ``(ntemps, Nc, nleaves_max, ndim)``.
            Nc (int): Number of candidates per temperature (``c.shape[1]``).
            Ns (int): Number of walkers being moved (usually ``nwalkers / nsplits``).
            ntemps (int): Number of temperature rows (folded).
            random_number_generator (object): Random state with ``randint``.
            **kwargs: Ignored (for subclasses with richer signatures).

        Returns:
            xp.ndarray: Chosen complement points, shape ``(ntemps, Ns, nleaves_max, ndim)``.
        """

        rint = random_number_generator.randint(
            Nc,
            size=(
                ntemps,
                Ns,
            ),
        )
        c_temp = self.xp.take_along_axis(c, rint[:, :, None, None], axis=1)
        return c_temp

    def get_new_points(
        self, name, s, c_temp, Ns, branch_shape, branch_i, random_number_generator
    ):
        """Apply the stretch ``y = c + z (s - c)`` to one branch.

        On the first branch (``branch_i == 0``) the stretch factors are drawn,
        ``z = ((a - 1) u + 1)**2 / a`` with ``u ~ U(0, 1)``, and stored in ``self.zz`` so every
        branch of a walker uses the same ``z``. With ``periodic`` set, ``s - c`` is the
        periodic distance and the result is wrapped.

        Args:
            name (str): Branch name (for the periodic lookup).
            s (xp.ndarray): Points to move, shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c_temp (xp.ndarray): Complement points, same shape as ``s``.
            Ns (int): Number of walkers being moved.
            branch_shape (tuple): ``s.shape``, ``(ntemps, Ns, nleaves_max, ndim)``.
            branch_i (int): Position of this branch in the loop over branches.
            random_number_generator (object): Random state with ``rand``.

        Returns:
            np.ndarray or cupy.ndarray: Proposed points, shape
            ``(ntemps, Ns, nleaves_max, ndim)``; host numpy when ``use_gpu`` is set and
            ``return_gpu`` is not.
        """
        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape

        # only for the first branch do we draw for zz
        if branch_i == 0:
            self.zz = (
                (self.a - 1.0) * random_number_generator.rand(ntemps, Ns) + 1
            ) ** 2.0 / self.a

        # get proper distance

        if self.periodic is not None:
            diff = self.periodic.distance(
                {name: s.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                {name: c_temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)
        else:
            diff = c_temp - s

        temp = c_temp - (diff) * self.zz[:, :, None, None]

        # wrap periodic values

        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
        return temp

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate a stretch proposal for one sub-ensemble.

        Args:
            s_all (dict): ``{branch_name: coords}`` of the walkers to move, shape
                ``(ntemps, Ns, nleaves_max, ndim)``.
            c_all (dict): ``{branch_name: list}`` of complement coordinate arrays
                ``(ntemps, Nc_j, nleaves_max, ndim)``; concatenated along the walker axis.
            random (object): Host random state (``model.random``); ``xp.random`` is used
                instead when ``use_gpu`` is set.
            gibbs_ndim (int, optional): Number of parameters actually moved; when given the
                factors are rescaled to it with :meth:`adjust_factors` (a scalar:
                :meth:`~eryn.moves.RedBlueMove.propose` always passes one, and an array fails
                there). Default ``None``.
            **kwargs: Ignored.

        Returns:
            tuple[dict, np.ndarray]: ``(newpos, factors)``: proposed coordinates per branch,
            shape ``(ntemps, Ns, nleaves_max, ndim)``, and log factors
            ``(d - 1) log z``, shape ``(ntemps, Ns)``.

        Raises:
            ValueError: A ``c_all`` value is not a list, or branches have different walker
                counts.
        """

        # needs to be set before we reach the end
        self.zz = None
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

            # get actual compliment values
            c_temp = self.choose_c_vals(c, Nc, Ns, ntemps, random_number_generator)

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
