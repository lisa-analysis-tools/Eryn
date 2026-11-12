# -*- coding: utf-8 -*-
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """Affine-Invariant Proposal

    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    This class was originally implemented in ``emcee``.

    On a periodic parameter ``x - c`` is known only modulo the period ``P``. The move stretches a
    mathematical lift ``r = delta + k P`` of it (``delta`` the shortest difference, ``|k| <= lift_kmax``) 
    drawn with probability ``w(r)`` proportional to ``exp(-r^2 / 2 s^2)``, and multiplies the acceptance
    by ``w(z r) / w(r)``; ``s`` is ``lift_scale`` times the complements' circular standard
    deviation, per temperature and parameter. This keeps detailed balance with moves that
    pass ``c + P/2``.

    Args:
        a (double, optional): The stretch scale parameter. (default: ``2.0``)
        return_gpu (bool, optional): If ``use_gpu == True and return_gpu == True``,
            the returned arrays will be returned as ``CuPy`` arrays. (default: ``False``)
        lift_scale (double, optional): Width of the lift weights in units of the complements'
            circular standard deviation. (default: ``2.0``)
        lift_kmax (int, optional): Most extra turns a lift may take. (default: ``8``)
        kwargs (dict, optional): Additional keyword arguments passed down through :class:`RedRedBlueMove`_.

    Attributes:
        a (double): The stretch scale parameter.
        return_gpu (bool): Whether the array being returned is in ``Cupy`` (``True``)
            or ``NumPy`` (``False``).

    """

    def __init__(
        self,
        a=2.0,
        return_gpu=False,
        random_seed=None,
        *,
        lift_scale=2.0,
        lift_kmax=2,
        **kwargs
    ):
        # store scale factor
        self.a = a

        # periodic parameters: lift weights, and per-proposal state set in get_proposal
        self.lift_scale = lift_scale
        self.lift_kmax = int(lift_kmax)
        self._lift = None
        self._inds_run = {}
        self._last_log_norm = None

        # pass kwargs up
        RedBlueMove.__init__(self, **kwargs)

        # precompute turns array on the chosen array library (avoiding per-step allocations)
        self._turns = self.xp.arange(-self.lift_kmax, self.lift_kmax + 1)
        self._periodic_cache = {}

        # set the random seed of the library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

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
        logzz = factors / (ndims_old - 1.0)
        factors[:] = logzz * (ndims_new - 1.0)

    def gibbs_sampling_setup_iterator(self, all_branch_names):
        """Iterate through Gibbs splits while tracking active parameters for periodic lift ratios.

        Overrides :meth:`eryn.moves.move.Move.gibbs_sampling_setup_iterator` to record
        the active parameter mask for the current Gibbs split in ``self._inds_run``.
        This ensures that when a Gibbs proposal updates only a subset of coordinates,
        only the periodic parameters being updated contribute to the lift detailed
        balance correction factor :math:`\\sum \\log [w(z r) / w(r)]`.

        Args:
            all_branch_names (list of str): List of all branch names across the model.

        Yields:
            tuple of (list, list): A 2-tuple ``(branch_names_run, inds_run)`` where
                ``branch_names_run`` contains the names of branches active in this split,
                and ``inds_run`` contains boolean mask arrays indicating which parameters
                are updated in this split.

        """
        for branch_names_run, inds_run in super().gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            self._inds_run = dict(zip(branch_names_run, inds_run))
            yield branch_names_run, inds_run

    def _get_periodic_info(self, name):
        """Retrieve and cache periodic indices and periods as xp arrays for branch ``name``."""
        info = self._periodic_cache.get(name)
        if info is None:
            inds = self.xp.asarray(self.periodic.inds_periodic[name])
            period = self.xp.asarray(self.periodic.periods[name])
            info = (inds, period)
            self._periodic_cache[name] = info
        return info

    def lift_widths(self, name, c):
        r"""Compute the adaptive lift scale :math:`s` from the complement ensemble.

        Calculates the circular standard deviation of the complement walkers across
        all active walkers and leaves for each periodic parameter, scaled by
        ``self.lift_scale``:

        .. math::

            \bar{R}_j = \sqrt{ \langle \cos(2\pi c_j / P_j) \rangle^2 + \langle \sin(2\pi c_j / P_j) \rangle^2 }

            \sigma_j = \frac{P_j}{2\pi} \sqrt{-2 \ln \bar{R}_j}

            s_j = \mathrm{clip}(\lambda \sigma_j, 10^{-3} P_j, 2 P_j)

        Because the complement ensemble remains fixed while the moving half proposes,
        both the forward proposal :math:`x \to y` and reverse proposal :math:`y \to x`
        share the exact same lift scale :math:`s`, satisfying detailed balance.
        ``nanmean`` is employed across walkers and leaves to safely support
        trans-dimensional models where inactive leaves contain NaNs.

        Args:
            name (str): Branch name containing periodic parameters.
            c (xp.ndarray): Complement coordinates with shape
                ``(ntemps, Nc, nleaves_max, ndim)``.

        Returns:
            xp.ndarray: Adaptive width array :math:`s` of shape ``(ntemps, nperiodic)``,
                clipped to :math:`[10^{-3} P, 2 P]`.

        """
        xp = self.xp
        inds, period = self._get_periodic_info(name)
        angle = 2.0 * np.pi * c[..., inds] / period
        rbar = xp.sqrt(
            xp.nanmean(xp.cos(angle), axis=(1, 2)) ** 2
            + xp.nanmean(xp.sin(angle), axis=(1, 2)) ** 2
        )
        rbar = xp.where(xp.isnan(rbar), 0.0, rbar)
        sigma = xp.sqrt(-2.0 * xp.log(xp.clip(rbar, 1e-12, 1.0))) * period / (2.0 * np.pi)
        return xp.clip(self.lift_scale * sigma, 1e-3 * period, 2.0 * period)

    def _lift_log_h(self, r, width):
        return -0.5 * (r / width) ** 2

    def lift_log_weight(self, r, period, width):
        r"""Compute :math:`\log w(r \mid \pi(r))` on the discrete fiber of lifts.

        Evaluates the normalized log-probability that lift :math:`r` is chosen
        among the finite candidate lifts :math:`\{\pi(r) + k P : |k| \le K\}`
        of its base projection :math:`\pi(r) = (r + P/2) \pmod P - P/2`:

        .. math::

            \log w(r \mid \pi(r)) = -\frac{r^2}{2 s^2} - \ln \sum_{k=-K}^K \exp\left( -\frac{(\pi(r) + k P)^2}{2 s^2} \right)

        Returns :math:`-\infty` for any lift where :math:`|k| = |(r - \pi(r))/P| > K`.

        Args:
            r (xp.ndarray): Lift coordinates with shape
                ``(ntemps, Ns, nleaves_max, nperiodic)``.
            period (xp.ndarray): 1-D array of period lengths for each periodic parameter,
                shape ``(nperiodic,)``.
            width (xp.ndarray): Lift scales :math:`s` broadcastable to ``r``,
                typically of shape ``(ntemps, 1, 1, nperiodic)``.

        Returns:
            xp.ndarray: Log-weights of shape ``(ntemps, Ns, nleaves_max, nperiodic)``,
                with :math:`-\infty` where :math:`|k| > \mathrm{lift\_kmax}`.

        """
        xp = self.xp
        base = (r + period / 2.0) % period - period / 2.0
        log_h = self._lift_log_h(base[..., None] + period[:, None] * self._turns, width[..., None])
        top = log_h.max(axis=-1, keepdims=True)
        log_norm = top[..., 0] + xp.log(xp.exp(log_h - top).sum(axis=-1))
        k = xp.rint((r - base) / period)
        return xp.where(
            xp.abs(k) <= self.lift_kmax, self._lift_log_h(r, width) - log_norm, -xp.inf
        )

    def draw_lift(self, delta, period, width, random_number_generator):
        r"""Draw a randomized lift :math:`r = \delta + k P` from the discrete fiber.

        For each shortest angular difference :math:`\delta \in [-P/2, P/2)`,
        evaluates candidate lifts :math:`r_k = \delta + k P` for integer turns
        :math:`|k| \le K` (``self.lift_kmax``). Computes the discrete categorical
        probability:

        .. math::

            P(k \mid \delta) = \frac{\exp(-(\delta + k P)^2 / 2s^2)}{\sum_{k'=-K}^K \exp(-(\delta + k' P)^2 / 2s^2)}

        and samples :math:`k` via inverse transform sampling on the cumulative
        distribution function (CDF).

        Also records the exact normalizer :math:`\ln Z(\delta)` on ``self._last_log_norm``
        so that the denominator :math:`\log w(r)` in the acceptance ratio can be evaluated
        in :math:`O(1)` time without recomputing the 5-D candidate tensor.

        Args:
            delta (xp.ndarray): Shortest periodic difference between walker and complement,
                wrapped to :math:`[-P/2, P/2)`, shape ``(ntemps, Ns, nleaves_max, nperiodic)``.
            period (xp.ndarray): 1-D array of period lengths, shape ``(nperiodic,)``.
            width (xp.ndarray): Lift scales :math:`s`, shape ``(ntemps, 1, 1, nperiodic)``.
            random_number_generator (object): Random state instance (NumPy or CuPy RNG).

        Returns:
            xp.ndarray: Sampled lift array :math:`r = \delta + k P` of shape
                ``(ntemps, Ns, nleaves_max, nperiodic)``.

        """
        xp = self.xp
        lifts = delta[..., None] + period[:, None] * self._turns
        log_h = self._lift_log_h(lifts, width[..., None])
        top = log_h.max(axis=-1, keepdims=True)
        exp_h = xp.exp(log_h - top)
        cdf = xp.cumsum(exp_h, axis=-1)
        u = random_number_generator.rand(*delta.shape)[..., None] * cdf[..., -1:]
        pick = xp.minimum((cdf < u).sum(axis=-1), 2 * self.lift_kmax)
        self._last_log_norm = top[..., 0] + xp.log(cdf[..., -1])
        return xp.take_along_axis(lifts, pick[..., None], axis=-1)[..., 0]

    def choose_c_vals(self, c, Nc, Ns, ntemps, random_number_generator, **kwargs):
        """Get the compliment array

        The compliment represents the points that are used to move the actual points whose position is
        changing.

        Args:
            c (np.ndarray): Possible compliment values with shape ``(ntemps, Nc, nleaves_max, ndim)``.
            Nc (int): Length of the ``...``: the subset of walkers proposed to move now (usually nwalkers/2).
            Ns (int): Number of generation points.
            ntemps (int): Number of temperatures.
            random_number_generator (object): Random state object.
            **kwargs (ignored): Ignored here. For modularity.

        Returns:
            np.ndarray: Compliment values to use with shape ``(ntemps, Ns, nleaves_max, ndim)``.

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
        r"""Compute new proposed points for walkers moving along branch ``name``.

        Computes affine stretch proposals :math:`y = c + z(s - c)` using complement
        points :math:`c`. For periodic parameters, draws randomized lifts
        :math:`r \sim w(r \mid s - c)` on the discrete fiber :math:`(s - c) + k P`
        via :meth:`draw_lift`, sets :math:`y = c + z r \pmod P`, and accumulates
        the log detailed balance correction factor :math:`\sum \log [w(z r) / w(r)]`
        into ``self._lift["log_ratio"]``. Uncorrected periodic wrapping is disallowed
        to guarantee detailed balance.

        Args:
            name (str): Branch name.
            s (xp.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c_temp (xp.ndarray): Complement points to move against, shape
                ``(ntemps, Ns, nleaves_max, ndim)``.
            Ns (int): Number of walkers proposing to move in this split.
            branch_shape (tuple): Full 4-tuple shape ``(ntemps, nwalkers, nleaves_max, ndim)``.
            branch_i (int): Index of current branch. On ``branch_i == 0``, draws
                stretch scale factors :math:`z \sim g(z)`.
            random_number_generator (object): Random state instance (NumPy or CuPy RNG).

        Returns:
            xp.ndarray: New proposed coordinates with shape ``(ntemps, Ns, nleaves_max, ndim)``.

        Raises:
            ValueError: If periodic parameters exist on ``name`` but ``self._lift``
                has not been initialized by :meth:`get_proposal`.

        """

        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape

        # only for the first branch do we draw for zz
        if branch_i == 0:
            self.zz = (
                (self.a - 1.0) * random_number_generator.rand(ntemps, Ns) + 1
            ) ** 2.0 / self.a

        # get proper distance

        lift = getattr(self, "_lift", None) or {}
        if (
            self.periodic is not None
            and name in self.periodic.periods
            and len(self.periodic.periods[name]) > 0
        ):
            if name not in lift:
                raise ValueError(
                    f"Periodic branch '{name}' requires lift setup in get_proposal. "
                    "Uncorrected periodic distance proposals violate detailed balance."
                )
            # periodic parameters: stretch a lift r of s - c drawn with probability w(r)
            diff = c_temp - s
            inds, period = self._get_periodic_info(name)
            width = lift[name][:, None, None, :]
            delta = (s[..., inds] - c_temp[..., inds] + period / 2.0) % period - period / 2.0
            r = self.draw_lift(delta, period, width, random_number_generator)
            diff[..., inds] = -r
            log_w_r = self._lift_log_h(r, width) - self._last_log_norm
            log_ratio = (
                self.lift_log_weight(self.zz[:, :, None, None] * r, period, width)
                - log_w_r
            )
            moved = self._inds_run.get(name)
            if moved is not None:  # Gibbs split: only the parameters it moves count
                log_ratio = self.xp.where(self.xp.asarray(moved)[:, inds], log_ratio, 0.0)
            lift["log_ratio"] = lift["log_ratio"] + log_ratio.sum(axis=(-2, -1))

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
        """Generate stretch proposal

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
        self.zz = None
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # lift widths per periodic branch, and log w(z r) - log w(r) summed by get_new_points
        self._lift = {"log_ratio": 0.0}

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

            if (
                self.periodic is not None
                and name in self.periodic.periods
                and len(self.periodic.periods[name]) > 0
            ):
                self._lift[name] = self.lift_widths(name, c)

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

        # the lift weights' ratio (after adjust_factors, which assumes (ndim - 1) log z)
        lift_log_ratio = self._lift["log_ratio"]
        self._lift = None
        self._last_log_norm = None
        if self.use_gpu and not self.return_gpu and hasattr(lift_log_ratio, "get"):
            lift_log_ratio = lift_log_ratio.get()

        return newpos, factors + lift_log_ratio
