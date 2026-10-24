from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..utils.utility import NDArrayLike
from .base import Prior

if TYPE_CHECKING:
    # * For type checkers only: JAX stays optional and loads when a prior first uses it.
    import jax


class DiscreteUniform(Prior):
    """A uniform discrete prior over a range of integers [minimum, maximum].
    
    This is useful for model selection (RJ-MCMC) or selecting discrete states.

    Args:
        minimum (int): Minimum integer bound.
        maximum (int): Maximum integer bound.
        name (str, optional): Name of the prior. (default: ``None``)
        name_phys (str, optional): Physical name. (default: ``None``)
        latex_label (str, optional): Latex label. (default: ``None``)
        unit (str, optional): Unit string. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments passed to base prior.
    """

    def __init__(
        self,
        minimum: int,
        maximum: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError(f"DiscreteUniform requires integer bounds. Got min: {minimum}, max: {maximum}")
        if minimum >= maximum:
            raise ValueError(f"maximum ({maximum}) must be strictly greater than minimum ({minimum}).")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            unit=unit,
            minimum=float(minimum),
            maximum=float(maximum),
            check_range_nonzero=True,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.min_int = minimum
        self.max_int = maximum
        self.n_states = self.max_int - self.min_int + 1
        self._logpdf_val = -np.log(self.n_states)

    def rvs(
        self, size: int | tuple[int, ...] = (1,), rng_key: jax.Array | None = None, **kwargs
    ) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        samples = self._random_integers(self.min_int, self.max_int + 1, size, rng_key)
        samples = samples.astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)

        mask = (x_arr >= self.min_int) & (x_arr <= self.max_int) & (self.xp.round(x_arr) == x_arr)
        # * where instead of a masked write, which JAX arrays do not allow.
        out = self.xp.where(mask, self._logpdf_val, np.float64(-np.inf))
        return self._to_device(out)

    logpmf = logpdf

    def cdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        states_below = self.xp.floor(x_arr) - self.min_int + 1
        states_below = self.xp.clip(states_below, 0, self.n_states)
        out = states_below / self.n_states
        return self._to_device(out)


class CategoricalDistribution(DiscreteUniform):
    """Convenience prior for sampling categorically among N possibilities.

    Samples uniformly from the integers [0, 1, ..., n_categories - 1].

    Args:
        n_categories (int): Number of categories.
        name (str, optional): Name of the prior. (default: ``None``)
        name_phys (str, optional): Physical name. (default: ``None``)
        latex_label (str, optional): Latex label. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments passed to base prior.
    """

    def __init__(
        self,
        n_categories: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(f"Categorical prior requires n_categories >= 2. Got {n_categories}.")

        super().__init__(
            minimum=0,
            maximum=n_categories - 1,
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )


class PoissonDistribution(Prior):
    """Poisson prior distribution for integer event counts.

    Probability mass function: p(k) = (lam^k * e^-lam) / k!

    Args:
        lam (float): Rate parameter (mean).
        name (str, optional): Name of the prior. (default: ``None``)
        name_phys (str, optional): Physical name. (default: ``None``)
        latex_label (str, optional): Latex label. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments passed to base prior.
    """

    def __init__(
        self,
        lam: float,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if lam <= 0:
            raise ValueError(f"Poisson rate parameter 'lam' must be strictly positive. Got {lam}.")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            minimum=0.0,
            maximum=np.inf,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.lam = lam
        self._log_lam = np.log(self.lam)

        # * Fails at construction when the backend's ln Gamma is missing.
        _ = self._gammaln

    @property
    def _gammaln(self):
        """ln Gamma of the backend, looked up on use so that ``to_backend`` stays correct."""
        if self.backend_name == "jax":
            from jax.scipy.special import gammaln

            return gammaln
        if self.use_cupy:
            try:
                from cupyx.scipy.special import gammaln # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "The Poisson prior requires 'cupyx' for the gammaln function "
                    "when use_cupy=True. Please install cupyx."
                )
            return gammaln
        from scipy.special import gammaln

        return gammaln

    def rvs(
        self, size: int | tuple[int, ...] = (1,), rng_key: jax.Array | None = None, **kwargs
    ) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        samples = self._random_poisson(self.lam, size, rng_key).astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)

        mask = (x_arr >= 0) & (self.xp.round(x_arr) == x_arr)
        valid_x = self.xp.where(mask, x_arr, 0.0)

        log_prob = valid_x * self._log_lam - self.lam - self._gammaln(valid_x + 1.0)

        out = self.xp.where(mask, log_prob, np.float64(-np.inf))
        return self._to_device(out)