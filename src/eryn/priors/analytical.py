"""Module containing analytical priors.

Priors that can be evaluated at any point in parameter space, and for which
we can draw samples from. This includes uniform, Gaussian, and log-uniform priors,
as well as any other prior for which the user can provide a logpdf and rvs method.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
from numpy.typing import ArrayLike

from ..utils.utility import NDArrayLike
from .base import Prior

ANGLE_SAFE = 1e-12


class UniformDistribution(Prior):
    """Standard Uniform prior.

    Args:
        minimum (float): Minimum value of the distribution.
        maximum (float): Maximum value of the distribution.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments passed to base prior.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=minimum,
            maximum=maximum,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.pdf_val = 1.0 / self.width
        self.logpdf_val = self.xp.log(self.pdf_val)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        samples = self.xp.random.uniform(self.minimum, self.maximum, size=size)
        return self._to_device(samples)

    def pdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.pdf_val * ((x_arr >= self.minimum) & (x_arr <= self.maximum))
        return self._to_device(out)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)

        mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
        out[mask] = self.logpdf_val

        return self._to_device(out)


class DeltaFunction(Prior):
    """Dirac delta function prior.

    Always returns the peak value during sampling and inf density at the peak.

    Args:
        peak (float): The delta function peak value.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments passed to base prior.
    """

    def __init__(
        self,
        peak: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=peak,
            maximum=peak,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.peak = peak

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.full(size, self.peak, dtype=self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr == self.peak, self.xp.inf, -self.xp.inf)
        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr >= self.peak, 1.0, 0.0)
        return self._to_device(out)


class PowerLaw(UniformDistribution):
    """Generalized Power Law distribution.

    Sampler space: Uniform in u = x^(alpha + 1).
    Physical space: x distributed as p(x) ~ x^alpha in [minimum, maximum].

    Args:
        alpha (float): Power law index.
        minimum (float): Minimum bound.
        maximum (float): Maximum bound.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        alpha: float,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if alpha == -1:
            raise ValueError("For alpha=-1, use the LogUniform class instead.")
        if minimum < 0 and (alpha + 1) % 1 != 0:
            raise ValueError("Fractional powers of negative numbers are undefined.")

        self.alpha = alpha
        self._power = alpha + 1.0
        self._inv_power = 1.0 / self._power
        self._log_abs_power = np.log(np.abs(self._power))

        u_bound_1 = minimum**self._power
        u_bound_2 = maximum**self._power

        super().__init__(
            minimum=min(u_bound_1, u_bound_2),
            maximum=max(u_bound_1, u_bound_2),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        return x_phys**self._power

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        return u_samp**self._inv_power

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        x_phys = self.xp.asarray(x_phys)
        valid_x = self.xp.where(x_phys > 0, x_phys, 1.0)
        out = self._log_abs_power + self.alpha * self.xp.log(valid_x)
        return self._to_device(out)


class LogUniform(UniformDistribution):
    """Log-Uniform distribution.

    Sampler space: Uniform in u = ln(x) between [ln(minimum), ln(maximum)].
    Physical space: x distributed as p(x) ~ 1/x between [minimum, maximum].

    Args:
        minimum (float): Minimum positive bound.
        maximum (float): Maximum bound.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("LogUniform minimum value must be strictly positive.")

        super().__init__(
            minimum=float(np.log(minimum)),
            maximum=float(np.log(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(self.xp.log(x_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.exp(u_samp))

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(-self.xp.log(x_phys))


class CosineUniform(UniformDistribution):
    """Uniform in Cosine prior.

    Used for angles such as inclinations (iota) or colatitudes.
    Sampler space: Uniform in u = cos(iota) in [-1, 1].
    Physical space: iota distributed as p(iota) ~ sin(iota) in [0, pi].

    Args:
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            minimum=-1.0 + ANGLE_SAFE,
            maximum=1.0 - ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, iota_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        iota_phys = self.xp.asarray(iota_phys)
        return self._to_device(self.xp.cos(iota_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.arccos(u_samp))

    def _jacobian(self, iota_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        iota_phys = self.xp.asarray(iota_phys)
        return self._to_device(self.xp.log(self.xp.sin(iota_phys)))


class SineUniform(UniformDistribution):
    """Uniform in Sine prior.

    Used for angles such as latitudes, declinations, or elevations (beta).
    Sampler space: Uniform in u = sin(beta) in [-1, 1].
    Physical space: beta distributed as p(beta) ~ cos(beta) in [-pi/2, pi/2].

    Args:
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            minimum=-1.0 + ANGLE_SAFE,
            maximum=1.0 - ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, beta_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        beta_phys = self.xp.asarray(beta_phys)
        return self._to_device(self.xp.sin(beta_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.arcsin(u_samp))

    def _jacobian(self, beta_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        beta_phys = self.xp.asarray(beta_phys)
        return self._to_device(self.xp.log(self.xp.cos(beta_phys)))


class Log10Uniform(UniformDistribution):
    """Log10-Uniform distribution.

    Sampler space: Uniform in u = log10(x) between [log10(minimum), log10(maximum)].
    Physical space: x distributed as p(x) ~ 1/x in [minimum, maximum].

    Args:
        minimum (float): Minimum positive bound.
        maximum (float): Maximum bound.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("Log10Uniform minimum value must be strictly positive.")

        super().__init__(
            minimum=float(np.log10(minimum)),
            maximum=float(np.log10(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self._log_log10 = np.log(np.log(10.0))

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(self.xp.log10(x_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.power(10.0, u_samp))

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        x_phys = self.xp.asarray(x_phys)
        out = -self.xp.log(x_phys) - self._log_log10
        return self._to_device(out)


class UniformInVolume(PowerLaw):
    """Euclidean Volume prior.

    Commonly used for distance (d).
    Sampler space: Uniform in u = d^3.
    Physical space: d distributed as p(d) ~ d^2.

    Args:
        minimum (float): Minimum positive bound.
        maximum (float): Maximum bound.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum < 0:
            raise ValueError("Distance cannot be negative.")

        super().__init__(
            alpha=2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )


class InverseUniform(PowerLaw):
    """Uniform in Inverse space.

    Sampler space: Uniform in u = 1/x.
    Physical space: x distributed as p(x) ~ 1/x^2.

    Args:
        minimum (float): Minimum positive bound.
        maximum (float): Maximum bound.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("InverseUniform minimum value must be strictly positive.")

        super().__init__(
            alpha=-2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )


class Gaussian(Prior):
    """Gaussian prior.

    Args:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.mu = mu
        self.sigma = sigma
        self._log_norm = 0.5 * np.log(2.0 * np.pi * self.sigma**2)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = -0.5 * ((x_arr - self.mu) / self.sigma) ** 2 - self._log_norm
        return self._to_device(out)


class Normal(Gaussian):
    """Synonym for the Gaussian distribution."""

    pass


class LogNormal(Prior):
    """Log-normal prior.

    Args:
        mu (float): Underlying log-mean.
        sigma (float): Underlying log-standard deviation.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if sigma <= 0.0:
            raise ValueError("Standard deviation sigma must be positive.")
        self.mu = mu
        self.sigma = sigma
        self._log_sqrt_2pi = np.log(np.sqrt(2.0 * np.pi))

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr > self.minimum

        valid_x = self.xp.where(mask, x_arr, 1.0)
        log_x = self.xp.log(valid_x)

        log_prob = (
            -0.5 * ((log_x - self.mu) / self.sigma) ** 2
            - self.xp.log(valid_x * self.sigma)
            - self._log_sqrt_2pi
        )
        out[mask] = log_prob[mask]

        return self._to_device(out)


class LogGaussian(LogNormal):
    """Synonym for LogNormal prior."""

    pass


class Exponential(Prior):
    """Exponential prior.

    Args:
        mu (float): Scale (mean) of the distribution.
        name (str, optional): Name of the prior. (default: ``None``)
        use_cupy (bool, optional): Use CuPy. (default: ``False``)
        return_gpu (bool, optional): Return arrays on GPU. (default: ``False``)
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        mu: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if mu <= 0.0:
            raise ValueError("Exponential mean 'mu' must be positive.")
        self.mu = mu
        self._log_mu = np.log(self.mu)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.exponential(scale=self.mu, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr >= self.minimum

        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = -(valid_x / self.mu) - self._log_mu

        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        out = self.xp.zeros_like(x_arr, dtype=self.xp.float64)
        mask = x_arr >= self.minimum

        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = 1.0 - self.xp.exp(-valid_x / self.mu)

        return self._to_device(out)


class GenericFunctionalDistribution(Prior):
    """
    Generic distribution class, based on user-provided callable functions for the probability density function (pdf) and the inverse cumulative density function (cdf).

    Args:
        pdf_function (callable): A callable function that computes the pdf of the distribution.
        inv_cdf_function (callable): A callable function that computes the inverse cdf of the distribution.
        minimum (float): The minimum value of the distribution's support.
        maximum (float): The maximum value of the distribution's support.
        name (str, optional): The name of the distribution. (default: None)
        clip (bool, optional): If ``True``, clip the samples to the range [minimum, maximum]. If ``False``, samples outside this range will be returned as is. (default: ``True``)
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy. The two callable functions must be compatible with the chosen array library. (default: ``False``)
        return_gpu (bool, optional): If ``True``, return CuPy array. If ``False``, return Numpy array. (default: ``False``)
    """

    def __init__(
        self,
        pdf_function: Callable,
        inv_cdf_function: Callable,
        minimum: float,
        maximum: float,
        name: str = None,
        clip: bool = True,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs,
    ):

        super().__init__(
            name=name,
            minimum=minimum,
            maximum=maximum,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.pdf_function = pdf_function
        self.inv_cdf_function = inv_cdf_function
        self.clip = clip

    def rvs(self, size: int | tuple[int, ...] = 1) -> NDArrayLike:
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        rand_unif = self.xp.random.rand(*size)

        out = self.inv_cdf_function(rand_unif)

        if self.clip:
            out = self.xp.clip(out, self.minimum, self.maximum)

        return self._to_device(out)

    def pdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        out = self.pdf_function(x)
    
        return self._to_device(out)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        out = self.xp.zeros_like(x)
        out[(x >= self.minimum) & (x <= self.maximum)] = self.xp.log(
            self.pdf_function(x[(x >= self.minimum) & (x <= self.maximum)])
        )
        out[(x < self.minimum) | (x > self.maximum)] = -np.inf

        return self._to_device(out)

    def copy(self):
        return deepcopy(self)
