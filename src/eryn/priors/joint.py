from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike

from .base import AbstractPrior

if TYPE_CHECKING:
    # * For type checkers only: JAX stays optional and loads when a prior first uses it.
    import jax


class JointPrior(AbstractPrior):
    _default_latex_labels = {}

    def __init__(
        self,
        names: Sequence[str],
        names_phys: Sequence[str] | None = None,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        boundaries: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform_nd: Callable[..., NDArrayLike] | None = None,
        inverse_transform_nd: Callable[..., NDArrayLike] | None = None,
        log_jacobian_nd: Callable[..., NDArrayLike] | None = None,
        force_backend: str | None = None,
    ):
        """Main joint prior constructor for N-dimensional parameter spaces.

        Args:
            names (list of str): Names associated with the parameters.
            names_phys (list of str, optional): Physical parameter names. (default: ``None``)
            latex_labels (list of str, optional): Latex labels for parameters. (default: ``None``)
            units (list of str, optional): Strings describing parameter units. (default: ``None``)
            boundaries (list of str, optional): Boundary conditions for each parameter. (default: ``None``)
            use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration. (default: ``False``)
            return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
            forward_transform_nd (callable, optional): Transform from physical to sampling space. (default: ``None``)
            inverse_transform_nd (callable, optional): Transform from sampling to physical space. (default: ``None``)
            log_jacobian_nd (callable, optional): Log-determinant of the Jacobian matrix. (default: ``None``)
            force_backend (str, optional): ``"cpu"``, a CUDA name or ``"jax"``; see
                :class:`~eryn.priors.base.AbstractPrior`. (default: ``None``)

        Raises:
            ValueError: If names sequence is empty, or lengths of provided metadata lists mismatch.
        """
        super().__init__(use_cupy=use_cupy, return_gpu=return_gpu, force_backend=force_backend)

        if not names:
            raise ValueError("Joint priors require a sequence of parameter names.")

        self.names = tuple(names)
        self.num_vars = len(self.names)

        if names_phys is not None:
            if len(names_phys) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} names_phys, got {len(names_phys)}")
            self.names_phys = tuple(
                np if np is not None else n for n, np in zip(self.names, names_phys)
            )
        else:
            self.names_phys = self.names
            
        if boundaries is not None:
            if len(boundaries) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} boundaries, got {len(boundaries)}")
            for boundary in boundaries:
                if boundary not in [None, "periodic", "reflective"]:
                    raise ValueError(
                        f"Invalid boundary condition '{boundary}'. "
                        "Supported values are None, 'periodic', or 'reflective'."
                    )
            self.boundaries = tuple(boundaries)
        else:
            self.boundaries = tuple(None for _ in range(self.num_vars))

        self.latex_label = latex_labels
        self.unit = units

        self.forward_transform_nd = forward_transform_nd
        self.inverse_transform_nd = inverse_transform_nd
        self.log_jacobian_nd = log_jacobian_nd

    def rvs(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draws random variable(s) from the joint prior distribution.

        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Random sample drawn from the prior distribution.

        Raises:
            ValueError: If size is not an integer or a tuple of integers.
        """
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        if isinstance(size, int):
            size = (size,)
        
        raise NotImplementedError("rvs method is implemented in subclass")

    def rvs_physical(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draw a random sample in the physical parameter space. 
        
        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments.
            
        Returns:
            NDArrayLike: A random sample in the physical parameter space.
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform_nd is None:
            return self._to_device(samples)
        
        output = self.inverse_transform_nd(samples, **kwargs)
        return self._to_device(output)

    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """The log probability density of the joint prior.

        Args:
            x (ArrayLike): The N-dimensional value(s) to evaluate.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: The log probability density.
        """
        raise NotImplementedError("logpdf method is implemented in subclass")

    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Log probability density evaluated in the physical parameter space. 
        
        Args:
            x_phys (ArrayLike): Value(s) in the physical space.
            **kwargs: Additional keyword arguments.
                
        Returns:
            NDArrayLike: Log probability density in physical space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = logp_samp + log_det_J
        return self._to_device(output)

    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Probability density of the joint prior distribution.

        Args:
            x (ArrayLike): N-dimensional value(s) to evaluate.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: The probability density.
        """
        output = self.xp.exp(self.logpdf(x, **kwargs))
        return self._to_device(output)

    def pdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Probability density evaluated in the physical parameter space.
        
        Args:
            x_phys (ArrayLike): N-dimensional value(s) in physical space.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Probability density in physical space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_det_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """CDF numerical integration for N-dimensional JointPriors.

        Raises:
            NotImplementedError: Mathematically undefined for general N-dimensions.
        """
        raise NotImplementedError(
            "Generic CDF numerical integration is mathematically undefined "
            "for N-dimensional JointPriors."
        )

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Boundary checking for N-dimensional bounds.

        Raises:
            NotImplementedError: Must be implemented by specific JointPrior subclass.
        """
        raise NotImplementedError(
            "Boundary checking must be implemented per JointPrior subclass."
        )

    @property
    def latex_label(self) -> tuple[str, ...]:
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: Sequence[str | None] | None = None):
        if latex_label is not None:
            if len(latex_label) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} latex_label, got {len(latex_label)}")
            
            labels = []
            for name, label in zip(self.names, latex_label):
                if label is not None:
                    labels.append(label)
                elif name in self._default_latex_labels:
                    labels.append(self._default_latex_labels[name])
                else:
                    labels.append(str(name))
            self.__latex_label = tuple(labels)
        else:
            self.__latex_label = tuple(
                self._default_latex_labels.get(name, str(name)) for name in self.names
            )

    @property
    def unit(self) -> tuple[str | None, ...]:
        return self.__unit

    @unit.setter
    def unit(self, unit: Sequence[str | None] | None = None):
        if unit is not None:
            if len(unit) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} units, got {len(unit)}")
            self.__unit = tuple(unit)
        else:
            self.__unit = tuple(None for _ in self.names)

    @property
    def latex_label_with_unit(self) -> tuple[str, ...]:
        """tuple: Strings of latex labels combined with units."""
        return tuple(
            f"{label} [{unit}]" if unit is not None else label
            for label, unit in zip(self.latex_label, self.unit)
        )
    
    
class MultivariateGaussian(JointPrior):
    """N-dimensional Multivariate Gaussian Prior.

    The mean, the covariance and the matrices derived from them are computed once on the
    host in float64 NumPy, whatever the backend, and served as arrays of the backend.

    Args:
        names (list of str): Names associated with parameters.
        mu (ArrayLike): Mean vector.
        cov (ArrayLike): Covariance matrix.
        latex_labels (list of str, optional): Latex labels for parameters. (default: ``None``)
        units (list of str, optional): Strings describing parameter units. (default: ``None``)
        use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration. (default: ``False``)
        return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
        force_backend (str, optional): ``"cpu"``, a CUDA name or ``"jax"``; see
            :class:`~eryn.priors.base.AbstractPrior`. (default: ``None``)
    """

    def __init__(
        self,
        names: Sequence[str],
        mu: ArrayLike,
        cov: ArrayLike,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        force_backend: str | None = None,
    ):
        super().__init__(
            names=names,
            latex_labels=latex_labels,
            units=units,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            force_backend=force_backend,
        )

        # * Host precompute: jax.numpy.linalg returns NaN where NumPy raises LinAlgError.
        mean = np.asarray(_to_host(mu), dtype=np.float64)
        covariance = np.asarray(_to_host(cov), dtype=np.float64)

        if mean.shape != (self.num_vars,):
            raise ValueError(f"'mu' must have shape ({self.num_vars},).")
        if covariance.shape != (self.num_vars, self.num_vars):
            raise ValueError(f"'cov' must have shape ({self.num_vars}, {self.num_vars}).")

        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance matrix must be symmetric.")

        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular and cannot be inverted.")

        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite.")

        self._mu_host = mean
        self._cov_host = covariance
        self._prec_host = precision
        self._cholesky_lower_host = np.linalg.cholesky(covariance)
        self._log_norm = -0.5 * (self.num_vars * np.log(2.0 * np.pi) + logdet)

    @property
    def mu(self) -> NDArrayLike:
        """Mean vector, as an array of the backend."""
        return self.xp.asarray(self._mu_host)

    @property
    def cov(self) -> NDArrayLike:
        """Covariance matrix, as an array of the backend."""
        return self.xp.asarray(self._cov_host)

    @property
    def prec(self) -> NDArrayLike:
        """Precision matrix, the inverse covariance, as an array of the backend."""
        return self.xp.asarray(self._prec_host)

    @property
    def cholesky_lower(self) -> NDArrayLike:
        """Lower Cholesky factor of the covariance, as an array of the backend."""
        return self.xp.asarray(self._cholesky_lower_host)

    def rvs(
        self, size: int | tuple[int, ...] = (1,), rng_key: jax.Array | None = None, **kwargs
    ) -> NDArrayLike:
        """Draw samples; the jax backend needs ``rng_key``, a ``jax.random`` key."""
        if isinstance(size, int):
            size = (size,)

        standard_draws = self._random_normal(0.0, 1.0, (*size, self.num_vars), rng_key)
        samples = self.mu + self.xp.einsum("ij,...j->...i", self.cholesky_lower, standard_draws)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)

        if x_arr.shape[-1] != self.num_vars:
            raise ValueError(
                f"Expected last dimension of x to be {self.num_vars}, got {x_arr.shape[-1]}"
            )

        diff = x_arr - self.mu
        quadratic_term = self.xp.einsum("...i,ij,...j->...", diff, self.prec, diff)
        out = -0.5 * quadratic_term + self._log_norm
        return self._to_device(out)


def _to_host(values: ArrayLike) -> ArrayLike:
    """Return CuPy arrays as NumPy arrays, which NumPy will not convert implicitly."""
    return getattr(values, "get")() if hasattr(values, "get") else values
