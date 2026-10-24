"""The base Prior classes.

Provides a functional, stateless base Prior class supporting GPU acceleration,
automatic conditional dependency tracking via method signatures, and 
seamless transformations between physical and sampling parameter spaces.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import math
import types
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike

if TYPE_CHECKING:
    # * For type checkers only: JAX stays optional and loads when a prior first uses it.
    import jax

# * Names accepted by ``force_backend``, as LISA Analysis Tools modules spell them, and the
# * array library behind each.
BACKEND_LIBRARIES = {
    "cpu": "numpy",
    "numpy": "numpy",
    "cuda": "cupy",
    "cuda11x": "cupy",
    "cuda12x": "cupy",
    "cuda13x": "cupy",
    "gpu": "cupy",
    "cupy": "cupy",
    "jax": "jax",
}
_ARRAY_MODULE_NAMES = {"numpy": "numpy", "cupy": "cupy", "jax": "jax.numpy"}


@functools.cache
def array_module(library: str) -> types.ModuleType:
    """Return the array module of a library: numpy, cupy or jax.numpy.

    Priors store the library name and look the module up here, because an array module
    stored on an instance breaks ``copy.deepcopy`` and ``pickle``.

    Args:
        library (str): ``"numpy"``, ``"cupy"`` or ``"jax"``.

    Returns:
        module: The array module.
    """
    return importlib.import_module(_ARRAY_MODULE_NAMES[library])


def resolve_backend(force_backend: str | None = None, use_cupy: bool = False) -> str:
    """Return the array library behind a backend name, after checking that it imports.

    Args:
        force_backend (str, optional): A key of :data:`BACKEND_LIBRARIES`: ``"cpu"`` or
            ``"numpy"`` for NumPy, a CUDA name or ``"cupy"`` for CuPy, ``"jax"`` for
            jax.numpy. (default: ``None``, which follows ``use_cupy``)
        use_cupy (bool, optional): Older spelling of ``force_backend="cupy"``.
            (default: ``False``)

    Returns:
        str: ``"numpy"``, ``"cupy"`` or ``"jax"``.

    Raises:
        ValueError: If the name is unknown, or both arguments are given.
        ImportError: If the array library is not installed.
    """
    if force_backend is None:
        library = "cupy" if use_cupy else "numpy"
    elif use_cupy:
        raise ValueError("Give either use_cupy or force_backend, not both.")
    elif force_backend in BACKEND_LIBRARIES:
        library = BACKEND_LIBRARIES[force_backend]
    else:
        raise ValueError(
            f"Unknown backend {force_backend!r}; expected one of {sorted(BACKEND_LIBRARIES)}."
        )
    try:
        array_module(library)
    except ImportError as error:
        raise ImportError(
            f"The {force_backend or library!r} backend needs {library}, which is not installed."
        ) from error
    return library


def require_rng_key(rng_key: jax.Array | None) -> jax.Array:
    """Return the key of a jax draw; jax has no global random state to fall back on.

    Args:
        rng_key (jax.Array or None): The key given to ``rvs``, from ``jax.random.key``.

    Returns:
        jax.Array: The key.

    Raises:
        ValueError: If no key is given.
    """
    if rng_key is None:
        raise ValueError("Draws on the jax backend need an explicit rng_key (a jax.random key).")
    return rng_key


def reject_rng_key(library: str, rng_key: jax.Array | None) -> None:
    """Raise if a NumPy or CuPy draw gets a key.

    Those draws come from the global random state of their library, so a key would promise
    a reproducibility that they do not have.

    Args:
        library (str): ``"numpy"`` or ``"cupy"``.
        rng_key (jax.Array or None): The key given to ``rvs``.

    Raises:
        TypeError: If a key is given.
    """
    if rng_key is not None:
        raise TypeError(
            f"rng_key applies to the jax backend only; {library} priors draw from the "
            "global random state of their library."
        )


def check_rng_key(library: str, rng_key: jax.Array | None, needed: bool = True) -> None:
    """Raise unless ``rng_key`` fits the array library.

    A jax draw needs a key (see :func:`require_rng_key`), and a NumPy or CuPy draw takes
    none (see :func:`reject_rng_key`).

    Args:
        library (str): ``"numpy"``, ``"cupy"`` or ``"jax"``.
        rng_key (jax.Array or None): The key given to ``rvs``, from ``jax.random.key``.
        needed (bool, optional): Whether the jax draws need a key; a fixed value does not.
            (default: ``True``)

    Raises:
        ValueError: If the jax backend needs a key and gets none.
        TypeError: If a NumPy or CuPy object gets a key.
    """
    if library != "jax":
        reject_rng_key(library, rng_key)
    elif needed:
        require_rng_key(rng_key)


def warn_if_jax_is_float32() -> None:
    """Warn when JAX computes in float32, since eryn priors expect float64.

    eryn never switches ``jax_enable_x64`` itself, because the setting holds for the whole
    process. Under the default warning filters the warning shows once.
    """
    import jax

    if not jax.config.jax_enable_x64:
        warnings.warn(
            "JAX computes in float32 because jax_enable_x64 is off, but eryn priors expect "
            "float64. Call jax.config.update('jax_enable_x64', True) before building priors.",
            UserWarning,
        )


class PriorException(Exception):
    """General base class for all prior exceptions"""
    pass


class AbstractPrior:
    """Base class for shared Prior functionality.

    Handles device resolution, automatic dependency inference, and metadata strings.

    The backend is chosen when the prior is built and stored as the name of its array
    library, so the prior stays deepcopy- and pickle-safe; :attr:`xp` looks the module up.
    On the jax backend every method runs under ``jax.jit``, ``jax.vmap`` and ``jax.grad``,
    and ``rvs`` draws from an explicit ``rng_key``.

    Args:
        use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration; the same as
            ``force_backend="cupy"``. (default: ``False``)
        return_gpu (bool, optional): If ``True``, returns GPU arrays when
            ``use_cupy`` is ``True``. (default: ``False``)
        force_backend (str, optional): ``"cpu"`` (NumPy), a CUDA name such as ``"cuda12x"``
            (CuPy) or ``"jax"``; see :data:`BACKEND_LIBRARIES`. (default: ``None``, NumPy
            unless ``use_cupy``)

    """

    def __init__(
        self, use_cupy: bool = False, return_gpu: bool = False, force_backend: str | None = None
    ):
        self._array_library = resolve_backend(force_backend, use_cupy)
        if self._array_library == "jax":
            warn_if_jax_is_float32()
        self.return_gpu = return_gpu

        self._infer_dependencies()

    @property
    def backend_name(self) -> str:
        """str: Array library of the prior: ``"numpy"``, ``"cupy"`` or ``"jax"``."""
        return self._array_library

    @property
    def xp(self) -> types.ModuleType:
        """numpy, cupy or jax.numpy, looked up from the stored library name.

        The module is never stored on the instance, which would break ``copy.deepcopy`` of
        settings trees that contain priors, and ``pickle``.
        """
        return array_module(self._array_library)

    @property
    def use_cupy(self) -> bool:
        """bool: Whether the prior computes with CuPy."""
        return self._array_library == "cupy"

    @use_cupy.setter
    def use_cupy(self, use_cupy: bool):
        # * Kept so that containers and user code can still switch CuPy on and off.
        self._array_library = resolve_backend(None, use_cupy)

    def to_backend(self, force_backend: str) -> AbstractPrior:
        """Return a deep copy of the prior on another backend.

        Args:
            force_backend (str): Backend name, as for the constructor.

        Returns:
            AbstractPrior: The copy; the prior itself is unchanged.
        """
        library = resolve_backend(force_backend)
        if library == "jax":
            warn_if_jax_is_float32()
        converted = deepcopy(self)
        converted._array_library = library
        return converted

    def _check_rng_key(self, rng_key: jax.Array | None, needed: bool = True) -> None:
        """Raise unless ``rng_key`` fits the backend; see :func:`check_rng_key`."""
        check_rng_key(self._array_library, rng_key, needed)

    # * The draw helpers below keep the exact xp.random calls of NumPy and CuPy, so their
    # * seeded random streams do not change; only the jax backend takes rng_key.

    def _random_uniform(
        self,
        low: float | NDArrayLike,
        high: float | NDArrayLike,
        size: tuple[int, ...],
        rng_key: jax.Array | None = None,
    ) -> NDArrayLike:
        if self._array_library == "jax":
            import jax

            return jax.random.uniform(require_rng_key(rng_key), size, minval=low, maxval=high)
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.uniform(low, high, size=size)

    def _random_normal(
        self,
        loc: float | NDArrayLike,
        scale: float | NDArrayLike,
        size: tuple[int, ...],
        rng_key: jax.Array | None = None,
    ) -> NDArrayLike:
        if self._array_library == "jax":
            import jax

            return loc + scale * jax.random.normal(require_rng_key(rng_key), size)
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.normal(loc, scale, size=size)

    def _random_lognormal(
        self,
        mean: float | NDArrayLike,
        sigma: float | NDArrayLike,
        size: tuple[int, ...],
        rng_key: jax.Array | None = None,
    ) -> NDArrayLike:
        if self._array_library == "jax":
            import jax

            return self.xp.exp(mean + sigma * jax.random.normal(require_rng_key(rng_key), size))
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.lognormal(mean=mean, sigma=sigma, size=size)

    def _random_exponential(
        self,
        scale: float | NDArrayLike,
        size: tuple[int, ...],
        rng_key: jax.Array | None = None,
    ) -> NDArrayLike:
        if self._array_library == "jax":
            import jax

            return scale * jax.random.exponential(require_rng_key(rng_key), size)
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.exponential(scale=scale, size=size)

    def _random_integers(
        self, low: int, high: int, size: tuple[int, ...], rng_key: jax.Array | None = None
    ) -> NDArrayLike:
        """Integers in [low, high), as ``numpy.random.randint`` draws them."""
        if self._array_library == "jax":
            import jax

            return jax.random.randint(require_rng_key(rng_key), size, low, high)
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.randint(low, high, size=size)

    def _random_poisson(
        self,
        lam: float | NDArrayLike,
        size: tuple[int, ...],
        rng_key: jax.Array | None = None,
    ) -> NDArrayLike:
        if self._array_library == "jax":
            import jax

            return jax.random.poisson(require_rng_key(rng_key), lam, size)
        reject_rng_key(self._array_library, rng_key)
        return self.xp.random.poisson(lam=lam, size=size)

    def _infer_dependencies(self):
        """Automatically infer dependencies from subclass signatures.
        
        Evaluates ``logpdf`` and ``rvs`` to determine required conditional 
        parameters and caches the class initialization keys for fast ``__repr__`` 
        evaluation.
        """
        deps = set()
        reserved_kwargs = {"self", "x", "size", "rng", "rng_key", "random_state", "kwargs", "args"}

        for method_name in ("logpdf", "rvs"):
            if method_name not in self.__class__.__dict__:
                continue

            method = getattr(self, method_name)
            sig = inspect.signature(method)

            for name, param in sig.parameters.items():
                if name in reserved_kwargs:
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                deps.add(name)

        self.required_variables: tuple[str, ...] = tuple(sorted(deps))

        # Cache __init__ arguments for fast __repr__ evaluation
        sig_init = inspect.signature(self.__class__.__init__)
        self._repr_keys = [
            k for k in sig_init.parameters.keys() 
            if k not in ("self", "kwargs", "args")
        ]

    def _to_device(self, array: NDArrayLike) -> NDArrayLike:
        """Safely move arrays back to CPU if requested.

        Args:
            array (NDArrayLike): The array to format.

        Returns:
            NDArrayLike: CPU or GPU array depending on instance settings.
        """
        if self.use_cupy and not self.return_gpu:
            if hasattr(array, "get"):
                return getattr(array, "get")()
            return np.asarray(array)
        return array

    def __repr__(self) -> str:
        """String representation of the prior for metadata serialization.

        Returns:
            str: Evaluatable string representation of the object.
        """
        class_name = self.__class__.__name__
        args = []
        for key in getattr(self, "_repr_keys", []):
            if hasattr(self, key):
                val = getattr(self, key)
                if isinstance(val, str):
                    args.append(f"{key}='{val}'")
                elif isinstance(val, (int, float, tuple, list, bool)):
                    args.append(f"{key}={val}")
                
        args_str = ", ".join(args)
        return f"{class_name}({args_str})"

    def copy(self) -> AbstractPrior:
        """Return a deep copy of the prior.

        Returns:
            AbstractPrior: Deep copy of self.
        """
        return deepcopy(self)


class Prior(AbstractPrior):
    _default_latex_labels = {}

    def __init__(
        self,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        minimum: float = -np.inf,
        maximum: float = np.inf,
        boundary: str | None = None,
        check_range_nonzero: bool = True,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform: Callable[..., NDArrayLike] | None = None,
        inverse_transform: Callable[..., NDArrayLike] | None = None,
        log_jacobian: Callable[..., NDArrayLike] | None = None,
        force_backend: str | None = None,
    ):
        """Main 1D prior constructor.

        Args:
            name (str, optional): Name associated with prior. (default: ``None``)
            name_phys (str, optional): Name of the associated physical parameter. 
                Defaults to name if not provided. (default: ``None``)
            latex_label (str, optional): Latex label associated with prior. (default: ``None``)
            unit (str, optional): A Latex string describing the units. (default: ``None``)
            minimum (float, optional): Minimum of the domain. (default: ``-np.inf``)
            maximum (float, optional): Maximum of the domain. (default: ``np.inf``)
            boundary (str, optional): Boundary condition (None, 'periodic', 'reflective'). (default: ``None``)
            check_range_nonzero (bool, optional): If ``True``, check that the range is > 0. (default: ``True``)
            use_cupy (bool, optional): If ``True``, uses CuPy for acceleration. (default: ``False``)
            return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
            forward_transform (callable, optional): Transform from physical to sampling space. (default: ``None``)
            inverse_transform (callable, optional): Transform from sampling to physical space. (default: ``None``)
            log_jacobian (callable, optional): Log Jacobian of the physical to sampling transform. (default: ``None``)
            force_backend (str, optional): ``"cpu"``, a CUDA name or ``"jax"``; see
                :class:`AbstractPrior`. (default: ``None``)

        Raises:
            ValueError: If ``check_range_nonzero`` is True and maximum <= minimum, or boundary is invalid.
        """
        super().__init__(use_cupy=use_cupy, return_gpu=return_gpu, force_backend=force_backend)

        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                f"maximum {maximum} <= minimum {minimum} "
                f"for {type(self).__name__} prior on {name}"
            )

        self.name = name
        self.name_phys = name_phys if name_phys is not None else name
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.latex_label = latex_label
        self.boundary = boundary
        
        if self.boundary not in [None, "periodic", "reflective"]:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary}' for prior '{self.name}'. "
                f"Supported values are None, 'periodic', or 'reflective'."
            )

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.log_jacobian = log_jacobian

    def rvs(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draws a random variable(s) from the prior distribution.

        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments for conditional priors.

        Returns:
            NDArrayLike: A random sample drawn from the prior distribution.

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
            **kwargs: Additional keyword arguments to pass to the rvs method.
            
        Returns:
            NDArrayLike: A random sample in the physical parameter space.
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform is None:
            return self._to_device(samples)
        
        output = self.inverse_transform(samples, **kwargs)
        return self._to_device(output)
    
    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """The log probability density at the given value(s).

        Args:
            x (ArrayLike): Value(s) to calculate logpdf.
            **kwargs: Additional keyword arguments for conditional priors.

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
            x_phys (ArrayLike): Value(s) in physical parameter space.
            **kwargs: Additional keyword arguments.
                
        Returns:
            NDArrayLike: Log probability density in physical space.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = logp_samp + log_J
        return self._to_device(output)
    
    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Probability density of the prior distribution.

        Args:
            x (ArrayLike): Value(s) to calculate density.
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
            x_phys (ArrayLike): Value(s) in physical space.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Probability density in physical space.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        n_points: int = 1000,
        **kwargs
    ) -> NDArrayLike:
        """Generic method to calculate CDF numerically.

        Args:
            x (ArrayLike): Value(s) to calculate the CDF at.
            n_points (int, optional): Grid points for numerical integration. (default: ``1000``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: CDF values.

        Raises:
            ValueError: If minimum or maximum is infinite.
        """
        # * math.isinf on the bounds: jax.numpy and CuPy functions reject a Python list.
        if math.isinf(self.minimum) or math.isinf(self.maximum):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with "
                "infinite support."
            )

        x_arr = self.xp.asarray(x)
        grid, dx = self.xp.linspace(self.minimum, self.maximum, n_points, retstep=True)

        pdf_grid = self.pdf(grid, **kwargs)
        # * Trapezoidal rule, assembled without in-place writes so that it also runs in JAX.
        steps = 0.5 * dx * self.xp.cumsum(pdf_grid[:-1] + pdf_grid[1:])
        cdf_grid = self.xp.concatenate([self.xp.zeros(1, dtype=steps.dtype), steps])
        total = cdf_grid[-1]
        cdf_grid = self.xp.where(
            total > 0, cdf_grid / self.xp.where(total > 0, total, 1.0), cdf_grid
        )

        output = self.xp.interp(x_arr, grid, cdf_grid, left=0.0, right=1.0)

        if isinstance(x, (int, float)) and not self.return_gpu:
            output = float(output)
        return self._to_device(output)

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        fallback_samples: int = 10_000,
        **kwargs
    ) -> NDArrayLike:
        """Check if values fall within the prior support.

        Args:
            x (ArrayLike): Values to evaluate.
            fallback_samples (int, optional): Number of samples to draw to estimate
                bounds if minimum/maximum are undefined. (default: ``10_000``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Boolean mask indicating inclusion in prior range.
        """
        x_arr = self.xp.asarray(x)
        if self.minimum is not None and self.maximum is not None:
            mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
            return self._to_device(mask)
        
        warnings.warn(
            "Prior range not defined, using samples to estimate bounds. "
            "This may result in an inaccurate estimate.",
            stacklevel=2
        )
        samples = self.xp.asarray(self.rvs(size=fallback_samples, **kwargs))
        minimum, maximum = samples.min(), samples.max()
        mask = (x_arr >= minimum) & (x_arr <= maximum)
        return self._to_device(mask)

    @property
    def latex_label(self) -> str:
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: str | None = None):
        if latex_label is not None:
            self.__latex_label = latex_label
        elif self.name in self._default_latex_labels:
            self.__latex_label = self._default_latex_labels[self.name]
        else:
            self.__latex_label = str(self.name)

    @property
    def unit(self) -> str | None:
        return self.__unit

    @unit.setter
    def unit(self, unit: str | None = None):
        self.__unit = unit

    @property
    def latex_label_with_unit(self) -> str:
        """str: Latex label combined with unit string."""
        if self.unit is not None:
            return f"{self.latex_label} [{self.unit}]"
        return self.latex_label

    @property
    def minimum(self) -> float:
        return self._minimum

    @minimum.setter
    def minimum(self, minimum: float):
        self._minimum = minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: float):
        self._maximum = maximum

    @property
    def width(self) -> float:
        return self.maximum - self.minimum