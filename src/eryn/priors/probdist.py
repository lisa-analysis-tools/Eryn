from __future__ import annotations

import inspect
from copy import deepcopy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from ..utils.utility import NDArrayLike
from .base import array_module, check_rng_key, resolve_backend, warn_if_jax_is_float32

if TYPE_CHECKING:
    # * For type checkers only: JAX stays optional and loads when a container first uses it.
    import jax

Element = TypeVar("Element", str, int)

class ProbDistContainer:
    """Container for holding and generating prior info.

    Args:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)
        return_gpu (bool, optional): If ``True``, return CuPy array. If ``False``,
            return Numpy array. (default: ``False``)
        force_backend (str, optional): ``"cpu"``, a CUDA name or ``"jax"``; see
            :class:`~eryn.priors.base.AbstractPrior`. The container sets this backend on
            every prior it holds, in place; :meth:`to_backend` returns a converted copy
            instead. (default: ``None``)

    On the jax backend, :meth:`logpdf`, :meth:`rvs` (with ``rng_key``) and the
    whole-vector transforms (:meth:`forward_transform`, :meth:`inverse_transform`,
    :meth:`log_jacobian`, :meth:`logpdf_physical`, :meth:`rvs_physical`) run under
    ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

    Attributes:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.
        priors (list): list of indexes and their associated distributions arranged
            in a list.
        ndim (int): Full dimensionality.
    
    Raises:
        ValueError: Missing parameters or incorrect index keys.
    """

    def __init__(
        self, 
        priors_in: Mapping[Element | Tuple[Element, ...], Any],
        use_cupy: bool = False,
        return_gpu: bool = False,
        force_backend: str | None = None,
    ):
        priors_in = dict(priors_in)
        self.priors_in = priors_in.copy()
        self.priors: list[list[Any]] = []
        self.has_strings: bool = False
        self.has_ints: bool = False
        self.name_to_idx: Dict[int | str, int] = {}
        self.key_order: list[int] | list[str] = []

        self.return_gpu: bool = return_gpu
        # Propagate the array backend RECURSIVELY: a child that is itself a
        # ProbDistContainer (a nested joint prior) forwards ``use_cupy`` only to
        # ITS direct children, so without recursion a grandchild distribution
        # keeps its original backend -- and a numpy grandchild fed cupy values
        # raises "Implicit conversion to a NumPy array is not allowed".
        self._set_backend(resolve_backend(force_backend, use_cupy))
        if self._array_library == "jax":
            warn_if_jax_is_float32()

        self._parse_priors()
        self._build_signatures()

    def _set_backend(self, library: str) -> None:
        """Set the array library on this container and all nested priors/containers.

        The library is stored as a name (``xp`` is derived from it) so this is safe to
        call post-construction as well; nested :class:`ProbDistContainer`
        children recurse so arbitrarily deep joints switch backend together.
        """
        self._array_library = library
        for prior in self.priors_in.values():
            if isinstance(prior, ProbDistContainer):
                prior._set_backend(library)
                prior.return_gpu = True
            elif hasattr(prior, "_array_library"):
                prior._array_library = library
                prior.return_gpu = True
            elif hasattr(prior, "use_cupy"):
                # * A distribution from outside eryn that knows only the CuPy flag.
                prior.use_cupy = library == "cupy"
                prior.return_gpu = True

    def _set_use_cupy(self, use_cupy: bool) -> None:
        """Set NumPy or CuPy on this container and all nested priors/containers."""
        self._set_backend(resolve_backend(None, use_cupy))

    @property
    def backend_name(self) -> str:
        """str: Array library of the container: ``"numpy"``, ``"cupy"`` or ``"jax"``."""
        return self._array_library

    @property
    def use_cupy(self) -> bool:
        """bool: Whether the container computes with CuPy."""
        return self._array_library == "cupy"

    @use_cupy.setter
    def use_cupy(self, use_cupy: bool):
        self._set_use_cupy(use_cupy)

    def to_backend(self, force_backend: str) -> ProbDistContainer:
        """Return a deep copy of the container, and of every prior in it, on another backend.

        Unlike ``force_backend`` in the constructor, which switches the given priors in
        place, this leaves the container and its priors unchanged.

        Args:
            force_backend (str): Backend name, as for the constructor.

        Returns:
            ProbDistContainer: The converted copy.
        """
        library = resolve_backend(force_backend)
        if library == "jax":
            warn_if_jax_is_float32()
        converted = deepcopy(self)
        converted._set_backend(library)
        return converted

    def _parse_priors(self) -> None:
        """Parse input keys, assert uniformity, and determine array shapes."""
        current_ind = 0
        key_order_tmp: list[str] = []
        temp_inds: list[NDArray[np.int_]] = []

        for inds, dist in self.priors_in.items():
            if isinstance(inds, tuple):
                inds_tmp: list[int] = []
                for i in range(len(inds)):
                    if isinstance(inds[i], str):
                        if self.has_ints:
                            raise ValueError("Cannot mix strings and integer keys.")
                        self.has_strings = True
                        inds_tmp.append(current_ind)
                        self.name_to_idx[inds[i]] = current_ind
                        key_order_tmp.append(str(inds[i]))
                        current_ind += 1

                    elif isinstance(inds[i], int):
                        if self.has_strings:
                            raise ValueError("Cannot mix strings and integer keys.")
                        self.has_ints = True
                        inds_tmp.append(int(inds[i]))
                        self.name_to_idx[inds[i]] = int(inds[i])
                        current_ind += 1
                    else:
                        raise ValueError("Index in tuple must be int or str.")

                inds_in = np.asarray(inds_tmp)
                self.priors.append([inds_in, dist])

            elif isinstance(inds, int):
                if self.has_strings:
                    raise ValueError("Cannot mix strings and integer keys.")
                self.has_ints = True
                inds_in = np.array([inds])
                self.name_to_idx[inds] = inds
                self.priors.append([inds_in, dist])
                current_ind += 1

            elif isinstance(inds, str):
                if self.has_ints:
                    raise ValueError("Cannot mix strings and integer keys.")
                self.has_strings = True
                key_order_tmp.append(inds)
                inds_in = np.array([current_ind])
                self.name_to_idx[inds] = current_ind
                self.priors.append([inds_in, dist])
                current_ind += 1
            else:
                raise ValueError("Keys for prior dictionary must be an integer, string, or tuple.")

            if self.has_strings:
                assert key_order_tmp is not None
                self.key_order = key_order_tmp
            if self.has_ints:
                self.key_order = [i for i in range(current_ind)]

            temp_inds.append(inds_in)

        uni_inds = np.unique(np.concatenate(temp_inds).flatten())
        if len(uni_inds) != len(np.arange(np.max(uni_inds) + 1)):
            raise ValueError("Please ensure all sampled parameters are included in priors.")

        self.ndim: int = int(uni_inds.max() + 1)
        
    def _build_signatures(self) -> None:
        """Cache method signatures to filter kwargs correctly."""
        self._prior_signatures: list[dict[str, dict[str, Any]]] = []
        
        for _, dist in self.priors:            
            sig_info: dict[str, dict[str, Any]] = {}
            for method_name in ("logpdf", "logpmf", "rvs"):
                if hasattr(dist, method_name):
                    sig = inspect.signature(getattr(dist, method_name))
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    allowed_keys = {
                        k for k, p in sig.parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD
                    }
                    sig_info[method_name] = {"has_varkw": has_varkw, "allowed_keys": allowed_keys}
            self._prior_signatures.append(sig_info)

    def _filter_kwargs(self, prior_idx: int, method_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter kwargs to match the exact signature of a prior method."""
        if not kwargs:
            return {}

        info = self._prior_signatures[prior_idx].get(method_name)
        if info is None:
            return {}

        if info["has_varkw"]:
            return kwargs

        return {k: v for k, v in kwargs.items() if k in info["allowed_keys"]}

    def _extract_dependencies(
        self, prior_i: Any, x: NDArray[np.float64], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Dynamically pull required conditional parameters from kwargs or state array."""
        deps: dict[str, Any] = {}
        required = getattr(prior_i, "required_variables", tuple())
        for req in required:
            if req in kwargs:
                deps[req] = kwargs[req]
            elif req in self.name_to_idx:
                idx = self.name_to_idx[req]
                deps[req] = x[..., idx]
            else:
                raise ValueError(f"Conditional variable '{req}' not found.")
        return deps

    @property
    def xp(self) -> Any:
        """Return the array module of the backend: NumPy, CuPy or jax.numpy.

        A missing CuPy or JAX raises ImportError when the container is built, not here.
        """
        return array_module(self._array_library)

    def logpdf(
        self,
        x: NDArray,
        keys: Sequence[str | int] | None = None,
        **kwargs: Any,
    ) -> NDArrayLike:
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[..., ndim]):
                Input parameters to get prior values.
            keys (list, optional): List of keys related to which parameters to gather the logpdf for.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__`` 
                function. Even when using this kwarg, must provide all ``ndim`` parameters as input. The prior will just not 
                be calculated if its associated key is not included. Default is ``None``.

        Returns:
            cp.ndarray or np.ndarray[``size + (self.ndim,)``]: Prior values.

        """
        x_arr = self.xp.asarray(x)
        squeeze = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            squeeze = True
        elif x_arr.ndim != 2:
            raise ValueError("x needs to be a 1 or 2 dimensional array.")

        nsamples = x_arr.shape[0]
        prior_vals = self.xp.zeros(nsamples, dtype=self.xp.float64)

        for i, (inds, prior_i) in enumerate(self.priors):
            if keys is not None:
                if len(inds) > 1 and tuple(inds) not in keys:
                    continue
                elif len(inds) == 1 and inds[0] not in keys:
                    continue

            vals_in = x_arr[:, inds]
            if len(inds) == 1:
                vals_in = vals_in[:,0]

            deps = self._extract_dependencies(prior_i, x_arr, kwargs)
            all_kwargs = {**kwargs, **deps}

            if hasattr(prior_i, "logpdf"):
                safe_kwargs = self._filter_kwargs(i, "logpdf", all_kwargs)
                temp = prior_i.logpdf(vals_in, **safe_kwargs)
            else:
                safe_kwargs = self._filter_kwargs(i, "logpmf", all_kwargs)
                temp = prior_i.logpmf(vals_in, **safe_kwargs)
            
            if temp.shape == (nsamples,):
                temp_out = temp
            elif temp.size == nsamples:
                temp_out = temp.reshape(nsamples)
            else:
                raise ValueError(
                    f"logpdf for prior at indices {tuple(int(v) for v in inds)} "
                    f"returned {temp.size} value(s) with shape {temp.shape}; expected "
                    f"exactly {nsamples} (one log-density per sample). This usually "
                    f"means a conditional dependency did not broadcast correctly "
                    f"against this prior's input shape (N, {len(inds)})."
                )
            prior_vals += temp_out.squeeze()

        if squeeze:
            # * .item() needs a concrete value, so under JAX a 0-d array is returned instead.
            prior_vals = prior_vals[0] if self._array_library == "jax" else prior_vals[0].item()

        return self._to_output(prior_vals)

    def _to_output(self, array: NDArrayLike) -> NDArrayLike:
        """Bring CuPy results to the host unless ``return_gpu``; other arrays pass through."""
        if self.use_cupy and not self.return_gpu and hasattr(array, "get"):
            return getattr(array, "get")()
        return array

    def ppf(self, x: NDArray, groups: Any = None, **kwargs: Any) -> NDArrayLike:
        """Percent point function (inverse of cdf)."""
        raise NotImplementedError("PPF is not yet implemented for the ProbDistContainer.")

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        keys: Sequence[str | int] | None = None,
        rng_key: jax.Array | None = None,
        **kwargs: Any,
    ) -> NDArrayLike:
        """Generate random values according to prior distribution

        The user will have to be careful if there are prior functions that
        do not have an ``rvs`` method. This means that generated points may lay
        inside the prior of all input priors that have ``rvs`` methods, but
        outside the prior if priors without the ``rvs`` method are included.

        Args:
            size (int or tuple of ints, optional): Output size for number of generated
                sources from prior distributions.
            keys (list, optional): List of keys related to which parameters to generate.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__``
                function. If used, it will produce and output array of ``tuple(size) + (len(keys),)``.
                Default is ``None``.
            rng_key (jax.Array, optional): Key of the draws, from ``jax.random.key``,
                required on the jax backend. Each prior draws from
                ``jax.random.fold_in(rng_key, position)``, with its position in
                :attr:`priors`, so equal priors give different columns. NumPy and CuPy
                containers draw from the global random state and reject a key.
                Default is ``None``.

        Returns:
            cp.ndarray or np.ndarray[``size + (self.ndim,)``]: Generated samples.

        Raises:
            ValueError: If size is not an int or tuple, or the jax backend gets no key.
            TypeError: If a NumPy or CuPy container gets a key.

        """
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple):
            raise ValueError("Size must be an int or tuple of ints.")
        check_rng_key(self._array_library, rng_key)

        # * One float64 column per parameter, stacked once at the end: no in-place writes, so
        # * the same code runs in JAX. Parameters left out by keys stay zero.
        zeros = self.xp.zeros(size, dtype=self.xp.float64)
        columns: list[Any] = [None] * self.ndim

        pending = [(i, inds, prior_i) for i, (inds, prior_i) in enumerate(self.priors)]
        evaluated_names = set(kwargs.keys())
        max_attempts = len(pending) * 2
        attempts = 0

        while pending:
            if attempts > max_attempts:
                raise RuntimeError("Circular dependency detected in prior conditionals.")
            attempts += 1

            i, inds, prior_i = pending.pop(0)

            if keys is not None:
                if len(inds) > 1 and tuple(inds) not in keys:
                    continue
                elif len(inds) == 1 and inds[0] not in keys:
                    continue

            if not hasattr(prior_i, "rvs"):
                continue

            required = getattr(prior_i, "required_variables", tuple())
            if not all(req in evaluated_names for req in required):
                pending.append((i, inds, prior_i))
                continue

            deps = {}
            for req in required:
                if req in kwargs:
                    deps[req] = kwargs[req]
                elif req in self.name_to_idx:
                    column = columns[self.name_to_idx[req]]
                    deps[req] = zeros if column is None else column
                else:
                    raise ValueError(f"Conditional variable '{req}' not found.")
            all_kwargs = {**kwargs, **deps}
            if rng_key is not None:
                import jax

                all_kwargs["rng_key"] = jax.random.fold_in(rng_key, i)

            safe_kwargs = self._filter_kwargs(i, "rvs", all_kwargs)

            samples_raw = self.xp.asarray(prior_i.rvs(size=size, **safe_kwargs))

            if len(inds) == 1:
                expected_size = int(np.prod(size))
                if samples_raw.shape == size:
                    block = samples_raw[..., None]
                elif samples_raw.size == expected_size:
                    block = samples_raw.reshape(size)[..., None]
                else:
                    raise ValueError(
                        f"rvs for prior at index {int(inds[0])} returned "
                        f"{samples_raw.size} value(s) with shape {samples_raw.shape}; expected "
                        f"exactly {expected_size} matching size {size}. Check that "
                        f"any conditional dependencies broadcast correctly."
                    )
            else:
                expected_shape = size + (len(inds),)
                expected_size = int(np.prod(expected_shape))
                if samples_raw.shape == expected_shape:
                    block = samples_raw
                elif samples_raw.size == expected_size:
                    block = samples_raw.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"rvs for prior at indices {tuple(int(v) for v in inds)} "
                        f"returned {samples_raw.size} value(s) with shape {samples_raw.shape}; "
                        f"expected exactly {expected_size} matching shape "
                        f"{expected_shape}. Check that any conditional dependencies "
                        f"broadcast correctly."
        )

            for column_index, position in enumerate(inds):
                columns[position] = self.xp.asarray(block[..., column_index], dtype=self.xp.float64)

            for idx in inds:
                name = next((k for k, v in self.name_to_idx.items() if v == idx), None)
                if name:
                    evaluated_names.add(name) # type: ignore

        out = self.xp.stack([zeros if column is None else column for column in columns], axis=-1)
        return self._to_output(out)

    def forward_transform(self, x_phys: NDArrayLike) -> NDArrayLike:
        """Map physical points, shaped ``(..., ndim)``, to the sampling space.

        Each prior maps its own columns with its ``forward_transform``, a
        :class:`~eryn.priors.joint.JointPrior` with ``forward_transform_nd``, a nested
        container recursively; parameters without a transform pass through. Conditional
        variables are not passed, since the built-in transforms take none.

        Args:
            x_phys (array): Points in the physical space.

        Returns:
            array: The points in the sampling space, shaped like ``x_phys``.
        """
        return self._map_entries(x_phys, "forward_transform", "forward_transform_nd")

    def inverse_transform(self, u_samp: NDArrayLike) -> NDArrayLike:
        """Map sampling-space points, shaped ``(..., ndim)``, to the physical space.

        The inverse of :meth:`forward_transform`, entry by entry.

        Args:
            u_samp (array): Points in the sampling space.

        Returns:
            array: The points in the physical space, shaped like ``u_samp``.
        """
        return self._map_entries(u_samp, "inverse_transform", "inverse_transform_nd")

    def log_jacobian(self, x_phys: NDArrayLike) -> NDArrayLike:
        """Return ln|det du/dx| of :meth:`forward_transform` at physical points ``(..., ndim)``.

        The transforms act on separate parameters, so the log-determinant is the sum of the
        priors' ``log_jacobian`` (``log_jacobian_nd`` for joint priors); parameters without a
        transform add zero.

        Args:
            x_phys (array): Points in the physical space.

        Returns:
            array: The log-Jacobian, shaped ``x_phys.shape[:-1]``.
        """
        values = self.xp.asarray(x_phys)
        total = self.xp.zeros(values.shape[:-1])
        for inds, prior_i in self.priors:
            if isinstance(prior_i, ProbDistContainer):
                term = prior_i.log_jacobian(values[..., inds])
            elif hasattr(prior_i, "log_jacobian_nd"):
                if prior_i.log_jacobian_nd is None:
                    continue
                term = prior_i.log_jacobian_nd(values[..., inds])
            else:
                function = getattr(prior_i, "log_jacobian", None)
                if function is None:
                    continue
                self._check_single_parameter(inds, prior_i)
                term = function(values[..., inds[0]])
            total = total + self.xp.asarray(term)
        return self._to_output(total)

    def logpdf_physical(self, x_phys: NDArrayLike, **kwargs: Any) -> NDArrayLike:
        """Log prior density at physical points: :meth:`logpdf` of the sampling-space
        points plus :meth:`log_jacobian`, as ``Prior.logpdf_physical`` does for one parameter.

        Args:
            x_phys (array): Points in the physical space, shaped ``(..., ndim)``.
            **kwargs: Passed on to :meth:`logpdf`.

        Returns:
            array: The log density in the physical space.
        """
        return self.logpdf(self.forward_transform(x_phys), **kwargs) + self.log_jacobian(x_phys)

    def rvs_physical(
        self,
        size: int | tuple[int, ...] = 1,
        keys: Sequence[str | int] | None = None,
        rng_key: jax.Array | None = None,
        **kwargs: Any,
    ) -> NDArrayLike:
        """Draw from the prior and map the draws to the physical space.

        Args:
            size (int or tuple of ints, optional): As for :meth:`rvs`.
            keys (list, optional): As for :meth:`rvs`.
            rng_key (jax.Array, optional): As for :meth:`rvs`.
            **kwargs: Passed on to :meth:`rvs`.

        Returns:
            array: Draws in the physical space, shaped ``size + (ndim,)``.
        """
        # TODO: decide whether keys stays: skipped parameters are zeros that inverse_transform
        # TODO: then maps (a LogUniform to 1, a CosineUniform to pi/2), so they look like draws.
        draws = self.rvs(size=size, keys=keys, rng_key=rng_key, **kwargs)
        return self.inverse_transform(draws)

    def _map_entries(
        self, points: NDArrayLike, method_name: str, joint_method_name: str
    ) -> NDArrayLike:
        """Apply each prior's transform to its own columns of the parameter vector.

        Serves :meth:`forward_transform` and :meth:`inverse_transform`: ``method_name`` names
        the one-parameter transform of a :class:`~eryn.priors.base.Prior`, and
        ``joint_method_name`` the matching ``*_nd`` transform of a joint prior.
        """
        values = self.xp.asarray(points)
        # * Parameters without a transform pass through unchanged.
        columns = [values[..., position] for position in range(self.ndim)]
        for inds, prior_i in self.priors:
            if isinstance(prior_i, ProbDistContainer):
                # * Nested container: maps its block recursively.
                mapped = getattr(prior_i, method_name)(values[..., inds])
            elif hasattr(prior_i, joint_method_name):
                # * Joint prior: maps its whole block with its *_nd transform.
                function = getattr(prior_i, joint_method_name)
                if function is None:
                    continue
                mapped = function(values[..., inds])
            else:
                # * One-parameter prior: maps its column, kept as a block of width one.
                function = getattr(prior_i, method_name, None)
                if function is None:
                    continue
                self._check_single_parameter(inds, prior_i)
                mapped = self.xp.asarray(function(values[..., inds[0]]))[..., None]
            # * Scatter the block back to the positions of its parameters.
            mapped = self.xp.asarray(mapped)
            for column_index, position in enumerate(inds):
                columns[position] = mapped[..., column_index]
        # * One stack instead of in-place writes, so the same code runs in JAX.
        return self._to_output(self.xp.stack(columns, axis=-1))

    @staticmethod
    def _check_single_parameter(inds: NDArray, prior_i: Any) -> None:
        """Raise when a one-parameter transform sits under a key of several parameters."""
        if len(inds) > 1:
            raise NotImplementedError(
                f"{type(prior_i).__name__} at indices {tuple(int(v) for v in inds)} has a "
                "one-parameter transform but spans several parameters; use a JointPrior "
                "with forward_transform_nd for a joint transform."
            )

    def reset_key_order(self, new_key_order: list[int] | list[str]) -> None:
        """Resets the key order to a new key order and reshuffles the prior to match the new key order.

        Args:
            new_key_order (Sequency[str] | Sequence[int]): The new desired order of keys.
                Must contain the exact same elements as `self.key_order` and be of the same data type.

        Raises:
            TypeError: Old and new key order must be of the same type
            ValueError: Old and new key order must have identical elements
        """
        if not isinstance(new_key_order[0], type(self.key_order[0])):
            raise TypeError("The new key order must be of the same type as the current key order.")

        if set(new_key_order) != set(self.key_order) or len(new_key_order) != len(self.key_order):
            raise ValueError("The new key order must be a perfect permutation of the original elements.")

        new_idx_map = {key: idx for idx, key in enumerate(new_key_order)}
        old_to_new_idx = {old_idx: new_idx_map[key] for old_idx, key in enumerate(self.key_order)}

        new_priors = []
        for inds_in, dist in self.priors:
            updated_inds = np.asarray([old_to_new_idx[idx] for idx in inds_in])
            new_priors.append([updated_inds, dist])

        new_priors.sort(key=lambda item: np.min(item[0]))

        self.priors = new_priors
        self.key_order = new_key_order

        for key, idx in new_idx_map.items():
            self.name_to_idx[key] = idx
        self._build_signatures()