from __future__ import annotations

import inspect
from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Sequence, TypeVar, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp # type: ignore[import]

except (ModuleNotFoundError, ImportError) as e:
    pass

from ..utils.utility import NDArrayLike

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
        self._set_use_cupy(use_cupy)

        self._parse_priors()
        self._build_signatures()

    def _set_use_cupy(self, use_cupy: bool) -> None:
        """Set ``use_cupy`` on this container and all nested priors/containers.

        ``use_cupy`` is a flag (``xp`` is derived from it) so this is safe to
        call post-construction as well; nested :class:`ProbDistContainer`
        children recurse so arbitrarily deep joints switch backend together.
        """
        self.use_cupy = bool(use_cupy)
        for prior in self.priors_in.values():
            if isinstance(prior, ProbDistContainer):
                prior._set_use_cupy(use_cupy)
                prior.return_gpu = True
            elif hasattr(prior, "use_cupy"):
                prior.use_cupy = use_cupy
                prior.return_gpu = True

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
        """Return the appropriate array module (NumPy or CuPy)."""
        if self.use_cupy:
            if cp is None:
                raise ImportError("CuPy is not installed but use_cupy is True.")
            return cp
        return np

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

            if len(inds) == 1:
                vals_in = x_arr[:, int(inds[0])].copy()
            elif np.all(np.diff(inds) == 1):
                vals_in = x_arr[:, int(inds[0]):int(inds[-1]) + 1].copy()
            else:
                vals_in = x_arr[:, inds]

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
            prior_vals = prior_vals[0].item()

        if self.use_cupy and not self.return_gpu:
            return prior_vals.get()

        return prior_vals

    def ppf(self, x: NDArray, groups: Any = None, **kwargs: Any) -> NDArrayLike:
        """Percent point function (inverse of cdf)."""
        raise NotImplementedError("PPF is not yet implemented for the ProbDistContainer.")

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        keys: Sequence[str | int] | None = None,
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

        Returns:
            cp.ndarray or np.ndarray[``size + (self.ndim,)``]: Generated samples.

        Raises:
            ValueError: If size is not an int or tuple.

        """
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple):
            raise ValueError("Size must be an int or tuple of ints.")

        out_inds = tuple([slice(None) for _ in range(len(size))])
        out = self.xp.zeros(size + (self.ndim,), dtype=self.xp.float64)

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

            inds_in = out_inds + (inds,)
            deps = self._extract_dependencies(prior_i, out, kwargs)
            all_kwargs = {**kwargs, **deps}

            safe_kwargs = self._filter_kwargs(i, "rvs", all_kwargs)

            samples_raw = self.xp.asarray(prior_i.rvs(size=size, **safe_kwargs))
            
            if len(inds) == 1:
                adjust_inds = out_inds + (None,)
                expected_size = int(np.prod(size))
                if samples_raw.shape == size:
                    out[inds_in] = samples_raw[adjust_inds]
                elif samples_raw.size == expected_size:
                    out[inds_in] = samples_raw.reshape(size)[adjust_inds]
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
                    out[inds_in] = samples_raw
                elif samples_raw.size == expected_size:
                    out[inds_in] = samples_raw.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"rvs for prior at indices {tuple(int(v) for v in inds)} "
                        f"returned {samples_raw.size} value(s) with shape {samples_raw.shape}; "
                        f"expected exactly {expected_size} matching shape "
                        f"{expected_shape}. Check that any conditional dependencies "
                        f"broadcast correctly."
        )

            for idx in inds:
                name = next((k for k, v in self.name_to_idx.items() if v == idx), None)
                if name:
                    evaluated_names.add(name) # type: ignore

        if self.use_cupy and not self.return_gpu:
            return out.get()
        return out

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