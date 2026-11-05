# src/eryn/flows/base.py
"""Abstract base classes for normalizing-flow proposals.

This module is torch-free.  All concrete backends (e.g. ``eryn.flows.torch``)
extend :class:`Flow` and live in optional subpackages.
"""
from __future__ import annotations

import copy
import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .transforms import DataTransform, IdentityTransform

__all__ = ["FlowHistory", "Flow", "FlowProposalDistribution"]


@dataclass
class FlowHistory:
    """Record of training and validation losses produced by :meth:`Flow.fit`.

    Parameters
    ----------
    training_loss : list of float, optional
        Per-epoch training losses.  Default is an empty list.
    validation_loss : list of float, optional
        Per-epoch validation losses.  Default is an empty list.

    Examples
    --------
    >>> h = FlowHistory(training_loss=[1.0, 0.8], validation_loss=[1.1, 0.9])
    >>> h.training_loss
    [1.0, 0.8]
    """

    training_loss: list = field(default_factory=list)
    validation_loss: list = field(default_factory=list)


class Flow(ABC):
    """Abstract base class for conditional normalizing flows.

    Subclasses must implement all abstract methods.  The concrete
    implementation is expected to live in an optional backend subpackage
    (e.g. ``eryn.flows.torch``).

    **Context contract**

    The ``context`` argument accepted by :meth:`log_prob`, :meth:`sample`,
    and :meth:`sample_and_log_prob` follows these conventions:

    - ``context=None``: unconditional flow; the backend receives no context
      vector.
    - ``context: int``: condition id.  The id is encoded via
      ``self.conditioning.encode(context)`` to produce the context vector,
      and the corresponding per-condition :attr:`data_transform` entry (if
      the transform is condition-aware) is used.
    - ``context: array-like``: raw context vector passed directly to the
      backend (data_transform condition 0 is used).

    **Correctness contract**

    :meth:`log_prob` must return the coords-space log density::

        log q(x) = backend_log_prob(data_transform.forward(x, c) | ctx)
                   + data_transform.log_abs_det_jacobian(x, z, c)

    The Jacobian correction is what makes the Metropolis-Hastings acceptance
    ratio unbiased.  Omitting it is a silent correctness bug.

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    device : str or None, optional
        Device identifier (e.g. ``"cpu"``, ``"cuda:0"``).  Passed through to
        the backend; ``None`` means use the backend default.
    data_transform : DataTransform or None, optional
        Invertible transform applied to samples before they are fed to the
        flow.  Defaults to :class:`eryn.flows.transforms.IdentityTransform`
        if ``None``.
    conditioning : ConditioningStrategy or None, optional
        Strategy for encoding integer condition ids as context vectors.
        ``None`` means unconditional or caller-supplied raw context.

    Attributes
    ----------
    dims : int
    device : str or None
    data_transform : DataTransform
    conditioning : ConditioningStrategy or None

    Notes
    -----
    The constructor captures all init arguments (including defaults) and stores
    them in ``_init_args``, enabling :meth:`config_dict` to reconstruct the
    configuration across process boundaries (e.g. multiprocessing spawn).

    **Subclass signature constraints for config_dict round-trips:**

    - All parameters must be named keyword parameters (no ``/`` positional-only).
    - ``**kwargs``-style parameters are supported: their contents are merged
      (flattened) into ``_init_args`` at capture time so that
      ``cls(**flow.config_dict())`` round-trips correctly.  A :exc:`ValueError`
      is raised if a ``**kwargs`` key collides with an explicitly named
      parameter.
    - ``*args``-style (VAR_POSITIONAL) parameters cannot be round-tripped
      through ``cls(**dict)`` and are therefore silently skipped at capture
      time.  Avoid ``*args`` in Flow subclass signatures.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        # Capture init args including defaults, excluding 'self'
        try:
            sig = inspect.signature(cls.__init__)
            bound = sig.bind(None, *args, **kwargs)
            bound.apply_defaults()
            params = list(sig.parameters.keys())
            self_param = params[0]
            init_args: dict = {}
            for k, v in bound.arguments.items():
                if k == self_param:
                    continue
                kind = sig.parameters[k].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    # Flatten **kwargs into top-level dict; detect collisions.
                    collisions = set(v.keys()) & set(init_args.keys())
                    if collisions:
                        raise ValueError(
                            f"{cls.__name__}: **kwargs key(s) {collisions} "
                            "collide with explicit parameter names."
                        )
                    init_args.update(v)
                elif kind is inspect.Parameter.VAR_POSITIONAL:
                    # *args cannot be reconstructed via cls(**dict); skip.
                    pass
                else:
                    init_args[k] = v
        except TypeError as exc:
            warnings.warn(
                f"{cls.__name__}: could not capture init args for config_dict(); "
                f"config round-trip will be unavailable ({exc}).",
                stacklevel=2,
            )
            init_args = {}
        obj._init_args = init_args
        return obj

    def __init__(
        self,
        dims: int,
        device=None,
        data_transform: Union[DataTransform, None] = None,
        conditioning=None,
    ):
        self.dims = int(dims)
        self.device = device
        self.data_transform = data_transform if data_transform is not None else IdentityTransform()
        self.conditioning = conditioning

    def config_dict(self) -> dict:
        """Return a copy of the constructor arguments used to create this instance.

        The returned dict can be unpacked as ``**kwargs`` to reconstruct an
        equivalent flow object (assuming the same class).  This is used for
        serialisation and for passing configuration across multiprocessing spawn
        boundaries.

        Returns
        -------
        dict
            Shallow copy of ``_init_args`` (excludes ``self``).

        Examples
        --------
        >>> f = SomeFlow(dims=4, device="cpu")
        >>> cfg = f.config_dict()
        >>> f2 = SomeFlow(**cfg)
        """
        return dict(self._init_args)

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every concrete backend
    # ------------------------------------------------------------------

    @abstractmethod
    def log_prob(self, x, context=None) -> np.ndarray:
        """Return the log-probability density at ``x`` in **coords space**.

        Implementations must apply :attr:`data_transform` and add the
        corresponding log-det term; see the class-level correctness contract.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Sample points in coords space.
        context : None, int, or array-like, optional
            Context following the class-level context contract.

        Returns
        -------
        log_prob : np.ndarray, shape (N,)
            Log density evaluated at each row of ``x``.
        """

    def log_prob_and_grad(self, x, context=None) -> tuple:
        """Return the coords-space log density and its gradient at ``x``.

        Optional capability: gradient-based proposals (e.g.
        :class:`eryn.moves.FlowNUTSMove`) need :math:`\\nabla_x \\log q(x)` in
        **coords space**, i.e. differentiated through :attr:`data_transform`
        (including its log-det term) as well as the flow network.  Backends
        built on an autodiff framework (torch, JAX) should override this with
        a native implementation; the log density returned here must agree with
        :meth:`log_prob` so that MCMC correction factors and gradient dynamics
        see one consistent density.

        This default raises: a backend that does not override it cannot be
        used with gradient-based moves.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Sample points in coords space.
        context : None, int, or array-like, optional
            Context following the class-level context contract.

        Returns
        -------
        log_prob : np.ndarray, shape (N,)
            Log density evaluated at each row of ``x``.
        grad : np.ndarray, shape (N, dims)
            Gradient of the log density with respect to ``x``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement log_prob_and_grad; "
            "gradient-based moves (e.g. FlowNUTSMove) require a differentiable "
            "backend that overrides this method."
        )

    @abstractmethod
    def sample(self, n: int, context=None) -> np.ndarray:
        """Draw ``n`` samples from the flow in coords space.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        context : None, int, or array-like, optional
            Context following the class-level context contract.

        Returns
        -------
        samples : np.ndarray, shape (n, dims)
            Samples drawn from the flow, mapped back to coords space via
            :attr:`data_transform`.
        """

    @abstractmethod
    def sample_and_log_prob(self, n: int, context=None) -> tuple:
        """Draw ``n`` samples and return their coords-space log-probabilities.

        More efficient than calling :meth:`sample` and :meth:`log_prob`
        separately when the backend can compute both in a single forward pass.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        context : None, int, or array-like, optional
            Context following the class-level context contract.

        Returns
        -------
        samples : np.ndarray, shape (n, dims)
            Samples drawn from the flow in coords space.
        log_prob : np.ndarray, shape (n,)
            Coords-space log density evaluated at each sample.
        """

    @abstractmethod
    def fit(self, samples, **kwargs) -> FlowHistory:
        """Train the flow on ``samples``.

        Parameters
        ----------
        samples : np.ndarray or dict[int, np.ndarray]
            Training data.  When a dict is provided, keys are condition ids
            and values are arrays of shape ``(N_i, dims)``.
        **kwargs
            Additional keyword arguments passed to the backend trainer (e.g.
            ``epochs``, ``batch_size``, ``lr``).

        Returns
        -------
        history : FlowHistory
            Training and validation losses recorded during fitting.
        """

    @abstractmethod
    def get_weights(self) -> dict:
        """Return the flow's trainable weights as a CPU-resident, picklable dict.

        This is used for checkpointing and for passing weights across process
        boundaries.

        Returns
        -------
        dict
            Backend-specific weight dictionary (e.g. ``state_dict()`` for
            PyTorch models).  Values are CPU-resident and picklable; backend
            implementations may return backend tensors (e.g.
            :class:`torch.Tensor`) rather than NumPy arrays.  Consumers on
            both sides of a process boundary must have the backend installed.
        """

    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """Restore the flow's trainable weights from a dict produced by :meth:`get_weights`.

        Backends should also accept the **snapshot** form produced by
        :meth:`get_snapshot` — a ``{"net", "data_transform"}`` dict — and install
        both the net and the matched transform atomically.  The two forms are
        disambiguated by the literal ``"net"`` key (a real weight dict's keys are
        parameter names, never ``"net"``).

        Parameters
        ----------
        weights : dict
            Either a bare weight dictionary in the format returned by
            :meth:`get_weights`, or a snapshot from :meth:`get_snapshot`.
        """

    def get_snapshot(self) -> dict:
        """Return a self-contained ``{"net", "data_transform"}`` snapshot.

        Bundles the net weights (:meth:`get_weights`) with a deep copy of the
        current :attr:`data_transform`, so the pair is a matched, directly-usable
        ``(transform, net)`` handoff: applying it to a fresh, identically
        configured flow — even one whose transform is still *unfitted* —
        reproduces this flow's densities.  This is the unit a
        :class:`~eryn.flows.executors.TrainerExecutor` ships back to the parent
        so the trained net and the transform it was trained against can never
        drift apart (a mismatch silently biases the density).

        The transform is deep-copied (mirroring the :meth:`get_weights` clone
        discipline) so the snapshot can never alias live training state.

        Returns
        -------
        dict
            ``{"net": <get_weights() dict>, "data_transform": <deep copy>}``.
            Pass it straight to :meth:`set_weights`, which detects the snapshot
            form by its ``"net"`` key.
        """
        return {
            "net": self.get_weights(),
            "data_transform": copy.deepcopy(self.data_transform),
        }

    @abstractmethod
    def save(self, h5_file, path: str = "flow") -> None:
        """Save the flow to an open HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            An open, writable HDF5 file handle.
        path : str, optional
            HDF5 group path under which to store the flow.  Default is
            ``"flow"``.
        """

    @classmethod
    @abstractmethod
    def load(cls, h5_file, path: str = "flow"):
        """Load a flow from an open HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            An open HDF5 file handle.
        path : str, optional
            HDF5 group path from which to load the flow.  Default is
            ``"flow"``.

        Returns
        -------
        Flow
            A new instance of the calling class with weights restored from
            the file.
        """


class FlowProposalDistribution:
    """Adapter exposing a :class:`Flow` as an eryn ``ProbDistContainer``-like object.

    Wraps a flow at a fixed condition id to provide ``logpdf`` / ``rvs``
    methods compatible with eryn's ``DistributionGenerate`` move.  This is a
    pure numpy boundary; no backend tensors cross it.

    Parameters
    ----------
    flow : Flow
        Fitted flow instance.
    condition : int, optional
        Condition id passed to every call of :meth:`Flow.log_prob` and
        :meth:`Flow.sample`.  Default is ``0``.

    Examples
    --------
    >>> fpd = FlowProposalDistribution(my_flow, condition=1)
    >>> samples = fpd.rvs(100)        # shape (100, dims)
    >>> log_q = fpd.logpdf(samples)   # shape (100,)
    """

    def __init__(self, flow: Flow, condition: int = 0):
        self.flow = flow
        self.condition = int(condition)

    def logpdf(self, x, **kwargs) -> np.ndarray:
        """Evaluate the coords-space log density at ``x``.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Points at which to evaluate the density.
        **kwargs
            Accepted and ignored for compatibility with eryn callers that may
            pass extra keyword arguments (e.g. ``random_state``).

        Returns
        -------
        log_prob : np.ndarray, shape (N,)
            Log density values.
        """
        return self.flow.log_prob(np.asarray(x), context=self.condition)

    def rvs(self, size, **kwargs) -> np.ndarray:
        """Draw samples from the flow.

        Follows the same shape contract as eryn's ``ProbDistContainer.rvs``:
        an integer ``size`` returns an array of shape ``(size, dims)``; a
        tuple ``size`` returns an array of shape ``size + (dims,)``.  For
        example, ``rvs((5, 3))`` returns shape ``(5, 3, dims)``.

        Parameters
        ----------
        size : int or tuple of int
            Number of samples.  An integer ``n`` draws ``n`` samples.  A
            tuple ``(d0, d1, ...)`` draws ``d0 * d1 * ...`` samples and
            reshapes the result to ``(d0, d1, ..., dims)``.
        **kwargs
            Accepted and ignored for compatibility with eryn callers that may
            pass extra keyword arguments (e.g. ``random_state``).

        Returns
        -------
        samples : np.ndarray, shape ``size + (dims,)`` or ``(size, dims)``
            Samples drawn from the flow.
        """
        if isinstance(size, (tuple, list)):
            shape = tuple(int(s) for s in size)
            n = int(np.prod(shape)) if shape else 1
            flat = self.flow.sample(n, context=self.condition)
            return flat.reshape(shape + (self.flow.dims,))
        n = int(size)
        return self.flow.sample(n, context=self.condition)
