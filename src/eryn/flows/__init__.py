# src/eryn/flows/__init__.py
"""Normalizing-flow proposals for the Eryn ensemble MCMC sampler.

This subpackage provides a backend-agnostic foundation for flow-based MCMC
proposals:

- :class:`Flow` — abstract base class with init-arg capture for config
  round-trips and cross-process transport.
- :class:`DataTransform` / :class:`IdentityTransform` — invertible transforms
  with the log-det Jacobian term required for exact MH correction factors.
- :class:`ConditioningStrategy` / :class:`OneHotLeafConditioning` — protocol
  and default implementation for encoding discrete condition ids as context
  vectors.
- :func:`get_flow_wrapper` — lazy factory; backend implementations (torch/zuko)
  are only imported when first requested.

**Optional extra**

The PyTorch backend (``ZukoFlow``, ``WhiteningTransform``) lives in the
``eryn.flows.torch`` subpackage and requires the ``flow`` extra::

    pip install eryn[flow]

Importing this module (``eryn.flows``) never triggers a torch import.

**Executor classes** (``TrainerExecutor``, ``InlineExecutor``,
``ProcessExecutor``, ``WorkerConfig``, ``TrainerError``) live in
``eryn.flows.executors``.  That module is torch-free, so these names could be
imported eagerly; they are kept in the lazy ``__getattr__`` map purely for
consistency with the other optional names and to keep the import of
``eryn.flows`` minimal.
"""
from __future__ import annotations

# NOTE: ZukoFlow and WhiteningTransform require the 'flow' extra (pip install eryn[flow]).
# `from eryn.flows import *` will therefore raise AttributeError in environments
# without torch/zuko — this is intentional and expected for an optional extra.
# TrainerExecutor/InlineExecutor/TrainerError come from the torch-free
# eryn.flows.executors module (resolved lazily in __getattr__).
__all__ = [
    "Flow",
    "FlowHistory",
    "FlowProposalDistribution",
    "DataTransform",
    "IdentityTransform",
    "ConditioningStrategy",
    "OneHotLeafConditioning",
    "get_flow_wrapper",
    "ZukoFlow",
    "WhiteningTransform",
    "TrainerError",
    "FlowSpec",
    "TrainerExecutor",
    "InlineExecutor",
    "ProcessExecutor",
    "WorkerConfig",
]

# No module-level import of torch, scipy, sklearn, .torch, or .executors.
# All heavy imports happen lazily inside __getattr__ or get_flow_wrapper.

# ---- always-available (torch-free) names imported eagerly ----
from .base import Flow, FlowHistory, FlowProposalDistribution  # noqa: E402
from .transforms import DataTransform, IdentityTransform  # noqa: E402
from .conditioning import ConditioningStrategy, OneHotLeafConditioning  # noqa: E402

# ---- lazy names — resolved by __getattr__ on first attribute access ----
# ZukoFlow/WhiteningTransform live in the optional torch subpackage; accessing
# them without torch raises AttributeError (hasattr-safe), with the install
# hint chained as cause.
_TORCH_NAMES = {"ZukoFlow", "WhiteningTransform"}

# Executor names live in the torch-free eryn.flows.executors module.  They are
# always importable (no optional backend), but resolved lazily for consistency.
_EXECUTOR_NAMES = {
    "TrainerError", "FlowSpec", "TrainerExecutor", "InlineExecutor",
    "ProcessExecutor", "WorkerConfig",
}


def get_flow_wrapper(backend: str = "zuko"):
    """Return the flow class and backend module for the given backend name.

    This is the recommended entry point for obtaining a backend-specific flow
    class.  The backend package is only imported on the first call, so
    importing ``eryn.flows`` never imports torch.

    Parameters
    ----------
    backend : str, optional
        Name of the flow backend.  Currently supported: ``"zuko"``.  Planned:
        ``"flowjax"``.  Default is ``"zuko"``.

    Returns
    -------
    flow_class : type
        The backend flow class (e.g. :class:`eryn.flows.torch.ZukoFlow`).
    backend_module : module
        The backend Python module (e.g. ``torch``).

    Raises
    ------
    ImportError
        If the backend package is not installed.  Install with
        ``pip install eryn[flow]``.
    ValueError
        If ``backend`` is not one of the known backends.

    Examples
    --------
    >>> FlowClass, torch = get_flow_wrapper("zuko")
    >>> flow = FlowClass(dims=4, ...)
    """
    _available = ["zuko"]
    if backend == "zuko":
        try:
            import torch  # noqa: F401
            from .torch.flows import ZukoFlow
            return ZukoFlow, torch
        except ImportError as exc:
            raise ImportError(
                "The 'zuko' flow backend requires torch and zuko.  "
                "Install the optional dependencies with:  pip install eryn[flow]"
            ) from exc
    else:
        raise ValueError(
            f"Unknown flow backend {backend!r}.  Available backends: {_available}.  "
            "Note: 'flowjax' is planned for a future release."
        )


def __getattr__(name: str):
    """Lazy attribute access for optional names.

    ZukoFlow/WhiteningTransform are resolved on first access.  If torch is not
    installed, AttributeError is raised (with the install hint chained as cause)
    so that ``hasattr`` returns ``False`` rather than propagating ImportError.

    Note: ``get_flow_wrapper`` (an explicit function call) keeps raising
    ``ImportError`` — that is the correct signal for a missing backend there.
    """
    if name in _TORCH_NAMES:
        try:
            if name == "ZukoFlow":
                from .torch.flows import ZukoFlow
                return ZukoFlow
            elif name == "WhiteningTransform":
                from .torch.transforms import WhiteningTransform
                return WhiteningTransform
        except ImportError as exc:
            raise AttributeError(
                f"'{name}' requires the torch backend.  "
                "Install the optional dependencies with:  pip install eryn[flow]"
            ) from exc

    if name in _EXECUTOR_NAMES:
        from . import executors
        return getattr(executors, name)

    raise AttributeError(f"module 'eryn.flows' has no attribute {name!r}")
