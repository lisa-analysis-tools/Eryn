# src/eryn/flows/torch/__init__.py
"""Torch-backend implementations for eryn.flows.

Requires the ``flow`` optional extra (``pip install eryn[flow]``), which
installs ``torch`` and ``zuko``.

This subpackage provides:

- :class:`WhiteningTransform` — periodic-aware per-condition whitening
  preprocessor implementing the :class:`eryn.flows.DataTransform` ABC.
- :class:`ZukoFlow` — conditional normalizing flow backed by the zuko library,
  implementing the full :class:`eryn.flows.base.Flow` ABC with coords-space
  densities and HDF5 serialisation.
"""
from __future__ import annotations

from .flows import BaseTorchFlow, ZukoFlow
from .transforms import WhiteningTransform

__all__ = [
    "BaseTorchFlow",
    "ZukoFlow",
    "WhiteningTransform",
]
