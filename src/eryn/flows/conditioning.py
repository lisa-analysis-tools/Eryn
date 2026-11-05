# src/eryn/flows/conditioning.py
"""Conditioning strategies for normalizing-flow proposals.

A :class:`ConditioningStrategy` maps a component identity (an integer
condition id) to a context vector fed to the flow, and maps a point in
parameter space back to the nearest condition id.  The same flow model
can then serve multiple components of a multi-component model unchanged.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["ConditioningStrategy", "OneHotLeafConditioning"]


@runtime_checkable
class ConditioningStrategy(Protocol):
    """Protocol for mapping a condition id to a flow-context vector.

    Implementations let the same :class:`eryn.flows.base.Flow` serve multiple
    components of a multi-component model, for example by labeling each
    component with its own one-hot vector.

    Attributes
    ----------
    context_dim : int
        Dimensionality of the context vector returned by :meth:`encode`.

    Notes
    -----
    :meth:`encode` is the only hot-path requirement: it is called on every flow
    forward pass and is what defines the protocol.  :meth:`assign` may require
    implementation-specific setup before it can be called — for example,
    :class:`OneHotLeafConditioning.assign` raises until
    :meth:`OneHotLeafConditioning.set_centroids` has been called.  Such setup
    hooks (``set_centroids`` here) are optional, implementation-specific
    extensions and are deliberately **not** part of this protocol; only
    ``context_dim``, ``encode``, and ``assign`` are.

    Examples
    --------
    Any class that exposes ``context_dim``, ``encode``, and ``assign`` is
    automatically recognised by ``isinstance`` checks:

    >>> from eryn.flows.conditioning import OneHotLeafConditioning, ConditioningStrategy
    >>> c = OneHotLeafConditioning(4)
    >>> isinstance(c, ConditioningStrategy)
    True
    """

    context_dim: int

    def encode(self, condition_id: int) -> np.ndarray: ...

    def assign(self, coords_summary: np.ndarray) -> int: ...


class OneHotLeafConditioning:
    """One-hot encoding of a discrete condition id.

    Typical use: labeling components of a multi-component model (e.g. the
    number of active sources in a trans-dimensional sampler) so that a single
    conditional flow can be shared across all component counts.

    Parameters
    ----------
    nleaves_max : int
        Number of possible condition ids.  Condition ids are integers in
        ``[0, nleaves_max)``.

    Attributes
    ----------
    nleaves_max : int
        Number of possible condition ids.
    context_dim : int
        Equal to ``nleaves_max``.

    Examples
    --------
    >>> import numpy as np
    >>> from eryn.flows.conditioning import OneHotLeafConditioning
    >>> cond = OneHotLeafConditioning(4)
    >>> ctx = cond.encode(2)
    >>> ctx.dtype
    dtype('float32')
    >>> int(np.argmax(ctx))
    2
    """

    def __init__(self, nleaves_max: int):
        self.nleaves_max = int(nleaves_max)
        self.context_dim = int(nleaves_max)
        # Filled by set_centroids() once per-component centroids are known.
        self._centroids: np.ndarray | None = None

    def encode(self, condition_id: int) -> np.ndarray:
        """Return the one-hot encoding of ``condition_id``.

        Parameters
        ----------
        condition_id : int
            Integer in ``[0, nleaves_max)``.

        Returns
        -------
        ctx : np.ndarray, shape (context_dim,), dtype float32
            One-hot vector with a ``1.0`` at position ``condition_id``.

        Raises
        ------
        ValueError
            If ``condition_id`` is outside ``[0, nleaves_max)``.
        """
        if not (0 <= condition_id < self.nleaves_max):
            raise ValueError(
                f"condition_id {condition_id} out of range [0, {self.nleaves_max})"
            )
        vec = np.zeros(self.nleaves_max, dtype=np.float32)
        vec[condition_id] = 1.0
        return vec

    def set_centroids(self, centroids: np.ndarray) -> None:
        """Store per-condition centroids used by :meth:`assign`.

        Parameters
        ----------
        centroids : np.ndarray, shape (nleaves_max, ndim)
            Representative point for each condition id.  :meth:`assign` maps
            an incoming summary vector to the nearest centroid.
        """
        self._centroids = np.asarray(centroids, dtype=float)

    def assign(self, coords_summary: np.ndarray) -> int:
        """Map a summary vector to the nearest condition id.

        Parameters
        ----------
        coords_summary : np.ndarray, shape (ndim,)
            Summary of the current state (e.g. the centroid of the current
            walkers for one component).

        Returns
        -------
        int
            Index of the closest centroid, i.e. the assigned condition id.

        Raises
        ------
        RuntimeError
            If :meth:`set_centroids` has not been called yet.
        """
        if self._centroids is None:
            raise RuntimeError("set_centroids() must be called before assign().")
        d = np.linalg.norm(
            self._centroids - np.asarray(coords_summary)[None, :], axis=1
        )
        return int(np.argmin(d))
