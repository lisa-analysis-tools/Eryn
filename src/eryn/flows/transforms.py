# src/eryn/flows/transforms.py
"""Data transforms for normalizing-flow proposals.

This module extends the fit/forward/inverse pattern from aspire with the
``log_abs_det_jacobian`` term required for exact Metropolis-Hastings correction
factors.  A per-sample scalar log-det is returned (shape ``(N,)``) rather than
a per-dimension value; returning the wrong shape silently biases MCMC acceptance
ratios.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["DataTransform", "IdentityTransform"]


class DataTransform(ABC):
    """Abstract base for invertible transforms applied to flow inputs.

    Subclasses couple a normalizing flow to a specific coordinate system by
    mapping samples ``x`` (in "coords space") to latent vectors ``z`` (in
    "flow space") via :meth:`forward`, and providing the inverse and the
    log-absolute-determinant of the Jacobian.

    The log-det term is required so that :meth:`eryn.flows.base.Flow.log_prob`
    can return the exact density in coords space::

        log q(x) = flow_log_prob(forward(x, c)) + log_abs_det_jacobian(x, z, c)

    This extends aspire's fit/forward/inverse trio with the Jacobian term
    needed for unbiased Metropolis-Hastings factors.

    Parameters
    ----------
    None.  Subclasses define their own ``__init__``.
    """

    @abstractmethod
    def fit(self, samples) -> None:
        """Fit the transform from data.

        Parameters
        ----------
        samples : np.ndarray or dict[int, np.ndarray]
            Training samples.  When a dict is provided, keys are condition ids
            and values are per-condition arrays of shape ``(N, ndim)``.  The
            transform may compute per-condition statistics or global statistics
            depending on the implementation.
        """

    @abstractmethod
    def forward(self, x, condition: int = 0):
        """Map coords-space samples to flow-space latents.

        Parameters
        ----------
        x : array-like, shape (N, ndim)
            Samples in coords space.
        condition : int, optional
            Condition id selecting per-condition parameters (e.g. per-leaf
            whitening statistics).  Default is ``0``.

        Returns
        -------
        z : np.ndarray, shape (N, ndim)
            Transformed samples in flow space (latent space).
        """

    @abstractmethod
    def inverse(self, z, condition: int = 0):
        """Map flow-space latents back to coords space.

        Parameters
        ----------
        z : array-like, shape (N, ndim)
            Samples in flow (latent) space.
        condition : int, optional
            Condition id.  Default is ``0``.

        Returns
        -------
        x : np.ndarray, shape (N, ndim)
            Reconstructed samples in coords space.
        """

    @abstractmethod
    def log_abs_det_jacobian(self, x, z, condition: int = 0):
        """Log absolute determinant of the Jacobian of ``forward``.

        Returns a **per-sample scalar**, shape ``(N,)`` — one value per row of
        ``x``.  A per-dimension return value would silently bias the MCMC
        acceptance ratio; implementors must sum over dimensions internally.

        Parameters
        ----------
        x : array-like, shape (N, ndim)
            Input samples in coords space (pre-transform).
        z : array-like, shape (N, ndim)
            Output samples in flow space (post-transform).  Provided so that
            implementations that already computed ``z`` need not recompute it.
        condition : int, optional
            Condition id.  Default is ``0``.

        Returns
        -------
        log_det : np.ndarray, shape (N,)
            Per-sample log |det J_forward(x)|.  Add this to the flow's
            log-density in latent space to obtain the density in coords space.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the transform has been fitted to data.

        Returns
        -------
        bool
            ``True`` if :meth:`fit` has been called successfully (or if the
            transform needs no fitting, as with :class:`IdentityTransform`).
        """


class IdentityTransform(DataTransform):
    """Pass-through transform (no rescaling).

    ``forward`` and ``inverse`` are identity maps; ``log_abs_det_jacobian``
    returns zeros.  Useful as a default when no preprocessing is desired.

    Parameters
    ----------
    None.

    Examples
    --------
    >>> import numpy as np
    >>> tr = IdentityTransform()
    >>> x = np.ones((5, 3))
    >>> np.testing.assert_array_equal(tr.forward(x), x)
    >>> np.testing.assert_array_equal(tr.log_abs_det_jacobian(x, x), np.zeros(5))
    """

    def fit(self, samples) -> None:
        """No-op: identity needs no fitting.

        Parameters
        ----------
        samples : array-like or dict
            Ignored.
        """

    def forward(self, x, condition: int = 0):
        """Return ``x`` unchanged.

        Parameters
        ----------
        x : array-like
            Input array.
        condition : int, optional
            Ignored.  Default is ``0``.

        Returns
        -------
        np.ndarray
            ``np.asarray(x)``.
        """
        return np.asarray(x)

    def inverse(self, z, condition: int = 0):
        """Return ``z`` unchanged.

        Parameters
        ----------
        z : array-like
            Input array.
        condition : int, optional
            Ignored.  Default is ``0``.

        Returns
        -------
        np.ndarray
            ``np.asarray(z)``.
        """
        return np.asarray(z)

    def log_abs_det_jacobian(self, x, z, condition: int = 0):
        """Return zeros (identity Jacobian determinant is 1, log is 0).

        Parameters
        ----------
        x : array-like, shape (N, ...)
            Input samples; only ``len(x)`` is used.
        z : array-like
            Ignored.
        condition : int, optional
            Ignored.  Default is ``0``.

        Returns
        -------
        log_det : np.ndarray, shape (N,)
            Array of zeros with length ``len(x)``.
        """
        return np.zeros(len(x))

    @property
    def is_fitted(self) -> bool:
        """Always ``True``: identity needs no fitting.

        Returns
        -------
        bool
            ``True``.
        """
        return True
