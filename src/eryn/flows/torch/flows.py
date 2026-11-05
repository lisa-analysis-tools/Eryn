# src/eryn/flows/torch/flows.py
"""Torch-backed normalizing-flow implementations for eryn.flows.

Provides :class:`BaseTorchFlow` (shared device/weight/serialisation logic) and
:class:`ZukoFlow` (conditional NSF via the zuko library).

Requires the ``flow`` optional extra (``pip install eryn[flow]``), which
installs ``torch`` and ``zuko``.

The correctness contract (coords-space densities) is documented on
:class:`eryn.flows.base.Flow`.  In brief::

    log q(x) = flow_net.log_prob(z | ctx) + transform.log_abs_det_jacobian(x, z, c)

where ``z = transform.forward(x, c)``.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import zuko
import zuko.flows
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from eryn.flows.base import Flow, FlowHistory
from eryn.flows.transforms import IdentityTransform

__all__ = ["BaseTorchFlow", "ZukoFlow"]

# ---------------------------------------------------------------------------
# Default NSF hyper-parameters (ported from lisatools FlowModel)
# ---------------------------------------------------------------------------
_DEFAULT_NSF = dict(
    passes=2,
    transforms=8,
    bins=6,
    hidden_features=(256, 256),
    residual=True,
    randperm=True,
)


# ---------------------------------------------------------------------------
# BaseTorchFlow
# ---------------------------------------------------------------------------

class BaseTorchFlow(Flow):
    """Shared PyTorch backend behaviour for eryn flow wrappers.

    Manages device placement, weight hand-off (CPU tensors, detached),
    and HDF5 serialisation.  Concrete backends (e.g. :class:`ZukoFlow`)
    extend this class and set ``self.flow`` via the property setter.

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    device : str, optional
        PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).  Default is
        ``"cpu"``.
    data_transform : DataTransform or None, optional
        Invertible transform applied to samples before the flow.  Defaults to
        :class:`eryn.flows.transforms.IdentityTransform` if ``None``.
    conditioning : ConditioningStrategy or None, optional
        Strategy for encoding integer condition ids as context vectors.
    seed : int, optional
        Random seed applied once at construction via ``torch.manual_seed``.
        Zuko's ``rsample`` / ``rsample_and_log_prob`` do not accept an explicit
        generator, so global seeding at construction is the simplest approach
        that keeps reproducibility without hidden state.  Default is ``1234``.

    Attributes
    ----------
    xp : module
        Array-API namespace marker (``torch``).  Follows the aspire pattern.
    device : str
        Active device string.
    """

    # Array-API namespace marker (aspire pattern).
    xp = torch

    def __init__(
        self,
        dims: int,
        device: str = "cpu",
        data_transform=None,
        conditioning=None,
        seed: int = 1234,
    ):
        super().__init__(
            dims=dims,
            device=device,
            data_transform=data_transform,
            conditioning=conditioning,
        )
        self.seed = int(seed)
        # Global seed at construction.  Zuko distributions (flow(ctx).rsample etc.)
        # operate on the global torch RNG and do not accept an explicit Generator,
        # so seeding globally at construction is both necessary and sufficient.
        torch.manual_seed(self.seed)
        self._flow: nn.Module | None = None

    # ------------------------------------------------------------------
    # flow property / setter
    # ------------------------------------------------------------------

    @property
    def flow(self) -> nn.Module:
        """The underlying ``nn.Module`` (zuko flow or other).

        Raises
        ------
        RuntimeError
            If the flow module has not been set yet.
        """
        if self._flow is None:
            raise RuntimeError("Flow module has not been set.  Call the subclass constructor.")
        return self._flow

    @flow.setter
    def flow(self, module: nn.Module) -> None:
        """Store and move the module to ``self.device``.

        Note: ``torch.compile`` is deliberately NOT applied here.  Compiled
        modules use transformed state_dict keys (``_orig_mod.*``) that break
        weight round-trips (get_weights/set_weights, HDF5 save/load) and
        pickling.  Compilation can be applied externally after serialisation if
        desired.
        """
        self._flow = module.to(self.device)

    # ------------------------------------------------------------------
    # Weight hand-off
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Return the flow's trainable weights as CPU-resident detached tensors.

        The returned dict can be pickled without torch (values are
        :class:`torch.Tensor`, which pickle safely on their own) and can be
        passed across process boundaries.

        Returns
        -------
        dict
            ``{key: tensor}`` where each tensor is detached, on CPU, and cloned
            so that mutating the returned dict does NOT affect the live flow.
        """
        return {k: v.detach().cpu().clone() for k, v in self._flow.state_dict().items()}

    def set_weights(self, obj: dict) -> None:
        """Restore weights from either a bare state_dict or a full snapshot.

        Two input forms are accepted (detected by the ``"net"`` sentinel key):

        - **Snapshot form** — ``{"net": <state_dict>, "data_transform": <transform>}``
          (the output of :meth:`get_snapshot`).  The net state_dict is loaded
          AND, when ``data_transform`` is not ``None``, that transform is
          installed onto this flow (``self.data_transform = ...``) so the net and
          its matched transform are swapped in **atomically** — they always agree
          on the coords-latent map.
        - **Bare form** — a plain net ``state_dict`` (the output of
          :meth:`get_weights`).  Loaded as-is; the data transform is left
          untouched.  This is the legacy contract and stays fully supported.

        The two forms are disambiguated by the literal key ``"net"``: a real
        torch ``state_dict`` is also a dict, but its keys are module-parameter
        names (``"transforms.0...."`` etc.) and can never literally equal
        ``"net"``, so the sentinel cannot collide.

        The module is left on its current device after loading.

        Note: net weights are independent of the data_transform; in the bare
        form, swapping weights does not change the transform's Jacobian.  In the
        snapshot form both move together by design.

        Parameters
        ----------
        obj : dict
            A snapshot (``{"net", ...}``) or a bare net state_dict.
        """
        if isinstance(obj, dict) and "net" in obj:
            # Snapshot form: install the matched transform first (if shipped),
            # then load the net it was trained against.  A real state_dict can
            # never reach this branch — its keys are parameter paths, never the
            # literal "net" sentinel.
            transform = obj.get("data_transform")
            if transform is not None:
                self.data_transform = transform
            state_dict = obj["net"]
        else:
            # Bare form: a plain net state_dict (legacy / get_weights() output).
            state_dict = obj
        self._flow.load_state_dict(state_dict)
        # Ensure the module stays on the correct device (load_state_dict may
        # leave parameters on CPU if the dict came from a CPU clone).
        self._flow.to(self.device)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str) -> "BaseTorchFlow":
        """Move the flow module to ``device`` and update ``self.device``.

        Parameters
        ----------
        device : str
            Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns
        -------
        self
        """
        self.device = device
        if self._flow is not None:
            self._flow.to(device)
        return self

    # ------------------------------------------------------------------
    # HDF5 serialisation helpers (shared)
    # ------------------------------------------------------------------

    def _open_h5(self, h5_file, mode: str):
        """Return (handle, should_close) for a path string or open h5py handle."""
        if isinstance(h5_file, (str, os.PathLike)):
            return h5py.File(h5_file, mode), True
        return h5_file, False

    def save(self, h5_file, path: str = "flow") -> None:
        """Save the flow to an HDF5 file.

        Accepts either a path string (file is opened and closed) or an already
        open :class:`h5py.File` / :class:`h5py.Group`.

        Layout::

            <path>/
                config          — JSON-serialisable subset of config_dict()
                data_transform  — pickle blob (bytes scalar dataset)
                conditioning    — pickle blob (bytes scalar dataset)
                weights/
                    <key>       — one float32 dataset per state_dict tensor

        .. warning::
            The ``data_transform`` and ``conditioning`` objects are stored as
            pickle blobs.  Only load flow files from **trusted sources**;
            ``pickle.loads`` executes arbitrary code.

        Parameters
        ----------
        h5_file : str or h5py.File
            Destination.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.
        """
        handle, should_close = self._open_h5(h5_file, "a")
        try:
            grp = handle.require_group(path)

            # --- config ---
            cfg = self.config_dict()
            serialisable = _make_serialisable(cfg)
            grp.attrs["config"] = json.dumps(serialisable)

            # --- data_transform ---
            blob = pickle.dumps(self.data_transform)
            if "data_transform" in grp:
                del grp["data_transform"]
            grp.create_dataset("data_transform", data=np.bytes_(blob))

            # --- conditioning ---
            blob = pickle.dumps(self.conditioning)
            if "conditioning" in grp:
                del grp["conditioning"]
            grp.create_dataset("conditioning", data=np.bytes_(blob))

            # --- weights ---
            # Delete the entire weights group before repopulating so that
            # orphan datasets from a prior (larger) flow do not remain and
            # cause load() to fail with "Unexpected key(s)".
            if "weights" in grp:
                del grp["weights"]
            wgrp = grp.create_group("weights")
            for k, v in self.get_weights().items():
                wgrp.create_dataset(k, data=v.numpy())
        finally:
            if should_close:
                handle.close()

    @classmethod
    def load(cls, h5_file, path: str = "flow") -> "BaseTorchFlow":
        """Load a flow from an HDF5 file.

        .. warning::
            The ``data_transform`` and ``conditioning`` objects are restored
            via ``pickle.loads``, which executes arbitrary code.  Only load
            flow files from **trusted sources**.

        Parameters
        ----------
        h5_file : str or h5py.File
            Source.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.

        Returns
        -------
        BaseTorchFlow
            Restored instance with weights, transform, and conditioning.
        """
        if isinstance(h5_file, (str, os.PathLike)):
            handle = h5py.File(h5_file, "r")
            should_close = True
        else:
            handle, should_close = h5_file, False

        try:
            grp = handle[path]

            # --- config ---
            cfg = json.loads(grp.attrs["config"])
            # Strip keys stored separately as pickle blobs (data_transform, conditioning)
            cfg = _strip_pickled_keys(cfg)

            # --- data_transform ---
            raw = bytes(grp["data_transform"][()])
            data_transform = pickle.loads(raw)

            # --- conditioning ---
            raw = bytes(grp["conditioning"][()])
            conditioning = pickle.loads(raw)

            # --- build instance ---
            cfg["data_transform"] = data_transform
            cfg["conditioning"] = conditioning
            instance = cls(**cfg)

            # --- weights ---
            wgrp = grp["weights"]
            weights = {k: torch.tensor(np.array(v)) for k, v in wgrp.items()}
            instance.set_weights(weights)

            return instance
        finally:
            if should_close:
                handle.close()


# ---------------------------------------------------------------------------
# ZukoFlow
# ---------------------------------------------------------------------------

class ZukoFlow(BaseTorchFlow):
    """Conditional normalizing flow backed by the zuko library.

    Wraps a zuko NSF (Neural Spline Flow) or any other ``zuko.flows``
    distribution class, composes it with a :class:`eryn.flows.DataTransform`,
    and implements the unified context contract (``None`` / int / raw vector).

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    flow_class : str or callable, optional
        Flow architecture.  When a string, resolved via
        ``getattr(zuko.flows, flow_class)``; valid names include
        ``"NSF"``, ``"MAF"``, ``"CNF"``, ``"FFJORD"``.  A callable is used
        directly (strings are preferred for config round-trips).  Default is
        ``"NSF"``.
    device : str, optional
        PyTorch device.  Default is ``"cpu"``.
    data_transform : DataTransform or None, optional
        Invertible preprocessing transform.
    conditioning : ConditioningStrategy or None, optional
        Strategy for encoding integer condition ids as context vectors.  If
        ``None``, the flow is unconditional (``context_dim=0``).
    seed : int, optional
        Random seed.  Default is ``1234``.
    periodic_aliases : int, optional
        Maximum ``|k|`` per periodic dimension used when summing the coords
        density over periodic aliases.  A periodic coordinate is only defined
        modulo its period, so the ``data_transform``'s coords → latent map is
        many-to-one and the density of the folded variable is the **wrapped**
        density ``sum_k q_Z(z + delta_k)``.  Summing it makes the density
        exactly normalized over the period and makes :meth:`log_prob` and
        :meth:`sample_and_log_prob` agree identically (the alias set does not
        depend on which representative was drawn).  ``0`` restores the legacy
        single-image density, which under-integrates by exactly the mass the
        flow placed outside the fundamental window.  The full
        ``(2 * order + 1) ** n_periodic`` alias lattice is used (single-dimension
        displacements alone are NOT sufficient when several periods are short in
        latent units), pruned per batch to the aliases that can actually
        contribute -- so the cost collapses to one evaluation whenever every
        alias is far away, and ``order = 1`` is enough in practice
        (``order = 2`` was measured to add < 0.2 nats).
        **No effect** when the data transform has no periodic dims (or does not
        expose ``periodic_alias_offsets``): that path is bit-identical to the
        pre-aliasing implementation.  Default is ``1``.
    **flow_kwargs
        Additional keyword arguments forwarded to the flow constructor.  When
        ``flow_class="NSF"`` (the default), the following defaults are applied
        and may be overridden::

            passes=2, transforms=8, bins=6,
            hidden_features=(256, 256), residual=True, randperm=True

        For other ``flow_class`` values, no default kwargs are injected (to
        avoid passing invalid arguments).

    Notes
    -----
    The ``data_transform`` always operates in float64 on the CPU, regardless
    of the ``device`` argument.  (MPS does not support float64.)  Only the
    flow network itself is placed on ``device``.

    Examples
    --------
    >>> from eryn.flows import ZukoFlow, OneHotLeafConditioning
    >>> cond = OneHotLeafConditioning(nleaves_max=2)
    >>> flow = ZukoFlow(dims=4, conditioning=cond, seed=0,
    ...                 transforms=3, hidden_features=(64, 64), bins=5)
    >>> import numpy as np
    >>> x = np.random.randn(10, 4)
    >>> lp = flow.log_prob(x, context=0)
    >>> lp.shape
    (10,)
    """

    def __init__(
        self,
        dims: int,
        flow_class: str | Any = "NSF",
        device: str = "cpu",
        data_transform=None,
        conditioning=None,
        seed: int = 1234,
        periodic_aliases: int = 1,
        **flow_kwargs,
    ):
        super().__init__(
            dims=dims,
            device=device,
            data_transform=data_transform,
            conditioning=conditioning,
            seed=seed,
        )
        self.flow_class = flow_class
        # Periodic dims are only defined modulo their period, so the coords
        # density is the WRAPPED density (a sum over latent aliases).  See
        # _alias_offsets / _log_flow_base.  0 restores the legacy single-image
        # behaviour; no effect at all when the data transform has no periodic
        # dims (the alias set is then trivial and the fast path is taken).
        self.periodic_aliases = int(periodic_aliases)

        # --- resolve context dimension ---
        context_dim = conditioning.context_dim if conditioning is not None else 0
        self._context_dim = context_dim

        # --- resolve flow constructor ---
        FlowCls = _resolve_flow_class(flow_class)

        # --- merge defaults (NSF only) ---
        if isinstance(flow_class, str) and flow_class == "NSF":
            kw = {**_DEFAULT_NSF, **flow_kwargs}
        else:
            kw = dict(flow_kwargs)

        # --- build and register the flow via the property setter ---
        self.flow = FlowCls(features=dims, context=context_dim, **kw)

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        """Coerce and validate a sample array.

        Parameters
        ----------
        x : array-like
            Input samples; must have shape ``(N, dims)`` with ``N >= 1``.

        Returns
        -------
        x : np.ndarray, shape (N, dims), dtype float64

        Raises
        ------
        ValueError
            If ``x`` is not 2-D or its second axis does not equal ``self.dims``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dims:
            raise ValueError(
                f"x must have shape (N, {self.dims}), got {x.shape}"
            )
        return x

    # ------------------------------------------------------------------
    # Context resolution
    # ------------------------------------------------------------------

    def _resolve(self, context) -> tuple:
        """Resolve the unified context argument to (ctx_tensor_or_None, condition_int).

        Parameters
        ----------
        context : None, int, or array-like
            Context following the class-level contract.

        Returns
        -------
        ctx : torch.Tensor or None
            Context tensor of shape ``(context_dim,)`` on ``self.device``,
            or ``None`` for unconditional flows.
        condition : int
            Integer condition id (used for the data_transform).
            Returns ``0`` when context is an array-like (raw vector path).

        Raises
        ------
        ValueError
            If an int context is provided but ``self.conditioning is None``.
        ValueError
            If a non-None context is provided but ``self._context_dim == 0``.
        """
        if context is None:
            return None, 0

        # Integer condition id path (including numpy integer types).
        # Check this before the context_dim==0 guard so the caller gets a
        # specific "requires a conditioning strategy" message rather than the
        # generic context_dim==0 one.
        if isinstance(context, (int, np.integer)):
            if self.conditioning is None:
                raise ValueError(
                    "An integer context id was provided but this flow has no "
                    "conditioning strategy.  Either supply a ConditioningStrategy "
                    "at construction or pass a raw context vector."
                )
            ctx_np = self.conditioning.encode(int(context))  # float32 (context_dim,)
            ctx = torch.as_tensor(ctx_np, device=self.device)
            return ctx, int(context)

        # Raw array-like context vector path.  context_dim==0 is only possible
        # here (integer path already handled above).
        if self._context_dim == 0:
            raise ValueError(
                "This flow was built with context_dim=0 (no conditioning).  "
                "Pass context=None or rebuild with a ConditioningStrategy."
            )

        ctx_np = np.asarray(context, dtype=np.float32)
        if ctx_np.shape != (self._context_dim,):
            raise ValueError(
                f"Raw context vector must have shape ({self._context_dim},), "
                f"got {ctx_np.shape}"
            )
        ctx = torch.as_tensor(ctx_np, device=self.device)
        return ctx, 0

    # ------------------------------------------------------------------
    # log_prob
    # ------------------------------------------------------------------

    def log_prob(self, x, context=None, base_scale: float | None = None) -> np.ndarray:
        """Return the coords-space log density at ``x``.

        Implements the correctness contract::

            log q(x) = flow_net.log_prob(z | ctx)
                       + data_transform.log_abs_det_jacobian(x, z, condition)

        When the data transform has periodic dims and ``periodic_aliases > 0``
        the first term is the alias sum ``logsumexp_k flow_net.log_prob(z +
        delta_k | ctx)`` -- the exact **wrapped** density of the folded
        variable, which is what makes this normalized over the period.  See the
        ``periodic_aliases`` constructor argument.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Sample points in coords space.
        context : None, int, or array-like, optional
            Context following the class-level contract.
        base_scale : float or None, optional
            Temperature-scaling factor for the flow's base distribution.
            ``None`` (default) evaluates the density under the flow's native
            base — bit-identical to the pre-``base_scale`` behaviour.  A
            float ``s`` evaluates the density as if the base were scaled by
            ``s`` (e.g. ``N(0, s**2 * I)`` for the usual standard-normal
            base) instead, via the exact change-of-variables identity (see
            :meth:`_scaled_base_log_prob`); all transform log-dets (network
            + ``data_transform``) are unaffected.  Used by
            :class:`eryn.moves.ConditionalFlowMove` to propose at
            ``s = beta**-0.5`` for hot chains from the SAME cold-trained flow.

        Returns
        -------
        log_prob : np.ndarray, shape (N,), dtype float64
        """
        x = self._validate_x(x)
        if x.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        if base_scale is not None and not (base_scale > 0.0):
            raise ValueError(f"base_scale must be > 0, got {base_scale}")
        ctx, condition = self._resolve(context)

        # Apply data transform: coords → flow space
        z = self.data_transform.forward(x, condition)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)

        offsets = self._alias_offsets(condition)
        with torch.no_grad():
            dist = self._get_dist(ctx)
            log_flow = self._log_flow_base(dist, z_t, base_scale, offsets)  # (N,)

        # Per-point log-det: correct for any data_transform (including
        # non-constant-Jacobian transforms such as LogTransform).  The cost
        # is negligible next to the flow forward pass.
        logdet_raw = self.data_transform.log_abs_det_jacobian(x, z, condition)
        if hasattr(logdet_raw, "detach"):
            logdet_raw = logdet_raw.detach().cpu().numpy()
        logdet = np.asarray(logdet_raw, dtype=np.float64)
        return (log_flow.cpu().numpy() + logdet).astype(np.float64)

    # ------------------------------------------------------------------
    # log_prob_and_grad
    # ------------------------------------------------------------------

    def log_prob_and_grad(self, x, context=None) -> tuple:
        """Return the coords-space log density and its gradient at ``x``.

        Same density as :meth:`log_prob` (identical ops, so the returned
        ``log_prob`` is bit-identical), but evaluated **in-graph** so torch
        autograd can differentiate through the data transform, the flow
        network, and the log-det term.  Used by gradient-based proposals
        (:class:`eryn.moves.FlowNUTSMove`).

        The data transform runs on CPU/float64 by contract; only the network
        forward/backward runs on ``self.device``.  ``IdentityTransform`` is
        special-cased (its numpy ``forward`` cannot carry a grad tensor); any
        other transform must be torch-native end-to-end (as
        :class:`~eryn.flows.torch.transforms.WhiteningTransform` is) or a
        :exc:`NotImplementedError` is raised.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Sample points in coords space.
        context : None, int, or array-like, optional
            Context following the class-level contract.

        Returns
        -------
        log_prob : np.ndarray, shape (N,), dtype float64
        grad : np.ndarray, shape (N, dims), dtype float64
            ``d log_prob / dx`` per row.
        """
        x = self._validate_x(x)
        if x.shape[0] == 0:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, self.dims), dtype=np.float64),
            )
        ctx, condition = self._resolve(context)

        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        if isinstance(self.data_transform, IdentityTransform):
            # IdentityTransform.forward is numpy (np.asarray) and would sever
            # the graph (and raise on a requires_grad tensor); the identity map
            # needs no transform pass and has zero log-det.
            z = x_t
            logdet = torch.zeros(x.shape[0], dtype=torch.float64)
        else:
            z = self.data_transform.forward(x_t, condition)
            if not (isinstance(z, torch.Tensor) and z.grad_fn is not None):
                raise NotImplementedError(
                    f"log_prob_and_grad requires a torch-differentiable "
                    f"data_transform; {type(self.data_transform).__name__}."
                    f"forward did not return a graph-connected tensor."
                )
            logdet = self.data_transform.log_abs_det_jacobian(x_t, z, condition)

        dist = self._get_dist(ctx)
        # The float64 -> float32 cast is differentiable and matches log_prob's
        # precision exactly, so the two methods return bit-identical densities.
        # The alias sum is differentiable too (logsumexp), so the periodic
        # wrapped density stays consistent with log_prob here as well.
        log_flow = self._log_flow_base(
            dist,
            z.to(device=self.device, dtype=torch.float32),
            None,
            self._alias_offsets(condition),
        )

        total = log_flow.double().cpu() + logdet
        (grad,) = torch.autograd.grad(total.sum(), x_t)

        return (
            total.detach().numpy().astype(np.float64),
            grad.detach().numpy().astype(np.float64),
        )

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(self, n: int, context=None) -> np.ndarray:
        """Draw ``n`` samples from the flow in coords space.

        Parameters
        ----------
        n : int
            Number of samples.
        context : None, int, or array-like, optional
            Context.

        Returns
        -------
        samples : np.ndarray, shape (n, dims), dtype float64
        """
        x, _ = self.sample_and_log_prob(int(n), context=context)
        return x

    # ------------------------------------------------------------------
    # sample_and_log_prob
    # ------------------------------------------------------------------

    def sample_and_log_prob(self, n: int, context=None, base_scale: float | None = None) -> tuple:
        """Draw ``n`` samples and return their coords-space log densities.

        Uses zuko's ``rsample_and_log_prob`` for a single forward pass.

        With periodic dims and ``periodic_aliases > 0`` the density is instead
        re-evaluated via :meth:`log_prob` at the returned (folded) points: the
        sampler may draw an alias outside the fundamental window, whose
        single-image density is not the density of the folded point.  The
        returned ``log_prob`` therefore matches ``log_prob(samples)`` exactly --
        which matters because the Metropolis-Hastings statistic in
        :class:`eryn.moves.ConditionalFlowMove` takes the proposal's density
        from here and the current point's from :meth:`log_prob`.

        Parameters
        ----------
        n : int
            Number of samples.
        context : None, int, or array-like, optional
            Context.
        base_scale : float or None, optional
            Temperature-scaling factor for the flow's base distribution.
            ``None`` (default): draws from the flow's native base —
            bit-identical to the pre-``base_scale`` behaviour.  A float ``s``
            draws the base latent scaled by ``s`` (e.g. ``N(0, s**2 * I)``
            for the usual standard-normal base) instead (see
            :meth:`_scaled_base_rsample_and_log_prob`) and reports the
            matching density; all transform log-dets are unaffected.  Used by
            :class:`eryn.moves.ConditionalFlowMove` to propose at
            ``s = beta**-0.5`` for hot chains from the SAME cold-trained flow.

        Returns
        -------
        samples : np.ndarray, shape (n, dims), dtype float64
            Samples in coords space (inverse-transformed from flow space).
        log_prob : np.ndarray, shape (n,), dtype float64
            Coords-space log density at each sample.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return np.empty((0, self.dims), dtype=np.float64), np.empty((0,), dtype=np.float64)
        if base_scale is not None and not (base_scale > 0.0):
            raise ValueError(f"base_scale must be > 0, got {base_scale}")
        ctx, condition = self._resolve(context)
        with torch.no_grad():
            dist = self._get_dist(ctx)
            if base_scale is None:
                z_t, log_flow = dist.rsample_and_log_prob((n,))
            else:
                z_t, log_flow = self._scaled_base_rsample_and_log_prob(dist, n, float(base_scale))

        z_t = z_t.reshape(n, self.dims)
        log_flow = log_flow.reshape(n)

        # Inverse-transform: flow space → coords space
        z_cpu = z_t.cpu()
        x = self.data_transform.inverse(z_cpu, condition)
        x = np.asarray(x, dtype=np.float64)

        if self._alias_offsets(condition) is not None:
            # With periodic dims the sampler may have drawn an alias OUTSIDE the
            # fundamental window; `log_flow` above is then the density at that
            # alias, which is not the density of the folded point x.  Re-evaluate
            # through log_prob to get the exact wrapped density -- and, because
            # it is literally the same function, a value that agrees with
            # log_prob(x) by construction rather than by luck.
            return x, self.log_prob(x, context=context, base_scale=base_scale)

        # Per-point log-det at the returned coords-space points.
        # z_cpu is the flow-space representation; pass both so transforms that
        # cached z during forward can reuse it.  Cost is negligible next to the
        # flow forward pass.
        logdet_raw = self.data_transform.log_abs_det_jacobian(x, z_cpu, condition)
        if hasattr(logdet_raw, "detach"):
            logdet_raw = logdet_raw.detach().cpu().numpy()
        logdet = np.asarray(logdet_raw, dtype=np.float64)
        logq = (log_flow.cpu().numpy() + logdet).astype(np.float64)
        return x, logq

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        samples,
        *,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 512,
        validation_fraction: float = 0.2,
        val_split: str = "random",
        train_noise: float = 0.0,
        clip_grad: float | None = None,
        optimizer: str | Any = "adam",
        optimizer_kwargs: dict | None = None,
        lr_annealing: bool = False,
        patience: int | None = None,
        refit_data_transform: bool = True,
        seed: int | None = None,
        verbose: bool = False,
    ) -> FlowHistory:
        """Train the flow in-place and return the loss history.

        Implements a plain Adam training loop with optional cosine-annealing
        learning-rate schedule and patience-based early stopping.  Safe to
        call inside a spawned trainer process (``num_workers=0``, no tqdm,
        silent unless ``verbose=True``).

        Parameters
        ----------
        samples : np.ndarray, shape (N, dims) or dict[int, np.ndarray]
            Training data.  A plain array is treated as a single condition
            ``{0: samples}``.  A dict maps integer condition ids to per-condition
            sample arrays.  The entire dataset is moved to ``device`` once before
            training; very large ``N`` may exhaust device memory, in which case
            chunk or subsample the data upstream.
        n_epochs : int, optional
            Maximum number of training epochs.  Default is ``100``.
        lr : float, optional
            Initial Adam learning rate.  Default is ``1e-3``.
        batch_size : int, optional
            Mini-batch size.  Default is ``512``.
        validation_fraction : float, optional
            Fraction of assembled samples held out as a validation set.
            Default is ``0.2``.
        val_split : {"random", "temporal"}, optional
            How the validation rows are chosen.  ``"random"`` (default)
            shuffles all rows globally before splitting.  ``"temporal"``
            holds out the **newest** ``validation_fraction`` of rows *per
            condition* (assembly preserves buffer order, oldest first).  Use
            temporal for online training on correlated chain buffers: a
            random split places near/exact duplicates of training rows in the
            validation set, so early stopping rewards memorization of the
            chain tracks; the temporal holdout instead measures the NLL of
            fresh points — the same quantity an independence-proposal MH
            factor depends on.
        train_noise : float, optional
            Standard deviation of Gaussian jitter added to the *training*
            latents, in units of the per-dimension std of the training set,
            redrawn every batch.  Acts as KDE-style smoothing that suppresses
            sub-posterior-scale structure (e.g. memorization of correlated
            walker tracks); validation rows stay clean.  ``0.0`` (default)
            disables it.
        clip_grad : float or None, optional
            If not ``None``, ``torch.nn.utils.clip_grad_norm_`` is applied with
            this max-norm before each optimizer step.  Default is ``None``.
        optimizer : str or type, optional
            Which optimizer to use.  A string is resolved against
            ``torch.optim`` (case-insensitive for the common names —
            ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, ...); a
            ``torch.optim.Optimizer`` subclass is used directly.  A string or
            class spec — unlike a bare lambda — pickles cleanly across the
            :class:`~eryn.flows.executors.ProcessExecutor` spawn boundary, so the
            same selection works for the online trainer.  Default is ``"adam"``,
            which reproduces the previous hard-coded behaviour exactly.
        optimizer_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the optimizer constructor (e.g.
            ``{"weight_decay": 1e-2}`` for AdamW, ``{"momentum": 0.9}`` for SGD).
            ``lr`` is the canonical top-level argument and always wins: any ``lr``
            present here is dropped so the two cannot silently disagree.  Default
            is ``None``.
        lr_annealing : bool, optional
            If ``True``, a :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`
            schedule is applied over ``n_epochs``.  Default is ``False``.
        patience : int or None, optional
            Early-stopping patience in epochs.  Training stops when the
            validation loss has not improved for ``patience`` consecutive epochs.
            ``None`` disables early stopping.  Default is ``None``.
        refit_data_transform : bool, optional
            If ``True`` (the default), always re-fit ``self.data_transform``
            before assembling latents.  If ``False``, re-fit only when the
            transform is not yet fitted (``not self.data_transform.is_fitted``).
        seed : int or None, optional
            Seed for the train/val shuffle generator.  Passing a fixed value
            makes ``fit`` deterministic given fixed data and flow initialisation.
            An int is expected; the value is coerced with ``int()`` at the
            boundary (whole floats such as ``7.0`` are accepted).  Defaults to
            ``self.seed`` when ``None``.
        verbose : bool, optional
            If ``True``, print one line per epoch (epoch, train_loss, val_loss).
            Default is ``False``.

        Returns
        -------
        history : FlowHistory
            Per-epoch training and validation losses.  ``best_state`` is loaded
            back into the flow before returning (best-val-loss model).

        Raises
        ------
        ValueError
            If ``validation_fraction`` is not strictly within ``(0.0, 1.0)``,
            ``val_split`` is not ``"random"`` or ``"temporal"``, or
            ``train_noise`` is negative.
        ValueError
            If the total assembled sample count is less than 2 (cannot split).
        ValueError
            If ``samples`` is a dict with more than one key but
            ``self.conditioning`` is ``None`` (conditions require a conditioning
            strategy).
        ValueError
            If any assembled latents contain non-finite values (NaN or Inf),
            naming the offending condition(s) and dimension(s).
        ValueError
            If ``optimizer`` is a string that does not resolve to a
            ``torch.optim`` class.
        TypeError
            If ``optimizer`` is neither a string nor a
            ``torch.optim.Optimizer`` subclass.

        Notes
        -----
        **Zuko per-row context batching** — during training, ``ctx_b`` is a
        ``(B, context_dim)`` batch with one context row per sample.  Calling
        ``self._flow(ctx_b).log_prob(z_b)`` returns a ``(B,)`` tensor of
        per-sample log-probabilities because zuko propagates the leading batch
        dimension through the context network and returns one scalar per row.
        (Verified against zuko source: a 2-D context argument is treated as a
        batch of independent contexts, not broadcast as a shared context.  This
        differs from the 1-D context in ``log_prob`` / ``sample``, which is
        broadcast over the sample batch.)

        **Tensor placement** — ``z`` and ``ctx`` are moved to ``self.device``
        once before building the :class:`~torch.utils.data.TensorDataset`.
        Datasets are small and keeping them on-device avoids per-batch
        host-to-device copies inside the DataLoader.
        """
        # ------------------------------------------------------------------
        # 0. Validate hyperparameters at the boundary
        # ------------------------------------------------------------------
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError(
                "validation_fraction must lie strictly in the open interval "
                f"(0.0, 1.0), got {validation_fraction}."
            )
        if val_split not in ("random", "temporal"):
            raise ValueError(
                f'val_split must be "random" or "temporal", got {val_split!r}.'
            )
        if train_noise < 0.0:
            raise ValueError(f"train_noise must be >= 0, got {train_noise}.")

        # ------------------------------------------------------------------
        # 1. Normalise samples to dict[int, np.ndarray]
        # ------------------------------------------------------------------
        if not isinstance(samples, dict):
            samples_dict = {0: np.asarray(samples, dtype=np.float64)}
        else:
            samples_dict = {k: np.asarray(v, dtype=np.float64) for k, v in samples.items()}

        # Guard: multi-condition dict requires a conditioning strategy
        if len(samples_dict) > 1 and self.conditioning is None:
            raise ValueError(
                "samples is a dict with multiple conditions but this flow has no "
                "conditioning strategy (conditioning=None).  Either pass a single "
                "array or supply a ConditioningStrategy at construction."
            )

        # ------------------------------------------------------------------
        # 2. Fit data transform (conditionally)
        # ------------------------------------------------------------------
        if refit_data_transform or not self.data_transform.is_fitted:
            self.data_transform.fit(samples_dict)

        # ------------------------------------------------------------------
        # 3. Assemble latents + context rows
        # ------------------------------------------------------------------
        z_list: list[torch.Tensor] = []
        ctx_list: list[torch.Tensor] = []
        bad_conditions: list[str] = []

        for cond_id, x_c in samples_dict.items():
            z_c = self.data_transform.forward(x_c, cond_id)
            # Coerce to float32 CPU tensor
            if isinstance(z_c, np.ndarray):
                z_c = torch.as_tensor(z_c.astype(np.float32))
            elif isinstance(z_c, torch.Tensor):
                z_c = z_c.float().cpu()
            z_list.append(z_c)

            if self.conditioning is not None:
                ctx_row = self.conditioning.encode(int(cond_id))  # float32 (context_dim,)
                ctx_t = torch.as_tensor(ctx_row)  # shape (context_dim,)
                ctx_expanded = ctx_t.unsqueeze(0).expand(z_c.shape[0], -1)  # (N_c, context_dim)
                ctx_list.append(ctx_expanded)

        z = torch.cat(z_list, dim=0)  # (M, dims)
        ctx = torch.cat(ctx_list, dim=0) if ctx_list else None  # (M, context_dim) or None

        # ------------------------------------------------------------------
        # 4. NaN / Inf guard
        # ------------------------------------------------------------------
        if not torch.isfinite(z).all():
            # Identify which conditions/dimensions are bad
            for cond_id, z_c in zip(samples_dict.keys(), z_list):
                bad_dims = (~torch.isfinite(z_c)).any(dim=0).nonzero(as_tuple=True)[0].tolist()
                if bad_dims:
                    bad_conditions.append(f"condition {cond_id}, dims {bad_dims}")
            raise ValueError(
                "Non-finite values (NaN or Inf) found in assembled latents after "
                f"data_transform.forward.  Offending: {'; '.join(bad_conditions)}.  "
                "Check data_transform or input samples."
            )

        # ------------------------------------------------------------------
        # 5. Train/val split using seeded generator
        # ------------------------------------------------------------------
        M = z.shape[0]
        if M < 2:
            raise ValueError(
                f"Need at least 2 samples after assembly to perform a train/val split, "
                f"got {M}."
            )

        # Coerce seed at the boundary: ints expected; int() accepts whole floats
        # (e.g. ``7.0``) and rejects non-numeric input with a clean TypeError.
        rng_seed = int(seed) if seed is not None else self.seed
        gen = torch.Generator()
        gen.manual_seed(rng_seed)

        if val_split == "temporal":
            # Hold out the NEWEST rows per condition (assembly preserves the
            # buffer's temporal order, oldest first). No pre-shuffle: the
            # DataLoader shuffles training batches per epoch anyway.
            tr_z, va_z, tr_c, va_c = [], [], [], []
            for k, z_c in enumerate(z_list):
                n_c = z_c.shape[0]
                n_val_c = min(max(1, int(n_c * validation_fraction)), n_c - 1)
                if n_c < 2:
                    n_val_c = 0  # single-row condition: keep it trainable
                tr_z.append(z_c[: n_c - n_val_c])
                va_z.append(z_c[n_c - n_val_c:])
                if ctx_list:
                    tr_c.append(ctx_list[k][: n_c - n_val_c])
                    va_c.append(ctx_list[k][n_c - n_val_c:])
            z_train = torch.cat(tr_z, dim=0)
            z_val = torch.cat(va_z, dim=0)
            ctx_train = torch.cat(tr_c, dim=0) if ctx_list else None
            ctx_val = torch.cat(va_c, dim=0) if ctx_list else None
            if z_val.shape[0] < 1 or z_train.shape[0] < 1:
                raise ValueError(
                    f"Temporal split produced train={z_train.shape[0]}, "
                    f"val={z_val.shape[0]} rows (M={M}).  Supply more samples "
                    "or reduce validation_fraction."
                )
        else:
            # Global shuffle
            perm = torch.randperm(M, generator=gen)
            z = z[perm]
            if ctx is not None:
                ctx = ctx[perm]

            n_val = max(1, int(M * validation_fraction))
            n_train = M - n_val
            if n_train < 1:
                raise ValueError(
                    f"Training set is empty after val split (M={M}, n_val={n_val}).  "
                    "Reduce validation_fraction or supply more samples."
                )

            z_train, z_val = z[:n_train], z[n_train:]
            ctx_train = ctx[:n_train] if ctx is not None else None
            ctx_val = ctx[n_train:] if ctx is not None else None

        # ------------------------------------------------------------------
        # 6. Move to device ONCE; build DataLoaders
        # ------------------------------------------------------------------
        z_train = z_train.to(self.device)
        z_val = z_val.to(self.device)
        if ctx_train is not None:
            ctx_train = ctx_train.to(self.device)
            ctx_val = ctx_val.to(self.device)

        if ctx_train is not None:
            train_ds = TensorDataset(z_train, ctx_train)
            val_ds = TensorDataset(z_val, ctx_val)
        else:
            train_ds = TensorDataset(z_train)
            val_ds = TensorDataset(z_val)

        train_gen = torch.Generator()
        train_gen.manual_seed(rng_seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=train_gen,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # ------------------------------------------------------------------
        # 7. Optimizer (and optional LR schedule)
        # ------------------------------------------------------------------
        opt_cls = _resolve_optimizer(optimizer)
        opt_kwargs = dict(optimizer_kwargs or {})
        # `lr` is the canonical top-level knob; it wins over any lr smuggled in
        # via optimizer_kwargs so the two never silently disagree.
        opt_kwargs.pop("lr", None)
        optimizer = opt_cls(self._flow.parameters(), lr=lr, **opt_kwargs)
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=n_epochs) if lr_annealing else None
        )

        # ------------------------------------------------------------------
        # 8. Training loop with early stopping and best-state tracking
        # ------------------------------------------------------------------
        history = FlowHistory()
        best_val = float("inf")
        best_state = copy.deepcopy(self._flow.state_dict())
        best_epoch = 0

        # Per-batch Gaussian jitter on the training latents (KDE smoothing;
        # validation stays clean). Scaled by per-dim training std so constant
        # dims get zero noise; correction=0 keeps n_train=1 finite.
        noise_scale = None
        noise_gen = None
        if train_noise > 0.0:
            noise_scale = train_noise * z_train.std(dim=0, keepdim=True, correction=0)
            noise_gen = torch.Generator(device=z_train.device)
            noise_gen.manual_seed(rng_seed + 1)

        for epoch in range(n_epochs):
            # --- train ---
            self._flow.train()
            train_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad()
                if ctx_train is not None:
                    z_b, ctx_b = batch
                else:
                    (z_b,) = batch
                    ctx_b = None
                if noise_scale is not None:
                    z_b = z_b + noise_scale * torch.randn(
                        z_b.shape,
                        generator=noise_gen,
                        device=z_b.device,
                        dtype=z_b.dtype,
                    )
                if ctx_b is not None:
                    loss = -self._flow(ctx_b).log_prob(z_b).mean()
                else:
                    loss = -self._flow().log_prob(z_b).mean()
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(self._flow.parameters(), clip_grad)
                optimizer.step()
                train_losses.append(loss.item())

            if scheduler is not None:
                scheduler.step()

            mean_train = float(np.mean(train_losses))

            # --- validate ---
            self._flow.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    if ctx_val is not None:
                        z_b, ctx_b = batch
                        val_loss = -self._flow(ctx_b).log_prob(z_b).mean()
                    else:
                        (z_b,) = batch
                        val_loss = -self._flow().log_prob(z_b).mean()
                    val_losses.append(val_loss.item())

            mean_val = float(np.mean(val_losses))

            history.training_loss.append(mean_train)
            history.validation_loss.append(mean_val)

            if verbose:
                print(f"epoch {epoch:4d}  train {mean_train:.6f}  val {mean_val:.6f}")

            # --- best-state bookkeeping ---
            if mean_val < best_val:
                best_val = mean_val
                best_state = copy.deepcopy(self._flow.state_dict())
                best_epoch = epoch

            # --- early stopping ---
            if patience is not None and (epoch - best_epoch) >= patience:
                break

        # ------------------------------------------------------------------
        # 9. Restore best state
        # ------------------------------------------------------------------
        self._flow.load_state_dict(best_state)
        self._flow.eval()

        return history

    # ------------------------------------------------------------------
    # Internal helper: build / cache distribution for a given context
    # ------------------------------------------------------------------

    def _alias_offsets(self, condition):
        """Latent-space periodic alias offsets, or ``None`` if there is no sum to do.

        Returns ``None`` (rather than a single zero row) whenever aliasing is
        disabled, the data transform does not expose alias offsets, or the
        transform has no periodic dims -- so those cases take a fast path that
        is bit-identical to the pre-aliasing implementation.

        Deliberately NOT cached: the trainer refits the data transform (and
        hence the whitening matrix these offsets derive from) as often as every
        round, and a stale cache would silently mis-score proposals.  The cost
        is a couple of tiny matrix products next to a network forward pass.
        """
        order = int(getattr(self, "periodic_aliases", 0) or 0)
        if order <= 0:
            return None
        getter = getattr(self.data_transform, "periodic_alias_offsets", None)
        if getter is None:
            return None
        # The FULL lattice is required, not just single-dimension displacements:
        # when several periods are short in latent units, simultaneous leakage in
        # two or three dims contributes materially (measured 43-55 nats on the
        # multimodal MBH leaves, where all three offset norms are < 20).  The
        # rows that cannot contribute are pruned per batch in _log_flow_base.
        offsets = np.asarray(getter(condition, order=order, shell="full"),
                             dtype=np.float64)
        return offsets if offsets.shape[0] > 1 else None

    def _log_flow_base(self, dist, z_t, base_scale, offsets):
        """Base-distribution log density at ``z_t``, summed over periodic aliases.

        With ``offsets`` the returned value is
        ``logsumexp_k log q_Z(z + delta_k)`` -- the exact wrapped density of the
        folded variable.  The sum is invariant to WHICH alias ``z_t`` is (the
        alias set of any representative is the same set), which is what makes
        :meth:`log_prob` and :meth:`sample_and_log_prob` agree identically.

        Kept as one helper so all three density entry points share the ops and
        stay mutually bit-identical.
        """

        def _eval(z):
            if base_scale is None:
                return dist.log_prob(z)
            return self._scaled_base_log_prob(dist, z, float(base_scale))

        if offsets is None:
            return _eval(z_t)

        off_t = torch.as_tensor(offsets, dtype=z_t.dtype, device=z_t.device)

        # Prune aliases that cannot contribute for THIS batch.  Outside the
        # spline core the base density decays with ||z||, so alias ``delta`` can
        # only rival the k = 0 term at ``z`` if it lands nearer the mode --
        # ``||z + delta|| < ||z||`` -- which requires ``||delta|| <= 2 ||z||``.
        # That argument does not hold INSIDE the core, where the density is an
        # arbitrary spline, so the core radius is added as slack: zuko's spline
        # domain is |z_i| <= 5 per dim, giving a core reach of 5 * sqrt(dims).
        # The batch maximum is used (not per-point) to keep one batched
        # evaluation; that is the conservative direction.
        # This keeps the full lattice where periods are short in latent units
        # while collapsing to a single evaluation when every alias is far
        # (measured: offset norms 22-6612 on the EMRI leaves and MBH leaves 0/2,
        # all with identically zero missing mass).  Validated bit-exact against
        # the unpruned lattice on every leaf of both production branches.
        cutoff = 2.0 * z_t.norm(dim=1).max() + 5.0 * float(np.sqrt(self.dims))
        keep = off_t.norm(dim=1) <= cutoff
        keep[0] = True  # the k = 0 image is always required
        off_t = off_t[keep]
        if off_t.shape[0] == 1:
            return _eval(z_t)

        n_alias = off_t.shape[0]
        stacked = (z_t.unsqueeze(0) + off_t.unsqueeze(1)).reshape(-1, z_t.shape[-1])
        return torch.logsumexp(_eval(stacked).reshape(n_alias, z_t.shape[0]), dim=0)

    def _get_dist(self, ctx):
        """Return the zuko conditional distribution for the given context tensor.

        Parameters
        ----------
        ctx : torch.Tensor or None
            Context tensor of shape ``(context_dim,)`` or ``None``.  A single
            1-D context vector is passed directly to ``self._flow(ctx)``.  Zuko
            broadcasts a 1-D context over the sample batch dimension, so this
            works for both ``log_prob(batch)`` and
            ``rsample_and_log_prob((n,))``.

        Returns
        -------
        zuko distribution
        """
        if ctx is None:
            return self._flow()
        # Pass a 1-D context vector: zuko broadcasts it over the batch.
        # Do NOT expand to (N, context_dim) — that makes zuko interpret the
        # leading N as a batch of *different* contexts, producing (N, n, dims)
        # shaped samples instead of (n, dims).
        return self._flow(ctx)

    # ------------------------------------------------------------------
    # Temperature-scaled base distribution (base_scale)
    # ------------------------------------------------------------------

    def _scaled_base_log_prob(self, dist, z: torch.Tensor, s: float) -> torch.Tensor:
        """Evaluate ``dist``'s log-density at flow-space ``z`` under a rescaled base.

        Mirrors zuko's ``NormalizingFlow.log_prob`` exactly (same transform
        forward pass and log-det bookkeeping — see ``zuko.distributions.
        NormalizingFlow.log_prob``) but replaces the base term with the
        density of the base distribution scaled by ``s``.  This is exact for
        ANY base density (Gaussian or otherwise): for the linear
        reparameterisation :math:`V = s U`, the change-of-variables formula
        gives ``log q_s(v) = log p(v / s) - dims * log(s)``.  The flow's own
        transform log-det is untouched — it does not depend on the base at
        all, only on the network + ``data_transform``.

        Parameters
        ----------
        dist : zuko.distributions.NormalizingFlow
            The (unscaled) flow distribution for the current context, as
            returned by :meth:`_get_dist`.
        z : torch.Tensor, shape (N, dims)
            Flow-space points (post ``data_transform``, pre-network base).
        s : float
            Base standard-deviation scale factor (``> 0``).

        Returns
        -------
        torch.Tensor, shape (N,)
        """
        u, ladj = dist.transform.call_and_ladj(z)
        ladj = _sum_rightmost(ladj, dist.reinterpreted)
        base_log_prob = dist.base.log_prob(u / s) - self.dims * math.log(s)
        return base_log_prob + ladj

    def _scaled_base_rsample_and_log_prob(self, dist, n: int, s: float) -> tuple:
        """Draw ``n`` samples from ``dist`` with an ``s``-scaled base, and their log-density.

        Mirrors zuko's ``NormalizingFlow.rsample_and_log_prob`` exactly (same
        inverse-transform pass and log-det bookkeeping) but draws the base
        latent from the ``s``-scaled base and reports the matching density —
        see :meth:`_scaled_base_log_prob` for the exact identity used.

        Parameters
        ----------
        dist : zuko.distributions.NormalizingFlow
            The (unscaled) flow distribution for the current context, as
            returned by :meth:`_get_dist`.
        n : int
            Number of samples to draw.
        s : float
            Base standard-deviation scale factor (``> 0``).

        Returns
        -------
        z : torch.Tensor, shape (n, dims)
            Flow-space samples (post ``data_transform`` space, pre-inverse
            transform of the coords-space map).
        log_prob : torch.Tensor, shape (n,)
        """
        if dist.base.has_rsample:
            u0 = dist.base.rsample((n,))
        else:
            u0 = dist.base.sample((n,))
        u_s = s * u0
        z, ladj = dist.transform.inv.call_and_ladj(u_s)
        ladj = _sum_rightmost(ladj, dist.reinterpreted)
        base_log_prob = dist.base.log_prob(u0) - self.dims * math.log(s)
        return z, base_log_prob - ladj


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sum_rightmost(value: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum out the rightmost ``dim`` dimensions of ``value``.

    Local copy of ``torch.distributions.utils._sum_rightmost`` (zuko imports
    this utility from torch.distributions) so :class:`ZukoFlow`'s base-scale
    machinery does not depend on zuko's private API surface.  ``dim == 0`` is
    a no-op — this is the common case, where the flow's transform codomain
    event-dim already equals the base distribution's event-dim (e.g. NSF/MAF
    with a ``DiagNormal`` base): no reinterpreted dims remain to sum.

    Parameters
    ----------
    value : torch.Tensor
    dim : int
        Number of rightmost dimensions to sum out.

    Returns
    -------
    torch.Tensor
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def _resolve_flow_class(flow_class):
    """Resolve a flow class from a string name or callable.

    Parameters
    ----------
    flow_class : str or callable
        If a string, resolved via ``getattr(zuko.flows, flow_class)``.

    Returns
    -------
    callable
        The flow constructor.

    Raises
    ------
    AttributeError
        If the string name is not found in ``zuko.flows``.
    """
    if callable(flow_class) and not isinstance(flow_class, str):
        return flow_class
    if isinstance(flow_class, str):
        try:
            return getattr(zuko.flows, flow_class)
        except AttributeError:
            valid = [n for n in dir(zuko.flows) if not n.startswith("_")]
            raise AttributeError(
                f"'{flow_class}' is not a valid zuko.flows class.  "
                f"Some valid names: {valid[:10]}"
            ) from None
    raise TypeError(f"flow_class must be a str or callable, got {type(flow_class)}")


def _resolve_optimizer(optimizer):
    """Resolve a torch.optim optimizer class from a name or a class.

    Mirrors :func:`_resolve_flow_class`: a string is looked up on
    ``torch.optim`` (with a small lowercase-alias table for the common
    optimizers), and a ``torch.optim.Optimizer`` subclass is returned as-is.

    Parameters
    ----------
    optimizer : str or type
        Optimizer name (e.g. ``"adam"``, ``"adamw"``, ``"sgd"``) or a
        ``torch.optim.Optimizer`` subclass.

    Returns
    -------
    type
        The optimizer class.

    Raises
    ------
    ValueError
        If a string name does not resolve to a ``torch.optim`` attribute.
    TypeError
        If ``optimizer`` is neither a string nor an Optimizer subclass.
    """
    if isinstance(optimizer, str):
        # torch.optim classes are CamelCase; accept common lowercase spellings.
        aliases = {
            "adam": "Adam", "adamw": "AdamW", "sgd": "SGD",
            "rmsprop": "RMSprop", "adamax": "Adamax", "adagrad": "Adagrad",
            "nadam": "NAdam", "radam": "RAdam",
        }
        name = aliases.get(optimizer.lower(), optimizer)
        try:
            return getattr(torch.optim, name)
        except AttributeError:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}.  Pass a torch.optim class name "
                f"(e.g. 'Adam', 'AdamW', 'SGD') or one of {sorted(aliases)}."
            ) from None
    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    raise TypeError(
        f"optimizer must be a str or a torch.optim.Optimizer subclass, "
        f"got {type(optimizer)!r}."
    )


def _make_serialisable(cfg: dict) -> dict:
    """Recursively convert a config dict to a JSON-serialisable form.

    Tuples become lists (JSON has no tuple type).  Non-serialisable objects
    (DataTransform, ConditioningStrategy) are dropped; they are stored separately
    as pickle blobs.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    dict
    """
    out = {}
    _skip_types = ("DataTransform", "IdentityTransform", "WhiteningTransform",
                   "ConditioningStrategy", "OneHotLeafConditioning")
    for k, v in cfg.items():
        # Skip objects that should be pickled separately
        type_name = type(v).__name__
        if type_name in _skip_types:
            continue
        if v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar_or_skip(i) for i in v]
        else:
            # Skip non-serialisable objects (e.g. DataTransform instances)
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                pass  # silently skip; restored via pickle
    return out


def _scalar_or_skip(v):
    """Convert a scalar to JSON-safe type; return it directly if already safe."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _strip_pickled_keys(cfg: dict) -> dict:
    """Remove keys from a JSON-derived config that are stored as pickle blobs.

    ``data_transform`` and ``conditioning`` are serialised separately and
    injected by :meth:`BaseTorchFlow.load`; removing them here prevents
    double-assignment when the config dict is unpacked into the constructor.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    dict
    """
    cfg.pop("data_transform", None)
    cfg.pop("conditioning", None)
    return cfg
