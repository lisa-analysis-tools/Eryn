# src/eryn/flows/executors.py
"""Trainer-executor seam for online flow training during MCMC.

This module defines the interface a normalizing-flow proposal uses to train a
flow *while sampling*, without ever blocking the proposal on the training step.
It is intentionally torch-free at import time: flows are manipulated only
through the :class:`eryn.flows.base.Flow` interface, and the concrete flow class
is carried by reference (a module-level type) inside :class:`FlowSpec`.  The
heavy backend (torch/zuko) is only imported when a worker actually builds and
trains the flow.

Three pieces live here:

- :class:`FlowSpec` — a picklable ``(class, config, weights)`` snapshot of a
  flow (its data transform may be unfitted), used to build an identically
  configured clone in a worker (or inline).
- :class:`TrainerExecutor` — the abstract, non-blocking training interface.
- :class:`InlineExecutor` — a synchronous reference implementation that trains a
  *cloned* flow on every Nth accepted batch and surfaces failures lazily, behind
  exactly the contract a future multiprocessing executor will implement.

Design contract
----------------
- A proposal NEVER blocks on training: :meth:`TrainerExecutor.submit` and
  :meth:`TrainerExecutor.latest_weights` are non-blocking.
- Failures surface loudly: a dead/failed trainer raises :class:`TrainerError`
  from :meth:`TrainerExecutor.latest_weights` (and, for the inline executor,
  from subsequent :meth:`submit` calls).  Callers do not swallow it.
- The data transform is fitted lazily, then frozen: the executor fits the
  transform on its FIRST training round (once ``min_train_samples`` are
  buffered), then keeps it frozen for the rest of its life (workers fit with
  ``refit_data_transform=False`` thereafter), so the coords-latent map is fixed
  and the handoff stays exact.  An optional ``refit_transform_every=K`` re-fits
  the transform every K rounds as a drift valve.  Because the executor (or its
  worker) is the side that fits the transform, every weight handoff is a
  **self-contained snapshot** carrying the fitted transform alongside the net
  (see :meth:`eryn.flows.torch.flows.BaseTorchFlow.get_snapshot`) — the parent
  installs both atomically and never disagrees with the trainer on the map.
- Everything that may cross a future process boundary is picklable;
  :meth:`FlowSpec.from_flow` eagerly round-trips ``pickle.dumps(spec)`` so
  unpicklable user objects fail fast in the parent with a clear message.  An
  *unfitted* transform is picklable (it holds only its config + ``None`` maps),
  so snapshotting before any samples have been seen is supported.
"""
from __future__ import annotations

import copy
import multiprocessing
import os
import pickle
import queue
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Torch-free and plotting-free at import time (h5py/matplotlib/corner are
# imported lazily inside its functions), so this keeps the module's
# import-hygiene contract intact.
from . import diagnostics as _diagnostics

# NOTE: ``multiprocessing`` is imported at module level — that is torch-free and
# fine.  torch is NEVER imported here: the worker (:func:`_trainer_worker`)
# imports it lazily inside the child process, so importing this module stays
# torch-free (verified by tests/test_flow_imports.py).

__all__ = [
    "TrainerError",
    "FlowSpec",
    "TrainerExecutor",
    "InlineExecutor",
    "WorkerConfig",
    "ProcessExecutor",
]


class TrainerError(RuntimeError):
    """Raised when the background trainer has failed.

    The exception carries the child-side error message (and, for a future
    process executor, the child traceback) so the failure is visible in the
    parent.  Proposals do not catch this: a long run that silently keeps
    sampling from a frozen or broken flow is worse than a loud crash.
    """


@dataclass
class FlowSpec:
    """Picklable snapshot of a flow: ``(class, config, initial_weights)``.

    A :class:`FlowSpec` carries everything needed to rebuild an identically
    configured flow elsewhere (in a worker process, or inline) *without*
    pickling the live flow object itself.  The flow class is referenced by
    type (a module-level class pickles by reference); the configuration and the
    CPU weights are deep-copied snapshots.

    The data transform inside ``config`` may be **unfitted**: the executor that
    builds from this spec fits it lazily on the first training round and then
    freezes it (training thereafter with ``refit_data_transform=False``).  The
    fitted transform travels back to the parent inside each weight snapshot, so
    the coords-latent map agrees on both sides regardless of which side fitted
    it.

    Parameters
    ----------
    flow_class : type
        Concrete :class:`~eryn.flows.base.Flow` subclass.  Pickled by reference,
        so it must be importable by its module path on both sides of any process
        boundary.
    config : dict
        Constructor keyword arguments (``flow.config_dict()``) with the
        ``device`` entry overridden for the worker.  Contains the
        ``data_transform`` / ``conditioning`` objects themselves; building from
        the spec reconstructs an equivalent flow.
    initial_weights : dict
        CPU-resident weights from ``flow.get_weights()``.

    Notes
    -----
    Construct via :meth:`from_flow` rather than directly: it snapshots the
    config/weights and verifies picklability eagerly.  The data transform may
    be **unfitted** — the executor fits it lazily on the first training round
    and ships the fitted transform back inside each weights snapshot.
    """

    flow_class: type
    config: dict
    initial_weights: dict

    @classmethod
    def from_flow(cls, flow, worker_device: str = "cpu") -> "FlowSpec":
        """Snapshot ``flow`` into a picklable :class:`FlowSpec`.

        Parameters
        ----------
        flow : eryn.flows.base.Flow
            Flow to snapshot.  Its data transform may be **unfitted** — the
            executor fits it lazily on the first training round.  The flow class
            must be importable by reference and its config/weights must be
            picklable (an unfitted transform is picklable: it holds only its
            config and ``None`` maps).
        worker_device : str, optional
            Device string written into ``config["device"]`` for the rebuilt
            flow.  Default is ``"cpu"`` — training a clone on CPU is the safe
            default and keeps this snapshot device-agnostic.

        Returns
        -------
        FlowSpec
            A snapshot that has been verified to pickle cleanly.

        Raises
        ------
        TypeError
            If the assembled spec is not picklable (e.g. the data transform or
            conditioning closes over a lambda or holds a device tensor).  The
            message names the offending concept.
        """
        # The data transform need not be fitted here: the executor fits it on
        # the first training round and ships it back inside each snapshot.

        # Deep-copy config so later mutation of the live flow cannot leak in,
        # and override device for the worker / clone.
        config = copy.deepcopy(flow.config_dict())
        config["device"] = worker_device

        spec = cls(
            flow_class=type(flow),
            config=config,
            initial_weights=flow.get_weights(),
        )

        # Eagerly round-trip so unpicklable user objects fail fast in the parent
        # with a clear, actionable message rather than deep inside a worker spawn.
        try:
            pickle.dumps(spec)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            raise TypeError(
                "FlowSpec must be picklable to cross a process boundary; "
                "check data_transform/conditioning for closures (e.g. lambdas) "
                "or device tensors.  Original error: " + repr(exc)
            ) from exc

        return spec

    def build(self):
        """Reconstruct a live flow from this spec.

        Returns
        -------
        eryn.flows.base.Flow
            A new ``flow_class`` instance built from ``config`` with
            ``initial_weights`` loaded.  Equivalent to the snapshotted flow
            (same config + weights → same densities).
        """
        flow = self.flow_class(**self.config)
        flow.set_weights(self.initial_weights)
        return flow


class TrainerExecutor(ABC):
    """Non-blocking interface for online flow training.

    Implementations train a flow asynchronously (inline-synchronously for
    :class:`InlineExecutor`, in a subprocess for the future process executor)
    and hand new weights back to the caller on request.  Every method on this
    interface is non-blocking with respect to the actual training step: a
    proposal calling :meth:`submit` / :meth:`latest_weights` never waits for a
    fit to finish.

    The executor trains a **clone** of the caller's flow (built from
    :class:`FlowSpec`), not the caller's object.  Weights flow back to the
    caller only through :meth:`latest_weights`; the caller then applies them to
    its own flow via ``flow.set_weights``.

    What :meth:`latest_weights` returns is a **self-contained snapshot** — a
    ``{"net", "data_transform"}`` dict pairing the trained net with the
    transform it was trained against (see
    :meth:`eryn.flows.torch.flows.BaseTorchFlow.get_snapshot`).  Passing it to
    ``flow.set_weights`` installs both atomically, so the caller's flow is always
    a matched, usable pair even when its own transform started unfitted.

    Failures are surfaced loudly: once training has failed,
    :meth:`latest_weights` raises :class:`TrainerError`.
    """

    @abstractmethod
    def submit(self, samples_by_condition: dict) -> bool:
        """Offer a batch of training samples to the trainer (non-blocking).

        Parameters
        ----------
        samples_by_condition : dict[int, np.ndarray]
            Mapping from condition id to an array of shape ``(N, dims)`` of
            coords-space samples to add to the trainer's buffer.

        Returns
        -------
        bool
            ``True`` if the batch was accepted, ``False`` if it was dropped or
            coalesced (e.g. the trainer is busy, or the executor is shut down).

        Empty harvests
        --------------
        Zero-length arrays carry no training data and must be dropped before
        buffering; a submit with no rows at all (an empty dict, or only empty
        arrays) returns ``False`` and is **not** counted as an accepted submit.
        An empty harvest — e.g. a sampling step with no active leaves — is a
        normal condition, not an error, and must never put the executor into a
        failed state.
        """

    @abstractmethod
    def latest_weights(self):
        """Return the newest ``(version, snapshot)`` seen so far (non-blocking).

        Returns
        -------
        tuple[int, dict] or None
            The newest ``(version, snapshot)`` observed, or ``None`` if no
            training has completed yet.  The ``snapshot`` is a
            ``{"net", "data_transform"}`` dict (see the class docstring): pass it
            straight to ``flow.set_weights``, which installs the matched net and
            transform atomically.  Calling this updates :attr:`version`.

        Ownership contract
        ------------------
        The returned ``snapshot`` dict is the executor's **private** copy.
        Callers may *load* it (``flow.set_weights(snapshot)`` copies the values
        into the module's parameters / installs the transform) but must **not**
        mutate it in place.  An executor must guarantee that successive snapshots
        are not aliased to live training state: mutating a previously returned
        dict, or training further, must never change a dict a caller is still
        holding.  A process-backed executor satisfies this naturally (each
        returned dict is a fresh deserialized copy); an inline executor must
        deep-copy the snapshot when stashing it so it matches that semantics.

        Raises
        ------
        TrainerError
            If the trainer has failed.
        """

    @property
    @abstractmethod
    def version(self) -> int:
        """Newest version OBSERVED via :meth:`latest_weights` (``0`` = none yet).

        This is *poll-advanced*: it tracks the highest version a caller has
        actually pulled through :meth:`latest_weights`, NOT the trainer's
        internal "produced" counter.  A trainer may have completed a newer fit
        that no one has polled yet; until :meth:`latest_weights` returns it,
        :attr:`version` does not advance.  This is THE contract for every
        executor, including process-based ones — callers use it to decide
        whether a hot-reload is needed.

        Per-executor-instance counter
        -----------------------------
        The version counter is local to THIS executor object and starts at
        ``0``.  A *recreated* executor (e.g. a fresh
        :class:`ProcessExecutor` spun up after a worker died) restarts its
        counter at ``1`` — it has no memory of a predecessor's versions.  Any
        consumer that remembers the highest version it has loaded and compares
        future versions against it MUST reset that memory when it swaps in a new
        executor, or it will silently ignore the replacement's hot-reloads
        forever.  :class:`~eryn.moves.flow.ConditionalFlowMove` does this automatically: its
        ``executor`` setter resets ``loaded_version`` on a real executor swap.
        """

    @property
    def latest_val_nll(self):
        """Most recent latent-space validation NLL, or ``None`` before any fit.

        After each trained version a concrete executor captures the **best**
        validation loss (``min(history.validation_loss)``) — the flow's negative
        log-likelihood in *latent* space for the weights that actually ship,
        since ``fit`` restores the best-val state before returning — so a user
        can watch training progress without touching the critical path.  Returns the value from the most
        recent completed fit, or ``None`` if no fit has finished yet.  This is a
        concrete default returning ``None`` so a minimal executor that does not
        track the metric need not override it.

        Latent-space NLL is comparable across rounds while the transform is
        frozen (the default).  Under ``refit_transform_every`` periodic
        re-fitting the latent space itself changes, so the metric shifts by the
        changing transform log-det at each re-fit and is only comparable
        *between* re-fits.  (A coords-space NLL would be invariant but is not
        computed here.)
        """
        return None

    @property
    def latest_history(self):
        """Full :class:`~eryn.flows.base.FlowHistory` of the most recent fit.

        Per-epoch training/validation loss curves of the newest completed
        training round, or ``None`` before any fit.  For asynchronous
        executors this is refreshed by :meth:`latest_weights` (poll first).
        Concrete default returning ``None`` so a minimal executor that does
        not track it need not override it.
        """
        return None

    @abstractmethod
    def shutdown(self, timeout: float = 10.0) -> None:
        """Shut the executor down (idempotent).

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for an in-flight trainer to stop.  Default
            is ``10.0``.  Ignored by purely synchronous implementations.
        """

    def __enter__(self) -> "TrainerExecutor":
        """Enter the runtime context, returning ``self``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context, calling :meth:`shutdown`."""
        self.shutdown()


def _validate_monitoring_args(save_path, diagnostics_dir, plot_corner) -> None:
    """Fail-fast validation for the checkpoint/diagnostics knobs.

    Shared by both executors so misconfiguration (missing optional packages,
    ``plot_corner`` without a ``diagnostics_dir``) surfaces in the parent's
    ``__init__`` with an actionable message — not deep inside a worker spawn.
    Also creates the output directories so the first save cannot fail on a
    missing path.
    """
    if plot_corner and diagnostics_dir is None:
        raise ValueError(
            "plot_corner=True requires diagnostics_dir to be set (the corner "
            "plots are written there)."
        )
    if save_path is not None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:  # pragma: no cover - h5py is a core eryn dep
            raise ImportError(
                "save_path requires h5py for the HDF5 checkpoint."
            ) from exc
        parent = os.path.dirname(os.path.abspath(str(save_path)))
        os.makedirs(parent, exist_ok=True)
    if diagnostics_dir is not None:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "diagnostics_dir requires matplotlib for the loss/trend plots."
            ) from exc
        os.makedirs(diagnostics_dir, exist_ok=True)
    if plot_corner:
        try:
            import corner  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "plot_corner=True requires the optional 'corner' package "
                "(pip install corner)."
            ) from exc


class InlineExecutor(TrainerExecutor):
    """Synchronous reference :class:`TrainerExecutor` that trains a cloned flow.

    On every ``train_every``-th *accepted* :meth:`submit` (once at least
    ``min_train_samples`` are buffered) the executor fits its own cloned flow on
    the concatenated buffer and stashes the new ``(version, weights)``.  Despite
    training synchronously inside :meth:`submit`, it honours the same
    non-blocking *contract* a multiprocessing executor will: the caller's flow
    is never touched, and weights only flow back through :meth:`latest_weights`.

    The clone is built via ``FlowSpec.from_flow(flow).build()`` — the executor
    does NOT train the caller's flow object.  The data transform is fitted
    **lazily**: the first training round fits it (``refit_data_transform=True``)
    and then it is frozen for the rest of the executor's life
    (``refit_data_transform=False``), unless ``refit_transform_every=K`` is set,
    in which case it is re-fitted every K rounds as a drift valve.  Each
    :meth:`latest_weights` hands back a self-contained snapshot
    (``{"net", "data_transform"}``) so the caller installs the trained net and
    the transform it was trained against atomically — matching the cross-process
    handoff semantics where parent and child must agree on the coords-latent
    map.

    Failure semantics
    -----------------
    If ``flow.fit`` raises during a :meth:`submit`, the executor does **not**
    propagate it from that ``submit`` call directly.  Instead it records the
    failure and raises :class:`TrainerError` from the **next**
    :meth:`latest_weights` call and from every subsequent :meth:`submit`.  This
    mirrors the deferred error surfacing of a process-backed trainer, where the
    failure is only observed when the parent next polls — and it keeps the
    "submit never blocks on / fails because of training" contract uniform across
    executors.

    Parameters
    ----------
    flow : eryn.flows.base.Flow
        Template flow.  Its data transform may be unfitted — the executor fits
        it on the first training round.  A clone is built from a
        :class:`FlowSpec` snapshot; the caller's object is never mutated.
    fit_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``flow.fit`` (e.g. ``n_epochs``,
        ``lr``).  ``refit_data_transform`` and ``verbose`` are controlled by the
        executor; supplying either raises :class:`ValueError` from ``__init__``.
        Default is ``None`` (empty).
    train_every : int, optional
        Train on every ``train_every``-th accepted submit.  Default is ``1``
        (train on every submit).
    refit_transform_every : int or None, optional
        Drift valve for the (otherwise frozen) data transform.  ``None`` (the
        default) fits the transform once on the first training round and freezes
        it forever after.  When set to an integer ``K``, the transform is
        ALSO re-fitted on every ``K``-th completed training round (rounds are
        counted from 1, so ``K=2`` re-fits on rounds 2, 4, ...).  Re-fitting
        changes the latent space, so it shifts :attr:`latest_val_nll`.
    min_train_samples : int, optional
        Minimum total buffered samples (summed over conditions) required before
        a fit runs.  Default is ``1``.  Training is always additionally guarded
        on a non-empty buffer, so this is self-documenting rather than a
        behaviour change: a fit never runs on zero rows.
    max_buffer_samples : int, optional
        Per-condition ring-buffer cap.  When a condition's buffered sample count
        exceeds this, the oldest arrays are dropped from the left — but the most
        recent array is always retained whole, so a single submit larger than
        the cap is kept in full (the effective cap is
        ``max(max_buffer_samples, one harvest)``).  Default is ``20_000``.
    seed : int or None, optional
        Reserved for parity with asynchronous executors; unused by the inline
        implementation (the clone carries the template flow's own seed).
        Default is ``None``.
    save_path : str or None, optional
        HDF5 checkpoint of the latest fit.  When set, every ``save_every``-th
        trained version is written atomically to this path via
        :func:`eryn.flows.diagnostics.save_checkpoint` — the file always holds
        the newest fit and round-trips a fully usable flow through
        ``FlowClass.load(save_path)`` (fitted transform included).  A save
        failure is a real training failure: it is recorded and surfaced as
        :class:`TrainerError` on the next poll/submit (deferred, like a fit
        failure).  Default is ``None`` (no checkpointing).
    save_every : int, optional
        Checkpoint cadence in trained versions.  Default is ``1``.
    diagnostics_dir : str or None, optional
        Directory for per-round diagnostic PNGs: per-round loss curves
        (``flow_loss_v####.png``) and a running validation-NLL trend
        (``flow_val_nll_trend.png``).  Plot failures are best-effort — they
        emit a :class:`RuntimeWarning` and never interrupt training.  Default
        is ``None`` (no plots).
    diagnostics_every : int, optional
        Plot cadence in trained versions.  Default is ``1``.
    plot_corner : bool, optional
        When ``True`` (requires ``diagnostics_dir`` and the optional
        ``corner`` package), each diagnostics round also writes one corner
        plot per condition overlaying that round's training buffer with an
        equal number of draws from the freshly trained flow
        (``flow_corner_v####_cond#.png``).  Default is ``False``.
    corner_max_samples : int, optional
        Cap on points per data set in each corner plot.  Default is ``5000``.
    """

    def __init__(
        self,
        flow,
        fit_kwargs: dict | None = None,
        *,
        train_every: int = 1,
        min_train_samples: int = 1,
        max_buffer_samples: int = 20_000,
        refit_transform_every: int | None = None,
        seed=None,
        save_path: str | None = None,
        save_every: int = 1,
        diagnostics_dir: str | None = None,
        diagnostics_every: int = 1,
        plot_corner: bool = False,
        corner_max_samples: int = 5000,
    ):
        self._flow = FlowSpec.from_flow(flow).build()
        self._fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        # refit_data_transform and verbose are controlled by the executor (the
        # transform is frozen; verbosity is forced off).  Reject them in
        # fit_kwargs so a caller's collision fails fast rather than being
        # silently overridden at fit time.
        _reserved = {"refit_data_transform", "verbose"} & self._fit_kwargs.keys()
        if _reserved:
            raise ValueError(
                f"fit_kwargs may not contain {sorted(_reserved)}: these are "
                "controlled by the executor (the data transform is frozen and "
                "verbosity is forced off).  Remove them from fit_kwargs."
            )
        self._train_every = int(train_every)
        self._min_train_samples = int(min_train_samples)
        self._max_buffer_samples = int(max_buffer_samples)
        self._refit_transform_every = (
            int(refit_transform_every) if refit_transform_every is not None else None
        )
        self._seed = seed

        _validate_monitoring_args(save_path, diagnostics_dir, plot_corner)
        self._save_path = str(save_path) if save_path is not None else None
        self._save_every = int(save_every)
        self._diagnostics_dir = (
            str(diagnostics_dir) if diagnostics_dir is not None else None
        )
        self._diagnostics_every = int(diagnostics_every)
        self._plot_corner = bool(plot_corner)
        self._corner_max_samples = int(corner_max_samples)

        # Per-condition ring buffers: condition id -> deque of (N_i, dims) arrays.
        self._buffers: dict[int, deque] = {}
        self._accepted_submits = 0
        self._trained_version = 0
        # Lazy-fit bookkeeping: the transform is fitted on the first training
        # round and (unless refit_transform_every is set) frozen thereafter.
        self._transform_fitted = False
        self._round_count = 0
        self._latest: tuple[int, dict] | None = None
        self._latest_val_nll: float | None = None
        self._latest_history = None
        # Per-round (version, val_nll) record feeding the trend plot.
        self._round_versions: list[int] = []
        self._round_val_nlls: list = []
        self._seen_version = 0
        self._error: BaseException | None = None
        self._shutdown = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self, condition: int) -> None:
        """Drop oldest arrays for ``condition`` until its buffer fits the cap.

        Ring-buffer policy: keep in sync with the worker's buffering in
        :func:`_trainer_worker` (the ``_buffer_item`` inner trim) — both drop
        the oldest arrays from the left but always retain the most recent array
        whole, so a single oversized harvest is kept in full.
        """
        buf = self._buffers[condition]
        total = sum(len(a) for a in buf)
        while total > self._max_buffer_samples and len(buf) > 1:
            removed = buf.popleft()
            total -= len(removed)

    def _buffered_total(self) -> int:
        """Total buffered samples summed over all conditions."""
        return sum(len(a) for buf in self._buffers.values() for a in buf)

    def _assemble(self) -> dict:
        """Concatenate per-condition buffers into ``{condition: (N, dims)}``."""
        return {
            cond: np.concatenate(list(buf), axis=0)
            for cond, buf in self._buffers.items()
            if len(buf) > 0
        }

    # ------------------------------------------------------------------
    # TrainerExecutor interface
    # ------------------------------------------------------------------

    def submit(self, samples_by_condition: dict) -> bool:
        """Append samples and, every ``train_every``-th accepted call, fit.

        Zero-length arrays are dropped before buffering; a submit that carries
        no rows at all (an empty dict, or only empty arrays) returns ``False``
        WITHOUT counting as an accepted submit and without bumping any version.
        Empty harvests — e.g. a sampling step with no active leaves — are a
        normal condition, not an error, and must not poison the executor.

        Returns ``False`` (drops the batch) after :meth:`shutdown`.  Otherwise
        returns ``True`` for any submit that buffered at least one row — the
        inline executor never coalesces or drops a non-empty live batch.

        Raises
        ------
        TrainerError
            If a previous fit failed (deferred surfacing; see class docstring).
        """
        if self._error is not None:
            raise TrainerError(
                "InlineExecutor training failed on a previous submit: "
                f"{self._error!r}"
            ) from self._error

        if self._shutdown:
            return False

        buffered_any = False
        for cond, arr in samples_by_condition.items():
            cond = int(cond)
            arr = np.asarray(arr)
            arr = arr.reshape(-1, self._flow.dims)
            if len(arr) == 0:
                # Drop empty conditions: a harvest with no rows is a normal
                # sampling condition, not data to train on.
                continue
            if cond not in self._buffers:
                self._buffers[cond] = deque()
            self._buffers[cond].append(arr)
            self._trim(cond)
            buffered_any = True

        # An empty harvest is not an accepted submit: don't count it, don't
        # train on it, and return False so callers can tell it was a no-op.
        if not buffered_any:
            return False

        self._accepted_submits += 1

        # Train on every train_every-th accepted submit once enough is buffered.
        # The _buffered_total() > 0 guard is independent of min_train_samples so
        # we never hand fit() an empty buffer (an empty fit is an error, not a
        # training step).
        if (
            self._accepted_submits % self._train_every == 0
            and self._buffered_total() > 0
            and self._buffered_total() >= self._min_train_samples
        ):
            # Lazy fit-once-then-freeze: fit the transform on the first round,
            # then keep it frozen — unless refit_transform_every=K asks for a
            # periodic re-fit (rounds counted from 1, so K=2 → rounds 2, 4, ...).
            next_round = self._round_count + 1
            do_refit = (not self._transform_fitted) or (
                self._refit_transform_every is not None
                and next_round % self._refit_transform_every == 0
            )
            train_data = self._assemble()
            try:
                history = self._flow.fit(
                    train_data,
                    refit_data_transform=do_refit,
                    verbose=False,
                    **self._fit_kwargs,
                )
            except BaseException as exc:  # noqa: BLE001 — record and defer.
                # Record and surface on the next poll/submit, mirroring a
                # process executor whose failure is only seen when next polled.
                self._error = exc
            else:
                self._transform_fitted = True
                self._round_count = next_round
                self._trained_version += 1
                # Capture the latent-space validation NLL for monitoring (None
                # if the fit produced no validation history).
                val_loss = getattr(history, "validation_loss", None)
                # min, not [-1]: fit restores the best-val state, so the
                # shipped weights score min(val_loss); the last epoch is
                # best + patience-worth of overfit tail.
                val_nll = (
                    float(min(val_loss))
                    if (val_loss is not None and len(val_loss))
                    else None
                )
                if val_nll is not None:
                    self._latest_val_nll = val_nll
                self._latest_history = history
                self._round_versions.append(self._trained_version)
                self._round_val_nlls.append(val_nll)
                # Stash a self-contained snapshot ({"net", "data_transform"}).
                # get_snapshot() already deep-copies the transform; only the net
                # dict can reference the clone's live torch parameters, so copy
                # just that — matching a process executor, whose returned dict is
                # always a fresh deserialized copy.
                snapshot = self._flow.get_snapshot()
                snapshot["net"] = copy.deepcopy(snapshot["net"])
                self._latest = (self._trained_version, snapshot)

                # Checkpoint the latest fit.  A save failure is a real failure
                # (silently losing persistence is worse than a crash) but keeps
                # the deferred-surfacing contract: recorded here, raised from
                # the next poll/submit — matching how a worker-side save
                # failure reaches the parent of a ProcessExecutor.
                if (
                    self._save_path is not None
                    and self._trained_version % self._save_every == 0
                ):
                    try:
                        _diagnostics.save_checkpoint(
                            self._flow,
                            self._save_path,
                            self._trained_version,
                            val_nll,
                        )
                    except BaseException as exc:  # noqa: BLE001
                        self._error = exc

                # Diagnostics plots are best-effort: warn, never interrupt
                # training.
                if (
                    self._diagnostics_dir is not None
                    and self._trained_version % self._diagnostics_every == 0
                ):
                    try:
                        _diagnostics.save_loss_plot(
                            history,
                            self._diagnostics_dir,
                            self._trained_version,
                            val_nll,
                        )
                        _diagnostics.save_val_nll_trend(
                            self._round_versions,
                            self._round_val_nlls,
                            self._diagnostics_dir,
                        )
                        if self._plot_corner:
                            _diagnostics.save_corner_plot(
                                train_data,
                                self._flow,
                                self._diagnostics_dir,
                                self._trained_version,
                                self._corner_max_samples,
                            )
                    except Exception:
                        warnings.warn(
                            "flow diagnostics plotting failed:\n"
                            + traceback.format_exc(),
                            RuntimeWarning,
                        )

        return True

    def latest_weights(self):
        """Return the newest stashed ``(version, snapshot)`` (or ``None``).

        The ``snapshot`` is a ``{"net", "data_transform"}`` dict: pass it to
        ``flow.set_weights`` to install the matched net + transform atomically.
        Updates :attr:`version` to the returned version.

        Raises
        ------
        TrainerError
            If a fit has failed.
        """
        if self._error is not None:
            raise TrainerError(
                f"InlineExecutor training failed: {self._error!r}"
            ) from self._error

        if self._latest is None:
            return None
        version, snapshot = self._latest
        self._seen_version = version
        return version, snapshot

    @property
    def version(self) -> int:
        """Latest version observed via :meth:`latest_weights` (``0`` = none yet)."""
        return self._seen_version

    @property
    def latest_val_nll(self):
        """Latent-space validation NLL of the most recent fit (``None`` if none yet)."""
        return self._latest_val_nll

    @property
    def latest_history(self):
        """:class:`~eryn.flows.base.FlowHistory` of the most recent fit (or ``None``)."""
        return self._latest_history

    def shutdown(self, timeout: float = 10.0) -> None:
        """Mark the executor shut down (idempotent).

        After shutdown, :meth:`submit` returns ``False`` and drops its batch.
        Synchronous, so ``timeout`` is unused.
        """
        self._shutdown = True


# ======================================================================
# ProcessExecutor — spawned-worker asynchronous trainer
# ======================================================================


@dataclass
class WorkerConfig:
    """Picklable training knobs handed to the child :func:`_trainer_worker`.

    Everything the worker needs to configure torch and drive ``flow.fit`` lives
    here so the parent can send a single picklable object across the spawn
    boundary.  Held separately from :class:`FlowSpec` (which carries *what* to
    build) — this carries *how* to train it.

    Parameters
    ----------
    epochs_per_round : int, optional
        ``n_epochs`` for each ``flow.fit`` call.  Default is ``20``.
    min_train_samples : int, optional
        Minimum total buffered samples (summed over conditions) before a fit
        runs.  Default is ``1000``.
    max_buffer_samples : int, optional
        Per-condition ring-buffer cap, with the same "keep the newest array
        whole" semantics as :class:`InlineExecutor`.  Default is ``20_000``.
    torch_num_threads : int, optional
        ``torch.set_num_threads`` value, applied *before* the flow is built so
        every op in the child is bounded.  Default is ``2``.
    seed : int, optional
        Seed for ``torch.manual_seed`` and ``np.random.seed`` in the child.
        Default is ``1234``.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``flow.fit`` (e.g. ``lr``).
        ``refit_data_transform`` and ``verbose`` are controlled by the executor
        and rejected in :class:`ProcessExecutor.__init__`.  Default is empty.
    refit_transform_every : int or None, optional
        Drift valve for the data transform.  ``None`` (the default) fits the
        transform once on the first training round and freezes it; an integer
        ``K`` re-fits it every ``K``-th round.  Mirrors
        :class:`InlineExecutor`'s knob of the same name.
    save_path : str or None, optional
        HDF5 checkpoint path for the latest fit, written atomically by the
        worker every ``save_every``-th version.  Default is ``None``.
    save_every : int, optional
        Checkpoint cadence in trained versions.  Default is ``1``.
    diagnostics_dir : str or None, optional
        Directory for the worker's per-round diagnostic PNGs.  Default is
        ``None``.
    diagnostics_every : int, optional
        Plot cadence in trained versions.  Default is ``1``.
    plot_corner : bool, optional
        Also write per-condition corner plots (training buffer vs flow draws)
        on each diagnostics round.  Default is ``False``.
    corner_max_samples : int, optional
        Cap on points per data set in each corner plot.  Default is ``5000``.
    """

    epochs_per_round: int = 20
    min_train_samples: int = 1000
    max_buffer_samples: int = 20_000
    torch_num_threads: int = 2
    seed: int = 1234
    fit_kwargs: dict = field(default_factory=dict)
    refit_transform_every: int | None = None
    save_path: str | None = None
    save_every: int = 1
    diagnostics_dir: str | None = None
    diagnostics_every: int = 1
    plot_corner: bool = False
    corner_max_samples: int = 5000


def _trainer_worker(spec, cfg, sample_q, weights_q, stop_event):
    """Child-process training loop (module-level so it is spawn-importable).

    Rebuilds the flow from ``spec``, then loops: pull sample batches
    off ``sample_q``, coalesce everything currently pending into the per-
    condition ring buffers, and — once at least ``cfg.min_train_samples`` are
    buffered — run exactly ONE ``flow.fit`` over the concatenated buffers.  The
    data transform is fitted on the FIRST training round and frozen thereafter
    (``cfg.refit_transform_every=K`` re-fits it every K rounds).  Each
    successful fit bumps an internal version and pushes
    ``("weights", version, snapshot, val_nll, history)`` onto the bounded
    ``weights_q``, dropping the oldest queued item if the parent has not
    drained it.  The ``snapshot`` is a self-contained
    ``{"net", "data_transform"}`` dict (the fitted transform travels with the
    net so the parent installs a matched pair), ``val_nll`` is the final-epoch
    latent-space validation loss, and ``history`` is the round's full
    :class:`~eryn.flows.base.FlowHistory` for monitoring.

    When ``cfg.save_path`` is set, every ``cfg.save_every``-th version is also
    checkpointed to HDF5 (atomic overwrite; a save failure is surfaced to the
    parent as a :class:`TrainerError` like any other worker failure).  When
    ``cfg.diagnostics_dir`` is set, per-round loss-curve / val-NLL-trend PNGs
    (and, with ``cfg.plot_corner``, per-condition corner plots of the training
    buffer vs flow draws) are written best-effort — plot failures warn and
    never interrupt training.

    Shutdown is driven by EITHER mechanism: a ``None`` sentinel on ``sample_q``
    (a clean wake from a blocked ``get``) or ``stop_event`` being set (works
    even when ``sample_q`` is full and the sentinel cannot be enqueued).

    On any exception the full child traceback is pushed as
    ``("error", traceback)`` so the parent can re-raise it as a
    :class:`TrainerError`.  Queue teardown in ``finally`` is deliberately
    asymmetric: ``sample_q`` (which the child only *reads*) has its feeder
    join thread cancelled so a half-flushed thread cannot deadlock the
    child's exit, while ``weights_q`` (which the child *writes*) is
    ``close()``d and joined so the final weights/error item is flushed
    before exit — cancelling it would silently discard that last item and
    turn worker failures into clean exitcode-0 deaths.
    """
    try:
        import torch

        # Bound thread usage BEFORE building the flow so every op the child runs
        # (including buffer allocs inside the flow constructor) is capped.
        torch.set_num_threads(cfg.torch_num_threads)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed % (2 ** 32))

        if cfg.diagnostics_dir is not None:
            # Force the non-interactive backend BEFORE anything (e.g. corner)
            # pulls in pyplot: the spawned child has no display.
            import matplotlib

            matplotlib.use("Agg")

        flow = spec.build()

        # Per-condition ring buffers: condition id -> deque of (N_i, dims) arrays.
        # Same trim semantics as InlineExecutor — drop oldest arrays but always
        # keep the most recent array whole (a lone oversized harvest is kept).
        # Keep in sync with InlineExecutor._trim (the parent-side implementation
        # of this identical ring-buffer policy).
        buffers: dict[int, deque] = {}

        def _buffer_item(item):
            for cond, arr in item.items():
                cond = int(cond)
                arr = np.asarray(arr).reshape(-1, flow.dims)
                if len(arr) == 0:
                    continue
                if cond not in buffers:
                    buffers[cond] = deque()
                buf = buffers[cond]
                buf.append(arr)
                total = sum(len(a) for a in buf)
                while total > cfg.max_buffer_samples and len(buf) > 1:
                    total -= len(buf.popleft())

        def _buffered_total():
            return sum(len(a) for buf in buffers.values() for a in buf)

        def _assemble():
            return {
                cond: np.concatenate(list(buf), axis=0)
                for cond, buf in buffers.items()
                if len(buf) > 0
            }

        version = 0
        round_count = 0
        transform_fitted = False
        # Per-round (version, val_nll) record feeding the trend plot.
        round_versions: list = []
        round_val_nlls: list = []
        while not stop_event.is_set():
            try:
                item = sample_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:  # sentinel: clean shutdown request
                break
            _buffer_item(item)

            # Coalesce: drain everything else currently pending so we run ONE
            # fit over the union, rather than once per queued batch.
            while True:
                try:
                    extra = sample_q.get_nowait()
                except queue.Empty:
                    break
                if extra is None:  # sentinel seen while draining → stop after
                    stop_event.set()
                    break
                _buffer_item(extra)

            total = _buffered_total()
            if total < cfg.min_train_samples or total == 0:
                continue

            # Lazy fit-once-then-freeze, mirroring InlineExecutor: fit the
            # transform on the first round, freeze it after, and (if
            # refit_transform_every=K) re-fit every K-th round.
            next_round = round_count + 1
            do_refit = (not transform_fitted) or (
                cfg.refit_transform_every is not None
                and next_round % cfg.refit_transform_every == 0
            )
            train_data = _assemble()
            history = flow.fit(
                train_data,
                n_epochs=cfg.epochs_per_round,
                refit_data_transform=do_refit,
                verbose=False,
                **cfg.fit_kwargs,
            )
            transform_fitted = True
            round_count = next_round
            version += 1
            val_loss = getattr(history, "validation_loss", None)
            # min, not [-1]: shipped weights = best-val state (see fit()).
            val_nll = (
                float(min(val_loss))
                if (val_loss is not None and len(val_loss))
                else None
            )
            round_versions.append(version)
            round_val_nlls.append(val_nll)

            # Checkpoint the latest fit.  Deliberately OUTSIDE any local
            # try/except: a save failure propagates to the outer handler and
            # reaches the parent as ("error", tb) → TrainerError.  Silently
            # losing persistence is worse than a crash.
            if cfg.save_path is not None and version % cfg.save_every == 0:
                _diagnostics.save_checkpoint(flow, cfg.save_path, version, val_nll)

            # Diagnostics plots are best-effort: warn on the child's stderr,
            # never interrupt training.
            if (
                cfg.diagnostics_dir is not None
                and version % cfg.diagnostics_every == 0
            ):
                try:
                    _diagnostics.save_loss_plot(
                        history, cfg.diagnostics_dir, version, val_nll
                    )
                    _diagnostics.save_val_nll_trend(
                        round_versions, round_val_nlls, cfg.diagnostics_dir
                    )
                    if cfg.plot_corner:
                        _diagnostics.save_corner_plot(
                            train_data,
                            flow,
                            cfg.diagnostics_dir,
                            version,
                            cfg.corner_max_samples,
                        )
                except Exception:
                    warnings.warn(
                        "flow diagnostics plotting failed in trainer worker:\n"
                        + traceback.format_exc(),
                        RuntimeWarning,
                    )

            # Ship a self-contained snapshot carrying the fitted transform with
            # the net (picklable: CPU net tensors + picklable transform), plus
            # the round's FlowHistory (plain lists of floats).
            _put_drop_oldest(
                weights_q,
                ("weights", version, flow.get_snapshot(), val_nll, history),
            )
    except Exception:  # noqa: BLE001 — surface ANY failure to the parent.
        _put_drop_oldest(weights_q, ("error", traceback.format_exc()))
    finally:
        # sample_q: the child is the READER, so it has no pending writes to flush
        # here; cancelling its join thread just guarantees the child never blocks
        # on the (parent-fed) queue at exit.  This is the documented macOS
        # feeder-deadlock guard.
        try:
            sample_q.cancel_join_thread()
        except Exception:
            pass
        # weights_q: the child is the WRITER.  We must NOT cancel its join thread
        # — doing so would discard the final "weights"/"error" item that the
        # feeder thread has not yet flushed to the pipe, and the parent would
        # never observe the failure (it would just see a clean exit code 0).  We
        # instead close() and let the normal feeder-thread join flush the last
        # item.  This cannot deadlock: weights_q is bounded (maxsize 2) with a
        # drop-oldest writer and a parent that drains it, so the pipe never wedges
        # for lack of a reader.
        try:
            weights_q.close()
            weights_q.join_thread()
        except Exception:
            pass


def _put_drop_oldest(q, item):
    """Put ``item`` on bounded queue ``q``, dropping the oldest item if full.

    Best-effort and non-blocking: on ``queue.Full`` we evict one item with
    ``get_nowait`` then retry ``put_nowait`` once.  A benign race with a
    concurrent consumer can still leave it full (a slot was taken either way);
    we then give up rather than block.  Used by the worker so a slow/absent
    parent poller can never wedge the trainer.
    """
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


class ProcessExecutor(TrainerExecutor):
    """Asynchronous :class:`TrainerExecutor` backed by a spawned worker process.

    A single child process (started with the ``"spawn"`` start method — see
    below) rebuilds the flow from a :class:`FlowSpec` and trains it on coalesced
    sample batches, returning versioned self-contained snapshots
    (``{"net", "data_transform"}``) through a small bounded queue.  The worker
    fits the data transform on its first round and ships it back inside every
    snapshot, so the parent always installs a matched net + transform.  The
    parent side is entirely non-blocking: :meth:`submit` offers a batch with
    ``put_nowait`` (applying ``drop_policy`` on a full queue) and
    :meth:`latest_weights` drains whatever the worker has produced so far,
    keeping the newest snapshot.

    This is the production executor behind the :class:`TrainerExecutor` seam; it
    honours exactly the ABC contract that :class:`InlineExecutor` does
    (poll-advanced :attr:`version`, non-aliased snapshots, empty-submit →
    ``False``, deferred failure surfacing) so :class:`~eryn.moves.flow.ConditionalFlowMove`
    works unchanged against either.

    Why ``spawn`` (always)
    ----------------------
    The child is started with ``multiprocessing.get_context("spawn")``
    unconditionally.  ``fork`` (the Linux default) copies the parent's address
    space, including torch's already-initialised thread pools and any held
    locks, which routinely deadlocks the first torch call in the child.
    ``spawn`` starts a fresh interpreter that imports torch cleanly — at the
    cost of a slow (~2-5 s) child startup, which callers must budget for.

    Single-use lifecycle
    --------------------
    An executor instance is **single-use**: once :meth:`shutdown` has run, the
    executor is permanently torn down — :meth:`start` afterwards is a no-op and
    a stopped worker cannot be restarted.  To resume online training, create a
    NEW :class:`ProcessExecutor` (and, if a :class:`~eryn.moves.flow.ConditionalFlowMove`
    holds it, assign it through ``move.executor =`` so the per-executor version
    counter reset is applied — see :attr:`TrainerExecutor.version`).

    Degenerate ``start=False`` state
    --------------------------------
    The sample queue, weights queue and stop event are created in
    ``__init__`` regardless of ``start``, and the worker process only in
    :meth:`start`.  This makes ``start=False`` a *supported* state:
    :meth:`submit` still buffers into the (bounded) queue and exercises the
    drop policy with no consumer attached; :meth:`latest_weights` returns
    ``None`` (nothing produced); :meth:`shutdown` is safe (no process to join).
    Calling :meth:`start` later drains the buffered batches into the worker.
    Because the queues/event are allocated in ``__init__``, you should call
    :meth:`shutdown` (or use the context manager) to release the feeder
    threads/pipes **even if :meth:`start` was never called** — a never-started
    executor still owns those resources (M2).

    Parameters
    ----------
    flow : eryn.flows.base.Flow
        Template flow.  Its data transform may be unfitted — the worker fits it
        on the first round.  Snapshotted into a ``FlowSpec``
        (``worker_device="cpu"``); the caller's object is never touched.
    fit_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the worker's ``flow.fit``.
        ``refit_data_transform`` / ``verbose`` collide with executor-controlled
        knobs and raise :class:`ValueError`.  Default is ``None``.
    worker_device : str, optional
        Device the worker builds the flow on.  Default is ``"cpu"``.
    epochs_per_round : int, optional
        ``n_epochs`` per fit in the worker.  Default is ``20``.
    min_train_samples : int, optional
        Minimum buffered samples before the worker fits.  Default is ``1000``.
    max_buffer_samples : int, optional
        Per-condition ring-buffer cap in the worker.  Default is ``20_000``.
    refit_transform_every : int or None, optional
        Drift valve for the data transform: ``None`` (default) fits it once and
        freezes it; an integer ``K`` re-fits it every ``K``-th worker round.
    max_pending_batches : int, optional
        ``maxsize`` of the sample queue.  Backpressure: once this many batches
        are queued, :meth:`submit` applies ``drop_policy``.  Default is ``4``.
    drop_policy : {"oldest", "newest"}, optional
        On a full sample queue, ``"oldest"`` evicts the oldest queued batch to
        make room for the incoming one; ``"newest"`` drops the incoming batch.
        Default is ``"oldest"``.
    torch_num_threads : int, optional
        ``torch.set_num_threads`` in the worker.  Default is ``2``.
    seed : int, optional
        Seed for the worker's torch/numpy RNGs.  Default is ``1234``.
    save_path : str or None, optional
        HDF5 checkpoint of the latest fit, written by the WORKER (where the
        trained flow lives) every ``save_every``-th version — atomic
        overwrite, so the file always holds the newest fit and round-trips a
        fully usable flow via ``FlowClass.load(save_path)`` (fitted transform
        included; ``version`` / ``val_nll`` stored as root attrs).  A save
        failure fails the worker loudly (:class:`TrainerError` on the next
        poll).  Default is ``None`` (no checkpointing).
    save_every : int, optional
        Checkpoint cadence in trained versions.  Default is ``1``.
    diagnostics_dir : str or None, optional
        Directory for per-round diagnostic PNGs written by the worker:
        per-round loss curves (``flow_loss_v####.png``) and a running
        validation-NLL trend (``flow_val_nll_trend.png``).  Plot failures are
        best-effort (warning on the child's stderr), never fatal.  Default is
        ``None`` (no plots).
    diagnostics_every : int, optional
        Plot cadence in trained versions.  Default is ``1``.
    plot_corner : bool, optional
        When ``True`` (requires ``diagnostics_dir`` and the optional
        ``corner`` package), each diagnostics round also writes one corner
        plot per condition overlaying that round's training buffer with an
        equal number of draws from the freshly trained flow
        (``flow_corner_v####_cond#.png``).  Default is ``False``.
    corner_max_samples : int, optional
        Cap on points per data set in each corner plot.  Default is ``5000``.
    start : bool, optional
        Start the worker process in ``__init__``.  Default is ``True``.  See the
        degenerate ``start=False`` state above.

    Attributes
    ----------
    version : int
        Newest version OBSERVED via :meth:`latest_weights` (poll-advanced).
    """

    # Always spawn: fork + torch deadlocks (copied thread pools / held locks).
    _CTX = multiprocessing.get_context("spawn")

    def __init__(
        self,
        flow,
        fit_kwargs: dict | None = None,
        *,
        worker_device: str = "cpu",
        epochs_per_round: int = 20,
        min_train_samples: int = 1000,
        max_buffer_samples: int = 20_000,
        refit_transform_every: int | None = None,
        max_pending_batches: int = 4,
        drop_policy: str = "oldest",
        torch_num_threads: int = 2,
        seed: int = 1234,
        save_path: str | None = None,
        save_every: int = 1,
        diagnostics_dir: str | None = None,
        diagnostics_every: int = 1,
        plot_corner: bool = False,
        corner_max_samples: int = 5000,
        start: bool = True,
    ):
        fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        # Same collision guard as InlineExecutor: the transform is frozen and
        # verbosity is forced off, so these are executor-controlled.
        _reserved = {"refit_data_transform", "verbose"} & fit_kwargs.keys()
        if _reserved:
            raise ValueError(
                f"fit_kwargs may not contain {sorted(_reserved)}: these are "
                "controlled by the executor (the data transform is frozen and "
                "verbosity is forced off).  Remove them from fit_kwargs."
            )
        if drop_policy not in ("oldest", "newest"):
            raise ValueError(
                f"drop_policy must be 'oldest' or 'newest', got {drop_policy!r}."
            )

        # Fail-fast in the parent (missing corner/matplotlib, plot_corner
        # without diagnostics_dir) rather than deep inside the worker spawn;
        # also pre-creates the output directories.
        _validate_monitoring_args(save_path, diagnostics_dir, plot_corner)

        # Snapshot the flow NOW (in the parent) so picklability fails fast here,
        # not deep inside a spawn. The transform may be unfitted; the worker
        # fits it on the first round and ships it back in each snapshot.
        self._worker_device = worker_device
        self._spec = FlowSpec.from_flow(flow, worker_device=worker_device)
        self._cfg = WorkerConfig(
            epochs_per_round=int(epochs_per_round),
            min_train_samples=int(min_train_samples),
            max_buffer_samples=int(max_buffer_samples),
            torch_num_threads=int(torch_num_threads),
            seed=int(seed),
            fit_kwargs=fit_kwargs,
            refit_transform_every=(
                int(refit_transform_every)
                if refit_transform_every is not None
                else None
            ),
            save_path=str(save_path) if save_path is not None else None,
            save_every=int(save_every),
            diagnostics_dir=(
                str(diagnostics_dir) if diagnostics_dir is not None else None
            ),
            diagnostics_every=int(diagnostics_every),
            plot_corner=bool(plot_corner),
            corner_max_samples=int(corner_max_samples),
        )
        self._drop_policy = drop_policy

        # Queues + event created up front so submit() works even with start=False
        # (degenerate buffering state).  weights_q is tiny: we only ever need the
        # newest weights, and _put_drop_oldest keeps it from backing up.
        self._sample_q = self._CTX.Queue(maxsize=int(max_pending_batches))
        self._weights_q = self._CTX.Queue(maxsize=2)
        self._stop_event = self._CTX.Event()

        self._process: multiprocessing.process.BaseProcess | None = None
        self._started = False
        self._shutdown = False
        self._error: BaseException | None = None
        self._latest: tuple[int, dict] | None = None
        self._latest_val_nll: float | None = None
        self._latest_history = None
        self._seen_version = 0
        self._exitcode: int | None = None

        if start:
            self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker process (idempotent; no-op if already started).

        Spawns the child with the queues/event created in ``__init__``.  Safe to
        call exactly once; subsequent calls are no-ops.  Any batches already
        buffered into the sample queue (via ``start=False`` + :meth:`submit`)
        are drained by the worker once it is up.
        """
        if self._started or self._shutdown:
            return
        self._process = self._CTX.Process(
            target=_trainer_worker,
            args=(self._spec, self._cfg, self._sample_q, self._weights_q,
                  self._stop_event),
            daemon=True,  # last-resort no-zombie guarantee if the parent dies.
        )
        self._process.start()
        self._started = True

    # ------------------------------------------------------------------
    # TrainerExecutor interface
    # ------------------------------------------------------------------

    def submit(self, samples_by_condition: dict) -> bool:
        """Offer a batch to the worker (non-blocking; never blocks on training).

        Empty harvests (an empty dict, or only zero-row arrays) are dropped and
        return ``False`` WITHOUT touching the queue — a sampling step with no
        active leaves is a normal condition, never a failure.

        On a full sample queue, ``drop_policy`` decides: ``"oldest"`` evicts one
        queued batch (``get_nowait``) and retries (a lost batch either way — the
        race with the consumer is benign); ``"newest"`` drops the incoming batch.
        Returns ``True`` iff the batch was enqueued.

        Returns ``False`` after :meth:`shutdown`.

        Raises
        ------
        TrainerError
            If the worker has already been observed to have failed (deferred
            surfacing, matching :class:`InlineExecutor`).
        """
        if self._error is not None:
            raise TrainerError(
                f"ProcessExecutor trainer failed: {self._error}"
            ) from self._error

        if self._shutdown:
            return False

        # Drop empty / all-empty harvests before touching the queue.
        cleaned: dict[int, np.ndarray] = {}
        for cond, arr in samples_by_condition.items():
            arr = np.asarray(arr)
            if arr.size == 0 or len(arr) == 0:
                continue
            cleaned[int(cond)] = arr
        if not cleaned:
            return False

        try:
            self._sample_q.put_nowait(cleaned)
            return True
        except queue.Full:
            pass

        if self._drop_policy == "newest":
            return False

        # "oldest": evict one queued batch to make room, then retry once.
        #
        # macOS pitfall: ``Queue.put_nowait`` enqueues via a background feeder
        # thread, so an item just put may not yet be readable from the pipe even
        # though the maxsize semaphore already counts it as present.  A bare
        # ``get_nowait`` can therefore see Empty (the item is mid-flight) and the
        # eviction would spuriously fail.  A tiny bounded ``get`` waits for the
        # in-flight item to land — effectively non-blocking (the item is already
        # queued; we are only waiting on the feeder thread to flush, ~ms), and
        # capped so a genuinely-drained queue cannot stall us.
        try:
            self._sample_q.get(timeout=0.1)
        except queue.Empty:
            pass
        try:
            self._sample_q.put_nowait(cleaned)
            return True
        except queue.Full:
            # Consumer raced and refilled the slot — drop this batch (benign:
            # a batch is lost either way, matching the documented contract).
            return False

    def latest_weights(self):
        """Drain the weights queue and return the newest ``(version, snapshot)``.

        Non-blocking: pulls every pending item with ``get_nowait``, keeping the
        newest ``"weights"`` item seen.  Each ``"weights"`` item is
        ``("weights", version, snapshot, val_nll, history)`` where ``snapshot``
        is a self-contained ``{"net", "data_transform"}`` dict; the latest
        item's ``val_nll`` / ``history`` update :attr:`latest_val_nll` /
        :attr:`latest_history`.  If an ``"error"`` item
        appears the failure is recorded and :class:`TrainerError` (carrying the
        child traceback) is raised.  If — after draining — the executor has not
        already failed and the (non-shutdown) worker process has died with a
        non-zero (or unknown) exit code, that silent death is recorded and
        surfaced as a :class:`TrainerError` too.  Crucially this check fires
        **regardless of whether weights have already been observed**: a worker
        that crashes mid-run after serving its first version must still fail
        loud rather than freeze the proposal on the last good snapshot.

        Ownership contract
        ------------------
        Each ``"weights"`` item off the queue is a freshly deserialized object,
        so returned snapshots are never aliased to the worker's live training
        state.  The stashed snapshot IS the object handed out; successive polls
        with no new version return the SAME dict object — callers must not mutate
        it in place (per the ABC contract).

        Returns
        -------
        tuple[int, dict] or None
            The newest observed ``(version, snapshot)``, or ``None`` if nothing
            has been produced yet.  Advances :attr:`version` on a new
            observation (poll-advanced).
        """
        if self._error is not None:
            raise TrainerError(
                f"ProcessExecutor trainer failed: {self._error}"
            ) from self._error

        newest: tuple[int, dict] | None = None
        newest_val_nll: float | None = None
        newest_history = None
        while True:
            try:
                kind, *rest = self._weights_q.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                tb = rest[0]
                self._error = TrainerError(
                    "ProcessExecutor trainer process raised:\n" + tb
                )
                raise self._error
            elif kind == "weights":
                version, snapshot, val_nll, history = rest
                if newest is None or version > newest[0]:
                    newest = (version, snapshot)
                    newest_val_nll = val_nll
                    newest_history = history

        if newest is not None:
            # Only keep it if it is strictly newer than what we already stashed
            # (queue draining is monotonic, but guard defensively).
            if self._latest is None or newest[0] > self._latest[0]:
                self._latest = newest
                self._latest_val_nll = newest_val_nll
                self._latest_history = newest_history

        # Silent-death detection: the worker has died with a bad exit code →
        # surface as a failure rather than freezing forever on stale weights.
        # This fires REGARDLESS of whether any weights have already been observed:
        # a mid-run crash (OOM-kill, segfault → nonzero exitcode) after the first
        # version must still be loud, or the proposal would silently sample from a
        # frozen flow for the rest of the run.  It is gated on ``not self._shutdown``
        # so a worker we deliberately terminated in :meth:`shutdown` (exitcode -15)
        # is not reported as a spurious failure.  Clean exit (0) or a still-living
        # / unknown-exitcode worker (None) is never an error here.
        if (
            not self._shutdown
            and self._error is None
            and self._process is not None
            and self._started
            and not self._process.is_alive()
        ):
            code = self._process.exitcode
            if code not in (0, None):
                self._exitcode = code
                self._error = TrainerError(
                    f"ProcessExecutor trainer process died, exitcode={code}"
                )
                raise self._error

        if self._latest is None:
            return None
        version, snapshot = self._latest
        self._seen_version = version
        return version, snapshot

    @property
    def version(self) -> int:
        """Newest version observed via :meth:`latest_weights` (``0`` = none yet)."""
        return self._seen_version

    @property
    def latest_val_nll(self):
        """Latent-space validation NLL of the newest observed version (or ``None``).

        Updated by :meth:`latest_weights` when a strictly-newer version is
        drained, so poll that first to refresh it.
        """
        return self._latest_val_nll

    @property
    def latest_history(self):
        """:class:`~eryn.flows.base.FlowHistory` of the newest observed version.

        Ships with each weights item, so — like :attr:`latest_val_nll` — it is
        refreshed by :meth:`latest_weights`; poll that first.  ``None`` before
        any version has been observed.
        """
        return self._latest_history

    @property
    def worker_device(self) -> str:
        """Device the worker builds the flow on (``"cpu"`` by default)."""
        return self._worker_device

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the worker and reclaim it (idempotent).

        Signals shutdown via BOTH mechanisms: sets ``stop_event`` (works even
        when the sample queue is full and the sentinel cannot be enqueued) and
        best-effort enqueues a ``None`` sentinel (a clean wake from a blocked
        ``get``).  Joins with ``timeout``; if the child is still alive escalates
        ``terminate()`` → join(2s) → ``kill()`` → join(1s).  Then drains both
        queues and calls ``cancel_join_thread`` on each so the parent's own
        feeder threads cannot deadlock at interpreter exit (macOS pitfall).
        Safe when ``start=False`` or the process never started.

        Drain-then-cancel ORDER matters: draining first frees queued items so
        the feeder threads have nothing to flush, then ``cancel_join_thread``
        makes the final teardown non-blocking.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # Signal via the event first (always effective), then the sentinel.
        try:
            self._stop_event.set()
        except Exception:
            pass
        try:
            self._sample_q.put_nowait(None)
        except queue.Full:
            pass  # full queue → stop_event is the wake mechanism instead.
        except Exception:
            pass

        p = self._process
        if p is not None:
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join(2.0)
            if p.is_alive():
                p.kill()
                p.join(1.0)
            self._exitcode = p.exitcode

        # Drain both queues BEFORE cancelling join threads so nothing is left to
        # flush, then cancel so teardown is non-blocking.
        for q in (self._sample_q, self._weights_q):
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    # queue.Empty ends the drain normally; any other queue error
                    # here is from an already-closed queue at teardown and is
                    # deliberately swallowed (both are subclasses of Exception).
                    break
            try:
                q.cancel_join_thread()
            except Exception:
                pass
