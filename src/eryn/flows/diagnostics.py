# -*- coding: utf-8 -*-
"""Checkpointing and per-round training diagnostics for trainer executors.

This module is torch-free AND plotting-free at import time: ``h5py``,
``matplotlib`` and the optional ``corner`` package are imported lazily inside
the functions, so ``import eryn.flows`` never drags in a plotting stack.

All figures are built through the object-oriented matplotlib API
(:class:`matplotlib.figure.Figure` — no ``pyplot``, no global figure state),
so these functions are safe to call both from a spawned trainer worker and
from the caller's own process without touching the interactive backend.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = [
    "save_checkpoint",
    "save_loss_plot",
    "save_val_nll_trend",
    "save_corner_plot",
]


def save_checkpoint(flow, save_path: str, version: int, val_nll=None) -> str:
    """Atomically write the flow's latest fit to an HDF5 checkpoint.

    Delegates the layout to :meth:`eryn.flows.Flow.save` (config + pickled
    ``data_transform``/``conditioning`` + weights), so the file round-trips a
    fully usable flow via ``FlowClass.load(save_path)`` — including the fitted
    whitening transform.  The trained ``version`` (and ``val_nll`` when
    available) are stored as root attributes.

    The write is atomic: the file is written to ``<save_path>.tmp`` and then
    ``os.replace``d over ``save_path``, so a reader never sees a half-written
    checkpoint and a crash mid-save leaves the previous checkpoint intact.

    Parameters
    ----------
    flow : eryn.flows.base.Flow
        The trained flow to persist (must implement ``save``).
    save_path : str
        Destination HDF5 file; overwritten on every call (latest-fit
        semantics).
    version : int
        Trained version, stored as the ``version`` root attribute.
    val_nll : float or None, optional
        Best (shipped-weights) latent-space validation NLL, stored as the ``val_nll``
        root attribute when not ``None``.

    Returns
    -------
    str
        The checkpoint path.
    """
    import h5py

    tmp_path = str(save_path) + ".tmp"
    with h5py.File(tmp_path, "w") as handle:
        flow.save(handle, path="flow")
        handle.attrs["version"] = int(version)
        if val_nll is not None:
            handle.attrs["val_nll"] = float(val_nll)
    os.replace(tmp_path, str(save_path))
    return str(save_path)


def save_loss_plot(history, out_dir: str, version: int, val_nll=None) -> str:
    """Plot one training round's loss curves to ``flow_loss_v<version>.png``.

    Parameters
    ----------
    history : eryn.flows.base.FlowHistory
        Per-epoch ``training_loss`` / ``validation_loss`` of the round.
    out_dir : str
        Directory the PNG is written into (must exist).
    version : int
        Trained version (used in the filename and title).
    val_nll : float or None, optional
        Best (shipped-weights) validation NLL, shown in the title when not
        ``None``.

    Returns
    -------
    str
        Path of the written PNG.
    """
    from matplotlib.figure import Figure

    training_loss = list(getattr(history, "training_loss", []) or [])
    validation_loss = list(getattr(history, "validation_loss", []) or [])

    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    if training_loss:
        ax.plot(np.arange(1, len(training_loss) + 1), training_loss,
                label="training")
    if validation_loss:
        ax.plot(np.arange(1, len(validation_loss) + 1), validation_loss,
                label="validation")
        # mark the best epoch: fit() restores this state, so everything to
        # its right is the discarded overfit tail, not the shipped model.
        best = int(np.argmin(validation_loss))
        ax.axvline(best + 1, color="gray", ls="--", lw=1.0,
                   label=f"best (shipped): epoch {best + 1}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (latent-space NLL)")
    title = f"flow training round v{version}"
    if val_nll is not None:
        title += f" — best val NLL {val_nll:.4g}"
    ax.set_title(title)
    if training_loss or validation_loss:
        ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"flow_loss_v{int(version):04d}.png")
    fig.savefig(out_path, dpi=120)
    return out_path


def save_val_nll_trend(versions, val_nlls, out_dir: str) -> str:
    """Plot validation NLL vs trained version to ``flow_val_nll_trend.png``.

    Overwritten on every call so the file always shows the full trend so far.
    Rounds whose fit produced no validation history (``val_nll is None``) are
    skipped.

    Parameters
    ----------
    versions : sequence of int
        Trained version of each completed round.
    val_nlls : sequence of float or None
        Best (shipped-weights) validation NLL per round (``None`` entries are skipped).
    out_dir : str
        Directory the PNG is written into (must exist).

    Returns
    -------
    str
        Path of the written PNG.
    """
    from matplotlib.figure import Figure

    pairs = [(v, n) for v, n in zip(versions, val_nlls) if n is not None]

    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    if pairs:
        vs, nlls = zip(*pairs)
        ax.plot(vs, nlls, marker="o")
    ax.set_xlabel("trained version")
    ax.set_ylabel("final validation NLL (latent space)")
    ax.set_title("flow training progress")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "flow_val_nll_trend.png")
    fig.savefig(out_path, dpi=120)
    return out_path


def save_corner_plot(
    samples_by_condition: dict,
    flow,
    out_dir: str,
    version: int,
    max_samples: int = 5000,
) -> list:
    """Corner-plot training samples against flow draws, one PNG per condition.

    For each condition the training buffer (black) is overlaid with an equal
    number of samples drawn from the freshly trained flow (blue), on shared
    axis ranges, to ``flow_corner_v<version>_cond<c>.png``.  Requires the
    optional ``corner`` package.

    Parameters
    ----------
    samples_by_condition : dict[int, np.ndarray]
        The assembled training buffers of this round, ``{condition: (N, dims)}``.
    flow : eryn.flows.base.Flow
        The freshly trained flow to draw from.  Drawn with ``context=cond``
        when the flow has a ``conditioning`` strategy, else unconditionally.
    out_dir : str
        Directory the PNGs are written into (must exist).
    version : int
        Trained version (used in filenames; also seeds the subsampling RNG so
        a given round's plot is reproducible).
    max_samples : int, optional
        Cap on the number of points per data set in each plot.  Default 5000.

    Returns
    -------
    list of str
        Paths of the written PNGs (one per plotted condition).
    """
    import corner
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(int(version))
    out_paths = []
    for cond in sorted(samples_by_condition):
        train = np.asarray(samples_by_condition[cond])
        train = train.reshape(-1, train.shape[-1])
        n = int(min(len(train), max_samples))
        if n < 2:
            continue
        if len(train) > n:
            train = train[rng.choice(len(train), size=n, replace=False)]

        context = int(cond) if getattr(flow, "conditioning", None) is not None else None
        flow_samples = np.asarray(flow.sample(n, context=context))

        # Shared axis ranges so both data sets are binned identically; pad
        # zero-width dimensions so corner's histogram never sees an empty range.
        combined = np.vstack([train, flow_samples])
        ranges = []
        for d in range(combined.shape[1]):
            lo, hi = float(combined[:, d].min()), float(combined[:, d].max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            ranges.append((lo, hi))

        ndim = train.shape[1]
        labels = [f"$x_{{{d}}}$" for d in range(ndim)]
        fig = Figure(figsize=(2.0 * ndim, 2.0 * ndim))
        corner.corner(train, fig=fig, color="k", labels=labels, range=ranges,
                      plot_datapoints=False, plot_density=False)
        corner.corner(flow_samples, fig=fig, color="C0", range=ranges,
                      plot_datapoints=False, plot_density=False)
        fig.legend(
            handles=[Line2D([0], [0], color="k"), Line2D([0], [0], color="C0")],
            labels=["training samples", "flow samples"],
            loc="upper right",
        )

        out_path = os.path.join(
            out_dir, f"flow_corner_v{int(version):04d}_cond{int(cond)}.png"
        )
        fig.savefig(out_path, dpi=120)
        out_paths.append(out_path)

    return out_paths
