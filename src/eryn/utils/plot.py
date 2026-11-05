# *-- coding: utf-8 --*
import os
from shutil import which
import typing

import corner
import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse, Rectangle

from eryn.utils.updates import UpdateStep
from eryn.utils.utility import get_integrated_act, stepping_stone_log_evidence, walk_moves

DEFAULT_PALETTE = "icefire"

class Backend:
    """A placeholder Backend class for type hinting."""
    pass

def setup_plotting():
    try:
        import scienceplots
        plt.style.use(['science'])
    except:
        pass

    # increase default font size
    # mpl.rcParams.update({'font.size': 16})

    if which("latex"):
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })
    else:
        print("LaTeX not found. Using default matplotlib fonts.")
        mpl.rcParams.update({
            "text.usetex": False,
        })

class Backend:
    """A placeholder Backend class for type hinting."""
    pass

def save_or_show(fig, filename=None):
    """
    Save the figure to a file or show it.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save or show.
        filename (str, optional): If provided, saves the figure to this filename.
    """
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def move_acceptance_rates(accepted, num_proposals, mode="interval"):
    """Derive acceptance rates from cumulative acceptance counters.

    Moves keep cumulative counters, so a plain ratio is the acceptance since
    the start of the run. That view lags: a proposal that only starts working
    late shows as a slow rise rather than a step. Storing the counters rather
    than the ratio lets either view be produced here.

    Args:
        accepted (np.ndarray): Cumulative accepted counts, shape
            ``(nsteps, ntemps, nwalkers)``.
        num_proposals (np.ndarray): Cumulative proposal counts, shape
            ``(nsteps,)``.
        mode (str, optional): ``"interval"`` for the rate within each interval
            between plot calls, ``"cumulative"`` for the rate since the start
            of the run. (default: ``"interval"``)

    Returns:
        np.ndarray: Acceptance rates, shape ``(nsteps, ntemps, nwalkers)``.
            Entries where the move was never drawn are ``np.nan``, so the line
            breaks rather than reading as a genuine zero acceptance.

    Raises:
        ValueError: ``mode`` is neither ``"interval"`` nor ``"cumulative"``.

    """
    accepted = np.asarray(accepted, dtype=float)
    num_proposals = np.asarray(num_proposals, dtype=float)

    if mode == "cumulative":
        counts, totals = accepted, num_proposals[:, None, None]
        drawn = totals > 0
    elif mode == "interval":
        counts = np.diff(accepted, axis=0, prepend=0.0)
        totals = np.diff(num_proposals, prepend=0.0)[:, None, None]
        # Cumulative counters are monotonic by construction, so a negative
        # increment here means something external reset them underneath us
        # (e.g. the same move instance handed to an EnsembleSampler, which
        # zeroes `accepted` in place while preserving its shape -- FIX 1
        # guards the shape-changing case, this is its silent sibling). Treat
        # that interval as not-drawn instead of plotting a false negative
        # acceptance rate.
        drawn = (totals > 0) & (counts >= 0)
    else:
        raise ValueError(f"mode must be 'interval' or 'cumulative', got {mode!r}.")

    # np.where evaluates both branches, so the denominator is guarded too --
    # an unguarded division would still warn on the branch that is discarded
    return np.where(drawn, counts / np.where(drawn, totals, 1.0), np.nan)


def move_counters(move, _seen=None):
    """Cumulative ``(accepted, num_proposals)`` for one move, or ``None``.

    A move that keeps its own counters reports them directly. A wrapper that
    does not is pooled from its children instead: :class:`eryn.moves.CombineMove`
    overrides ``accepted`` to return its children's arrays (and its setter never
    stores ``_accepted``, so reading it raises), and never increments
    ``num_proposals``. Pooling sums counts rather than averaging fractions, so
    children drawn at different rates are weighted correctly. The same move
    instance can appear more than once in a subtree (shared between a wrapper
    and one of its own descendants), so pooling tracks visited ids and counts
    each instance once, the same guard :func:`eryn.utils.utility.walk_moves`
    keeps for the same reason.

    Args:
        move (:class:`eryn.moves.Move`): Move to read.
        _seen (set, optional): ``id()`` values already pooled within this call's
            subtree. Set during recursion; callers pass nothing. (default:
            ``None``)

    Returns:
        tuple or None: ``(accepted, num_proposals)`` with ``accepted`` of shape
            ``(ntemps, nwalkers)``, or ``None`` when no counters are available
            anywhere in this move's subtree. ``accepted`` is always a snapshot,
            never a view: moves mutate their counters in place (``self.accepted
            += ...``), so returning a view would alias every recorded history
            entry to the same live buffer and collapse the whole history to
            the final value.

    """
    if _seen is None:
        _seen = set()
    if id(move) in _seen:
        return None
    _seen.add(id(move))

    try:
        # np.array (not np.asarray) always copies, even when move.accepted is
        # already a float64 ndarray -- see the snapshot note above.
        accepted = np.array(move.accepted, dtype=float)
        num_proposals = float(move.num_proposals)
        if accepted.ndim == 2 and num_proposals > 0:
            return accepted, num_proposals
    except (AttributeError, ValueError, TypeError):
        # counters not initialised, or an aggregating override such as
        # CombineMove.accepted -- fall through to pooling
        pass

    pooled = [
        counters
        for counters in (
            move_counters(child, _seen) for child in getattr(move, "sub_moves", [])
        )
        if counters is not None
    ]
    if not pooled:
        return None

    return (
        np.sum([accepted for accepted, _ in pooled], axis=0),
        float(np.sum([num_proposals for _, num_proposals in pooled])),
    )


def _tex_safe(label):
    """Escape characters that break LaTeX rendering when ``text.usetex`` is on.

    Move paths contain underscores once a class is disambiguated
    (``MHMove_0``), which LaTeX reads as a subscript and rejects outside math
    mode.
    """
    if not mpl.rcParams.get("text.usetex", False):
        return label
    return label.replace("_", r"\_")


def move_tree_colors(paths, palette=None):
    """Assign one hue family per top-level move, lightening with depth.

    Args:
        paths (list): Move paths as produced by
            :func:`eryn.utils.utility.walk_moves`.
        palette (str or list, optional): Seaborn palette for the top-level
            moves. Default is ``'tab10'``.

    Returns:
        dict: ``path -> RGB tuple`` for every path given.

    """
    roots = []
    for path in paths:
        root = path.split("/")[0]
        if root not in roots:
            roots.append(root)

    base_colors = sns.color_palette(
        palette if palette is not None else "tab10", max(len(roots), 1)
    )

    colors = {}
    for root, base in zip(roots, base_colors):
        family = sorted(
            (path for path in paths if path.split("/")[0] == root),
            key=lambda path: (path.count("/"), path),
        )
        # reverse=True runs dark -> light so the root keeps the base hue; the
        # +2 keeps the deepest entries away from white
        shades = sns.light_palette(base, n_colors=len(family) + 2, reverse=True)
        for path, shade in zip(family, shades):
            colors[path] = shade

    return colors


def cov_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    """
    Plot a covariance ellipse using eigendecomposition.

    The ellipse axes are aligned with the eigenvectors of the covariance matrix,
    and scaled by sqrt(eigenvalue) * n_std.

    Args:
        mean (array-like): Center of the ellipse (mean_x, mean_y).
        cov (np.ndarray): 2x2 covariance matrix.
        ax (matplotlib.axes.Axes): Axes object on which to plot the ellipse.
        n_std (float, optional): Number of standard deviations for ellipse radius. Default is 1.0.
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.

    Returns:
        matplotlib.patches.Ellipse: The covariance ellipse added to the axes.
    """
    # Eigendecomposition: eigenvalues are variances along principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first) for consistent orientation
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Ellipse dimensions: 2 * n_std * sqrt(eigenvalue) for width/height
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # Rotation angle from the first eigenvector (major axis direction)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    return ax.add_patch(ellipse)

def overlay_fim_covariance(
    fig,
    covariance,
    means=None,
    nsigmas=[1, 2, 3],
    plot_1d=False,
    colors=None,
    linestyles=None,
    linewidths=None,
    alpha=0.7,
    labels=None,
):
    """
    Overlay Fisher Information Matrix confidence contours on corner plot axes.
    
    For 2D subplots, draws elliptical contours at specified confidence levels.
    For 1D diagonal plots, draws vertical lines at ±nσ from the mean.
    
    Args:
        fig (matplotlib.figure.Figure): Figure object containing corner plot axes.
        covariance (np.ndarray): Covariance matrix from Fisher analysis, shape (n_params, n_params).
        means (np.ndarray, optional): Mean values for each parameter. If None, uses origin (0, 0, ...).
        nsigmas (list, optional): List of sigma levels to plot (e.g., [1, 2, 3]). Default [1, 2, 3].
        plot_1d (bool, optional): Whether to plot 1D contours on diagonal plots. Default is False.
        colors (list, optional): Colors for each sigma level. If None, uses default color cycle.
        linestyles (list, optional): Line styles for each sigma level. If None, uses solid lines.
        linewidths (list, optional): Line widths for each sigma level. If None, uses default (1.5).
        alpha (float, optional): Transparency of contours. Default 0.7.
        labels (list, optional): Labels for legend entries. If None, uses "nσ FIM".
    
    Returns:
        matplotlib.figure.Figure: The figure with overlaid contours.
    """
    
    # Convert axs to numpy array if it's a list
    axs = np.array(fig.get_axes())
    
    # Infer number of parameters from covariance matrix
    n_params = covariance.shape[0]
    
    if covariance.shape != (n_params, n_params):
        raise ValueError(f"Covariance matrix must be square, got shape {covariance.shape}")
    
    # Set default means to origin
    if means is None:
        means = np.zeros(n_params)
    elif len(means) != n_params:
        raise ValueError(f"means must have length {n_params}, got {len(means)}")
    
    # Set default colors
    if colors is None:
        colors = [f'C{i}' for i in range(len(nsigmas))]
    elif isinstance(colors, str):
        colors = [colors] * len(nsigmas)
    elif len(colors) != len(nsigmas):
        raise ValueError(f"colors must have length {len(nsigmas)}, got {len(colors)}")
    
    # Set default linestyles
    if linestyles is None:
        linestyles = ['-'] * len(nsigmas)
    elif len(linestyles) != len(nsigmas):
        raise ValueError(f"linestyles must have length {len(nsigmas)}, got {len(linestyles)}")
    
    # Set default linewidths
    if linewidths is None:
        linewidths = [1.5] * len(nsigmas)
    elif len(linewidths) != len(nsigmas):
        raise ValueError(f"linewidths must have length {len(nsigmas)}, got {len(linewidths)}")
    
    # Set default labels
    if labels is None:
        labels = [f'{n}$\\sigma$ FIM' for n in nsigmas]
    elif len(labels) != len(nsigmas):
        raise ValueError(f"labels must have length {len(nsigmas)}, got {len(labels)}")
    
    # Extract standard deviations for 1D plots
    sigmas = np.sqrt(np.diag(covariance))
    
    # Reshape axes into 2D grid if needed
    if axs.ndim == 1:
        # Corner plot axes are typically returned as 1D array
        # Reshape to (n_params, n_params) grid
        n_axs = int(np.sqrt(len(axs)))
        axs_grid = np.empty((n_axs, n_axs), dtype=object)
        idx = 0
        for i in range(n_axs):
            for j in range(n_axs):
                axs_grid[j, i] = axs[idx]
                idx += 1
                # if j <= i:
                #     axs_grid[j, i] = axs[idx]
                #     idx += 1
                # else:
                #     axs_grid[j, i] = None
    else:
        axs_grid = axs
    
    # Loop over each sigma level
    for sigma_idx, (n_sigma, color, ls, lw, label) in enumerate(
        zip(nsigmas, colors, linestyles, linewidths, labels)
    ):        
        # Loop over all subplots
        for i in range(n_params):
            for j in range(i, n_params):
                ax = axs_grid[i, j]
                
                if ax is None:
                    continue
                
                if i == j:
                    if plot_1d:
                        # 1D diagonal plot - draw vertical lines at mean ± n*sigma
                        mean_val = means[i]
                        sigma_val = sigmas[i]
                        
                        # Get y-limits for vertical lines
                        ylim = ax.get_ylim()
                        
                        # Draw vertical lines
                        for sign in [-1, 1]:
                            line = ax.axvline(
                                mean_val + sign * n_sigma * sigma_val,
                                color=color,
                                linestyle=ls,
                                linewidth=lw,
                                alpha=alpha,
                                zorder=10,
                            )
                        else:
                            continue
                
                else:
                    # 2D off-diagonal plot - draw ellipse
                    # Extract 2x2 subcovariance for parameters j and i
                    cov = np.array(
                    (
                        (covariance[i][i], covariance[i][j]),
                        (covariance[j][i], covariance[j][j]),
                    )
                )
                # print(cov)

                    mean = np.array((means[i], means[j]))
                    cov_ellipse(mean, cov, ax, n_std=n_sigma, 
                                edgecolor=color, facecolor='none', linestyle=ls, linewidth=lw, 
                                zorder=10, alpha=alpha
                                )
    
    return fig

def cornerplot(data, *args, means=None, overlay_covariance=None, legend_label='Samples', overlay_label='Information Matrix Covariance', filename=None, **kwargs):
    """
    Create a corner plot with optional Information Matrix covariance overlay. This is centered around the means if provided.
    Wrapper around `corner.corner()` that adds Fisher Information Matrix covariance contours.

    Args:
        data (array-like): Input data for corner plot (e.g., MCMC samples).
        *args: Positional arguments passed to `corner.corner()`.
        means (array-like, optional): Mean values for each parameter to center the overlay. If None, uses 'truths' from kwargs or mean of data.
        overlay_covariance (np.ndarray, optional): Covariance matrix to overlay. If None, no overlay is added.
        legend_label (str, optional): Label for the sample distribution in the legend. Default is 'Samples'.
        overlay_label (str, optional): Label for the overlay covariance in the legend. Default is 'Information Matrix Covariance', assuming FIM.
        filename (str, optional): If provided, saves the figure to this filename.
        **kwargs: Keyword arguments passed to `corner.corner()`.

    Returns:
        matplotlib.figure.Figure: The corner plot figure with optional overlays. If `filename` is provided, the figure is saved instead.
    """

    corner_kwargs = {
    #'quantiles': [0.16, 0.5, 0.84],
    'levels': (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
    'show_titles': True,
    'title_fmt': '.1e',
    'title_kwargs': {'fontsize': 12},
    'label_kwargs': {'fontsize': 14},
    'hist_kwargs': {'density': True, 'linewidth': 2},
    'plot_datapoints': False,
    'fill_contours': True,
    'color': 'steelblue',
    'truth_color': 'red',
    }

    # Update base kwargs with any user-provided kwargs
    corner_kwargs.update(kwargs)

    # how do we deal with 2d truths for reversible jump?
    truths = corner_kwargs.get('truths', None)
    if truths is not None and len(truths.shape) > 1:  
        corner_kwargs['truths'] = None # add truths later

    fig = corner.corner(data=data, *args, **corner_kwargs)

    # add the other truths as vertical/horizontal lines
    if truths is not None and len(truths.shape) > 1:  
        corner_kwargs['truths'] = truths # add back in for legend
        n_params = data.shape[1]
        axs = np.array(fig.get_axes()).reshape((n_params, n_params))
        for truth in truths:
            for i in range(n_params):
                ax = axs[i, i]
                ax.axvline(truth[i], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)

            for i in range(n_params):
                for j in range(i):
                    ax = axs[i, j]
                    ax.axhline(truth[i], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)
                    ax.axvline(truth[j], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)

    # prepare handles for legend
    handles = []
    handles_labels = []

    handles.append(mpl.lines.Line2D([], [], color=corner_kwargs['color'],
                                    linestyle='-', linewidth=2))
    handles_labels.append(legend_label)
    if 'truths' in corner_kwargs and corner_kwargs['truths'] is not None:
        handles.append(mpl.lines.Line2D([], [], color=corner_kwargs['truth_color'],
                                        linestyle='-', linewidth=1))
        handles_labels.append('Truths')


    # Overlay covariance contours if provided
    if overlay_covariance is not None:
        if means is None:
            means = kwargs.get('truths', None)
        
        if means is None:
            # If no means or truths provided, default to the mean of the samples
            means = np.mean(data, axis=0)

        overlay_fim_covariance(
            fig,
            overlay_covariance,
            means=means,
            plot_1d=False,
            alpha=0.8,
            nsigmas=[1, 2 ,3],   
            colors='k'
        )

        handles.append(mpl.lines.Line2D([], [], color='k', linestyle='-', linewidth=1.5))
        handles_labels.append(overlay_label)
    
    # Add legend to the second leftmost top subplot
    ax_legend = fig.get_axes()[1]
    ax_legend.legend(handles, handles_labels, loc='upper left', fontsize=18)
    
    save_or_show(fig, filename)


def traceplot(chain, labels=None, truths=None, filename=None):
    """
    Create trace plots for MCMC chains.

    Args:
        chain (np.ndarray): MCMC chain of shape (nsteps, nwalkers, nleaves, ndim).
        labels (list, optional): List of parameter names for x-axis labels. 
        truths (array-like, optional): True parameter values to overlay as horizontal lines.
        filename (str, optional): If provided, saves the figure to this filename.

    Returns:
        matplotlib.figure.Figure: The trace plot figure. If `filename` is provided, the figure is saved instead.
    """

    nsteps, nwalkers, nleaves, ndim = chain.shape
    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    
    for i in range(ndim):
        for w in range(nwalkers):
            axs[i].plot(chain[:, w, :, i], alpha=0.5, rasterized=True)
            if truths is not None:
                truths = np.atleast_2d(truths)
                for j in range(truths.shape[0]):
                    axs[i].axhline(truths[j, i], color='k', linestyle='--')
            if labels is not None:
                axs[i].set_ylabel(labels[i])

    axs[-1].set_xlabel('Step')
    plt.tight_layout()

    save_or_show(fig, filename)


def plot_loglikelihood(logl, filename=None):
    nsteps, nwalkers = logl.shape
    fig = plt.figure(figsize=(10, 6))
    for j in range(nwalkers):
        plt.plot(logl[:, j], color=f"C{j % 10}", alpha=0.8, rasterized=True)
    plt.xlabel("Sampler Iteration")
    plt.ylabel("Log-Likelihood")
    
    save_or_show(fig, filename)

    # plot a facet grid of loglikelihood evolution for each walker.
    # This is a *diagnostic* and must never abort a (multi-hour) run: with few
    # walkers/steps and large per-walker logL offsets the small facets can make
    # matplotlib's tight_layout produce a NaN axis length ("cannot convert float
    # NaN to integer"). Guard the whole block so a layout failure is skipped, not
    # fatal; the primary log-likelihood plot above is already saved.
    grid = None
    try:
        # Reshape: logl is (nsteps, nwalkers), need to flatten properly
        max_logl = np.max(logl, axis=(0,1))
        facet_logl = logl - max_logl

        step = np.tile(range(nsteps), nwalkers)
        walker = np.repeat(np.arange(nwalkers, dtype=int), nsteps)

        df = pd.DataFrame(np.c_[facet_logl.flat, step, walker],
                          columns=[r"$\Delta \log\mathcal{L}$", "step", "walker"])

        # Initialize a grid of plots with an Axes for each walker
        grid = sns.FacetGrid(df, col="walker", hue="walker", #palette="tab20c",
                         col_wrap=int(np.floor(np.sqrt(nwalkers))), height=1.5)

        # Draw a line plot to show the trajectory of each random walk
        grid.map(plt.plot, "step", r"$\Delta \log\mathcal{L}$", marker=".", rasterized=True)

        grid.refline(y=0, linestyle=":") # Add a horizontal reference line at y=0 ~ average loglikelihood at each step

        # Adjust the arrangement of the plots
        grid.set_titles(col_template="Walker {col_name:.0f}")
        grid.set_axis_labels("Step", r"$\Delta \log\mathcal{L}$")
        # Disable tight_layout to avoid warning
        grid.tight_layout = lambda *args, **kwargs: None
        grid.tight_layout()

        # add overall title
        plt.subplots_adjust(top=0.9)
        grid.figure.suptitle(r"$\Delta \log\mathcal{L}_w = \log\mathcal{L}_w - \max(\log\mathcal{L})$", fontsize=16)

        save_or_show(grid.figure, filename.replace('.png', '_facet.png') if filename else None)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Skipping per-walker log-likelihood facet plot: %s: %s", type(e).__name__, e
        )
        if grid is not None:
            plt.close(grid.figure)

def tempering_ridgeplot(chain, labels=None, palette=None, 
                        bw_adjust=0.5, aspect=5, height=0.5, hspace=-0.25,
                        max_samples=10000, filename=None):
    """
    Create ridge plots of tempered distributions using overlapping KDE plots for all parameters.

    This creates a visually appealing ridge plot (also known as joy plot) showing
    how the posterior distribution broadens at higher temperatures. Each temperature
    level is shown as a separate row with overlapping density estimates.
    All parameters are shown as columns in a single FacetGrid figure.

    Args:
        chain (np.ndarray): MCMC chain of shape (nsteps, ntemps, nwalkers, nleaves, ndim).
        labels (list, optional): List of parameter names. If provided, uses labels for column titles.
        palette (str or list, optional): Seaborn color palette name or list of colors. 
            Default uses cubehelix_palette.
        bw_adjust (float, optional): Bandwidth adjustment factor for KDE. Default is 0.5.
        aspect (float, optional): Aspect ratio of each facet. Default is 5.
        height (float, optional): Height of each temperature row in inches. Default is 0.5.
        hspace (float, optional): Vertical spacing between temperature rows (negative for overlap). Default is -0.25.
        max_samples (int, optional): Maximum number of samples per temperature for KDE. Default is 5000.
            Subsampling speeds up KDE computation for large chains.
        filename (str, optional): If provided, saves figure to this filename.
        
    Returns:
        matplotlib.figure.Figure: The figure containing all ridge plots.
            If `filename` is provided, the figure is saved instead.
    """
    # Use seaborn context manager to temporarily set theme
    with sns.axes_style("white", {"axes.facecolor": (0, 0, 0, 0)}):

        nsteps, ntemps, nwalkers, nleaves, ndim = chain.shape
        
        # Create color palette (blue=cold/β=1 at top, red=hot/β→0 at bottom)
        if palette is None:
            # Use coolwarm reversed: blue for cold (β=1), red for hot (β→0)
            pal = sns.color_palette(DEFAULT_PALETTE, ntemps)
        elif isinstance(palette, str):
            pal = sns.color_palette(palette, ntemps)
        else:
            pal = palette
        
        # Subsampling RNG
        rng = np.random.default_rng(42)
        
        # Build dataframe with samples from all temperatures and parameters
        data_list = []
        for param_idx in range(ndim):
            param_label = labels[param_idx] if labels is not None else fr'$x_{param_idx}$'
            for t in range(ntemps):
                # Flatten samples across steps, walkers, leaves for the selected parameter
                samples = chain[:, t, :, :, param_idx].reshape(-1)
                
                # Remove NaNs
                samples = samples[~np.isnan(samples)]
                
                # Subsample if needed for faster KDE
                if len(samples) > max_samples:
                    samples = rng.choice(samples, size=max_samples, replace=False)
                
                temp_label = rf"$\beta_{{{t}}}$" #fr"$\beta$={betas[t]:.1e}"
                data_list.append(pd.DataFrame({
                    'x': samples,
                    'temp': temp_label,
                    'temp_idx': t,
                    'param': param_label,
                    'param_idx': param_idx
                }))
        
        df = pd.concat(data_list, ignore_index=True)
        
        # Get unique temps in order (beta=1 first at top)
        temp_order = df.drop_duplicates('temp_idx').sort_values('temp_idx', ascending=True)['temp'].tolist()
        
        # Get unique params in order
        param_order = df.drop_duplicates('param_idx').sort_values('param_idx', ascending=True)['param'].tolist()
        
        # Compute x-axis limits from the cold posterior (β=1, temp_idx=0) for each parameter
        # This ensures the cold posterior is always visible even when hot distributions are much wider
        xlims = {}
        cold_df = df[df['temp_idx'] == 0]
        for param in param_order:
            param_data = cold_df[cold_df['param'] == param]['x']
            q_low, q_high = param_data.quantile([0.001, 0.999])
            margin = (q_high - q_low) * 0.3  # Add 30% margin to show some broadening
            xlims[param] = (q_low - margin, q_high + margin)
        
        # Initialize the FacetGrid with row=temp, col=param
        # Suppress tight_layout warnings during initialization
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Tight layout.*")
            g = sns.FacetGrid(df, row="temp", col="param", hue="temp", 
                            aspect=aspect, height=height, 
                            palette=pal, row_order=temp_order, col_order=param_order,
                            sharex=False, sharey=False)
        
        # Disable tight_layout to avoid warning with negative hspace
        g.tight_layout = lambda *args, **kwargs: None
        
        # Custom plotting function that clips KDE to the parameter's xlim
        def plot_kde_clipped(x, color, label, **kwargs):
            ax = plt.gca()
            # Get the parameter for this column
            col_idx = ax.get_subplotspec().colspan.start
            param = param_order[col_idx]
            clip = xlims[param]
            
            sns.kdeplot(x, ax=ax, bw_adjust=bw_adjust, clip_on=False,
                        fill=True, alpha=1, linewidth=1.5, color=color, clip=clip)
            sns.kdeplot(x, ax=ax, clip_on=False, color="w", lw=2, bw_adjust=bw_adjust, clip=clip)
            ax.set_xlim(clip)
        
        g.map(plot_kde_clipped, "x")
        
        # Add horizontal reference line at y=0
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        
        # Label function for temperature labels (only on first column)
        def label_temp(x, color, label):
            ax = plt.gca()
            # Only add label if this is the first column
            if ax.get_subplotspec().colspan.start == 0:
                ax.text(-0.2, 0.2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax.transAxes, fontsize=14)
        
        g.map(label_temp, "x")
        
        # Set subplots to overlap vertically
        g.figure.subplots_adjust(hspace=hspace, wspace=0.1)

        # use parameter labels in the x axes (on the last row)
        for ax, param in zip(g.axes[-1], param_order):
            ax.set_xlabel(param, fontsize=14)

        # Remove xticks from all rows except the bottom
        for row_idx in range(len(temp_order) - 1):
            for col_idx in range(len(param_order)):
                g.axes[row_idx, col_idx].set_xticks([])

                # set xlims again to ensure consistency
                param = param_order[col_idx]
                g.axes[row_idx, col_idx].set_xlim(xlims[param])
        
        # Remove axes details that don't work well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        # Add overall title
        g.figure.suptitle('Tempered Distributions', y=1.02, fontsize=14)
        
        save_or_show(g.figure, filename)

    # # Properly restore the original seaborn style
    # sns.set_theme(style=original_style['style'], rc=original_style)
    
    # return g.figure

def plot_swap_acceptance(swap_acceptance_fraction, palette=None, filename=None):
    """
    Plot the temperature swap acceptance fraction between adjacent temperature levels.

    Args:
        swap_acceptance_fraction (np.ndarray): Swap acceptance fraction between adjacent 
            temperatures, shape (ntemps-1,). Element i corresponds to swaps between 
            temperature i and i+1.
        palette (str or list, optional): Seaborn color palette name or list of colors.
        filename (str, optional): If provided, saves the figure to this filename.

    Returns:
        matplotlib.figure.Figure: The swap acceptance plot figure.
    """
    ntemps = swap_acceptance_fraction.shape[0] + 1
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # X-axis: temperature pair indices (swap between temp i and i+1)
    x = np.arange(ntemps - 1)
    
    # Create labels for each swap pair
    labels = [fr'{i}$\leftrightarrow${i+1}' for i in range(ntemps - 1)]
    
    # Color by temperature (use coolwarm to match ridgeplot)
    palette = palette if palette is not None else DEFAULT_PALETTE
    colors = sns.color_palette(palette, ntemps - 1)
    
    # Bar plot
    bars = ax.bar(x, swap_acceptance_fraction, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add horizontal reference line at common target acceptance (0.2-0.4 is often good)
    ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0.25')
    
    # Add beta values as secondary labels
    # ax2 = ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(x)
    # beta_labels = [fr'$\beta$={betas[i]:.1e}$\leftrightarrow${betas[i+1]:.1e}' for i in range(ntemps - 1)]
    # ax2.set_xticklabels(beta_labels, fontsize=8, rotation=45, ha='left')
    
    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_xlabel('Temperature Pair (index)', fontsize=12)
    ax.set_ylabel('Swap Acceptance Fraction', fontsize=12)
    ax.set_title('Temperature Swap Acceptance', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add value annotations on bars
    for bar, val in zip(bars, swap_acceptance_fraction):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_or_show(fig, filename)
    
    return fig
    
def plot_logl_betas(betas: np.ndarray, 
                    logl: np.ndarray, 
                    palette: str = None,
                    filename: str = None
                    ):
        """
        Plots the evolution of log-likelihood values for each temperature.

        Args:
            betas (numpy.ndarray): Array of inverse temperatures.
            logl (numpy.ndarray): Array of log-likelihood values.
            palette (str, optional): Seaborn color palette name or list of colors.
            filename (str, optional): If provided, saves the figure to this filename.
        Returns:
            None
        """
        fig = plt.figure(figsize=(10, 6))
        ntemp = betas.shape[1]
        tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemp)
        for temp in range(ntemp):
            plt.semilogx(betas[-1, temp], np.mean(logl[:, temp]), '.', c=tempcolors[temp], label=f'$T_{temp}$')

        logZ, dlogZ = stepping_stone_log_evidence(betas[-1], logl)
        
        plt.ylabel(r'$<\log{\mathcal{L}}>_{\beta}$')
        plt.xlabel(r'$\beta$')
        plt.title(r'$\log{\mathcal{Z}} = %.2f \pm %.2f$' % (logZ, dlogZ))
        
        save_or_show(fig, filename)

def plot_betas_evolution(betas: np.ndarray, palette: str = None, filename: str = None):
    """
    Plots the evolution of inverse temperatures (betas) over sampling steps.

    Args:
        betas (numpy.ndarray): Array of inverse temperatures of shape (nsteps, ntemps).
        palette (str, optional): Seaborn color palette name or list of colors.
        filename (str, optional): If provided, saves the figure to this filename.
    Returns:
        None
    """
    nsteps, ntemps = betas.shape
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)
    fig = plt.figure(figsize=(10, 6))
    for temp in range(ntemps):
        plt.plot(range(nsteps), betas[:, temp], color=tempcolors[temp], linewidth=1.5, alpha=0.8, rasterized=True)
    plt.xlabel('Sampler Iteration')
    plt.ylabel(r'Inverse Temperature ($\beta$)')
    plt.title('Evolution of Inverse Temperatures')
    
    # Create temperature color gradient legend
    
    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / ntemps, legend_y),
            legend_width / ntemps, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)
    
    save_or_show(fig, filename)


# RJ plots
def plot_leaves(nleaves: np.ndarray,
                nleaves_min: int,
                nleaves_max: int,
                palette: str = None,
                iteration: int = 0,
                filename: str = None):
    """
    Plot the histogram of the number of leaves for each temperature.

    This method plots a histogram of the number of leaves for each temperature in the `rj_branches` dictionary.
    It uses the `self.backend` object to get the number of leaves for each temperature.
    The histogram is plotted using the `plt.hist` function from the `matplotlib.pyplot` module.
    The plot includes temperature-specific colors and a legend for the colors.

    Returns:
    None
    """
    bns = (np.arange(nleaves_min, nleaves_max + 2) - 0.5)
    ntemps = nleaves.shape[1]
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)

    fig = plt.figure(figsize=(8, 5))

    for temp, tempcolor in enumerate(tempcolors):
        plt.hist(nleaves[:, temp].flatten(), bins=bns, histtype="stepfilled", edgecolor=tempcolor,
                    facecolor=to_rgba(tempcolor, 0.2), density=True, ls='-', zorder=100 - temp, rasterized=True)

    plt.xlabel('Number of leaves')
    plt.ylabel('Density')
    
    # Create temperature color gradient legend
    # Use a horizontal gradient showing cold (blue) to hot (red)
    
    # Add a color bar as legend showing temperature progression
    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    n_gradient = len(tempcolors)
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / n_gradient, legend_y),
            legend_width / n_gradient, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)

    fig.text(0.07, 0.08, f"Step: {iteration}", ha='left', va='top', fontfamily='serif', c='k')
    #plt.title(key)
    save_or_show(fig, filename)

def plot_leaves_evolution(nleaves: np.ndarray,
                          filename: str = None):
    """
    Plot the evolution of the number of leaves per walker in the cold chain over sampling steps.
    Args:
        nleaves (np.ndarray): Array of number of leaves in the cold chain, shape (nsteps, nwalkers).
        filename (str, optional): If provided, saves the figure to this filename.
    Returns:
        None
    """
    nsteps, nwalkers = nleaves.shape
    fig = plt.figure(figsize=(10, 6))
    for w in range(nwalkers):
        plt.plot(range(nsteps), nleaves[:, w], color=f"C{w % 10}", linewidth=1.5, alpha=0.8, rasterized=True)
    plt.xlabel('Sampler Iteration')
    plt.ylabel('Number of Leaves')
    plt.title('Evolution of Number of Leaves in Cold Chain')
    save_or_show(fig, filename)

def plot_acceptance_fraction(steps: typing.Union[np.ndarray, list],
                             total_acceptance_fraction: np.ndarray,
                             moves_acceptance_fraction: dict,
                             moves_steps: dict = None,
                             rate_label: str = 'Acceptance Fraction',
                             filename: str = None):
    """Plot the acceptance fraction of every move in the tree over sampling steps.

    Args:
        steps (np.ndarray or list): Sampling steps for the total curve.
        total_acceptance_fraction (np.ndarray): Total acceptance fraction,
            shape ``(nsteps, ntemps, nwalkers)``.
        moves_acceptance_fraction (dict): ``path -> (nsteps, ntemps, nwalkers)``
            rates, keyed by the ``/``-separated paths from
            :func:`eryn.utils.utility.walk_moves`. Nesting depth is styled with
            the line style and a lighter shade of the top-level move's hue.
        moves_steps (dict, optional): ``path -> (nsteps,)`` steps. Each path may
            have its own history length, so it cannot share ``steps``. If
            ``None``, ``steps`` is used for every path.
        rate_label (str, optional): Y-axis label. Default ``'Acceptance Fraction'``.
        filename (str, optional): If provided, saves the figure to this filename.

    """
    fig = plt.figure(figsize=(10, 6))
    # cold chain total acceptance fraction
    plt.plot(steps, total_acceptance_fraction[:, 0].mean(axis=1), label='Total',
             color='black', linewidth=2)

    if len(moves_acceptance_fraction) != 0:
        paths = list(moves_acceptance_fraction)
        colors = move_tree_colors(paths)
        linestyles = ['-', '--', ':', '-.']

        for path in paths:
            depth = path.count('/')
            x = steps if moves_steps is None else moves_steps[path]
            # Leading spaces alone don't distinguish depths: text.usetex
            # collapses them, so e.g. depth 1 and depth 2 would render as the
            # same '- <name>' legend entry. Repeat the marker instead, which
            # survives LaTeX rendering.
            label = '- ' * depth + path.split('/')[-1]

            plt.plot(x,
                     moves_acceptance_fraction[path][:, 0].mean(axis=1),
                     marker='o',
                     markersize=3,
                     color=colors[path],
                     linestyle=linestyles[depth % len(linestyles)],
                     alpha=max(0.4, 1.0 - 0.15 * depth),
                     label=_tex_safe(label))

    plt.axhline(y=0.234, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0.234')
    plt.legend(fontsize=9)
    plt.xlabel('Sampler Iteration')
    plt.ylabel(_tex_safe(rate_label))
    plt.title('Acceptance Fraction Over Time')

    save_or_show(fig, filename)


def plot_move_tree_acceptance(moves_acceptance_fraction: dict,
                               moves_steps: dict,
                               parent_folder: str = ".",
                               rate_label: str = "Acceptance Fraction"):
    """Write one acceptance figure per wrapper move, showing its direct children.

    A path is a wrapper when another path starts with ``path + "/"``. Output
    lands in ``<parent_folder>/moves/<path>/acceptance_fraction.png``, so the
    folder tree mirrors the move tree and documents how the moves are nested.

    Args:
        moves_acceptance_fraction (dict): ``path -> (nsteps, ntemps, nwalkers)``
            rates.
        moves_steps (dict): ``path -> (nsteps,)`` steps.
        parent_folder (str, optional): Folder to write into. Default is the
            current directory.
        rate_label (str, optional): Y-axis label.

    """
    paths = list(moves_acceptance_fraction)

    for path in paths:
        prefix = path + "/"
        children = [
            other for other in paths
            if other.startswith(prefix) and "/" not in other[len(prefix):]
        ]
        if not children:
            continue

        folder = os.path.join(parent_folder, "moves", *path.split("/"))
        os.makedirs(folder, exist_ok=True)

        colors = move_tree_colors(children)

        fig = plt.figure(figsize=(10, 6))
        for child in children:
            plt.plot(moves_steps[child],
                     moves_acceptance_fraction[child][:, 0].mean(axis=1),
                     marker="o",
                     markersize=3,
                     color=colors[child],
                     label=_tex_safe(child.split("/")[-1]))

        plt.axhline(y=0.234, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                    label="0.234")
        plt.legend(fontsize=10)
        plt.xlabel("Sampler Iteration")
        plt.ylabel(_tex_safe(rate_label))
        plt.title(_tex_safe("Acceptance Fraction - " + path.split("/")[-1]))

        save_or_show(fig, os.path.join(folder, "acceptance_fraction.png"))


def plot_tempered_acceptance_fraction(steps: typing.Union[np.ndarray, list],
                            total_acceptance_fraction: np.ndarray,
                            palette: str = None,
                            filename: str = None):
    """
    Plot the acceptance fraction for different moves over sampling steps.

    Args:
        steps (np.ndarray or list): Array of sampling steps.
        total_acceptance_fraction (np.ndarray): Array of total acceptance fractions, shape (nsteps, ntemps, nwalkers).
        palette (str or list, optional): Seaborn color palette name or list of colors.
        filename (str, optional): If provided, saves the figure to this filename.
    """
    ntemps = total_acceptance_fraction.shape[1]
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)

    fig = plt.figure(figsize=(10, 6))
    
    for temp in range(ntemps):
        plt.plot(steps, total_acceptance_fraction[:, temp].mean(axis=1), color=tempcolors[temp], linewidth=1.5, marker='o', alpha=0.8, rasterized=True)

    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / ntemps, legend_y),
            legend_width / ntemps, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)

    plt.xlabel('Sampler Iteration')
    plt.ylabel('Acceptance Fraction')  

    ymin, ymax = plt.ylim()
    plt.ylim(ymin, 1.2 * ymax)

    plt.title('Acceptance Fraction Over Time')

    save_or_show(fig, filename)

def plot_act_evolution(chain: dict,
                       iteration: int = 0,
                       parent_folder: str = '.'):
    
    """
    Plot the evolution of the autocorrelation time for each branch in the chain. Also plots the ACT values per parameter in each branch.

    Args:
        chain (Dict): Dictionary of MCMC chains for different branches.
        iteration (int, optional): Current iteration number for labeling. Default is 0.
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
    """

    NPOINTS = 10
    points = np.exp(np.linspace(np.log(min(100, iteration)), np.log(iteration), NPOINTS)).astype(int)
    
    taus = {}
    for branch, samples in chain.items():
        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        nsteps, ntemps, nwalkers, nleaves, ndim = samples.shape
        cold_chain = samples[:, 0, :, :, :].reshape(nsteps, 1, nwalkers, nleaves * ndim)
        #cold_chain = cold_chain[~np.isnan(cold_chain).any(axis=-1)] # remove NaNs
        if np.isnan(cold_chain).any(axis=-1).any():
            print(f"Skipping ACT plot for branch {branch} due to NaNs in the cold chain.")
            continue
        tmp = []
        for i, point in enumerate(points):
            if point > cold_chain.shape[0]:
                continue
            
            tau = get_integrated_act(cold_chain[:point], average=True)
            tmp.append(tau.squeeze())

        taus[branch] = np.array(tmp)
        
        fig = plt.figure(figsize=(10, 6))
        for d in range(ndim):
            plt.loglog(points[:len(taus[branch])], taus[branch][:, d], marker='o', label=fr'$x_{d}$')

        # add tau = iteration / 50 line for reference
        xaxis = np.logspace(0, np.log10(iteration), 100)
        #xaxis[0] = 1  # avoid plotting tau=0 at step 0

        plt.loglog(xaxis, xaxis / 50, linestyle='--', color='k', lw=2, label=r'$\tau$ = Nsteps / 50')
        plt.xlabel('Number of Steps')
        plt.ylabel('Integrated Autocorrelation Time')
        plt.title(f'Autocorrelation Time Evolution - Branch: {branch}')

        ymax = np.max(taus[branch]) * 1.2
        ymin = np.min(taus[branch]) * 0.5
        if ymin < ymax:
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(0, ymax)
        plt.xlim(min(points) * 0.9, iteration * 1.1)
        plt.legend()
        save_or_show(fig, os.path.join(branch_folder, f'act_evolution.png'))
    
        # plot the maximum ACT across all parameters for each branch
    if len(taus) > 0:
        fig = plt.figure(figsize=(10, 6))
        for branch in taus.keys():
            max_tau = np.max(taus[branch], axis=1)
            plt.loglog(points[:len(max_tau)], max_tau, marker='o', label=f'Branch: {branch}')
        
        plt.loglog(xaxis, xaxis / 50, linestyle='--', color='k', lw=2, label=r'$\tau$ = Nsteps / 50')
        plt.xlabel('Number of Steps')
        plt.ylabel('Maximum Integrated Autocorrelation Time')
        plt.title('Maximum Autocorrelation Time Evolution Across Branches')
        ymax = max([np.max(np.max(taus[branch], axis=1)) for branch in taus.keys()]) * 1.2
        ymin = min([np.min(np.max(taus[branch], axis=1)) for branch in taus.keys()]) * 0.5
        if ymin < ymax:
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(0, ymax)
        plt.xlim(min(points) * 0.9, iteration * 1.1)
        plt.legend()
        save_or_show(fig, os.path.join(parent_folder, f'max_act_evolution.png'))
            

def produce_base_plots(chain: dict, 
                       logl: np.ndarray, 
                       truths: dict = None, 
                       overlay_covariance: dict = None, 
                       labels: dict = None, 
                       iteration: int = 0,
                       parent_folder: str = '.',
                       ):

    """
    Produce a set of standard diagnostic plot. These include:
    
    * corner plots for the cold chain per branch,
    * trace plots for the cold chain per branch,
    * log-likelihood evolution plots.

    Args:
        chain (Dict): Dictionary of MCMC chains for different branches.
        logl (np.ndarray): Log-likelihood array of shape (nsteps, ntemperatures, nwalkers).
        truths (Dict, optional): Dictionary of true parameter values for different branches.
        overlay_covariance (Dict, optional): Dictionary of covariance matrices to overlay on corner plots.
        labels (Dict, optional): Dictionary of parameter labels for different branches.
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
    """

    # create dictionary with a color shade for each branch
    branches = list(chain.keys())
    legend_label = 'Samples at step %d' % iteration

    palette = 'Blues'
    colors = sns.color_palette(palette, n_colors=len(branches))
    #colors = sns.color_palette("mako", n_colors=len(branches)+2)[2:]  # avoid very light colors
    branch_colors = {branch: colors[i] for i, branch in enumerate(branches)}

    for branch, samples in chain.items():
        branch_labels = labels.get(branch, None) if labels else None
        branch_truths = truths.get(branch, None) if truths else None
        branch_cov = overlay_covariance.get(branch, None) if overlay_covariance else None

        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        nsteps, ntemps, nwalkers, nleaves, ndim = samples.shape
        cold_chain = samples[:, 0, :, :, :].reshape(-1, ndim)
        cold_chain = cold_chain[~np.isnan(cold_chain).any(axis=1)] # remove NaNs
        if cold_chain.shape[0] == 0:
            print(f"Skipping plots for branch {branch} due to no valid samples in the cold chain.")
            continue

        cornerplot(
            cold_chain,
            means=branch_truths,
            overlay_covariance=branch_cov,
            legend_label=legend_label,
            truths=branch_truths,
            overlay_label='Information Matrix Covariance' if branch_cov is not None else None,
            labels=branch_labels,
            color=branch_colors[branch],
            filename=os.path.join(branch_folder, f'cornerplot.png')
        )

        traceplot(
            samples[:, 0, :, :, :],
            labels=branch_labels,
            truths=branch_truths,
            filename=os.path.join(branch_folder, f'traceplot.png')
        )

    plot_loglikelihood(
        logl[:, 0, :],
        filename=os.path.join(parent_folder, f'loglikelihood.png')
    )

def produce_tempering_plots(chain: dict, 
                            betas: np.ndarray,
                            logl: np.ndarray,
                            swap_acceptance_fraction: np.ndarray,
                            labels: dict = None, 
                            parent_folder: str = '.',
                            palette: str = None
                            ):
    """
    Produce tempering ridge plots for each branch in the chain. These include:

    * ridge plots of the tempered distributions per parameter per branch,
    * the swap acceptance fraction between adjacent temperatures,
    * the averaged log-likelihood vs. betas plot,
    * the evolution of betas over sampling steps.

    Args:
        chain (Dict): Dictionary of MCMC chains for different branches.
        betas (np.ndarray): Inverse temperatures of shape (nsteps, ntemps,).
        swap_acceptance_fraction (np.ndarray): Swap acceptance fraction between adjacent temperatures.
        labels (Dict, optional): Dictionary of parameter labels for different branches.
        logl (np.ndarray): Log-likelihood array of shape (nsteps, ntemperatures, nwalkers).
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
        palette (str or list, optional): Seaborn color palette name or list of colors.
    """

    for branch, samples in chain.items():
        branch_labels = labels.get(branch, None) if labels else None

        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        tempering_ridgeplot(
            samples,
            labels=branch_labels,
            palette=palette,
            filename=os.path.join(branch_folder, f'tempering_ridgeplot.png')
        )

    plot_swap_acceptance(
        swap_acceptance_fraction,
        palette=palette,
        filename=os.path.join(parent_folder, f'swap_acceptance.png')
    )

    plot_logl_betas(
        betas,
        logl,
        palette=palette,
        filename=os.path.join(parent_folder, f'logl_betas.png')
    )

    plot_betas_evolution(
        betas,
        palette=palette,
        filename=os.path.join(parent_folder, f'betas_evolution.png')
    )

def produce_advanced_plots(steps: typing.Union[np.ndarray, list],
                           total_acceptance_fraction: np.ndarray,
                           moves_acceptance_fraction: dict,
                           moves_steps: dict = None,
                           rate_label: str = 'Acceptance Fraction',
                           palette: str = None,
                           iteration: int = 0,
                           chain: dict = None,
                           parent_folder: str = '.'):
    """
    Produce advanced diagnostic plots. These include:

    * autocorrelation time evolution per parameter per branch in the cold chain,
    * the comparison of the maximum autocorrelation  time in each branch against the number of steps,
    * the acceptance fraction evolution over steps in the cold chain (both overall and per move),
    * the overall acceptance fraction evolution over steps per temperature.

    Args:
        steps (Union[np.ndarray, list]): Array or list of sampling steps.
        total_acceptance_fraction (np.ndarray): Total acceptance fraction array of shape (nsteps, ntemps, nwalkers).
        moves_acceptance_fraction (Dict): Dictionary of acceptance fractions for different moves.
        moves_steps (Dict, optional): Per-move-path sampling steps. Each path can
            have its own history length, so it cannot share ``steps``.
        rate_label (str, optional): Y-axis label for the acceptance plots.
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
    """

    plot_acceptance_fraction(
        steps,
        total_acceptance_fraction,
        moves_acceptance_fraction,
        moves_steps=moves_steps,
        rate_label=rate_label,
        filename=os.path.join(parent_folder, f'acceptance_fraction.png')
    )

    plot_move_tree_acceptance(
        moves_acceptance_fraction,
        moves_steps
        if moves_steps is not None
        else {path: steps for path in moves_acceptance_fraction},
        parent_folder=parent_folder,
        rate_label=rate_label,
    )

    plot_tempered_acceptance_fraction(
        steps,
        total_acceptance_fraction,
        palette=palette,
        filename=os.path.join(parent_folder, f'tempered_acceptance_fraction.png')
    )

    plot_act_evolution(
        chain,
        iteration=iteration,
        parent_folder=parent_folder
    )

def produce_rj_plots(nleaves: dict,
                     nleaves_min: dict,
                     nleaves_max: dict,
                     palette: str = None,
                     parent_folder: str = '.',
                     iteration: int = 0):
    
    """
    Produce RJ diagnostic plots for each branch in the chain. At present, only plots the histogram of the number of leaves across temperatures.
    
    Args:
        nleaves (Dict): Dictionary of number of leaves arrays for different branches.
        nleaves_min (Dict): Dictionary of minimum number of leaves for different branches.
        nleaves_max (Dict): Dictionary of maximum number of leaves for different branches.
        palette (str or list, optional): Seaborn color palette name or list of colors.
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
        iteration (int, optional): Current iteration number for labeling plots.
    """

    for branch, leaves in nleaves.items():
        branch_nleaves_min = nleaves_min.get(branch, 1)
        branch_nleaves_max = nleaves_max.get(branch, 1)

        # skip if there is no reversible-jump sampling for this branch
        if branch_nleaves_max <= branch_nleaves_min:
            continue
        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)
        
        plot_leaves(
            leaves,
            branch_nleaves_min,
            branch_nleaves_max,
            palette=palette,
            iteration=iteration,
            filename=os.path.join(branch_folder, f'rj_nleaves.png')
        )

        # plot_leaves_evolution(
        #     leaves[:, 0],
        #     filename=os.path.join(branch_folder, f'rj_nleaves_evolution.png')
        # )
    
    


class PlotContainer:
    """
    An Update that generates diagnostic plots at specified intervals
    
    Args:
        plots (list or str): List of plot types to generate. Options are 'base', 'tempering', 'advanced', 'rj', or 'all'.  If multiple plot types are desired, provide a list of strings.
        branches (list, optional): List of branch names to generate plots for. If None, all branches are used.
        truths (dict, optional): Dictionary of true parameter values for different branches.
        overlay_covariance (dict, optional): Dictionary of covariance matrices to overlay on corner plots.
        tempering_palette (str or list, optional): Seaborn color palette name or list of colors for tempering plots. If None, it defaults to `icefire`.
        parent_folder (str, optional): Folder to save the plots. Default is current directory.
        discard (float, optional): Number of initial samples to discard from the chain before plotting. If between 0 and 1, it is treated as fraction of total samples. Default is 0.
        move_rate_mode (str, optional): ``'interval'`` plots each move's acceptance
            within the interval since the previous plot, ``'cumulative'`` plots it
            since the start of the run. Default is ``'interval'``.
        stop (int, optional): Maximum number of steps to generate plots for. Default is 10000.
    """
    
    def __init__(self, 
                 backend: Backend = None,
                 plots : typing.Union[list, str] = 'base',
                 branches : list = None,
                 truths: dict = None,
                 overlay_covariance: dict = None,
                 tempering_palette: str = None,
                 parent_folder: str = '.',
                 discard: float = 0,
                 move_rate_mode: str = "interval",
                 stop: int = int(1e4),
                 ):
        """
        Initialize the PlotContainer.
        """
        setup_plotting()
        self.backend = backend

        self.parent_folder = parent_folder
        os.makedirs(self.parent_folder, exist_ok=True)

        allowable_plots = ['base', 'tempering', 'advanced', 'rj']
        self.branches = branches

        if isinstance(plots, str):
            if plots == 'all':
                plots = allowable_plots
            else:
                plots = [plots]

        for plot in plots:
            if plot not in allowable_plots:
                raise ValueError(f"Plot type '{plot}' not recognized. Allowable types: {allowable_plots}")
        self.plots = plots
        
        #self.labels = labels
        self.truths = truths
        self.overlay_covariance = overlay_covariance
        self.discard = discard
        self.tempering_palette = tempering_palette

        self.steps = []
        self.total_acceptance_fraction = None
        # counters, not ratios: storing accepted/num_proposals separately lets
        # either a per-interval or a cumulative rate be produced at plot time
        self.move_accepted = {}
        self.move_num_proposals = {}
        self.move_steps = {}
        self.move_rate_mode = move_rate_mode

        self.stop = stop

    @property
    def backend(self):
        return self._backend
    @backend.setter
    def backend(self, value):
        self._backend = value

    @property
    def truths(self):
        return self._truths
    @truths.setter
    def truths(self, value):
        self._truths = value

    @property
    def overlay_covariance(self):
        return self._overlay_covariance
    @overlay_covariance.setter
    def overlay_covariance(self, value):
        self._overlay_covariance = value

    def _rates_by_path(self, mode):
        """Acceptance rates per move path in the requested mode."""
        return {
            path: move_acceptance_rates(
                np.array(accepted),
                np.array(self.move_num_proposals[path]),
                mode=mode,
            )
            for path, accepted in self.move_accepted.items()
        }

    @property
    def move_acceptance_fractions(self):
        """Cumulative acceptance fraction per move path, derived from the counters."""
        return self._rates_by_path("cumulative")

    def move_rates(self):
        """Acceptance rates and steps per move path, in ``self.move_rate_mode``.

        Returns:
            tuple: ``(rates, steps)``, both dicts keyed by move path. ``rates``
                values have shape ``(nsteps, ntemps, nwalkers)``.

        """
        return (
            self._rates_by_path(self.move_rate_mode),
            {path: np.array(step) for path, step in self.move_steps.items()},
        )

    def _collect_move_acceptance(self, moves):
        """Record the acceptance counters of every move in the tree.

        Each path keeps its own step list. A move whose counters are not yet
        initialised is skipped, so its history is shorter than ``self.steps``
        and plotting it against that shared list would misalign it.
        """
        for path, move in walk_moves(moves):
            counters = move_counters(move)
            if counters is None:
                continue

            accepted, num_proposals = counters
            self.move_accepted.setdefault(path, []).append(accepted)
            self.move_num_proposals.setdefault(path, []).append(num_proposals)
            self.move_steps.setdefault(path, []).append(self.backend.iteration)

    def produce_plots(self, sampler=None) -> None:
        """
        Generate diagnostic plots at specified intervals.

        Args:
            sampler: The sampler object. If not provided, uses self.backend. In this cases not all the plots could be available.
        Returns:
            None
        """

        if getattr(self.backend, "nsamplers", 1) > 1:
            raise NotImplementedError(
                "PlotContainer does not support nsamplers > 1; plot per sampler "
                "with get_chain(sampler_index=...)."
            )

        if self.backend.iteration > self.stop:
            return

        labels = self.backend.key_order
    
        discard = int(self.discard) if self.discard >= 1 else int(self.discard * self.backend.iteration) 
        chain = self.backend.get_chain(discard=discard)
        logl = self.backend.get_log_like(discard=discard)
        betas = self.backend.get_betas(discard=discard)

        if self.branches is not None:
            chain = {branch: chain[branch] for branch in self.branches if branch in chain}

        for plot in self.plots:
            base_folder = os.path.join(self.parent_folder, plot)
            os.makedirs(base_folder, exist_ok=True)    

            if plot == 'base':           
                produce_base_plots(
                    chain=chain,
                    logl=logl,
                    truths=self.truths,
                    overlay_covariance=self.overlay_covariance,
                    iteration=self.backend.iteration,
                    labels=labels,
                    parent_folder=base_folder
                )
            elif plot == 'tempering':      
                swap_acceptance_fraction = self.backend.swaps_accepted / float(self.backend.iteration * self.backend.nwalkers)         
                produce_tempering_plots(
                    chain=chain,
                    betas=betas,
                    logl=logl,
                    swap_acceptance_fraction=swap_acceptance_fraction,
                    labels=labels,
                    parent_folder=base_folder,
                    palette=self.tempering_palette
                )

            elif plot == 'advanced':
                self.steps.append(self.backend.iteration)
                if self.total_acceptance_fraction is None:
                    self.total_acceptance_fraction = (self.backend.accepted / float(self.backend.iteration))[np.newaxis, ...]
                else:
                    self.total_acceptance_fraction = np.vstack((self.total_acceptance_fraction, (self.backend.accepted / float(self.backend.iteration))[np.newaxis, ...])) # shape (niterations, ntemps, nwalkers)
                
                if sampler is not None:
                    moves = sampler.moves
                elif hasattr(self.backend, "moves"):
                    moves = self.backend.moves
                else:
                    moves = None

                if moves is not None:
                    self._collect_move_acceptance(moves)

                full_chain = self.backend.get_chain(discard=0) if discard > 0 else chain

                move_rates, move_steps = self.move_rates()
                rate_label = (
                    'Acceptance Fraction (per interval)'
                    if self.move_rate_mode == 'interval'
                    else 'Acceptance Fraction (cumulative)'
                )

                produce_advanced_plots(steps=self.steps,
                                        total_acceptance_fraction=self.total_acceptance_fraction,
                                        moves_acceptance_fraction=move_rates,
                                        moves_steps=move_steps,
                                        rate_label=rate_label,
                                        palette=self.tempering_palette,
                                        iteration=self.backend.iteration,
                                        chain=full_chain,
                                        parent_folder=base_folder
                                    )

            elif plot == 'rj':
                if self.backend.rj is False:
                    continue

                if len(self.backend.rj_branches) == 0:
                    continue
    
                nleaves = self.backend.get_nleaves(discard=discard)

                nleaves_min = sampler.nleaves_min if (sampler is not None and hasattr(sampler, 'nleaves_min')) else dict(zip(self.backend.rj_branches, [0]*len(self.backend.rj_branches)))
                nleaves_max = self.backend.nleaves_max

                produce_rj_plots(
                    nleaves=nleaves,
                    nleaves_min=nleaves_min,
                    nleaves_max=nleaves_max,
                    palette=self.tempering_palette,
                    parent_folder=base_folder,
                    iteration=self.backend.iteration
                )
