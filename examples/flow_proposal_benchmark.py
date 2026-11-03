#!/usr/bin/env python
"""P0 go/no-go gate: does a frozen flow beat StretchMove AND a GMM baseline on
ESS-per-likelihood-eval for a realistic non-Gaussian target?

Usage::

    uv run python Eryn/examples/flow_proposal_benchmark.py --out p0_report.md
    uv run python Eryn/examples/flow_proposal_benchmark.py \\
        --chain chain_leaf.npy --periodic-index 5 --out p0_report.md

The synthetic target (no --chain) is a smoke/dev check. The real gate uses
--chain on a saved leaf cold-chain (e.g. one MBH leaf).

Ported from LISAanalysistools/scripts/run_p0_flow_benchmark.py.
API changes vs. original:
  - FlowModel(ndim, conditioning, normalizer, nsf_kwargs) ->
      ZukoFlow(dims, conditioning, data_transform, **nsf_kwargs)
  - SourceNormalizer / norm.build_transforms -> WhiteningTransform(...).fit(...)
    (WhiteningTransform is fitted lazily on first ZukoFlow.fit call, but we
    build it explicitly so we can pass it as data_transform)
  - train_flow(fm, {0: train}, max_epochs=N, batch_size=B, accelerator="auto") ->
      flow.fit({0: train}, n_epochs=N, batch_size=B)
    (ZukoFlow.fit uses Adam + optional LR schedule; no Lightning / accelerator)
  - hist['val_loss'][-1] -> hist.validation_loss[-1]  (FlowHistory dataclass)
  - fmove.active_condition = 0  (unchanged)
  - IndependentProposalMove now lives in eryn.moves (not in benchmark)
"""
from __future__ import annotations

import argparse

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.moves import ConditionalFlowMove, IndependentProposalMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer

from eryn.flows import OneHotLeafConditioning, ZukoFlow, WhiteningTransform
from eryn.flows.benchmark import (
    BenchmarkResult,
    EvalCountingTarget,
    GMMProposalDistribution,
    banana_periodic_target,
    chain_kde_target,
    effective_sample_size,
)


def _priors(ndim: int, bounds: dict) -> dict:
    return {"x": ProbDistContainer({d: uniform_dist(*bounds[d]) for d in range(ndim)})}


def _run(move, log_prob, ndim, periodic, bounds, train_samples, nwalkers, nsteps, burn, seed=0):
    target = EvalCountingTarget(log_prob)
    priors = _priors(ndim, bounds)
    pc = (
        PeriodicContainer({"x": {k: (v[1] - v[0]) for k, v in periodic.items()}})
        if periodic
        else None
    )
    rng = np.random.default_rng(seed)
    start_pts = train_samples[rng.choice(len(train_samples), nwalkers, replace=False)]
    sampler = EnsembleSampler(
        nwalkers,
        {"x": ndim},
        target,
        priors,
        tempering_kwargs=dict(ntemps=1),
        vectorize=True,
        periodic=pc,
        moves=[move],
        branch_names=["x"],
    )
    # DELIBERATE IMPROVEMENT over the original script, whose verdict was
    # run-to-run flaky: the EnsembleSampler builds an *unseeded* internal
    # RandomState (eryn.ensemble ~line 679). Seed it deterministically via the
    # public ``random_state`` setter so the GO/NO-GO gate is reproducible.
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = State({"x": start_pts.reshape(1, nwalkers, 1, ndim)})
    sampler.run_mcmc(state, nsteps, burn=burn, progress=True)
    chain = sampler.get_chain()["x"][:, 0, :, 0, :]  # (nsteps, nwalkers, ndim)
    ess = effective_sample_size(chain)                # ensemble-aware (per-walker ACF)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, ess, target.n_evals, acc


def main():
    ap = argparse.ArgumentParser(
        description="P0 go/no-go gate for the flow proposal (ESS-per-eval comparison)."
    )
    ap.add_argument("--chain", default=None,
                    help="npy file of one leaf's cold chain (n, ndim)")
    ap.add_argument("--out", default="p0_flow_gate_report.md")
    ap.add_argument("--periodic-index", type=int, default=None,
                    help="index of a periodic (2pi) dim in --chain")
    ap.add_argument("--nwalkers", type=int, default=100)
    ap.add_argument("--nsteps", type=int, default=1500)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42,
                    help="master RNG seed for reproducible GO/NO-GO verdicts "
                         "(DELIBERATE IMPROVEMENT: the original gate was unseeded "
                         "and could flake across the decision thresholds)")
    args = ap.parse_args()

    # Seed every source of randomness so the verdict is identical run-to-run:
    # numpy global RNG (start-coord draws when no explicit rng is threaded) and
    # torch's global RNG (flow weight init + flow.sample draws). Each sampler
    # arm is additionally seeded inside _run; see _run for the per-arm seeding.
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    if args.chain:
        try:
            chain = np.load(args.chain)
        except FileNotFoundError:
            raise SystemExit(f"--chain file not found: {args.chain}")
        if chain.ndim != 2:
            raise SystemExit(
                f"--chain array must be 2-D (n_samples, ndim); "
                f"got shape {chain.shape} from {args.chain}"
            )
        if args.periodic_index is not None and not (
            0 <= args.periodic_index < chain.shape[1]
        ):
            raise SystemExit(
                f"--periodic-index {args.periodic_index} out of range for a "
                f"chain with ndim={chain.shape[1]} (valid: 0..{chain.shape[1] - 1})"
            )
        periodic = (
            {args.periodic_index: (0.0, 2 * np.pi)}
            if args.periodic_index is not None
            else {}
        )
        log_prob, sampler_fn, ndim, periodic = chain_kde_target(chain, periodic)
        train_samples = chain
        target_name = f"KDE on {args.chain} (ndim={ndim})"
    else:
        log_prob, sampler_fn, ndim, periodic = banana_periodic_target(ndim_gauss=4)
        train_samples = sampler_fn(20000, seed=1)
        target_name = "synthetic banana+periodic (ndim=5)"

    # generous bounds covering the support
    lo = train_samples.min(0) - 2.0
    hi = train_samples.max(0) + 2.0
    bounds = {d: (float(lo[d]), float(hi[d])) for d in range(ndim)}
    for k, (a, b) in periodic.items():
        bounds[k] = (a, b)

    # --- train the frozen flow ---
    # Original: SourceNormalizer("bench", ...) + norm.build_transforms({0: train})
    #           -> FlowModel(ndim, conditioning, normalizer, nsf_kwargs)
    #           -> train_flow(fm, {0: train}, max_epochs=epochs, batch_size=2048, accelerator="auto")
    # Eryn API: WhiteningTransform(ndim, periodic) + ZukoFlow(dims, conditioning, data_transform, **nsf_kwargs)
    #           -> flow.fit({0: train}, n_epochs=epochs, batch_size=2048)
    # Hyperparameter mapping:
    #   max_epochs=args.epochs   -> n_epochs=args.epochs   (same semantics)
    #   batch_size=2048          -> batch_size=2048         (unchanged)
    #   accelerator="auto"       -> not supported; ZukoFlow always uses device= at construction
    #   patience=20              -> DELIBERATE CHANGE: the original Lightning trainer had no
    #                               early stopping. fit() restores best-validation weights, so
    #                               patience can only help the flow arm (no overfit tail) while
    #                               saving wall-time on long gate runs.
    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=ndim, periodic=periodic)
    flow = ZukoFlow(
        dims=ndim,
        conditioning=cond,
        data_transform=wt,
        # NSF hyper-parameters identical to original nsf_kwargs
        transforms=8,
        hidden_features=(256, 256),
        bins=8,
    )
    hist = flow.fit(
        {0: train_samples},
        n_epochs=args.epochs,
        batch_size=2048,
        patience=20,
    )

    # --- three competitors ---
    runargs = dict(nwalkers=args.nwalkers, nsteps=args.nsteps, burn=args.burn)
    results = []

    # Each arm is seeded deterministically but DIFFERENTLY (seed, seed+1,
    # seed+2). Identical seeds across arms is not required for fairness (the
    # arms run different move types); reproducibility run-to-run is what the
    # gate needs.
    fmove = ConditionalFlowMove(flow, branch_name="x")
    fmove.active_condition = 0
    _, ess, nev, acc = _run(fmove, log_prob, ndim, periodic, bounds, train_samples, seed=args.seed, **runargs)
    results.append(BenchmarkResult("flow", float(ess.min()), float(ess.min()) / nev, nev, acc))

    gmm = GMMProposalDistribution(train_samples, n_components=10, seed=args.seed)
    gmove = IndependentProposalMove(gmm, "x")
    _, ess, nev, acc = _run(gmove, log_prob, ndim, periodic, bounds, train_samples, seed=args.seed + 1, **runargs)
    results.append(BenchmarkResult("gmm", float(ess.min()), float(ess.min()) / nev, nev, acc))

    smove = StretchMove()
    _, ess, nev, acc = _run(smove, log_prob, ndim, periodic, bounds, train_samples, seed=args.seed + 2, **runargs)
    results.append(BenchmarkResult("stretch", float(ess.min()), float(ess.min()) / nev, nev, acc))

    # --- verdict (same thresholds as original) ---
    by_name = {r.name: r for r in results}
    flow_epe = by_name["flow"].ess_per_eval
    win_vs_stretch = flow_epe / by_name["stretch"].ess_per_eval
    win_vs_gmm = flow_epe / by_name["gmm"].ess_per_eval
    decisive = (win_vs_stretch >= 2.0) and (win_vs_gmm >= 1.5)

    # hist.validation_loss is a list (FlowHistory dataclass); original used hist['val_loss']
    final_val_nll = hist.validation_loss[-1] if hist.validation_loss else float("nan")

    lines = [
        "# P0 Flow Proposal Gate - Report", "",
        f"**Target:** {target_name}",
        f"**Walkers:** {args.nwalkers}  **Steps:** {args.nsteps} (burn {args.burn})",
        f"**Seed:** {args.seed}",
        f"**Final val NLL:** {final_val_nll:.4f}", "",
        "| proposal | min-ESS | evals | ESS/eval | acceptance |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.ess_min:.1f} | {r.n_evals} | {r.ess_per_eval:.3e} | {r.acceptance:.3f} |"
        )
    lines += [
        "",
        f"**Flow ESS/eval vs stretch:** {win_vs_stretch:.2f}x",
        f"**Flow ESS/eval vs GMM:** {win_vs_gmm:.2f}x", "",
        f"## VERDICT: {'GO' if decisive else 'NO-GO / MARGINAL'}",
        "",
        "- GO       -> proceed to P1 (in-process ConditionalFlowMove integration via InlineExecutor).",
        "- MARGINAL (beats stretch, ~GMM) -> a cheap GMM proposal may suffice; reconsider the flow.",
        "- NO-GO (no win over stretch) -> stop; the flow proposal is not worth building here.",
    ]
    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
