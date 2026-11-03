# tests/test_flow_online_training.py
"""End-to-end slow gate for the ONLINE flow-training loop.

This is the integration counterpart to ``test_flow_detailed_balance.py``: where
that test froze a trained flow and checked detailed balance, this one drives the
*whole* online machinery in anger and proves it holds together:

    EnsembleSampler  --harvest cold chain-->  ProcessExecutor (spawned trainer)
          ^                                            |
          |  hot-load versioned weights (poll)  <------/

A :class:`ConditionalFlowMove` (mixed 30/70 with a plain :class:`StretchMove`, so the chain
stays ergodic while the flow learns) harvests the cold chain into a spawned
:class:`ProcessExecutor`, which trains a clone in a *second process* and hands
back versioned weights that the move hot-loads mid-run.  The single test asserts
three properties:

  (a) weights advance + hot-load   — executor reaches version >= 2 and the move
                                      loads them (loaded_version >= 2);
  (b) posterior correct            — the pooled cold chain matches ground-truth
                                      empirical moments (including the periodic
                                      dim's circular moments);
  (c) clean shutdown, no zombie    — the trainer process exits with exitcode 0
                                      (graceful — neither terminate nor kill
                                      fired), a second shutdown is a no-op, and
                                      no zombie is left in active_children().

It also asserts NO ``TrainerError`` escaped: the test passing means none was
raised, and a final wrapped ``latest_weights()`` poll fails with a clear message
if the trainer died.

spawn re-imports this module in each child as a NON-main module, so everything
at module level here is import-safe (only defs + constants).  Child startup is
slow (~2-5 s), so every wait is a wall-clock DEADLINE loop, never a tight sleep.
"""
import multiprocessing
import time

import numpy as np
import pytest

# Module-level guards: the whole file requires the optional 'flow' extra.
pytest.importorskip("torch")
pytest.importorskip("zuko")

import torch  # noqa: E402

from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.moves import ConditionalFlowMove, StretchMove  # noqa: E402
from eryn.prior import ProbDistContainer, uniform_dist  # noqa: E402
from eryn.state import State  # noqa: E402
from eryn.utils import PeriodicContainer  # noqa: E402

from eryn.flows import (  # noqa: E402
    OneHotLeafConditioning,
    ProcessExecutor,
    WhiteningTransform,
    ZukoFlow,
)
from eryn.flows.benchmark import banana_periodic_target  # noqa: E402
from eryn.flows.executors import TrainerError  # noqa: E402

PERIOD = 2 * np.pi

# Trimmed budget (vs the example's defaults) so the whole loop runs in ~3 min.
NWALKERS = 48
NSTEPS = 800
BURN = 200
WARMUP_STEPS = 250
WARMUP_BURN = 80
HARVEST_EVERY = 5
SEED = 20240611

# Generous wall-clock deadline for "wait until the trainer produced something".
# Child startup is ~2-5 s on macOS CPU; we never tight-loop.
_POLL_DEADLINE_S = 60.0


# ---------------------------------------------------------------------------
# Circular statistics for the periodic dim — same formulas as
# test_flow_detailed_balance.py (a bias here would be a periodic-Jacobian bug).
# ---------------------------------------------------------------------------
def _circular_mean(ang: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * ang))) % PERIOD)


def _circular_std(ang: np.ndarray) -> float:
    R = np.abs(np.mean(np.exp(1j * ang)))
    return float(np.sqrt(-2.0 * np.log(R)))


def _build_sampler(moves, log_prob, ndim, bounds, seed):
    """Seeded, single-temperature EnsembleSampler for branch 'x'."""
    sampler = EnsembleSampler(
        NWALKERS,
        {"x": ndim},
        log_prob,
        {"x": ProbDistContainer(
            {d: uniform_dist(*bounds[d]) for d in range(ndim)}
        )},
        tempering_kwargs=dict(ntemps=1),
        vectorize=True,
        periodic=PeriodicContainer({"x": {ndim - 1: PERIOD}}),
        moves=moves,
        branch_names=["x"],
    )
    # WHY seed via the setter: EnsembleSampler builds its OWN *unseeded*
    # RandomState (np.random.seed never reaches it), so without this the
    # accept/reject draws — and therefore version timing and the posterior —
    # would be nondeterministic and the gate would flake.
    sampler.random_state = np.random.RandomState(seed).get_state()
    return sampler


def _cold_chain(sampler, ndim):
    """Flatten the post-run cold chain to (nsteps*nwalkers, ndim).

    Chain axes are (step, temp, walker, leaf, dim); ntemps=1 and single-leaf,
    so dropping temp=0 and leaf=0 is exact.
    """
    return sampler.get_chain()["x"][:, 0, :, 0, :].reshape(-1, ndim)


@pytest.mark.slow
def test_online_training_full_loop():
    # WHY save/restore the global RNG states: this test reseeds the global numpy
    # and torch RNGs (flow init + flow.sample read the global torch RNG), and
    # leaking a reseeded global state would make unrelated tests in the suite
    # order-sensitive.  Mirror the detailed-balance test's hygiene exactly.
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        # Seed every global source: numpy (warmup start draws + start selection)
        # and torch (flow weight init + flow.sample).  Each sampler additionally
        # seeds its own RandomState in _build_sampler.
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # --- target: banana (2 Gaussian dims) + 1 periodic dim => ndim = 3 ---
        log_prob, sampler_fn, ndim, periodic = banana_periodic_target(ndim_gauss=2)
        periodic_dim = ndim - 1

        # ground truth: large exact draw → empirical moments are the reference.
        truth = sampler_fn(200_000, seed=SEED + 1)
        truth_mean = np.array([
            _circular_mean(truth[:, d]) if d == periodic_dim
            else float(truth[:, d].mean())
            for d in range(ndim)
        ])
        truth_std = np.array([
            _circular_std(truth[:, d]) if d == periodic_dim
            else float(truth[:, d].std())
            for d in range(ndim)
        ])

        # generous box prior over the support; periodic dim fixed to (0, 2pi).
        lo = truth.min(0) - 2.0
        hi = truth.max(0) + 2.0
        bounds = {d: (float(lo[d]), float(hi[d])) for d in range(ndim)}
        bounds[periodic_dim] = (0.0, PERIOD)

        # --- warmup: short stretch-only run to seed the flow + fit the transform
        warmup_sampler = _build_sampler(
            [StretchMove()], log_prob, ndim, bounds, SEED
        )
        warmup_sampler.run_mcmc(
            State({"x": sampler_fn(NWALKERS, seed=SEED).reshape(
                1, NWALKERS, 1, ndim)}),
            WARMUP_STEPS, burn=WARMUP_BURN, progress=False,
        )
        warmup = _cold_chain(warmup_sampler, ndim)

        # --- flow + frozen whitening transform (fit here freezes the map) ---
        wt = WhiteningTransform(ndim=ndim, periodic={periodic_dim: (0.0, PERIOD)})
        flow = ZukoFlow(
            dims=ndim,
            device="cpu",
            conditioning=OneHotLeafConditioning(nleaves_max=1),
            data_transform=wt,
            seed=SEED,
            transforms=3,
            hidden_features=(64, 64),
            bins=5,
        )
        flow.fit(warmup, n_epochs=10, batch_size=512,
                 validation_fraction=0.1, seed=SEED)

        proc = None  # captured inside the context for the post-exit assertions
        with ProcessExecutor(
            flow,
            epochs_per_round=10,
            min_train_samples=1500,
            torch_num_threads=2,
            seed=SEED,
        ) as ex:
            flow_move = ConditionalFlowMove(
                flow, "x", executor=ex, harvest_every=HARVEST_EVERY
            )
            sampler = _build_sampler(
                [(StretchMove(), 0.7), (flow_move, 0.3)],
                log_prob, ndim, bounds, SEED,
            )
            start = State({"x": warmup[
                np.random.RandomState(SEED).choice(
                    len(warmup), NWALKERS, replace=False)
            ].reshape(1, NWALKERS, 1, ndim)})
            sampler.run_mcmc(start, NSTEPS, burn=BURN, progress=False)

            # ----------------------------------------------------------------
            # (a) weights advance + hot-load.
            # The spawned trainer starts slowly (~2-5 s), so the main run may
            # finish before 2 fit rounds land.  Drive the chain forward in short
            # continuation segments (which keep harvesting + polling) on a
            # wall-clock deadline until the executor reports version >= 2.
            # ----------------------------------------------------------------
            t0 = time.monotonic()
            while ex.version < 2 and time.monotonic() < t0 + _POLL_DEADLINE_S:
                sampler.run_mcmc(
                    sampler.get_last_sample(), 50, burn=0, progress=False
                )
            assert ex.version >= 2, (
                f"trainer never reached version 2 within {_POLL_DEADLINE_S}s "
                f"(reached version {ex.version}); the spawned trainer is too "
                "slow at these budgets — report timings rather than relaxing."
            )

            # A few more steps so the move's setup() polls and hot-loads the
            # newest weights into its own flow.
            sampler.run_mcmc(
                sampler.get_last_sample(), 30, burn=0, progress=False
            )
            assert flow_move.loaded_version >= 2, (
                f"ConditionalFlowMove hot-load lagged: loaded_version="
                f"{flow_move.loaded_version}, executor version={ex.version}."
            )

            # Explicit final poll, wrapped: if the trainer died, surface it as a
            # clear assertion rather than a bare TrainerError from teardown.
            try:
                ex.latest_weights()
            except TrainerError as exc:  # pragma: no cover - failure path
                pytest.fail(f"trainer raised TrainerError during the run: {exc}")

            # ----------------------------------------------------------------
            # (b) posterior correct — pooled post-burn cold chain of the ONLINE
            # run vs ground-truth empirical moments.
            # ----------------------------------------------------------------
            online = _cold_chain(sampler, ndim)
            for d in range(ndim):
                if d == periodic_dim:
                    om, os_ = _circular_mean(online[:, d]), _circular_std(online[:, d])
                    dmean = abs(((om - truth_mean[d] + np.pi) % PERIOD) - np.pi)
                else:
                    om, os_ = float(online[:, d].mean()), float(online[:, d].std())
                    dmean = abs(om - truth_mean[d])
                dstd = abs(os_ - truth_std[d])
                assert dmean < 0.15, (
                    f"dim {d} mean off: online={om:.4f} truth={truth_mean[d]:.4f} "
                    f"|delta|={dmean:.4f} >= 0.15"
                )
                assert dstd < 0.15, (
                    f"dim {d} std off: online={os_:.4f} truth={truth_std[d]:.4f} "
                    f"|delta|={dstd:.4f} >= 0.15"
                )

            # capture the live process handle for the clean-shutdown checks
            proc = ex._process
        # <-- context manager exit: graceful shutdown of the trainer process.

        # --------------------------------------------------------------------
        # (c) clean shutdown, no zombie.
        # exitcode 0 means the worker stopped on the sentinel/stop-event and
        # neither terminate() (-15) nor kill() (-9) had to fire.
        # --------------------------------------------------------------------
        assert proc is not None
        assert not proc.is_alive(), "trainer process still alive after shutdown"
        assert proc.exitcode == 0, (
            f"trainer did not exit gracefully: exitcode={proc.exitcode} "
            "(0 expected; a negative code means terminate/kill had to fire)"
        )
        ex.shutdown()  # idempotent second call: must be a no-op, no raise
        assert proc not in multiprocessing.active_children(), (
            "trainer left a zombie in multiprocessing.active_children()"
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


@pytest.mark.slow
def test_online_training_lazy_no_prefit():
    """End-to-end NO-PRE-FIT path: the worker fits the transform, hands it back.

    The companion to :func:`test_online_training_full_loop`, but proving the
    *lazy* bootstrapping the user actually runs: a flow is built with an
    **UNFITTED** shared :class:`WhiteningTransform` and handed straight to a
    :class:`ProcessExecutor` with no warm ``flow.fit`` — something that raised
    before CT2 (``FlowSpec.from_flow`` asserted ``is_fitted``).  The harvested
    cold chain trains the clone in the worker; the first trained snapshot ships
    the *fitted* transform back, and the live flow hot-loads it via
    ``set_weights`` (the same call :meth:`ConditionalFlowMove.setup` makes).  The core new
    guarantee asserted here is that the LIVE flow's transform goes False -> True
    via that handoff — it was NEVER fitted locally.

    Bootstrapping note (see also the module finding): a :class:`ConditionalFlowMove` cannot
    *propose* from a flow whose transform is still unfitted — ``get_proposal``
    samples the flow, which raises ``RuntimeError`` on an unfitted transform.  So
    the live flow must reach version 1 BEFORE a mixed Stretch/Flow run lets the
    ConditionalFlowMove fire.  We seed that first version exactly the way a production
    harness does: submit the warmup samples (already in hand) to the executor and
    poll the first snapshot into the live flow — no local fit, the worker does it.
    Once usable, the mixed run continues harvesting and advancing versions.

    Asserts:
      (a) ``flow.data_transform.is_fitted`` is False right after construction and
          True after the worker's first snapshot is hot-loaded — the fitted
          transform came from the worker via the snapshot, NOT a local fit (THE
          point of CT2);
      (b) ``ex.version >= 1`` and ``flow_move.loaded_version >= 1`` — a trained
          snapshot was produced and the move hot-loaded it during the mixed run;
      (c) the now-usable flow returns finite ``log_prob`` /
          ``sample_and_log_prob`` (the installed transform + net are a matched,
          working pair);
      (d) ``ex.latest_val_nll`` is finite (the worker reported a real fit);
      (e) clean shutdown: ``proc.exitcode == 0``, not alive, no zombie.

    Same RNG hygiene as the pre-fit test (save/restore global numpy + torch RNG;
    seed each sampler's own RandomState) and the same bounded wall-clock catch-up
    continuation so a fast toy MCMC that outruns the ~2-5 s spawn startup still
    observes a trained version before the deadline.
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        log_prob, sampler_fn, ndim, periodic = banana_periodic_target(ndim_gauss=2)
        periodic_dim = ndim - 1

        # generous box prior over the support; periodic dim fixed to (0, 2pi).
        truth = sampler_fn(50_000, seed=SEED + 1)
        lo = truth.min(0) - 2.0
        hi = truth.max(0) + 2.0
        bounds = {d: (float(lo[d]), float(hi[d])) for d in range(ndim)}
        bounds[periodic_dim] = (0.0, PERIOD)

        # --- warmup: short stretch-only run.  Used to seed chain start points AND
        # to bootstrap the executor's first version below.  Crucially we NEVER
        # call flow.fit on it — the worker fits the transform lazily.
        warmup_sampler = _build_sampler(
            [StretchMove()], log_prob, ndim, bounds, SEED
        )
        warmup_sampler.run_mcmc(
            State({"x": sampler_fn(NWALKERS, seed=SEED).reshape(
                1, NWALKERS, 1, ndim)}),
            WARMUP_STEPS, burn=WARMUP_BURN, progress=False,
        )
        warmup = _cold_chain(warmup_sampler, ndim)

        # --- flow with an UNFITTED, shared whitening transform: NO warm fit. ---
        wt = WhiteningTransform(
            ndim=ndim, periodic={periodic_dim: (0.0, PERIOD)}, shared=True
        )
        flow = ZukoFlow(
            dims=ndim,
            device="cpu",
            conditioning=OneHotLeafConditioning(nleaves_max=1),
            data_transform=wt,
            seed=SEED,
            transforms=3,
            hidden_features=(64, 64),
            bins=5,
        )
        # THE precondition: the transform is unfitted at hand-off time.  Building
        # a ProcessExecutor from this flow would have raised before CT2.
        assert flow.data_transform.is_fitted is False, (
            "transform must be UNFITTED before the executor — this test proves "
            "the no-pre-fit path."
        )

        proc = None
        with ProcessExecutor(
            flow,
            # epochs_per_round is the ProcessExecutor's n_epochs knob; passing
            # n_epochs in fit_kwargs would collide with it in the worker.
            epochs_per_round=10,
            min_train_samples=1500,
            torch_num_threads=2,
            seed=SEED,
        ) as ex:
            # ----------------------------------------------------------------
            # BOOTSTRAP version 1 from the worker WITHOUT a local fit: submit the
            # warmup samples and wait (bounded) for the first snapshot, then
            # install it with the very same set_weights() call ConditionalFlowMove.setup
            # makes.  This is what turns the unfitted flow usable.
            # ----------------------------------------------------------------
            ex.submit({0: warmup})
            t0 = time.monotonic()
            lw = None
            while lw is None and time.monotonic() < t0 + _POLL_DEADLINE_S:
                ex.submit({0: warmup})  # keep feeding so the worker has >= min
                lw = ex.latest_weights()
            assert lw is not None, (
                f"trainer never produced a first snapshot within "
                f"{_POLL_DEADLINE_S}s; the spawned trainer is too slow at these "
                "budgets — report timings rather than relaxing."
            )
            version, snapshot = lw
            assert version >= 1

            # (a) THE core new guarantee: the LIVE flow's transform was UNFITTED,
            # and installing the worker's snapshot makes it fitted — never a local
            # fit.  The snapshot carries {"net", "data_transform"}; set_weights
            # installs the matched pair atomically (exactly ConditionalFlowMove.setup's call).
            assert flow.data_transform.is_fitted is False
            flow.set_weights(snapshot)
            assert flow.data_transform.is_fitted is True, (
                "live flow transform should be fitted AFTER installing the "
                "worker snapshot: the snapshot must carry the fitted transform."
            )

            # (c) the flow is now usable: finite log_prob + sample_and_log_prob.
            probe = flow.sample(64, context=0)
            lp = flow.log_prob(probe, context=0)
            assert np.all(np.isfinite(lp)), "log_prob produced non-finite values"
            xs, logq = flow.sample_and_log_prob(64, context=0)
            assert np.all(np.isfinite(xs)) and np.all(np.isfinite(logq)), (
                "sample_and_log_prob produced non-finite values"
            )

            # (d) the worker reported a real fit — its latent-space val NLL is finite.
            assert ex.latest_val_nll is not None and np.isfinite(ex.latest_val_nll), (
                f"executor latest_val_nll not finite: {ex.latest_val_nll!r}"
            )

            # ----------------------------------------------------------------
            # Now the flow is usable, run the realistic mixed Stretch/Flow loop.
            # ConditionalFlowMove harvests its cold chain into the SAME executor and
            # hot-loads newer snapshots as they land — proving the online story
            # past the bootstrap.  The move starts at loaded_version 0 (its own
            # bookkeeping), so it re-loads version 1 (or newer) on its first poll.
            # ----------------------------------------------------------------
            flow_move = ConditionalFlowMove(
                flow, "x", executor=ex, harvest_every=HARVEST_EVERY
            )
            sampler = _build_sampler(
                [(StretchMove(), 0.7), (flow_move, 0.3)],
                log_prob, ndim, bounds, SEED,
            )
            start = State({"x": warmup[
                np.random.RandomState(SEED).choice(
                    len(warmup), NWALKERS, replace=False)
            ].reshape(1, NWALKERS, 1, ndim)})
            sampler.run_mcmc(start, NSTEPS, burn=BURN, progress=False)

            # A few more steps so the move's setup() polls + hot-loads at least
            # the bootstrap version into its own bookkeeping.
            t1 = time.monotonic()
            while (flow_move.loaded_version < 1
                   and time.monotonic() < t1 + _POLL_DEADLINE_S):
                sampler.run_mcmc(
                    sampler.get_last_sample(), 30, burn=0, progress=False
                )

            # (b) versions advanced + the move hot-loaded a trained snapshot.
            assert ex.version >= 1
            assert flow_move.loaded_version >= 1, (
                f"ConditionalFlowMove hot-load lagged: loaded_version="
                f"{flow_move.loaded_version}, executor version={ex.version}."
            )
            # transform stays fitted across the mixed run (live flow is the same
            # object the move hot-loads into).
            assert flow.data_transform.is_fitted is True

            # Surface a dead trainer as a clear failure rather than a teardown crash.
            try:
                ex.latest_weights()
            except TrainerError as exc:  # pragma: no cover - failure path
                pytest.fail(f"trainer raised TrainerError during the run: {exc}")

            proc = ex._process
        # <-- context manager exit: graceful shutdown of the trainer process.

        # (e) clean shutdown, no zombie.
        assert proc is not None
        assert not proc.is_alive(), "trainer process still alive after shutdown"
        assert proc.exitcode == 0, (
            f"trainer did not exit gracefully: exitcode={proc.exitcode} "
            "(0 expected; a negative code means terminate/kill had to fire)"
        )
        ex.shutdown()  # idempotent second call: must be a no-op, no raise
        assert proc not in multiprocessing.active_children(), (
            "trainer left a zombie in multiprocessing.active_children()"
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
