# tests/test_flow_detailed_balance.py
"""End-to-end detailed-balance gate for ConditionalFlowMove.

A frozen trained flow used as the ONLY proposal must leave a known target
invariant, including a periodic dimension.  If the periodic Jacobian or the
Hastings factor sign were wrong, the recovered periodic marginal would be
biased.  This is the third and strongest correctness gate on the coords-space
Hastings factor: it exercises the whole stack (ZukoFlow + WhiteningTransform +
OneHotLeafConditioning + ConditionalFlowMove) through eryn's EnsembleSampler end to end.

Ported from ``LISAanalysistools/tests/test_ml_detailed_balance.py`` (branch
feat/flow-proposal-p0).  The assertions and tolerances are carried over
unchanged in substance — they define the gate.  API renames applied:

    FlowModel(ndim=, conditioning=, normalizer=, nsf_kwargs=)
        -> ZukoFlow(dims, conditioning=, data_transform=, **nsf_kwargs)
    SourceNormalizer(..., periodic=) + build_transforms
        -> WhiteningTransform(ndim, periodic=) (refit by flow.fit)
    hand-rolled full-batch Adam loop
        -> flow.fit(...)  (mapping documented at the call site)
    imports from lisatools.globalfit.ml.* -> eryn.flows / eryn.moves
"""
import numpy as np
import pytest

# Module-level guards: the whole file requires the optional 'flow' extra.
pytest.importorskip("torch")
pytest.importorskip("zuko")

import torch  # noqa: E402
from scipy import stats  # noqa: E402

from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.prior import ProbDistContainer, uniform_dist  # noqa: E402
from eryn.utils import PeriodicContainer  # noqa: E402
from eryn.state import State  # noqa: E402

from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning  # noqa: E402
from eryn.moves import ConditionalFlowMove  # noqa: E402

PERIOD = 2 * np.pi


def _target_samples(n, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(1.0, 0.7, size=n)
    x1 = (rng.normal(2.5, 0.5, size=n)) % PERIOD  # periodic, bulk away from wrap
    return np.column_stack([x0, x1]).astype(np.float64)


def _log_target(x):
    """Vectorized analytic log target (unnormalised): Normal(1,0.7) x WrappedNormal(2.5,0.5)."""
    x = np.atleast_2d(x)
    lp0 = -0.5 * ((x[:, 0] - 1.0) / 0.7) ** 2
    # wrapped normal approx via nearest image (sigma small vs period -> dominant image ok)
    d = (x[:, 1] - 2.5 + np.pi) % PERIOD - np.pi
    lp1 = -0.5 * (d / 0.5) ** 2
    return lp0 + lp1


def _train_toy_flow(samples, steps=400, seed=0):
    """Train a small NSF flow on ``samples``.

    Hyperparameter mapping vs the original ``_train_toy_flow`` (lisatools)::

        nsf_kwargs(transforms=4, hidden_features=(64,64), bins=6)
            -> ZukoFlow(transforms=4, hidden_features=(64,64), bins=6)  [unchanged]
        torch.optim.Adam(lr=1e-3)                  -> flow.fit(lr=1e-3)  [unchanged]
        steps=400 full-batch gradient updates      -> n_epochs=400 with
            batch_size >= n_train so each epoch is exactly one full-batch step
            (400 gradient updates total — identical update budget).

    Deliberate deviations (each forced by the flow.fit API, none affecting the
    gate semantics):

    - ``flow.fit`` holds out ``validation_fraction`` of the data and restores
      the best-validation state at the end.  The original trained on the full
      8000 samples and kept the final state.  We set validation_fraction small
      (held-out for best-state selection only) and keep batch_size >= the
      training split so the per-epoch update remains a single full-batch step.
    - ``WhiteningTransform`` replaces ``SourceNormalizer`` + ``build_transforms``;
      ``flow.fit`` refits the transform from ``samples`` (refit_data_transform
      defaults to True), mirroring the original ``norm.build_transforms`` call.
    - Seeding is explicit (torch + the fit seed) so the gate is deterministic
      and not flaky; the original seeded only ``torch.manual_seed(seed)``.
    """
    torch.manual_seed(seed)

    cond = OneHotLeafConditioning(nleaves_max=1)
    # SourceNormalizer("toy", num_dims=2, periodic={1: (0.0, PERIOD)})
    #   -> WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})
    wt = WhiteningTransform(ndim=2, periodic={1: (0.0, PERIOD)})

    # FlowModel(ndim=2, conditioning=cond, normalizer=norm, nsf_kwargs=...)
    #   -> ZukoFlow(dims=2, conditioning=cond, data_transform=wt, **nsf_kwargs)
    flow = ZukoFlow(
        dims=2,
        device="cpu",
        conditioning=cond,
        data_transform=wt,
        seed=seed,
        flow_class="NSF",
        transforms=4,
        hidden_features=(64, 64),
        bins=6,
    )

    # n_epochs == original `steps` (400). batch_size >= the training split makes
    # each epoch exactly one full-batch Adam step, matching the original
    # 400-full-batch-update loop. lr=1e-3 unchanged. fit refits the transform.
    flow.fit(
        samples,
        n_epochs=steps,
        lr=1e-3,
        batch_size=len(samples),  # full-batch: one gradient step per epoch
        validation_fraction=0.1,
        seed=seed,
    )
    return flow


@pytest.mark.slow
def test_flowmove_recovers_periodic_target():
    # Save/restore the global RNG states: every other test in this suite uses
    # local default_rng generators, and leaking a reseeded global state would
    # make unrelated tests ordering-sensitive.
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)  # load-bearing: flow init + flow sampling read the global torch RNG

        train = _target_samples(8000, seed=1)
        fm = _train_toy_flow(train)

        ndim, nwalkers = 2, 100
        priors = {"x": ProbDistContainer({0: uniform_dist(-5.0, 7.0), 1: uniform_dist(0.0, PERIOD)})}
        periodic = PeriodicContainer({"x": {1: PERIOD}})

        move = ConditionalFlowMove(fm, branch_name="x")
        move.active_condition = 0  # select the single one-hot condition (nleaves_max=1)

        sampler = EnsembleSampler(
            nwalkers, {"x": ndim}, _log_target, priors,
            tempering_kwargs=dict(ntemps=1), vectorize=True,
            periodic=periodic, moves=[move], branch_names=["x"],
        )
        # EnsembleSampler builds its own *unseeded* RandomState (np.random.seed
        # does not reach it); seed it via the public setter so the accept/reject
        # draws — and hence the whole gate — are deterministic run-to-run.
        sampler.random_state = np.random.RandomState(0).get_state()
        start = State({"x": _target_samples(nwalkers, seed=7).reshape(1, nwalkers, 1, ndim)})
        sampler.run_mcmc(start, 600, burn=200, progress=False)

        # chain axes: (step, temp, walker, leaf, dim) -> drop temp=0, leaf=0
        # (valid because ntemps=1 and the branch is single-leaf)
        chain = sampler.get_chain()["x"][:, 0, :, 0, :].reshape(-1, ndim)  # (nsteps*nwalkers, ndim)
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    # non-periodic dim: KS vs analytic Normal(1, 0.7)
    ks_p = stats.kstest((chain[:, 0] - 1.0) / 0.7, "norm").pvalue
    assert ks_p > 1e-3, f"non-periodic marginal off (KS p={ks_p})"

    # periodic dim: circular mean ~ 2.5 and circular std ~ 0.5 (bias here == Jacobian bug)
    ang = chain[:, 1]
    cmean = np.angle(np.mean(np.exp(1j * ang))) % PERIOD
    R = np.abs(np.mean(np.exp(1j * ang)))
    cstd = np.sqrt(-2 * np.log(R))
    assert abs(((cmean - 2.5 + np.pi) % PERIOD) - np.pi) < 0.15, f"circular mean {cmean}"
    assert abs(cstd - 0.5) < 0.15, f"circular std {cstd}"
