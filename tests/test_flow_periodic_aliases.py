# tests/test_flow_periodic_aliases.py
"""Periodic aliasing in the coords-space density of a flow with periodic dims.

A periodic coordinate lives on a circle, but the flow models its latent as an
UNBOUNDED real coordinate.  The coords->latent map is therefore many-to-one:
every representative ``u + k*T`` (k integer) is the same angle.  The density of
the folded variable is consequently the WRAPPED density -- a sum over aliases:

    q(theta) = |det| * sum_k q_Z(z_0 + delta_k)

Evaluating only the ``k = 0`` image (the legacy behaviour) reports one term of a
positive sum, so it under-integrates over a period by exactly the mass the flow
placed outside the fundamental window.

These tests pin the two observable consequences:
  * normalization -- the coords-space density must integrate to 1 over the full
    domain (one period in the periodic dim, the real line in the others);
  * self-consistency -- ``sample_and_log_prob`` and ``log_prob`` must agree at
    the same point (they disagree in the legacy path because the sampler reports
    the density at the alias it DREW while ``log_prob`` re-derives k = 0).

The toy deliberately uses a broad (near-uniform) angle so the fundamental
window spans only ~+-1.7 latent sigma and the leaked mass is percent-level; a
concentrated angle leaks ~0 and would not discriminate.  Fit quality is
irrelevant -- normalization is a property of the density definition -- so the
tests use an untrained net on a fitted transform.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning


def _fitted_mixed_transform(seed=3):
    """3-D transform: dims 0 and 2 periodic (2*pi and pi), dim 1 ordinary."""
    tr = WhiteningTransform(
        ndim=3, periodic={0: (0.0, PERIOD), 2: (0.0, np.pi)}, shared=False
    )
    rng = np.random.default_rng(seed)
    tr.fit({0: np.column_stack([
        rng.uniform(0.0, PERIOD, 3000),
        rng.normal(0.0, 2.0, 3000),
        rng.uniform(0.0, np.pi, 3000),
    ])})
    return tr


def _coords_gap(tr, x, x_ref):
    """Max discrepancy, comparing periodic dims circularly."""
    worst = 0.0
    for d in range(x.shape[1]):
        diff = np.abs(x[:, d] - x_ref[:, d])
        if d in tr.periodic:
            period = tr.periodic[d][1] - tr.periodic[d][0]
            diff = np.minimum(diff % period, period - (diff % period))
        worst = max(worst, float(diff.max()))
    return worst

PERIOD = 2 * np.pi
NPER = 241     # grid points across one period
NREAL = 481    # grid points across the real dim


def _make_flow(periodic=True, seed=7, **flow_kw):
    """2-D flow: dim 0 (optionally) periodic and broad, dim 1 ordinary Gaussian.

    The net is left untrained on purpose: the density must be normalized for
    ANY flow, so training would only obscure what is being tested.
    """
    per = {0: (0.0, PERIOD)} if periodic else {}
    tr = WhiteningTransform(ndim=2, periodic=per, shared=False)
    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.uniform(0.0, PERIOD, size=4000),   # broad angle -> narrow window
        rng.normal(0.0, 1.0, size=4000),
    ])
    tr.fit({0: samples})
    flow = ZukoFlow(
        dims=2,
        flow_class="NSF",
        device="cpu",
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        data_transform=tr,
        seed=seed,
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
        **flow_kw,
    )
    return flow


def _integrate_over_period(flow):
    """Integrate exp(log_prob) over one period in dim 0 x the real line in dim 1.

    Dim 0 is integrated over exactly one period, which is exact (the density is
    periodic).  Dim 1's range is derived from the flow's OWN draws so the
    quadrature domain provably contains the mass -- otherwise a truncated tail
    would masquerade as a normalization defect.
    """
    probe, _ = flow.sample_and_log_prob(20000, context=0)
    real_half = float(np.abs(probe[:, 1]).max()) + 3.0
    a = np.linspace(0.0, PERIOD, NPER)
    b = np.linspace(-real_half, real_half, NREAL)
    A, B = np.meshgrid(a, b, indexing="ij")
    lp = np.asarray(flow.log_prob(np.column_stack([A.ravel(), B.ravel()]),
                                  context=0), dtype=np.float64)
    dens = np.exp(lp).reshape(NPER, NREAL)
    return float(np.trapezoid(np.trapezoid(dens, b, axis=1), a))


def test_periodic_density_integrates_to_one_over_a_period():
    """The coords-space density of a periodic dim must be normalized on [0, T)."""
    flow = _make_flow(periodic=True)
    total = _integrate_over_period(flow)
    assert total == pytest.approx(1.0, abs=5e-3), (
        f"periodic density integrates to {total:.6f}, not 1.0 -- the mass the "
        "flow placed outside the fundamental window is being dropped instead "
        "of summed over aliases"
    )


def test_sample_and_log_prob_agrees_with_log_prob_on_periodic_dim():
    """Both density entry points must report the same value at the same point."""
    flow = _make_flow(periodic=True)
    x, logq_draw = flow.sample_and_log_prob(3000, context=0)
    logq_eval = flow.log_prob(x, context=0)
    gap = np.abs(np.asarray(logq_draw, dtype=np.float64)
                 - np.asarray(logq_eval, dtype=np.float64))
    assert gap.max() < 1e-4, (
        f"max |logq| self-inconsistency {gap.max():.3e} nats over {len(gap)} draws "
        f"({(gap > 1e-4).mean():.2%} of draws affected) -- the sampler reports the "
        "density at the alias it drew, log_prob at the k=0 alias"
    )


def test_non_periodic_flow_density_is_bit_identical_to_legacy():
    """With no periodic dims the alias machinery must be an exact no-op."""
    default = _make_flow(periodic=False)
    legacy = _make_flow(periodic=False, periodic_aliases=0)
    pts = np.column_stack([
        np.linspace(-6.0, 6.0, 400), np.linspace(-4.0, 4.0, 400),
    ])
    lp_default = np.asarray(default.log_prob(pts, context=0), dtype=np.float64)
    lp_legacy = np.asarray(legacy.log_prob(pts, context=0), dtype=np.float64)
    assert np.array_equal(lp_default, lp_legacy), (
        "alias summation changed a non-periodic density; it must be a no-op "
        f"(max diff {np.abs(lp_default - lp_legacy).max():.3e})"
    )


def test_pruned_path_is_still_normalized_for_a_concentrated_angle():
    """A razor-thin angle prunes every alias -- and must stay normalized.

    The whitening scale for a sigma ~ 0.01 angle puts the period ~600 latent
    sigma away, so no alias can contribute and ZukoFlow collapses to a single
    evaluation.  This guards the prune: dropping negligible aliases must not
    cost normalization.  Integrated over a window that brackets the (very
    narrow) support, since a whole-period grid would simply under-resolve it.
    """
    per = {0: (0.0, PERIOD)}
    tr = WhiteningTransform(ndim=2, periodic=per, shared=False)
    rng = np.random.default_rng(4)
    tr.fit({0: np.column_stack([rng.normal(3.0, 0.01, 3000) % PERIOD,
                                rng.normal(0.0, 1.0, 3000)])})
    flow = ZukoFlow(
        dims=2, flow_class="NSF", device="cpu",
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        data_transform=tr, seed=3, transforms=3, hidden_features=(64, 64), bins=5,
    )
    # every alias is far away -> pruned down to the k=0 image
    assert np.linalg.norm(tr.periodic_alias_offsets(0, 1, "full")[1:], axis=1).min() > 100.0

    probe, _ = flow.sample_and_log_prob(20000, context=0)
    lo, hi = probe[:, 0].min() - 0.05, probe[:, 0].max() + 0.05
    a = np.linspace(lo, hi, 2001)
    b = np.linspace(-14.0, 14.0, 601)
    A, B = np.meshgrid(a, b, indexing="ij")
    dens = np.exp(np.asarray(flow.log_prob(np.column_stack([A.ravel(), B.ravel()]),
                                           context=0), dtype=np.float64))
    total = float(np.trapezoid(np.trapezoid(dens.reshape(len(a), len(b)), b, axis=1), a))
    assert total == pytest.approx(1.0, abs=5e-3), (
        f"pruned-alias density integrates to {total:.6f}, not 1.0"
    )


def test_alias_offsets_map_back_to_the_same_coords_point():
    """Defining property: displacing a latent by an alias offset is a no-op in coords."""
    tr = _fitted_mixed_transform()
    offsets = tr.periodic_alias_offsets(0, order=1, shell="full")
    assert offsets.shape == (9, 3)          # 3**2 lattice for 2 periodic dims
    assert not offsets[0].any()             # row 0 is the k=0 image

    z = torch.as_tensor(
        np.random.default_rng(11).normal(0.0, 1.0, size=(500, 3)), dtype=torch.float64
    )
    x_ref = tr.inverse(z, 0)
    for k, delta in enumerate(offsets):
        x = tr.inverse(z + torch.as_tensor(delta, dtype=torch.float64), 0)
        assert _coords_gap(tr, x, x_ref) < 1e-8, (
            f"alias offset {k} does not map back to the same coords point"
        )

    # negative control: a perturbed offset must NOT be an alias, otherwise the
    # assertion above would be vacuous.
    bad = offsets[1] * 1.1
    x_bad = tr.inverse(z + torch.as_tensor(bad, dtype=torch.float64), 0)
    assert _coords_gap(tr, x_bad, x_ref) > 1e-3


def test_alias_offsets_are_trivial_when_there_is_nothing_to_sum():
    """order=0 and no-periodic-dims both collapse to the single zero offset."""
    tr = _fitted_mixed_transform()
    assert tr.periodic_alias_offsets(0, order=0).shape == (1, 3)
    assert not tr.periodic_alias_offsets(0, order=0).any()

    plain = WhiteningTransform(ndim=2, periodic={}, shared=False)
    plain.fit({0: np.random.default_rng(5).normal(size=(500, 2))})
    assert plain.periodic_alias_offsets(0, order=1).shape == (1, 2)


def test_alias_shell_axis_is_a_subset_of_full():
    """The axis shell must be exactly the single-dimension displacements."""
    tr = _fitted_mixed_transform()
    axis = tr.periodic_alias_offsets(0, order=1, shell="axis")
    full = tr.periodic_alias_offsets(0, order=1, shell="full")
    assert axis.shape == (5, 3)   # 2 * n_periodic * order + 1
    for row in axis:
        assert any(np.allclose(row, f) for f in full)


def test_periodic_aliases_survives_checkpoint_round_trip(tmp_path):
    """periodic_aliases must round-trip through save/load (config is JSON)."""
    flow = _make_flow(periodic=True, periodic_aliases=0)
    pts = np.column_stack([np.linspace(0.1, 6.0, 50), np.linspace(-2.0, 2.0, 50)])
    before = np.asarray(flow.log_prob(pts, context=0), dtype=np.float64)

    path = tmp_path / "aliased_flow.h5"
    flow.save(str(path))
    reloaded = ZukoFlow.load(str(path))

    assert reloaded.periodic_aliases == 0, (
        f"periodic_aliases lost in round-trip: got {reloaded.periodic_aliases}"
    )
    after = np.asarray(reloaded.log_prob(pts, context=0), dtype=np.float64)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)


def test_alias_order_zero_reproduces_single_image_density():
    """periodic_aliases=0 must reproduce the legacy (k=0 only) density."""
    legacy = _make_flow(periodic=True, periodic_aliases=0)
    total = _integrate_over_period(legacy)
    # legacy drops the leaked mass, so it under-integrates -- pinned both as the
    # opt-out contract and as the baseline the fix improves on.
    assert total < 0.99, (
        f"periodic_aliases=0 integrated to {total:.6f}; expected the legacy "
        "single-image deficit (< 0.99) for this deliberately broad angle"
    )
