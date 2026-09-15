"""Fancy (walker-permuting) temperature-swap acceptance pairing.

Found 2026-09-14 in the LISA 6mo campaign: ``temperature_swaps`` applies
slot ``k``'s swap to the PERMUTED pair ``(i, iperm[k]) <-> (i-1,
i1perm[k])``, but ``perform_fancy_swap_acceptance_fraction`` built the
acceptance from ``new_like[.][k] - old_like[.][k]`` -- the deltas of
walker SLOT ``k``, which belong to two *different* proposed pairs
(``iperm^-1(k)`` and ``i1perm^-1(k)``). Exact only for the identity
permutation, i.e. never on the path that permutes: a detailed-balance
violation for every non-trivial fancy swap. The correct ratio for slot
``k`` indexes each rung's delta by that pair's own walker:

    beta_i  * (new[1][iperm[k]]  - old[1][iperm[k]])
  + beta_i1 * (new[0][i1perm[k]] - old[0][i1perm[k]])

Simultaneous evaluation of all pairs in one ``compute_log_like`` call
stays valid because each (rung, walker) likelihood depends only on its
own coordinates and data.
"""

import numpy as np
import unittest

from eryn.moves.tempering import TemperatureControl


NW = 4
BETAS = np.array([1.0, 0.4])
# per-walker "data" offset -- emulates walker-indexed residuals/PSD; it
# must cancel inside each delta (same walker slot on both sides).
OFFSET = np.array([0.0, 10.0, 20.0, 30.0])


def _lnl(coords):
    """Deterministic per-(rung, walker) likelihood: value + walker data."""
    v = coords[:, :, 0, 0]
    return v + OFFSET[None, :]


def _fake_compute_log_like(x, inds=None, supps=None, branch_supps=None,
                           logp=None):
    return _lnl(x["a"]), None


class FancySwapPairingTest(unittest.TestCase):
    def _setup(self):
        tc = TemperatureControl(1, NW, ntemps=2, betas=BETAS.copy(),
                                adaptive=False)
        rng = np.random.default_rng(3)
        coords = rng.normal(0.0, 1.0, (2, NW, 1, 1))
        x = {"a": coords.copy()}
        logl = _lnl(coords)
        return tc, x, logl, coords

    def test_acceptance_matches_the_applied_pair(self):
        tc, x, logl, coords = self._setup()
        iperm = np.array([1, 2, 3, 0])
        i1perm = np.array([2, 0, 3, 1])
        got = tc.perform_fancy_swap_acceptance_fraction(
            BETAS[0] - BETAS[1], 1, iperm, i1perm, x, logl,
            inds=None, blobs=None, supps=None,
            branch_supps={"a": None},
            compute_log_like=_fake_compute_log_like,
        )
        v = coords[:, :, 0, 0]
        # slot k proposes (1, iperm[k]) <-> (0, i1perm[k]); the OFFSET
        # terms cancel per rung, leaving
        # beta_1*(v[0,i1perm[k]] - v[1,iperm[k]])
        #   + beta_0*(v[1,iperm[k]] - v[0,i1perm[k]])
        want = (BETAS[1] * (v[0, i1perm] - v[1, iperm])
                + BETAS[0] * (v[1, iperm] - v[0, i1perm]))
        np.testing.assert_allclose(np.asarray(got).ravel(), want,
                                   rtol=1e-12)

    def test_identity_permutation_unchanged(self):
        tc, x, logl, coords = self._setup()
        ident = np.arange(NW)
        got = tc.perform_fancy_swap_acceptance_fraction(
            BETAS[0] - BETAS[1], 1, ident, ident, x, logl,
            inds=None, blobs=None, supps=None,
            branch_supps={"a": None},
            compute_log_like=_fake_compute_log_like,
        )
        v = coords[:, :, 0, 0]
        # beta_i*(v0-v1) + beta_{i-1}*(v1-v0) = (beta_{i-1}-beta_i)(v1-v0)
        want = (BETAS[0] - BETAS[1]) * (v[1] - v[0])
        np.testing.assert_allclose(np.asarray(got).ravel(), want,
                                   rtol=1e-12)

    def test_offsets_never_leak_into_acceptance(self):
        # doubling every per-walker offset must not move the ratio at
        # all -- the walker-indexed "noise term" cancels pair-wise.
        tc, x, logl, coords = self._setup()
        iperm = np.array([3, 0, 1, 2])
        i1perm = np.array([1, 3, 0, 2])
        got1 = tc.perform_fancy_swap_acceptance_fraction(
            BETAS[0] - BETAS[1], 1, iperm, i1perm,
            {"a": coords.copy()}, _lnl(coords),
            inds=None, blobs=None, supps=None,
            branch_supps={"a": None},
            compute_log_like=_fake_compute_log_like,
        )
        global OFFSET
        old = OFFSET
        OFFSET = 2.0 * OFFSET
        try:
            got2 = tc.perform_fancy_swap_acceptance_fraction(
                BETAS[0] - BETAS[1], 1, iperm, i1perm,
                {"a": coords.copy()}, _lnl(coords),
                inds=None, blobs=None, supps=None,
                branch_supps={"a": None},
                compute_log_like=_fake_compute_log_like,
            )
        finally:
            OFFSET = old
        np.testing.assert_allclose(np.asarray(got1), np.asarray(got2),
                                   rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
