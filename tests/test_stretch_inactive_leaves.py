"""Stretch moves on a branch whose walkers have inactive leaf slots.

Each active leaf is an independent draw from a known density and no reversible-jump move is
used, so the exact marginal of every active coordinate is known. The stretch factor must
count only the parameters a walker actually moves, i.e. those of its active leaves; counting
the inactive slots too over-disperses those walkers (variance ~2.5 instead of 1 with 1 of 3
slots active).
"""

import unittest

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.moves import GroupStretchMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer

NSLOTS = 3
NWALKERS = 64


class PoolGroupStretch(GroupStretchMove):
    """Group stretch whose friends are random active leaves of a stationary pool."""

    def setup_friends(self, branches):
        branch = branches["leaf"]
        self.pool = branch.coords[0][branch.inds[0]]

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        return self.pool[np.random.randint(len(self.pool), size=s.shape[:-1])]


def gaussian_log_like(x, groups, *args, **kwargs):
    out = np.zeros(int(groups.max()) + 1)
    np.add.at(out, groups, -0.5 * np.sum(x**2, axis=-1))
    return out


def angle_log_like(x, groups, *args, **kwargs):
    # x[:, 0] ~ N(0, 1); x[:, 1] on [0, 2 pi) with density ~ exp(-(x - pi)^2 / (2 * 0.8^2))
    out = np.zeros(int(groups.max()) + 1)
    np.add.at(out, groups, -0.5 * x[:, 0] ** 2 - 0.5 * ((x[:, 1] - np.pi) / 0.8) ** 2)
    return out


def per_slot_setup(name, ndim):
    setup = []
    for i in range(NSLOTS):
        mask = np.zeros((NSLOTS, ndim), dtype=bool)
        mask[i] = True
        setup.append((name, mask))
    return setup


def sample_active(move, n_active, log_like, priors, init, periodic=None, steps=2500, burn=500,
                  seed=3, active=None):
    """Run ``move`` with the first ``n_active`` of NSLOTS slots active in every walker, or with
    the boolean ``(NWALKERS, NSLOTS)`` mask ``active``; return the active leaves of all stored
    samples."""
    rng = np.random.default_rng(seed)
    ndim = len(init)
    if active is None:
        active = np.broadcast_to(np.arange(NSLOTS) < n_active, (NWALKERS, NSLOTS))
    coords = np.zeros((1, NWALKERS, NSLOTS, ndim))
    # stale values in the inactive slots, as left behind by deaths in a reversible-jump run
    coords[0][~active] = rng.uniform(0.5, 5.5, ((~active).sum(), ndim))
    coords[0][active] = init + 0.1 * rng.standard_normal((active.sum(), ndim))
    inds = active[None].copy()

    np.random.seed(seed)  # the sampler copies the global RNG state when it is built
    sampler = EnsembleSampler(
        NWALKERS, ndim, log_like, priors, branch_names=["leaf"],
        nleaves_max={"leaf": NSLOTS}, nleaves_min={"leaf": 1}, moves=move,
        periodic=periodic, vectorize=True, provide_groups=True,
    )
    sampler.run_mcmc(State({"leaf": coords}, inds={"leaf": inds}), steps, burn=burn,
                     progress=False)
    chain = sampler.get_chain()["leaf"][:, 0]
    keep = sampler.get_inds()["leaf"][:, 0]
    return chain[keep]


class StretchInactiveLeavesTest(unittest.TestCase):
    gauss_priors = {"leaf": ProbDistContainer({i: uniform_dist(-10, 10) for i in range(2)})}

    def assert_unit_variance(self, leaves, tol=0.15):
        var = leaves.var(axis=0)
        self.assertTrue(np.all(np.abs(var - 1.0) < tol), f"variance {var}, expected 1")

    def test_stretch_all_slots_together(self):
        for n_active in (1, 2):
            leaves = sample_active(StretchMove(live_dangerously=True), n_active,
                                   gaussian_log_like, self.gauss_priors, np.zeros(2))
            self.assert_unit_variance(leaves)

    def test_stretch_one_slot_per_gibbs_block(self):
        move = StretchMove(gibbs_sampling_setup=per_slot_setup("leaf", 2), live_dangerously=True)
        leaves = sample_active(move, 1, gaussian_log_like, self.gauss_priors, np.zeros(2))
        self.assert_unit_variance(leaves)

    def test_stretch_one_slot_per_gibbs_block_different_slots(self):
        # ! each block runs while some walkers have no active leaf in it; those walkers must
        # be rejected, not stored with log_prior = 0 (which biases their next move)
        active = np.arange(NSLOTS) == (np.arange(NWALKERS) % NSLOTS)[:, None]
        move = StretchMove(gibbs_sampling_setup=per_slot_setup("leaf", 2), live_dangerously=True)
        leaves = sample_active(move, None, gaussian_log_like, self.gauss_priors, np.zeros(2),
                               active=active)
        self.assert_unit_variance(leaves)

    def test_group_stretch_all_slots_together(self):
        for n_active in (1, 2):
            leaves = sample_active(PoolGroupStretch(nfriends=1), n_active, gaussian_log_like,
                                   self.gauss_priors, np.zeros(2))
            self.assert_unit_variance(leaves)

    def test_stretch_periodic_parameter(self):
        # inactive slots also hold periodic values, whose lift weights must not enter the
        # acceptance of a walker that does not move them
        priors = {"leaf": ProbDistContainer({0: uniform_dist(-10, 10),
                                             1: uniform_dist(0, 2 * np.pi)})}
        periodic = PeriodicContainer({"leaf": {1: 2 * np.pi}})
        leaves = sample_active(StretchMove(live_dangerously=True), 1, angle_log_like, priors,
                               np.array([0.0, np.pi]), periodic=periodic)
        self.assertLess(abs(leaves[:, 0].var() - 1.0), 0.15)
        self.assertLess(abs(leaves[:, 1].mean() - np.pi), 0.1)
        self.assertLess(abs(leaves[:, 1].var() - 0.8**2), 0.1)


if __name__ == "__main__":
    unittest.main()
