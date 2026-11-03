"""Tests for walking nested move trees and reporting per-move acceptance."""
from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eryn.moves import CombineMove, Move
from eryn.utils import plot as plot_module
from eryn.utils.plot import (
    PlotContainer,
    _tex_safe,
    move_acceptance_rates,
    move_counters,
    move_tree_colors,
    plot_acceptance_fraction,
    plot_move_tree_acceptance,
    produce_advanced_plots,
)
from eryn.utils.utility import walk_moves


class LeafA(Move):
    """Minimal concrete move; only the acceptance counters are exercised."""

    def propose(self, model, state):  # pragma: no cover - never called
        raise NotImplementedError


class LeafB(LeafA):
    """A second leaf class, so paths can be told apart by name."""


def test_sub_moves_defaults_to_empty():
    assert LeafA().sub_moves == []


def test_combine_move_sub_moves_unwraps_weight_tuples():
    a, b = LeafA(), LeafB()
    combine = CombineMove([(a, 0.5), (b, 0.5)])
    assert combine.sub_moves == [a, b]


def test_walk_moves_flat_uses_class_names():
    a, b = LeafA(), LeafB()
    assert [path for path, _ in walk_moves([a, b])] == ["LeafA", "LeafB"]


def test_walk_moves_disambiguates_repeated_class_at_same_level():
    a, b = LeafA(), LeafA()
    assert [path for path, _ in walk_moves([a, b])] == ["LeafA_0", "LeafA_1"]


def test_walk_moves_composes_nested_paths():
    leaf, inner_a, inner_b = LeafA(), LeafA(), LeafB()
    inner = CombineMove([inner_a, inner_b])
    outer = CombineMove([leaf, inner])

    assert [path for path, _ in walk_moves([outer])] == [
        "CombineMove",
        "CombineMove/LeafA",
        "CombineMove/CombineMove",
        "CombineMove/CombineMove/LeafA",
        "CombineMove/CombineMove/LeafB",
    ]


def test_walk_moves_yields_the_move_objects():
    leaf = LeafA()
    outer = CombineMove([leaf])
    found = dict(walk_moves([outer]))
    assert found["CombineMove/LeafA"] is leaf
    assert found["CombineMove"] is outer


def test_walk_moves_reports_a_shared_instance_once():
    shared = LeafA()
    outer = CombineMove([shared, CombineMove([shared])])
    paths = [path for path, _ in walk_moves([outer])]
    assert paths.count("CombineMove/LeafA") == 1
    assert "CombineMove/CombineMove/LeafA" not in paths


def test_walk_moves_reports_a_repeated_sibling_once():
    shared = LeafA()
    paths = [path for path, _ in walk_moves([shared, shared])]
    assert paths == ["LeafA"]


def test_walk_moves_tolerates_a_move_without_the_hook():
    class NotAMove:
        pass

    assert [path for path, _ in walk_moves([NotAMove()])] == ["NotAMove"]


# cumulative counters over 4 plot calls, shape (nsteps, ntemps=1, nwalkers=2).
# Between call 1 and 2 the move was never drawn: num_proposals does not move.
COUNTS = np.array([[[1.0, 2.0]], [[3.0, 5.0]], [[3.0, 5.0]], [[7.0, 9.0]]])
NUM_PROPOSALS = np.array([10.0, 20.0, 20.0, 40.0])


def test_cumulative_rate_is_the_plain_ratio():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="cumulative")
    expected = np.array([[[0.1, 0.2]], [[0.15, 0.25]], [[0.15, 0.25]], [[0.175, 0.225]]])
    np.testing.assert_allclose(rates, expected)


def test_interval_rate_differences_the_counters():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
    # first row is measured from the start of the run; rows 1 and 3 are the
    # increments (2/10, 3/10) and (4/20, 4/20)
    np.testing.assert_allclose(rates[0], [[0.1, 0.2]])
    np.testing.assert_allclose(rates[1], [[0.2, 0.3]])
    np.testing.assert_allclose(rates[3], [[0.2, 0.2]])


def test_interval_rate_is_nan_when_the_move_was_not_drawn():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
    # NaN, not 0.0 -- the line must break rather than read as a real collapse
    assert np.all(np.isnan(rates[2]))


def test_interval_rate_is_nan_when_the_counter_decreases():
    # a decreasing cumulative counter means something external reset it
    # underneath us (e.g. a shared move re-owned by an EnsembleSampler that
    # zeroed `accepted` in place); that interval must break the line rather
    # than plot a negative acceptance rate.
    counts = np.array([[[1.0, 2.0]], [[3.0, 5.0]], [[1.0, 1.0]], [[4.0, 3.0]]])
    num_proposals = np.array([10.0, 20.0, 30.0, 40.0])

    rates = move_acceptance_rates(counts, num_proposals, mode="interval")

    # interval 2 (index 2) decreased from the previous cumulative value -> NaN
    assert np.all(np.isnan(rates[2]))
    # all other intervals are computed normally
    np.testing.assert_allclose(rates[0], [[0.1, 0.2]])
    np.testing.assert_allclose(rates[1], [[0.2, 0.3]])
    np.testing.assert_allclose(rates[3], [[0.3, 0.2]])


def test_no_divide_by_zero_warning_is_emitted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
        move_acceptance_rates(COUNTS, np.zeros(4), mode="cumulative")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="interval"):
        move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="sideways")


def _leaf_with_counters(accepted_value, num_proposals):
    move = LeafA()
    move.accepted = np.full((2, 3), float(accepted_value))
    move.num_proposals = num_proposals
    return move


def test_move_counters_reads_a_leafs_own_counters():
    move = _leaf_with_counters(4.0, 10)
    accepted, num_proposals = move_counters(move)
    np.testing.assert_allclose(accepted, np.full((2, 3), 4.0))
    assert num_proposals == 10.0


def test_move_counters_returns_none_when_uninitialised():
    assert move_counters(LeafA()) is None


def test_move_counters_pools_a_combine_move_from_its_children():
    # CombineMove.accepted raises AttributeError (its setter never stores
    # _accepted) and its num_proposals is never incremented, so it must be
    # pooled rather than read directly.
    combine = CombineMove([_leaf_with_counters(4.0, 10), _leaf_with_counters(6.0, 30)])
    accepted, num_proposals = move_counters(combine)
    np.testing.assert_allclose(accepted, np.full((2, 3), 10.0))
    assert num_proposals == 40.0


def test_move_counters_pooling_weights_by_proposal_count():
    # 4/10 and 6/30 pool to 10/40 = 0.25, not the mean of 0.4 and 0.2 = 0.3
    combine = CombineMove([_leaf_with_counters(4.0, 10), _leaf_with_counters(6.0, 30)])
    accepted, num_proposals = move_counters(combine)
    assert accepted[0, 0] / num_proposals == pytest.approx(0.25)


class _FakeBackend:
    def __init__(self, iteration):
        self.iteration = iteration


def test_collect_move_acceptance_records_every_node(tmp_path):
    leaf = _leaf_with_counters(2.0, 10)
    combine = CombineMove([leaf])
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))

    container._collect_move_acceptance([combine])

    assert set(container.move_accepted) == {"CombineMove", "CombineMove/LeafA"}
    assert container.move_steps["CombineMove/LeafA"] == [100]
    assert container.move_num_proposals["CombineMove/LeafA"] == [10.0]


def test_collect_move_acceptance_keeps_per_path_steps(tmp_path):
    ready = _leaf_with_counters(2.0, 10)
    late = LeafA()
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))

    container._collect_move_acceptance([ready, late])
    # the second node's counters only become valid on the next call
    late.accepted = np.full((2, 3), 1.0)
    late.num_proposals = 5
    container.backend = _FakeBackend(200)
    container._collect_move_acceptance([ready, late])

    # the late starter must not be plotted against the earlier step
    assert container.move_steps["LeafA_0"] == [100, 200]
    assert container.move_steps["LeafA_1"] == [200]


def test_move_acceptance_fractions_property_is_cumulative(tmp_path):
    leaf = _leaf_with_counters(2.0, 10)
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))
    container._collect_move_acceptance([leaf])

    fractions = container.move_acceptance_fractions
    assert fractions["LeafA"].shape == (1, 2, 3)
    np.testing.assert_allclose(fractions["LeafA"][0], np.full((2, 3), 0.2))


def test_move_counters_counts_a_shared_instance_once():
    shared = _leaf_with_counters(4.0, 10)
    outer = CombineMove([shared, CombineMove([shared])])
    accepted, num_proposals = move_counters(outer)
    np.testing.assert_allclose(accepted, np.full((2, 3), 4.0))
    assert num_proposals == 10.0


def test_move_counters_returns_none_when_no_child_has_counters():
    combine = CombineMove([LeafA(), LeafA()])
    assert move_counters(combine) is None


def test_tex_safe_escapes_underscores_only_under_usetex():
    original = matplotlib.rcParams["text.usetex"]
    try:
        matplotlib.rcParams["text.usetex"] = False
        assert _tex_safe("MHMove_0") == "MHMove_0"
        matplotlib.rcParams["text.usetex"] = True
        assert _tex_safe("MHMove_0") == r"MHMove\_0"
    finally:
        matplotlib.rcParams["text.usetex"] = original


def test_move_tree_colors_gives_each_root_its_own_hue():
    colors = move_tree_colors(["A", "A/x", "B"])
    assert colors["A"] != colors["B"]
    # descendants stay in their parent's family, but are distinguishable
    assert colors["A"] != colors["A/x"]


def test_move_tree_colors_covers_every_path():
    paths = ["A", "A/x", "A/x/y", "B", "B/z"]
    assert set(move_tree_colors(paths)) == set(paths)


def test_plot_acceptance_fraction_writes_a_file_for_a_nested_tree(tmp_path):
    steps = np.array([10, 20])
    total = np.full((2, 3, 4), 0.3)
    rates = {
        "CombineMove": np.full((2, 3, 4), 0.3),
        "CombineMove/LeafA": np.full((2, 3, 4), 0.25),
        "CombineMove/LeafA_0": np.full((1, 3, 4), 0.4),
    }
    move_steps = {
        "CombineMove": steps,
        "CombineMove/LeafA": steps,
        "CombineMove/LeafA_0": np.array([20]),
    }
    out = tmp_path / "acceptance_fraction.png"

    plot_acceptance_fraction(steps, total, rates, moves_steps=move_steps,
                             filename=str(out))

    assert out.exists()


def test_plot_acceptance_fraction_still_accepts_no_moves(tmp_path):
    out = tmp_path / "empty.png"
    plot_acceptance_fraction(np.array([10]), np.full((1, 3, 4), 0.3), {},
                             filename=str(out))
    assert out.exists()


def test_plot_acceptance_fraction_falls_back_to_shared_steps(tmp_path):
    # the back-compat path: without moves_steps every path is plotted
    # against the shared steps array, which requires matching lengths
    steps = np.array([10, 20])
    total = np.full((2, 3, 4), 0.3)
    rates = {
        "CombineMove": np.full((2, 3, 4), 0.3),
        "CombineMove/LeafA": np.full((2, 3, 4), 0.25),
    }
    out = tmp_path / "fallback.png"

    plot_acceptance_fraction(steps, total, rates, filename=str(out))

    assert out.exists()


def test_plot_acceptance_fraction_labels_distinguish_depth(monkeypatch):
    # text.usetex collapses leading whitespace, so depth used to be encoded
    # only via leading spaces and two different depths of the same leaf class
    # rendered as byte-identical legend entries. Capture the figure instead
    # of saving it so the legend labels can be inspected directly.
    captured = {}

    def fake_save_or_show(fig, filename=None):
        captured["fig"] = fig

    monkeypatch.setattr(plot_module, "save_or_show", fake_save_or_show)

    steps = np.array([10])
    rates = {
        "CombineMove/StretchMove": np.full((1, 3, 4), 0.3),
        "CombineMove/Residual/StretchMove": np.full((1, 3, 4), 0.3),
    }
    move_steps = {path: steps for path in rates}

    plot_module.plot_acceptance_fraction(
        steps, np.full((1, 3, 4), 0.3), rates, moves_steps=move_steps
    )

    try:
        _, labels = captured["fig"].axes[0].get_legend_handles_labels()
    finally:
        plt.close(captured["fig"])

    assert "- StretchMove" in labels
    assert "- - StretchMove" in labels


NESTED_RATES = {
    "CombineMove": np.full((2, 3, 4), 0.30),
    "CombineMove/LeafA": np.full((2, 3, 4), 0.25),
    "CombineMove/Residual": np.full((2, 3, 4), 0.20),
    "CombineMove/Residual/StretchMove": np.full((2, 3, 4), 0.35),
    "CombineMove/Residual/FlowMove": np.full((2, 3, 4), 0.05),
}
NESTED_STEPS = {path: np.array([10, 20]) for path in NESTED_RATES}


def test_move_tree_plots_mirror_the_move_tree(tmp_path):
    plot_move_tree_acceptance(NESTED_RATES, NESTED_STEPS, parent_folder=str(tmp_path))

    assert (tmp_path / "moves" / "CombineMove" / "acceptance_fraction.png").exists()
    assert (
        tmp_path / "moves" / "CombineMove" / "Residual" / "acceptance_fraction.png"
    ).exists()


def test_move_tree_plots_skip_leaves(tmp_path):
    plot_move_tree_acceptance(NESTED_RATES, NESTED_STEPS, parent_folder=str(tmp_path))

    # LeafA and FlowMove have no children, so they get no figure of their own
    assert not (tmp_path / "moves" / "CombineMove" / "LeafA").exists()
    assert not (
        tmp_path / "moves" / "CombineMove" / "Residual" / "FlowMove"
    ).exists()


def test_move_tree_plots_handle_a_flat_tree(tmp_path):
    plot_move_tree_acceptance(
        {"LeafA": np.full((2, 3, 4), 0.3)},
        {"LeafA": np.array([10, 20])},
        parent_folder=str(tmp_path),
    )
    assert not (tmp_path / "moves").exists()


class _FakeAdvancedBackend:
    """Minimal backend stand-in exercising the 'advanced' branch of
    ``PlotContainer.produce_plots``.

    Only the attributes/methods that branch actually touches are provided:
    ``iteration``, ``key_order``, ``accepted``, ``moves``, and the three
    unconditionally-called getters (``get_chain``, ``get_log_like``,
    ``get_betas``); their return values are irrelevant to the acceptance-
    fraction wiring under test, so they are kept as cheap stand-ins.
    """

    def __init__(self, iteration, moves):
        self.iteration = iteration
        self.moves = moves
        self.key_order = []
        self.accepted = np.full((2, 3), 0.3)

    def get_chain(self, discard=0):
        return {}

    def get_log_like(self, discard=0):
        return np.zeros((1, 2, 3))

    def get_betas(self, discard=0):
        return np.zeros((1, 2))


def test_produce_plots_advanced_survives_a_late_starting_move(tmp_path):
    """Regression test for the wiring this task closes.

    Before this task, ``produce_advanced_plots`` called
    ``plot_acceptance_fraction`` with the old 3-argument form (no
    ``moves_steps``), which forces every move to be plotted against the
    container's shared ``steps``. As of Task 3, a move whose counters are
    not yet valid is skipped by ``_collect_move_acceptance``, so its own
    step history is shorter than ``steps`` -- exactly what happens to
    ``late`` here, which only gains valid counters on the second call. That
    mismatch used to raise ``ValueError: x and y must have same first
    dimension``. This drives ``PlotContainer`` end-to-end (two real
    ``produce_plots`` calls) to prove the per-path steps now reach
    ``plot_acceptance_fraction`` and ``plot_move_tree_acceptance`` intact.
    """
    ready = _leaf_with_counters(2.0, 10)
    late = LeafA()
    combine = CombineMove([ready, late])

    backend = _FakeAdvancedBackend(iteration=100, moves=[combine])
    container = PlotContainer(backend=backend, plots="advanced", parent_folder=str(tmp_path))

    # first call: `late` has no counters yet and is skipped, so its history
    # (empty) is already shorter than `ready`'s and the container's `steps`
    container.produce_plots()

    # `late` becomes valid only now, on the second recorded step
    late.accepted = np.full((2, 3), 1.0)
    late.num_proposals = 5
    backend.iteration = 200

    container.produce_plots()

    assert (tmp_path / "advanced" / "acceptance_fraction.png").exists()
    assert (
        tmp_path / "advanced" / "moves" / "CombineMove" / "acceptance_fraction.png"
    ).exists()


def test_produce_advanced_plots_without_moves_steps_falls_back_for_the_move_tree(tmp_path):
    """Regression test: omitting ``moves_steps`` used to raise ``KeyError``.

    ``plot_acceptance_fraction`` already falls back to the shared ``steps``
    for every path when ``moves_steps`` is ``None``, but the
    ``plot_move_tree_acceptance`` call was wired with a plain ``{}``
    substitution, so it did an unconditional ``moves_steps[child]`` lookup
    into an empty dict and raised ``KeyError`` for any wrapper with
    children. ``produce_advanced_plots`` should give both calls the same
    shared-steps fallback instead.
    """
    steps = np.array([10, 20])
    total = np.full((2, 3, 4), 0.3)
    rates = {
        "CombineMove": np.full((2, 3, 4), 0.30),
        "CombineMove/LeafA": np.full((2, 3, 4), 0.25),
    }

    produce_advanced_plots(
        steps,
        total,
        rates,
        iteration=20,
        chain={},
        parent_folder=str(tmp_path),
    )

    assert (
        tmp_path / "moves" / "CombineMove" / "acceptance_fraction.png"
    ).exists()


def test_move_counters_returns_a_snapshot_not_a_view():
    """Regression test for the counter-aliasing bug.

    ``move_counters`` used to read a move's own counters with
    ``np.asarray(move.accepted, dtype=float)``, which does not copy when
    ``move.accepted`` is already a float64 ndarray -- it hands back the same
    object. Moves accumulate their counters in place (``self.accepted +=
    ...``), so every caller holding that "snapshot" was actually watching the
    live buffer keep mutating underneath it.
    """
    move = _leaf_with_counters(2.0, 10)
    snapshot, _ = move_counters(move)

    assert snapshot is not move.accepted

    move.accepted += 5.0

    np.testing.assert_allclose(snapshot, np.full((2, 3), 2.0))


def test_collect_move_acceptance_records_independent_snapshots_across_calls(tmp_path):
    """Regression test through the real collection path.

    Before the fix, every entry ``_collect_move_acceptance`` appended for a
    leaf move was an alias of the same live ``move.accepted`` buffer, so the
    whole recorded history collapsed to the final value. Differencing two
    identical snapshots for an interval rate produced nonsense: an
    impossible (>1) rate for the interval that swallowed the whole jump, and
    exactly 0.0 for every interval after. This drives
    ``_collect_move_acceptance`` and ``move_rates`` end-to-end the way a
    sampler run would, to prove the two collected snapshots differ and that
    the derived interval rates stay within the only physically valid range,
    [0, 1].
    """
    leaf = _leaf_with_counters(2.0, 10)
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))

    container._collect_move_acceptance([leaf])

    # mutate in place, the way a sampler accumulates counters between plot
    # calls
    leaf.accepted += 48.0
    leaf.num_proposals += 90
    container.backend = _FakeBackend(200)
    container._collect_move_acceptance([leaf])

    history = container.move_accepted["LeafA"]
    assert len(history) == 2
    assert not np.allclose(history[0], history[1])

    rates, _ = container.move_rates()
    leaf_rates = rates["LeafA"]
    valid = ~np.isnan(leaf_rates)
    assert np.all(leaf_rates[valid] >= 0.0)
    assert np.all(leaf_rates[valid] <= 1.0)
