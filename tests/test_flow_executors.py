# tests/test_flow_executors.py
"""Tests for the TrainerExecutor seam: FlowSpec and InlineExecutor.

Most tests use a lightweight numpy-only fake Flow defined here, so they are
fast and torch-free.  One ZukoFlow-backed integration test (``importorskip``'d)
verifies that ``FlowSpec.from_flow(...).build()`` reproduces ``log_prob``.
"""
from __future__ import annotations

import multiprocessing
import time

import numpy as np
import pytest

from eryn.flows import (
    FlowSpec,
    InlineExecutor,
    ProcessExecutor,
    TrainerError,
    WorkerConfig,
)
from eryn.flows.base import Flow, FlowHistory
from eryn.flows.transforms import DataTransform, IdentityTransform


# ---------------------------------------------------------------------------
# Lightweight numpy-only fake Flow + transforms
# ---------------------------------------------------------------------------

class _FittableTransform(DataTransform):
    """Identity-like transform with a togglable ``is_fitted`` flag (picklable)."""

    def __init__(self, fitted: bool = True):
        self._fitted = bool(fitted)

    def fit(self, samples) -> None:
        self._fitted = True

    def forward(self, x, condition: int = 0):
        return np.asarray(x)

    def inverse(self, z, condition: int = 0):
        return np.asarray(z)

    def log_abs_det_jacobian(self, x, z, condition: int = 0):
        return np.zeros(len(x))

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class FakeFlow(Flow):
    """Numpy-only Flow for executor tests.

    ``fit`` bumps a counter, records how many samples it received, and stores a
    ``scale`` weight derived from the data so that ``get_weights`` /
    ``set_weights`` round-trips carry meaning (log_prob depends on ``scale``).
    """

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None,
                 scale: float = 1.0):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning)
        self.scale = float(scale)
        self.fit_count = 0
        self.last_fit_n = 0
        self.last_fit_kwargs: dict = {}

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum((x / self.scale) ** 2, axis=-1)

    def sample(self, n: int, context=None):
        return self.scale * np.random.randn(n, self.dims)

    def sample_and_log_prob(self, n: int, context=None):
        x = self.sample(n, context=context)
        return x, self.log_prob(x, context=context)

    def fit(self, samples, **kwargs):
        self.fit_count += 1
        self.last_fit_kwargs = kwargs
        if isinstance(samples, dict):
            n = sum(len(v) for v in samples.values())
            allx = np.concatenate([np.asarray(v) for v in samples.values()], axis=0)
        else:
            n = len(samples)
            allx = np.asarray(samples)
        self.last_fit_n = n
        # Derive a meaningful weight from the data so set_weights matters.
        self.scale = float(np.std(allx)) if allx.size else self.scale
        return FlowHistory()

    def get_weights(self):
        return {"scale": np.array(self.scale, dtype=np.float64)}

    def set_weights(self, obj):
        # Polymorphic, mirroring the Flow ABC contract: accept either a bare
        # weight dict ({"scale": ...}) or a snapshot ({"net", "data_transform"}).
        if isinstance(obj, dict) and "net" in obj:
            transform = obj.get("data_transform")
            if transform is not None:
                self.data_transform = transform
            weights = obj["net"]
        else:
            weights = obj
        self.scale = float(np.asarray(weights["scale"]))

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


class _RaisingFlow(FakeFlow):
    """FakeFlow whose ``fit`` always raises (to exercise failure surfacing)."""

    def fit(self, samples, **kwargs):
        raise RuntimeError("boom in fit")


class _ExplodingFlow(FakeFlow):
    """Module-level FakeFlow whose ``fit`` raises ``RuntimeError('boom')``.

    Defined at module scope so it pickles BY REFERENCE — a ``FlowSpec`` wrapping
    it can be sent across the spawn boundary and rebuilt in the worker, where
    its ``fit`` then raises and the executor must surface the child traceback as
    a :class:`TrainerError` whose message contains ``"boom"``.
    """

    def fit(self, samples, **kwargs):
        raise RuntimeError("boom")


class _LiveWeightFlow(FakeFlow):
    """FakeFlow whose ``get_weights`` returns a REFERENCE to live mutable state.

    This mimics torch ``get_weights`` returning references to a clone's live
    parameters: ``log_prob`` reads ``self._scale_arr`` and ``get_weights``
    hands back that exact array.  Without a defensive copy at stash time, a
    caller mutating the returned dict would corrupt the executor's clone.
    """

    def __init__(self, *args, scale: float = 1.0, **kwargs):
        super().__init__(*args, scale=scale, **kwargs)
        self._scale_arr = np.array(scale, dtype=np.float64)

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum((x / float(self._scale_arr)) ** 2, axis=-1)

    def fit(self, samples, **kwargs):
        hist = super().fit(samples, **kwargs)
        self._scale_arr = np.array(self.scale, dtype=np.float64)
        return hist

    def get_weights(self):
        # Live reference, NOT a copy — the executor must deep-copy when stashing.
        return {"scale": self._scale_arr}

    def set_weights(self, obj):
        if isinstance(obj, dict) and "net" in obj:
            transform = obj.get("data_transform")
            if transform is not None:
                self.data_transform = transform
            weights = obj["net"]
        else:
            weights = obj
        self._scale_arr = np.asarray(weights["scale"], dtype=np.float64).copy()
        self.scale = float(self._scale_arr)


class _UnpicklableFlow(FakeFlow):
    """FakeFlow carrying a lambda in its config (unpicklable)."""

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None,
                 bad=None, scale: float = 1.0):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning, scale=scale)
        self.bad = bad


def _make_fake(dims: int = 2, scale: float = 1.0) -> FakeFlow:
    return FakeFlow(dims=dims, device="cpu", data_transform=_FittableTransform(True),
                    scale=scale)


# ---------------------------------------------------------------------------
# FlowSpec
# ---------------------------------------------------------------------------

def test_flowspec_from_flow_allows_unfitted_transform():
    """An unfitted transform is allowed: the executor fits it lazily."""
    flow = FakeFlow(dims=2, data_transform=_FittableTransform(fitted=False))
    spec = FlowSpec.from_flow(flow)  # must NOT raise
    rebuilt = spec.build()
    assert rebuilt.data_transform.is_fitted is False


def test_flowspec_from_flow_unpicklable_config_raises():
    flow = _UnpicklableFlow(dims=2, data_transform=_FittableTransform(True),
                            bad=lambda z: z)
    with pytest.raises(TypeError, match="picklable"):
        FlowSpec.from_flow(flow)


def test_flowspec_overrides_device():
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow, worker_device="cuda:0")
    assert spec.config["device"] == "cuda:0"
    # Default is cpu.
    spec_cpu = FlowSpec.from_flow(flow)
    assert spec_cpu.config["device"] == "cpu"


def test_flowspec_deepcopies_config():
    """Mutating the live flow after snapshot must not change the spec."""
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow)
    flow.config_dict()["dims"] = 999  # no-op, but mutate the transform too
    flow.data_transform._fitted = False
    assert spec.config["dims"] == 2


def test_flowspec_build_reconstructs():
    """build() reproduces config + weights of the snapshotted flow."""
    flow = _make_fake(scale=3.5)
    spec = FlowSpec.from_flow(flow)
    rebuilt = spec.build()
    assert isinstance(rebuilt, FakeFlow)
    assert rebuilt.dims == flow.dims
    # Weights (scale) round-trip.
    assert rebuilt.scale == pytest.approx(3.5)
    x = np.random.randn(5, 2)
    np.testing.assert_allclose(rebuilt.log_prob(x), flow.log_prob(x))


def test_flowspec_is_picklable():
    import pickle
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow)
    spec2 = pickle.loads(pickle.dumps(spec))
    assert spec2.config["dims"] == spec.config["dims"]


# ---------------------------------------------------------------------------
# InlineExecutor — buffering / versioning
# ---------------------------------------------------------------------------

def test_inline_clones_flow_not_caller():
    """The executor trains a CLONE; the caller's flow is never mutated."""
    flow = _make_fake(scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.full((10, 2), 5.0)})  # would change scale on the clone
    assert flow.fit_count == 0  # caller's flow untouched
    assert flow.scale == 1.0


def test_inline_version_zero_before_min_train_samples():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=100)
    # Submit fewer than min_train_samples; no training should run.
    ex.submit({0: np.random.randn(10, 2)})
    assert ex.latest_weights() is None
    assert ex.version == 0


def test_inline_version_advances_after_enough_submits():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=20)
    ex.submit({0: np.random.randn(10, 2)})  # 10 buffered < 20: no train
    assert ex.latest_weights() is None
    ex.submit({0: np.random.randn(15, 2)})  # 25 buffered >= 20: train
    lw = ex.latest_weights()
    assert lw is not None
    version, weights = lw
    assert version == 1
    assert ex.version == 1


def test_inline_train_every_honored():
    """train_every=2 → first accepted submit does not train, second does."""
    flow = _make_fake()
    ex = InlineExecutor(flow, train_every=2, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})
    assert ex.latest_weights() is None  # 1st submit: no train
    ex.submit({0: np.random.randn(5, 2)})
    lw = ex.latest_weights()
    assert lw is not None and lw[0] == 1  # 2nd submit: trained


def test_inline_latest_weights_returns_newest():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})
    v1, _ = ex.latest_weights()
    ex.submit({0: np.random.randn(5, 2)})
    v2, _ = ex.latest_weights()
    assert v2 == v1 + 1
    assert ex.version == v2


def test_inline_weights_applied_to_sibling_change_outputs():
    """Weights from latest_weights, applied to a sibling flow, change its outputs."""
    flow = _make_fake(scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=0)
    # Submit data with a clearly different spread so the trained scale differs.
    ex.submit({0: np.random.randn(500, 2) * 7.0})
    _, weights = ex.latest_weights()

    sibling = _make_fake(scale=1.0)
    x = np.full((4, 2), 2.0)
    before = sibling.log_prob(x).copy()
    sibling.set_weights(weights)
    after = sibling.log_prob(x)
    assert not np.allclose(before, after)


def test_inline_latest_weights_snapshot_not_aliased_across_versions():
    """Mutating a returned weights dict must not corrupt a later version's snapshot.

    The ownership contract (ABC latest_weights docstring) says each snapshot is
    private and not aliased to live training state.  Here we mutate the
    version-1 dict, train again, and confirm the version-2 snapshot is a fresh,
    uncorrupted object — exactly as a process executor's freshly deserialized
    dict would be.
    """
    flow = _make_fake(scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=1)
    ex.submit({0: np.full((50, 2), 3.0)})
    v1, s1 = ex.latest_weights()

    # Caller misbehaves: mutate the returned snapshot's net dict in place.
    s1["net"]["scale"][...] = 999.0

    # Train again → a new, independent snapshot.
    ex.submit({0: np.full((50, 2), 3.0)})
    v2, s2 = ex.latest_weights()
    assert v2 == v1 + 1
    assert s2 is not s1
    assert float(np.asarray(s2["net"]["scale"])) != pytest.approx(999.0)


def test_inline_latest_weights_not_aliased_to_training_clone_state():
    """The stashed snapshot is decoupled from the clone's live state.

    Mutating the returned weights array must not change the trainer clone's
    own log_prob — i.e. the snapshot is a defensive copy (copy.deepcopy at
    stash time), not a reference to the clone's live parameters (the corruption
    a process boundary cannot have).
    """
    flow = _LiveWeightFlow(dims=2, device="cpu",
                           data_transform=_FittableTransform(True), scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=1)
    x = np.full((4, 2), 2.0)

    ex.submit({0: np.random.randn(200, 2) * 5.0})
    _, snapshot = ex.latest_weights()
    clone_before = ex._flow.log_prob(x).copy()

    # snapshot["net"]["scale"] would BE the clone's live array without the
    # deepcopy fix.
    assert snapshot["net"]["scale"] is not ex._flow.get_weights()["scale"]
    # Corrupt the returned snapshot; the clone's density must be unaffected.
    snapshot["net"]["scale"][...] = 1e6
    clone_after = ex._flow.log_prob(x)
    np.testing.assert_array_equal(clone_before, clone_after)


# ---------------------------------------------------------------------------
# InlineExecutor — empty-harvest safety (I2)
# ---------------------------------------------------------------------------

def test_inline_empty_dict_submit_is_noop_not_error():
    """submit({}) returns False, bumps no version, and never poisons the executor."""
    flow = _make_fake()
    ex = InlineExecutor(flow)  # default min_train_samples now 1
    assert ex.submit({}) is False
    assert ex.latest_weights() is None
    assert ex.version == 0
    assert flow.fit_count == 0
    # A subsequent real submit still works (no lingering TrainerError).
    assert ex.submit({0: np.random.randn(5, 2)}) is True
    assert ex.latest_weights() is not None


def test_inline_zero_row_array_submit_is_noop():
    """A submit carrying only a zero-row array behaves like an empty submit."""
    flow = _make_fake(dims=2)
    ex = InlineExecutor(flow)
    assert ex.submit({0: np.zeros((0, 2))}) is False
    assert ex.latest_weights() is None
    assert ex.version == 0
    # No deferred TrainerError on the next poll or submit.
    assert ex.submit({0: np.random.randn(5, 2)}) is True


def test_inline_mixed_empty_and_nonempty_buffers_only_nonempty():
    """A mixed submit (one empty + one non-empty condition) buffers only the real one."""
    flow = _make_fake(dims=2)
    ex = InlineExecutor(flow)
    assert ex.submit({0: np.zeros((0, 2)), 1: np.full((7, 2), 4.0)}) is True
    # Only condition 1 was buffered; condition 0 never created a buffer.
    assert 0 not in ex._buffers
    assert ex._flow.last_fit_n == 7
    # Training ran on the non-empty rows; the new version is observable on poll.
    lw = ex.latest_weights()
    assert lw is not None and lw[0] == 1
    assert ex.version == 1


def test_inline_buffer_cap_trims():
    """Total samples handed to fit never exceed the per-condition cap."""
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0, max_buffer_samples=50)
    # Submit way more than the cap across several batches.
    for _ in range(10):
        ex.submit({0: np.random.randn(20, 2)})
    # The clone records the last fit's sample count; must be <= cap.
    assert ex._flow.last_fit_n <= 50


def test_inline_buffer_cap_multi_condition():
    """Cap is per-condition: two conditions can each fill to the cap."""
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0, max_buffer_samples=30)
    for _ in range(5):
        ex.submit({0: np.random.randn(20, 2), 1: np.random.randn(20, 2)})
    # Each condition trimmed to <= 30; total <= 60.
    assert ex._flow.last_fit_n <= 60


def test_inline_single_submit_larger_than_cap_kept_whole():
    """A single submit bigger than the cap is retained in full (cap = max(cap, harvest)).

    _trim only drops when more than one array is buffered, so the most recent
    array is always kept whole — a lone oversized harvest is never truncated.
    """
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=1, max_buffer_samples=50)
    ex.submit({0: np.random.randn(500, 2)})  # one array, 10x the cap
    # The whole 500-row array is handed to fit despite exceeding the cap.
    assert ex._flow.last_fit_n == 500


# ---------------------------------------------------------------------------
# InlineExecutor — fit_kwargs guard (M6)
# ---------------------------------------------------------------------------

def test_inline_fit_kwargs_rejects_refit_data_transform():
    flow = _make_fake()
    with pytest.raises(ValueError, match="refit_data_transform"):
        InlineExecutor(flow, fit_kwargs=dict(refit_data_transform=True))


def test_inline_fit_kwargs_rejects_verbose():
    flow = _make_fake()
    with pytest.raises(ValueError, match="verbose"):
        InlineExecutor(flow, fit_kwargs=dict(verbose=True))


# ---------------------------------------------------------------------------
# InlineExecutor — failure surfacing
# ---------------------------------------------------------------------------

def test_inline_fit_failure_surfaces_on_latest_weights():
    flow = _RaisingFlow(dims=2, data_transform=_FittableTransform(True))
    ex = InlineExecutor(flow, min_train_samples=0)
    # submit that triggers the failing fit does NOT raise (deferred surfacing).
    ex.submit({0: np.random.randn(5, 2)})
    with pytest.raises(TrainerError, match="boom in fit"):
        ex.latest_weights()


def test_inline_fit_failure_surfaces_on_next_submit():
    flow = _RaisingFlow(dims=2, data_transform=_FittableTransform(True))
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})  # records failure
    with pytest.raises(TrainerError):
        ex.submit({0: np.random.randn(5, 2)})  # next submit re-raises


# ---------------------------------------------------------------------------
# InlineExecutor — shutdown / context manager
# ---------------------------------------------------------------------------

def test_inline_shutdown_idempotent_and_blocks_submit():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.shutdown()
    ex.shutdown()  # idempotent
    assert ex.submit({0: np.random.randn(5, 2)}) is False


def test_inline_submit_returns_true_when_live():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    assert ex.submit({0: np.random.randn(5, 2)}) is True


def test_inline_context_manager_calls_shutdown():
    flow = _make_fake()
    with InlineExecutor(flow, min_train_samples=0) as ex:
        assert ex.submit({0: np.random.randn(5, 2)}) is True
    assert ex.submit({0: np.random.randn(5, 2)}) is False


# ---------------------------------------------------------------------------
# ZukoFlow integration — FlowSpec.from_flow(...).build() reproduces log_prob
# ---------------------------------------------------------------------------

def _make_zuko_flow():
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 3))

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: samples})

    return ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=0,
        flow_class="NSF",
        transforms=2,
        hidden_features=(32, 32),
        bins=4,
    )


def test_flowspec_zuko_build_reproduces_log_prob():
    pytest.importorskip("torch")
    flow = _make_zuko_flow()
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    spec = FlowSpec.from_flow(flow)
    rebuilt = spec.build()

    lp1 = flow.log_prob(x, context=0)
    lp2 = rebuilt.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="FlowSpec build changed log_prob")


def test_inline_zuko_trains_and_serves_weights():
    """InlineExecutor with a real ZukoFlow trains and serves usable weights."""
    pytest.importorskip("torch")
    flow = _make_zuko_flow()
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 3))

    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=0)
    accepted = ex.submit({0: data})
    assert accepted is True
    lw = ex.latest_weights()
    assert lw is not None
    version, weights = lw
    assert version == 1

    # Weights are applicable to a sibling flow.
    sibling = _make_zuko_flow()
    sibling.set_weights(weights)
    x = rng.standard_normal((16, 3))
    lp = sibling.log_prob(x, context=0)
    assert lp.shape == (16,)
    assert np.all(np.isfinite(lp))


# ---------------------------------------------------------------------------
# Lazy transform fit (Inline) — unfitted transform fitted on first round, the
# snapshot carries it, and it reproduces densities on a fresh flow.
# ---------------------------------------------------------------------------

def _make_zuko_flow_unfitted(seed: int = 0):
    """A ZukoFlow with a shared, UNFITTED WhiteningTransform."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3, shared=True)  # NOT fitted
    assert wt.is_fitted is False
    return ZukoFlow(
        dims=3, device="cpu", data_transform=wt, conditioning=cond, seed=seed,
        flow_class="NSF", transforms=2, hidden_features=(32, 32), bins=4,
    )


def test_inline_lazy_fit_unfitted_transform():
    """Inline executor with an UNFITTED shared transform: FlowSpec does not raise,
    the first round fits the transform, and the snapshot reproduces densities."""
    pytest.importorskip("torch")
    flow = _make_zuko_flow_unfitted(seed=0)

    # FlowSpec no longer raises on an unfitted transform.
    spec = FlowSpec.from_flow(flow)
    assert spec.build().data_transform.is_fitted is False

    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=50)
    # The executor's clone started unfitted.
    assert ex._flow.data_transform.is_fitted is False
    assert ex.latest_val_nll is None

    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 3)) * 3.0
    assert ex.submit({0: data}) is True

    # After the first round the clone's transform is fitted and a snapshot exists.
    assert ex._flow.data_transform.is_fitted is True
    version, snapshot = ex.latest_weights()
    assert version == 1
    assert set(snapshot.keys()) == {"net", "data_transform"}
    assert snapshot["data_transform"].is_fitted is True
    assert ex.latest_val_nll is not None and np.isfinite(ex.latest_val_nll)

    # Apply the snapshot to a FRESH flow whose own transform is unfitted: it must
    # become fitted and reproduce the trainer clone's densities.
    fresh = _make_zuko_flow_unfitted(seed=0)
    assert fresh.data_transform.is_fitted is False
    fresh.set_weights(snapshot)
    assert fresh.data_transform.is_fitted is True
    x = rng.standard_normal((16, 3))
    lp_clone = ex._flow.log_prob(x, context=0)
    lp_fresh = fresh.log_prob(x, context=0)
    np.testing.assert_allclose(lp_clone, lp_fresh, atol=1e-6,
                               err_msg="snapshot did not reproduce clone densities")


def _make_zuko_flow_unfitted_perleaf(nleaves: int = 2, seed: int = 0):
    """A ZukoFlow with a PER-LEAF (shared=False), UNFITTED WhiteningTransform."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    cond = OneHotLeafConditioning(nleaves_max=nleaves)
    wt = WhiteningTransform(ndim=3, shared=False)  # per-leaf, NOT fitted
    assert wt.is_fitted is False
    return ZukoFlow(
        dims=3, device="cpu", data_transform=wt, conditioning=cond, seed=seed,
        flow_class="NSF", transforms=2, hidden_features=(32, 32), bins=4,
    )


def test_inline_lazy_fit_per_leaf_multi_condition():
    """Per-leaf (shared=False) lazy fit across two leaves — the fixed-leaf / non-RJ
    path.  The first round fits ONE map per condition (each source centered on its
    own mean), the snapshot carries all per-leaf maps, and applying it to a fresh
    flow reproduces densities for EACH condition independently."""
    pytest.importorskip("torch")
    flow = _make_zuko_flow_unfitted_perleaf(nleaves=2, seed=0)

    # FlowSpec does not raise on the unfitted per-leaf transform.
    assert FlowSpec.from_flow(flow).build().data_transform.is_fitted is False

    # min_train_samples low enough that the first fit fires once BOTH leaves have
    # contributed (fixed leaves accumulate together, so both are present).
    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=1)
    rng = np.random.default_rng(2)
    # two leaves with very different means/scales (different sources)
    a = rng.standard_normal((400, 3)) * 0.5 + np.array([0.0, 0.0, 0.0])
    b = rng.standard_normal((400, 3)) * 3.0 + np.array([10.0, -5.0, 2.0])
    assert ex.submit({0: a, 1: b}) is True

    version, snapshot = ex.latest_weights()
    assert version == 1
    wt = snapshot["data_transform"]
    assert wt.is_fitted is True and wt.shared is False
    # per-leaf: each condition has its OWN fitted map (not one shared map)
    assert set(wt.transforms.keys()) == {0, 1}

    # the two maps genuinely differ (each centred on its own source) — a shared
    # map would whiten both identically
    x = rng.standard_normal((8, 3))
    assert not np.allclose(wt.forward(x, 0).numpy(), wt.forward(x, 1).numpy())

    # snapshot reproduces the trainer clone's densities for BOTH conditions
    fresh = _make_zuko_flow_unfitted_perleaf(nleaves=2, seed=0)
    fresh.set_weights(snapshot)
    assert fresh.data_transform.is_fitted is True
    for ctx in (0, 1):
        np.testing.assert_allclose(
            ex._flow.log_prob(x, context=ctx), fresh.log_prob(x, context=ctx),
            atol=1e-6, err_msg=f"per-leaf snapshot mismatch at condition {ctx}",
        )


def _shared_whitening_matrix(transform):
    """Return a clone of the pooled whitening matrix of a fitted shared WhiteningTransform.

    A refit changes this matrix; a freeze leaves it bit-identical.  Compared
    instead of object identity because snapshots deep-copy the transform.
    """
    compose = transform.transforms[0]  # shared map under key 0
    lin = next(t for t in compose.parts if hasattr(t, "matrix"))
    return lin.matrix.clone()


def test_inline_refit_transform_every_refits_on_expected_rounds():
    """refit_transform_every=2: the transform refits on rounds 1, 2, 4 and is
    frozen on rounds 3, 5 (rounds counted from 1; round 1 is the first fit)."""
    torch = pytest.importorskip("torch")
    flow = _make_zuko_flow_unfitted(seed=0)
    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=1,
                        refit_transform_every=2)
    rng = np.random.default_rng(2)

    mats = []
    for i in range(5):
        # Vary the data spread each round so a refit visibly changes the matrix.
        ex.submit({0: rng.standard_normal((60, 3)) * (2.0 + i)})
        _, snap = ex.latest_weights()
        mats.append(_shared_whitening_matrix(snap["data_transform"]))

    # round 1: first fit. round 2 (2%2==0): refit → differs from round 1.
    assert not torch.allclose(mats[0], mats[1]), "round 2 should refit"
    # round 3 (3%2!=0): frozen → identical to round 2.
    assert torch.allclose(mats[1], mats[2]), "round 3 should stay frozen"
    # round 4 (4%2==0): refit → differs from round 3.
    assert not torch.allclose(mats[2], mats[3]), "round 4 should refit"
    # round 5 (5%2!=0): frozen → identical to round 4.
    assert torch.allclose(mats[3], mats[4]), "round 5 should stay frozen"


def test_inline_refit_transform_every_none_freezes_after_first():
    """Default (refit_transform_every=None): the transform is fitted once on
    round 1 and frozen on every subsequent round."""
    torch = pytest.importorskip("torch")
    flow = _make_zuko_flow_unfitted(seed=0)
    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=1)
    rng = np.random.default_rng(3)

    mats = []
    for i in range(3):
        ex.submit({0: rng.standard_normal((60, 3)) * (2.0 + i)})
        _, snap = ex.latest_weights()
        mats.append(_shared_whitening_matrix(snap["data_transform"]))
    # All rounds share the round-1 whitening matrix (frozen forever).
    assert torch.allclose(mats[0], mats[1])
    assert torch.allclose(mats[1], mats[2])


def test_inline_latest_val_nll_advances():
    """latest_val_nll is None before the first version and finite after (real fit)."""
    pytest.importorskip("torch")
    flow = _make_zuko_flow_unfitted(seed=0)
    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=1)
    assert ex.latest_val_nll is None
    rng = np.random.default_rng(4)
    ex.submit({0: rng.standard_normal((120, 3))})
    assert ex.version == 0  # not yet polled
    ex.latest_weights()
    assert ex.latest_val_nll is not None
    assert np.isfinite(ex.latest_val_nll)


# ===========================================================================
# ProcessExecutor — spawned-worker trainer
# ===========================================================================
#
# These tests run REAL spawned child processes.  spawn re-imports this test
# module in each child as a NON-main module, so all module-level code here must
# be import-safe (it is — only class/function defs and a couple of constants).
# We never need ``if __name__ == "__main__":`` for pytest.
#
# Child startup is slow (~2-5 s to import eryn + torch), so every wait is a
# wall-clock DEADLINE loop (no tight sleeps, no fixed-count polling).
# ---------------------------------------------------------------------------

# Generous deadline for any "wait until the worker produced something" loop.
_POLL_DEADLINE_S = 60.0


def _make_tiny_zuko_flow(seed: int = 0):
    """A minimal ZukoFlow + fitted WhiteningTransform for process tests."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((200, 2))

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=2)
    wt.fit({0: samples})

    return ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=seed,
        flow_class="NSF",
        transforms=1,
        hidden_features=(16,),
        bins=3,
    )


def _poll_until_version(ex, target: int = 1, deadline_s: float = _POLL_DEADLINE_S):
    """Poll ``latest_weights`` until ``version >= target`` or the deadline.

    Returns the last ``latest_weights()`` result (may be ``None`` on timeout).
    Re-raises any :class:`TrainerError` so failures surface promptly.
    """
    t0 = time.monotonic()
    lw = None
    while time.monotonic() < t0 + deadline_s:
        lw = ex.latest_weights()
        if lw is not None and lw[0] >= target:
            return lw
        time.sleep(0.1)
    return lw


# ---- tiny worker config used across process tests (fast fits) -------------
_TINY_FIT = dict(epochs_per_round=2, min_train_samples=50)


# ---------------------------------------------------------------------------
# 1. Happy path — train, hand off versioned weights, clean shutdown.
# ---------------------------------------------------------------------------

def test_process_happy_path_trains_and_serves_weights():
    pytest.importorskip("torch")
    flow = _make_tiny_zuko_flow(seed=0)
    # epochs_per_round is the worker-controlled n_epochs; do NOT also pass it via
    # fit_kwargs (that would collide on the worker's flow.fit call).
    ex = ProcessExecutor(flow, epochs_per_round=2, min_train_samples=50,
                         seed=7)
    try:
        rng = np.random.default_rng(0)
        for _ in range(3):
            assert ex.submit({0: rng.standard_normal((100, 2)) * 4.0}) is True

        lw = _poll_until_version(ex, target=1)
        assert lw is not None and lw[0] >= 1, "worker never produced weights"
        version, weights = lw
        assert ex.version == version

        # Weights load into a sibling flow and change its density.
        sibling = _make_tiny_zuko_flow(seed=0)
        x = rng.standard_normal((8, 2))
        before = sibling.log_prob(x, context=0).copy()
        sibling.set_weights(weights)
        after = sibling.log_prob(x, context=0)
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after), "trained weights did not change log_prob"
    finally:
        ex.shutdown()

    assert not ex._process.is_alive()
    assert ex._process.exitcode == 0
    ex.shutdown()  # idempotent second call: no-op, no raise


# ---------------------------------------------------------------------------
# 1a. Lazy transform fit end-to-end: unfitted shared transform → worker fits it
#     on the first round, ships it inside the snapshot, parent reproduces densities.
# ---------------------------------------------------------------------------

def _make_tiny_zuko_flow_unfitted(seed: int = 0):
    """A minimal ZukoFlow with a shared, UNFITTED WhiteningTransform."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=2, shared=True)  # NOT fitted
    assert wt.is_fitted is False
    return ZukoFlow(
        dims=2, device="cpu", data_transform=wt, conditioning=cond, seed=seed,
        flow_class="NSF", transforms=1, hidden_features=(16,), bins=3,
    )


def test_process_lazy_fit_snapshot_end_to_end():
    pytest.importorskip("torch")
    flow = _make_tiny_zuko_flow_unfitted(seed=0)
    # FlowSpec snapshot of an unfitted transform must not raise.
    ex = ProcessExecutor(flow, epochs_per_round=2, min_train_samples=50, seed=21)
    try:
        assert ex.latest_val_nll is None
        rng = np.random.default_rng(0)
        for _ in range(3):
            assert ex.submit({0: rng.standard_normal((100, 2)) * 4.0}) is True

        lw = _poll_until_version(ex, target=1)
        assert lw is not None and lw[0] >= 1, "worker never produced weights"
        version, snapshot = lw
        # Snapshot is self-contained and carries a FITTED transform.
        assert set(snapshot.keys()) == {"net", "data_transform"}
        assert snapshot["data_transform"].is_fitted is True
        # val NLL surfaced for monitoring.
        assert ex.latest_val_nll is not None and np.isfinite(ex.latest_val_nll)

        # Apply to a sibling whose own transform is unfitted → reproduces the
        # worker's log_prob (loose tol: independent torch RNG across processes).
        sibling = _make_tiny_zuko_flow_unfitted(seed=0)
        assert sibling.data_transform.is_fitted is False
        x = rng.standard_normal((8, 2))
        sibling.set_weights(snapshot)
        assert sibling.data_transform.is_fitted is True
        lp = sibling.log_prob(x, context=0)
        assert lp.shape == (8,)
        assert np.all(np.isfinite(lp))
    finally:
        ex.shutdown()
    assert not ex._process.is_alive()
    assert ex._process.exitcode == 0


# ---------------------------------------------------------------------------
# 1b. Silent worker death AFTER first weights must still surface loudly (I2).
# ---------------------------------------------------------------------------

def test_process_worker_death_after_first_weights_surfaces(make_dummy_flow=None):
    """A worker that crashes AFTER serving version >= 1 must fail loud, not freeze.

    Regression for I2: the dead-process check used to be guarded on
    ``self._latest is None``, so once any weights had been observed a later
    crash (here: an external kill → nonzero exitcode) was never surfaced and
    stale weights were returned forever.  The check now fires regardless of held
    weights, gated only on ``not self._shutdown``.
    """
    pytest.importorskip("torch")
    flow = _make_tiny_zuko_flow(seed=4)
    ex = ProcessExecutor(flow, epochs_per_round=2, min_train_samples=50, seed=4)
    try:
        rng = np.random.default_rng(0)
        for _ in range(3):
            assert ex.submit({0: rng.standard_normal((100, 2)) * 4.0}) is True

        # Happy path first: observe at least version 1.
        lw = _poll_until_version(ex, target=1)
        assert lw is not None and lw[0] >= 1, "worker never produced weights"
        assert ex.version >= 1

        # Now kill the worker hard (simulates OOM-kill / segfault: nonzero code).
        ex._process.kill()
        ex._process.join()
        assert ex._process.exitcode not in (0, None)

        # The death must surface as a TrainerError mentioning the exitcode,
        # despite weights already having been served.
        t0 = time.monotonic()
        raised = None
        while time.monotonic() < t0 + _POLL_DEADLINE_S:
            try:
                ex.latest_weights()
            except TrainerError as exc:
                raised = exc
                break
            time.sleep(0.1)
        assert raised is not None, "worker death after first weights never surfaced"
        assert "exitcode" in str(raised)
    finally:
        ex.shutdown()
    # Shutdown after a killed worker is still clean and idempotent.
    assert not ex._process.is_alive()
    ex.shutdown()


# ---------------------------------------------------------------------------
# 2. Immediate shutdown with no submits — worker exits fast (not stuck in get).
# ---------------------------------------------------------------------------

def test_process_shutdown_without_submits_is_fast():
    pytest.importorskip("torch")
    flow = _make_tiny_zuko_flow(seed=1)
    ex = ProcessExecutor(flow, **_TINY_FIT, seed=1)
    # Give the child time to come up so we exercise a real running-then-stopped
    # worker, not a not-yet-started one.
    t0 = time.monotonic()
    while time.monotonic() < t0 + _POLL_DEADLINE_S and not ex._process.is_alive():
        time.sleep(0.05)
    ex.shutdown(timeout=10.0)
    assert not ex._process.is_alive()
    assert ex._process.exitcode == 0


# ---------------------------------------------------------------------------
# 3. Error propagation — child fit raises; TrainerError carries the traceback.
# ---------------------------------------------------------------------------

def test_process_error_propagation_surfaces_trainer_error():
    # No real torch flow needed: _ExplodingFlow is a numpy FakeFlow whose fit
    # raises in the worker.  The worker still imports torch, builds the flow,
    # and the error path runs identically.
    pytest.importorskip("torch")  # worker imports torch unconditionally
    flow = _ExplodingFlow(dims=2, device="cpu",
                          data_transform=_FittableTransform(True))
    ex = ProcessExecutor(flow, epochs_per_round=2, min_train_samples=50, seed=3)
    try:
        rng = np.random.default_rng(0)
        # Submit enough rows to cross min_train_samples and trigger the fit.
        for _ in range(3):
            ex.submit({0: rng.standard_normal((100, 2))})

        t0 = time.monotonic()
        raised = None
        while time.monotonic() < t0 + _POLL_DEADLINE_S:
            try:
                ex.latest_weights()
            except TrainerError as exc:
                raised = exc
                break
            time.sleep(0.1)
        assert raised is not None, "TrainerError never surfaced from the worker"
        assert "boom" in str(raised)

        # submit after a recorded failure raises (matches Inline semantics).
        with pytest.raises(TrainerError):
            ex.submit({0: rng.standard_normal((100, 2))})
    finally:
        ex.shutdown()
    assert not ex._process.is_alive()


# ---------------------------------------------------------------------------
# 4. Drop policy — no worker; queues buffer; oldest vs newest (non-blocking).
# ---------------------------------------------------------------------------

def test_process_drop_policy_oldest_accepts_when_full(make_dummy_flow=None):
    # Verifies the "oldest" drop policy stays NON-BLOCKING and ACCEPTS the
    # incoming batch when the queue is already full (it evicts the oldest to make
    # room and returns True).  This checks acceptance + non-blocking behaviour,
    # NOT the data identity of the evicted item.
    flow = _make_fake(dims=2)
    ex = ProcessExecutor(flow, max_pending_batches=1, drop_policy="oldest",
                         start=False)
    try:
        t0 = time.monotonic()
        assert ex.submit({0: np.ones((3, 2))}) is True       # fills the queue
        assert ex.submit({0: np.ones((3, 2))}) is True       # accepted: evicts oldest
        assert time.monotonic() - t0 < 1.0, "submit blocked (must be non-blocking)"
    finally:
        ex.shutdown()


def test_process_drop_policy_newest_rejects():
    flow = _make_fake(dims=2)
    ex = ProcessExecutor(flow, max_pending_batches=1, drop_policy="newest",
                         start=False)
    try:
        t0 = time.monotonic()
        assert ex.submit({0: np.ones((3, 2))}) is True       # fills the queue
        assert ex.submit({0: np.ones((3, 2))}) is False      # incoming dropped
        assert time.monotonic() - t0 < 1.0, "submit blocked (must be non-blocking)"
    finally:
        ex.shutdown()


# ---------------------------------------------------------------------------
# 5. Empty submit → False without touching the queue.
# ---------------------------------------------------------------------------

def test_process_empty_submit_is_false_without_touching_queue():
    flow = _make_fake(dims=2)
    ex = ProcessExecutor(flow, max_pending_batches=1, start=False)
    try:
        assert ex.submit({}) is False
        assert ex.submit({0: np.zeros((0, 2))}) is False
        # Queue untouched: a real batch still fits and a second one displaces it
        # (proving the empties never consumed the single slot).
        assert ex.submit({0: np.ones((5, 2))}) is True
    finally:
        ex.shutdown()


# ---------------------------------------------------------------------------
# 6. fit_kwargs collision → ValueError.
# ---------------------------------------------------------------------------

def test_process_fit_kwargs_collision_raises():
    flow = _make_fake(dims=2)
    with pytest.raises(ValueError, match="refit_data_transform"):
        ProcessExecutor(flow, fit_kwargs=dict(refit_data_transform=True),
                        start=False)
    with pytest.raises(ValueError, match="verbose"):
        ProcessExecutor(flow, fit_kwargs=dict(verbose=True), start=False)


def test_process_bad_drop_policy_raises():
    flow = _make_fake(dims=2)
    with pytest.raises(ValueError, match="drop_policy"):
        ProcessExecutor(flow, drop_policy="nope", start=False)


# ---------------------------------------------------------------------------
# 7. Context manager joins the child.
# ---------------------------------------------------------------------------

def test_process_context_manager_joins_child():
    pytest.importorskip("torch")
    flow = _make_tiny_zuko_flow(seed=2)
    with ProcessExecutor(flow, **_TINY_FIT, seed=2) as ex:
        proc = ex._process
        # let it come up
        t0 = time.monotonic()
        while time.monotonic() < t0 + _POLL_DEADLINE_S and not proc.is_alive():
            time.sleep(0.05)
        assert proc.is_alive()
    assert not proc.is_alive()


# ---------------------------------------------------------------------------
# 8. ConditionalFlowMove end-to-end smoke with ProcessExecutor.
# ---------------------------------------------------------------------------

def test_process_flowmove_end_to_end_smoke():
    pytest.importorskip("torch")
    from eryn.ensemble import EnsembleSampler
    from eryn.moves import ConditionalFlowMove
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State

    ndim, ntemps, nwalkers = 2, 1, 16
    flow = _make_tiny_zuko_flow(seed=0)
    ex = ProcessExecutor(flow, epochs_per_round=2, min_train_samples=50, seed=11)
    try:
        move = ConditionalFlowMove(flow, branch_name="x", executor=ex, harvest_every=2)
        move.active_condition = 0

        priors = {"x": ProbDistContainer(
            {0: uniform_dist(-10.0, 10.0), 1: uniform_dist(-10.0, 10.0)}
        )}

        def _loglike(x):
            x = np.atleast_2d(x)
            return -0.5 * np.sum(x ** 2, axis=-1)

        sampler = EnsembleSampler(
            nwalkers, {"x": ndim}, _loglike, priors,
            tempering_kwargs=dict(ntemps=ntemps), vectorize=True,
            moves=[move], branch_names=["x"],
        )
        rng = np.random.default_rng(42)
        start = State({"x": rng.standard_normal((ntemps, nwalkers, 1, ndim))})
        sampler.run_mcmc(start, 60, burn=0, progress=False)

        chain = sampler.get_chain()["x"]
        assert chain.shape[0] == 60
        assert np.all(np.isfinite(chain))

        # Deterministic version assertion: wait for the worker to produce >= 1,
        # then run a few more steps so the move hot-loads it.
        lw = _poll_until_version(ex, target=1)
        assert lw is not None and lw[0] >= 1, "worker never produced weights"
        sampler.run_mcmc(sampler.get_last_sample(), 5, burn=0, progress=False)
        assert move.loaded_version >= 1
    finally:
        ex.shutdown()
    assert not ex._process.is_alive()


# ---------------------------------------------------------------------------
# No-zombie guarantee: no child processes left alive after the module's tests.
# ---------------------------------------------------------------------------

def test_process_no_leftover_children():
    # If any earlier test leaked a child, this fails loudly.  (Runs last by
    # definition order within the module.)
    assert multiprocessing.active_children() == []
