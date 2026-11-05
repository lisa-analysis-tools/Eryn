# tests/test_flow_base_contract.py
"""Tests for the torch-free Flow ABC and FlowProposalDistribution.

NOTE: ZukoFlow contract tests are added in T3.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from eryn.flows.base import Flow, FlowHistory, FlowProposalDistribution
from eryn.flows.transforms import IdentityTransform


# ---------------------------------------------------------------------------
# Minimal concrete Flow subclasses for testing (no torch, standard normal)
# ---------------------------------------------------------------------------

class GaussianFlow(Flow):
    """Standard-normal flow for testing the ABC contract."""

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning)

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum(x ** 2, axis=-1) - 0.5 * x.shape[-1] * np.log(2 * np.pi)

    def sample(self, n: int, context=None):
        return np.random.randn(n, self.dims)

    def sample_and_log_prob(self, n: int, context=None):
        x = self.sample(n, context=context)
        lp = self.log_prob(x, context=context)
        return x, lp

    def fit(self, samples, **kwargs):
        return FlowHistory()

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


class GaussianFlowKwargs(Flow):
    """Subclass with extra **flow_kwargs to test VAR_KEYWORD flattening in __new__."""

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None,
                 **flow_kwargs):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning)
        self.flow_kwargs = flow_kwargs

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum(x ** 2, axis=-1) - 0.5 * x.shape[-1] * np.log(2 * np.pi)

    def sample(self, n: int, context=None):
        return np.random.randn(n, self.dims)

    def sample_and_log_prob(self, n: int, context=None):
        x = self.sample(n, context=context)
        lp = self.log_prob(x, context=context)
        return x, lp

    def fit(self, samples, **kwargs):
        return FlowHistory()

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


# ---------------------------------------------------------------------------
# Flow ABC tests
# ---------------------------------------------------------------------------

def test_gaussian_flow_instantiates():
    f = GaussianFlow(dims=3)
    assert f.dims == 3


def test_flow_default_data_transform():
    """If no data_transform given, defaults to IdentityTransform."""
    f = GaussianFlow(dims=3)
    assert isinstance(f.data_transform, IdentityTransform)


def test_flow_custom_data_transform():
    tr = IdentityTransform()
    f = GaussianFlow(dims=3, data_transform=tr)
    assert f.data_transform is tr


def test_flow_device_stored():
    f = GaussianFlow(dims=3, device="cpu")
    assert f.device == "cpu"


def test_flow_conditioning_none():
    f = GaussianFlow(dims=3)
    assert f.conditioning is None


def test_config_dict_contains_dims():
    f = GaussianFlow(dims=5)
    cfg = f.config_dict()
    assert cfg["dims"] == 5


def test_config_dict_contains_defaults():
    """config_dict must include default values (device, data_transform, conditioning)."""
    f = GaussianFlow(dims=2)
    cfg = f.config_dict()
    assert "device" in cfg
    assert "data_transform" in cfg
    assert "conditioning" in cfg


def test_config_dict_excludes_self():
    f = GaussianFlow(dims=2)
    cfg = f.config_dict()
    assert "self" not in cfg


def test_init_args_excludes_self():
    f = GaussianFlow(dims=2)
    assert "self" not in f._init_args


def test_config_dict_round_trip():
    """config_dict() produces kwargs that reconstruct an equivalent instance."""
    f = GaussianFlow(dims=7)
    cfg = f.config_dict()
    f2 = GaussianFlow(**cfg)
    assert f2.dims == f.dims
    assert f2.device == f.device
    assert f2.conditioning == f.conditioning


def test_flow_cannot_instantiate_abc():
    """Flow ABC itself cannot be instantiated."""
    with pytest.raises(TypeError):
        Flow(dims=3)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FlowHistory tests
# ---------------------------------------------------------------------------

def test_flow_history_default():
    h = FlowHistory()
    assert h.training_loss == []
    assert h.validation_loss == []


def test_flow_history_fields():
    h = FlowHistory(training_loss=[1.0, 0.8], validation_loss=[1.1, 0.9])
    assert h.training_loss == [1.0, 0.8]
    assert h.validation_loss == [1.1, 0.9]


# ---------------------------------------------------------------------------
# FlowProposalDistribution tests
# ---------------------------------------------------------------------------

def test_fpd_logpdf_shape():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    x = np.random.randn(10, 3)
    lp = fpd.logpdf(x)
    assert lp.shape == (10,)


def test_fpd_rvs_int_size():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs(5)
    assert samples.shape == (5, 3)


def test_fpd_rvs_tuple_size():
    """rvs accepts a tuple with a single element (eryn DistributionGenerate compat)."""
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs((7,))
    assert samples.shape == (7, 3)


def test_fpd_logpdf_finite():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    x = np.zeros((5, 3))
    lp = fpd.logpdf(x)
    assert np.all(np.isfinite(lp))


def test_fpd_condition_stored():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=2)
    assert fpd.condition == 2


def test_fpd_rvs_2d_tuple_shape():
    """rvs((5, 3)) must return shape (5, 3, dims) — size + (dims,) contract."""
    f = GaussianFlow(dims=4)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs((5, 3))
    assert samples.shape == (5, 3, 4)


def test_fpd_rvs_kwargs_accepted():
    """rvs and logpdf accept and ignore extra kwargs (eryn random_state compat)."""
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs(5, random_state=42)
    assert samples.shape == (5, 3)
    x = np.random.randn(5, 3)
    lp = fpd.logpdf(x, random_state=42)
    assert lp.shape == (5,)


# ---------------------------------------------------------------------------
# VAR_KEYWORD flattening in Flow.__new__ (Fix 3)
# ---------------------------------------------------------------------------

def test_var_keyword_flattened_in_init_args():
    """**flow_kwargs contents are merged into _init_args, not nested."""
    f = GaussianFlowKwargs(dims=5, n_transforms=8, lr=1e-3)
    cfg = f.config_dict()
    # Extra kwargs must be top-level, not nested under 'flow_kwargs'
    assert "flow_kwargs" not in cfg
    assert cfg["n_transforms"] == 8
    assert cfg["lr"] == 1e-3
    assert cfg["dims"] == 5


def test_var_keyword_round_trip():
    """A subclass with **flow_kwargs round-trips through cls(**config_dict())."""
    f = GaussianFlowKwargs(dims=5, n_transforms=8, lr=1e-3)
    cfg = f.config_dict()
    f2 = GaussianFlowKwargs(**cfg)
    assert f2.dims == f.dims
    assert f2.flow_kwargs == f.flow_kwargs


def test_config_dict_pickle_survives():
    """config_dict() output survives pickle round-trip (Task-9 spawn-boundary scenario)."""
    f = GaussianFlowKwargs(dims=5, n_transforms=8, lr=1e-3)
    cfg = f.config_dict()
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg2 == cfg


# ---------------------------------------------------------------------------
# ZukoFlow contract — torch-dependent tests
# Each function calls pytest.importorskip("torch") so the torch-free tests
# above remain collectible and runnable without torch installed.
# ---------------------------------------------------------------------------

def _make_zuko_flow_for_contract():
    """Build a small ZukoFlow with WhiteningTransform + OneHotLeafConditioning."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")

    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 3))

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: samples})

    flow = ZukoFlow(
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
    return flow


def test_zukoflow_config_dict_round_trip_reproduces_log_prob():
    """ZukoFlow(**flow.config_dict()) + set_weights reproduces log_prob (atol 1e-6)."""
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow

    flow = _make_zuko_flow_for_contract()
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    cfg = flow.config_dict()
    flow2 = ZukoFlow(**cfg)
    flow2.set_weights(flow.get_weights())

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="config_dict round-trip changed log_prob")


def test_zukoflow_get_weights_cpu_detached_isolated():
    """get_weights() tensors are CPU and detached; mutating them leaves the flow unchanged."""
    torch = pytest.importorskip("torch")

    flow = _make_zuko_flow_for_contract()
    x = np.zeros((4, 3), dtype=np.float64)
    lp_before = flow.log_prob(x, context=0)

    weights = flow.get_weights()
    # All on CPU
    assert all(v.device.type == "cpu" for v in weights.values())
    # Mutating the returned dict must not affect the flow
    for v in weights.values():
        v.fill_(999.0)
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8)


def test_zukoflow_get_snapshot_round_trip_reproduces_log_prob():
    """get_snapshot() → set_weights() on a fresh flow with an UNFITTED transform
    reproduces log_prob to ~1e-6, proving the transform handoff makes the parent
    usable.  Mutating the returned snapshot does not change the live flow."""
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    flow = _make_zuko_flow_for_contract()  # transform already fitted
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    snapshot = flow.get_snapshot()
    assert set(snapshot.keys()) == {"net", "data_transform"}
    assert snapshot["data_transform"].is_fitted is True

    # Fresh, identically-configured flow but with an UNFITTED transform.
    cond = OneHotLeafConditioning(nleaves_max=1)
    fresh_transform = WhiteningTransform(ndim=3)
    assert fresh_transform.is_fitted is False
    fresh = ZukoFlow(
        dims=3, device="cpu", data_transform=fresh_transform, conditioning=cond,
        seed=0, flow_class="NSF", transforms=2, hidden_features=(32, 32), bins=4,
    )
    # The snapshot installs the fitted transform AND the net atomically.
    fresh.set_weights(snapshot)
    assert fresh.data_transform.is_fitted is True

    lp1 = flow.log_prob(x, context=0)
    lp2 = fresh.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="snapshot handoff changed log_prob")


def test_zukoflow_get_snapshot_isolation():
    """Mutating the snapshot must not change the live flow (deep-copy isolation)."""
    torch = pytest.importorskip("torch")

    flow = _make_zuko_flow_for_contract()
    x = np.zeros((4, 3), dtype=np.float64)
    lp_before = flow.log_prob(x, context=0)

    snapshot = flow.get_snapshot()
    for v in snapshot["net"].values():
        v.fill_(999.0)
    # The deep-copied transform is a different object than the live one.
    assert snapshot["data_transform"] is not flow.data_transform

    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8)


def test_zukoflow_set_weights_bare_state_dict_still_works():
    """Legacy set_weights(get_weights()) (bare state_dict) must keep working."""
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow

    flow = _make_zuko_flow_for_contract()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 3))
    lp_before = flow.log_prob(x, context=0)

    # Bare state_dict round-trip — the legacy contract.
    flow.set_weights(flow.get_weights())
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8,
                               err_msg="bare state_dict round-trip changed log_prob")


def test_zukoflow_h5_save_load_round_trip(tmp_path):
    """h5 save/load round-trip reproduces log_prob exactly and restores transform+conditioning."""
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    from eryn.flows import ZukoFlow

    flow = _make_zuko_flow_for_contract()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 3))

    h5_path = str(tmp_path / "contract_flow.h5")
    flow.save(h5_path)
    flow2 = ZukoFlow.load(h5_path)

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-8,
                               err_msg="h5 save/load changed log_prob")

    # Conditioning and transform must be functional
    assert flow2.conditioning is not None
    assert flow2.conditioning.context_dim == flow.conditioning.context_dim
    z = flow2.data_transform.forward(x, condition=0)
    assert z.shape == (64, 3)
