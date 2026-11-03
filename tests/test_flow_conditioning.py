# tests/test_flow_conditioning.py
"""Tests for eryn.flows.conditioning — numpy-based one-hot encoding and assignment."""
from __future__ import annotations

import numpy as np
import pytest

from eryn.flows.conditioning import OneHotLeafConditioning


def test_onehot_context_dim():
    cond = OneHotLeafConditioning(nleaves_max=6)
    assert cond.context_dim == 6


def test_onehot_encode_shape():
    cond = OneHotLeafConditioning(nleaves_max=6)
    ctx = cond.encode(2)
    assert ctx.shape == (6,)


def test_onehot_encode_argmax():
    cond = OneHotLeafConditioning(nleaves_max=6)
    ctx = cond.encode(2)
    assert np.argmax(ctx) == 2


def test_onehot_encode_sum_one():
    cond = OneHotLeafConditioning(nleaves_max=6)
    ctx = cond.encode(2)
    assert np.isclose(ctx.sum(), 1.0)


def test_onehot_encode_dtype_float32():
    cond = OneHotLeafConditioning(nleaves_max=4)
    ctx = cond.encode(0)
    assert ctx.dtype == np.float32


def test_onehot_encode_first_index():
    cond = OneHotLeafConditioning(nleaves_max=4)
    ctx = cond.encode(0)
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(ctx, expected)


def test_onehot_encode_last_index():
    cond = OneHotLeafConditioning(nleaves_max=4)
    ctx = cond.encode(3)
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(ctx, expected)


def test_onehot_rejects_out_of_range_high():
    cond = OneHotLeafConditioning(nleaves_max=3)
    with pytest.raises(ValueError):
        cond.encode(3)


def test_onehot_rejects_out_of_range_negative():
    cond = OneHotLeafConditioning(nleaves_max=3)
    with pytest.raises(ValueError):
        cond.encode(-1)


def test_set_centroids_and_assign():
    cond = OneHotLeafConditioning(nleaves_max=3)
    centroids = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], dtype=float)
    cond.set_centroids(centroids)
    # Nearest to [0.1, 0.1] is centroid 0
    assert cond.assign(np.array([0.1, 0.1])) == 0
    # Nearest to [4.9, 0.1] is centroid 1
    assert cond.assign(np.array([4.9, 0.1])) == 1
    # Nearest to [0.1, 4.9] is centroid 2
    assert cond.assign(np.array([0.1, 4.9])) == 2


def test_assign_requires_set_centroids():
    cond = OneHotLeafConditioning(nleaves_max=3)
    with pytest.raises(RuntimeError):
        cond.assign(np.array([0.0, 0.0]))


def test_conditioning_strategy_protocol():
    """OneHotLeafConditioning satisfies the ConditioningStrategy Protocol."""
    from eryn.flows.conditioning import ConditioningStrategy

    cond = OneHotLeafConditioning(nleaves_max=4)
    assert isinstance(cond, ConditioningStrategy)
