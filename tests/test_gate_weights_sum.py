"""Verifies reliability_estimator outputs w_v + w_a == 1 for arbitrary inputs.

Status: implemented — tests gate weight normalization under arbitrary inputs,
extreme values, various ablation settings, and missing modality conditions.
See AGENTS.md for build order.
"""

from __future__ import annotations

import pytest
import torch

from src.models.reliability_estimator import ReliabilityEstimator


def test_gate_weights_sum_to_one_standard() -> None:
    """Standard random inputs produce w_v + w_a == 1."""
    estimator = ReliabilityEstimator(
        visual_feature_dim=256,
        audio_feature_dim=256,
        sync_feature_dim=4,
        hidden_dim=128,
        use_sync_features=True,
        use_quality_features=True,
    )
    estimator.eval()

    batch_sizes = [1, 4, 16]
    for b in batch_sizes:
        h_v = torch.randn(b, 256)
        h_a = torch.randn(b, 256)
        u_v = torch.rand(b, 1)
        u_a = torch.rand(b, 1)
        sync = torch.randn(b, 4)
        q_v = torch.rand(b, 7)
        q_a = torch.rand(b, 5)

        out = estimator(
            h_v=h_v,
            h_a=h_a,
            uncertainty_v=u_v,
            uncertainty_a=u_a,
            sync_features=sync,
            quality_v=q_v,
            quality_a=q_a,
        )

        w_v = out["w_v"]
        w_a = out["w_a"]
        weights = out["weights"]

        # w_v + w_a must sum to 1
        sum_weights = w_v + w_a
        assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

        # weights.sum(dim=-1) must sum to 1
        assert torch.allclose(
            weights.sum(dim=-1, keepdim=True),
            torch.ones(b, 1),
            atol=1e-6,
        )

        # Non-negativity and <= 1 bounds
        assert (w_v >= 0.0).all() and (w_v <= 1.0).all()
        assert (w_a >= 0.0).all() and (w_a <= 1.0).all()


def test_gate_weights_sum_to_one_extreme_inputs() -> None:
    """Extreme / edge-case inputs must still sum to 1 and remain finite."""
    estimator = ReliabilityEstimator(
        visual_feature_dim=64,
        audio_feature_dim=64,
        sync_feature_dim=4,
        hidden_dim=64,
    )
    estimator.eval()

    test_cases = [
        ("all_zeros", torch.zeros(4, 64), torch.zeros(4, 64)),
        ("large_positive", torch.ones(4, 64) * 1e4, torch.ones(4, 64) * 1e4),
        ("large_negative", torch.ones(4, 64) * -1e4, torch.ones(4, 64) * -1e4),
        ("asymmetric", torch.ones(4, 64) * 1e3, torch.ones(4, 64) * -1e3),
    ]

    for name, h_v, h_a in test_cases:
        u_v = torch.zeros(4, 1)
        u_a = torch.zeros(4, 1)

        out = estimator(h_v=h_v, h_a=h_a, uncertainty_v=u_v, uncertainty_a=u_a)
        w_v = out["w_v"]
        w_a = out["w_a"]

        assert not torch.isnan(w_v).any(), f"NaN in w_v for {name}"
        assert not torch.isnan(w_a).any(), f"NaN in w_a for {name}"
        assert torch.allclose(w_v + w_a, torch.ones_like(w_v), atol=1e-6)


def test_gate_weights_sum_to_one_ablations_and_temperatures() -> None:
    """Verify sum-to-one constraint across ablation configurations and temperatures."""
    temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
    configs = [
        {"use_sync_features": False, "use_quality_features": False},
        {"use_sync_features": True, "use_quality_features": False},
        {"use_sync_features": False, "use_quality_features": True},
        {"use_sync_features": True, "use_quality_features": True},
    ]

    for temp in temperatures:
        for cfg in configs:
            estimator = ReliabilityEstimator(
                visual_feature_dim=32,
                audio_feature_dim=32,
                gate_temperature=temp,
                **cfg,
            )
            estimator.eval()

            h_v = torch.randn(8, 32)
            h_a = torch.randn(8, 32)
            u_v = torch.rand(8, 1)
            u_a = torch.rand(8, 1)

            out = estimator(
                h_v=h_v,
                h_a=h_a,
                uncertainty_v=u_v,
                uncertainty_a=u_a,
            )
            w_sum = out["w_v"] + out["w_a"]
            assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-6)
            assert (out["w_v"] >= 0.0).all()
            assert (out["w_a"] >= 0.0).all()


def test_gate_weights_with_missingness_flags() -> None:
    """Missingness flags preserve sum-to-one property."""
    estimator = ReliabilityEstimator(visual_feature_dim=32, audio_feature_dim=32)
    estimator.eval()

    b = 6
    h_v = torch.randn(b, 32)
    h_a = torch.randn(b, 32)
    u_v = torch.rand(b, 1)
    u_a = torch.rand(b, 1)

    # Various missingness combinations
    vm = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    am = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.5], [0.5]])

    out = estimator(
        h_v=h_v,
        h_a=h_a,
        uncertainty_v=u_v,
        uncertainty_a=u_a,
        visual_missing=vm,
        audio_missing=am,
    )

    w_sum = out["w_v"] + out["w_a"]
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-6)
