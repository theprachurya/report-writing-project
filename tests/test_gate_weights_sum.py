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


def test_gate_temperature_t_g_alias() -> None:
    """T_g parameter works both in __init__ and per-forward call."""
    estimator = ReliabilityEstimator(
        visual_feature_dim=16,
        audio_feature_dim=16,
        T_g=0.5,
    )
    assert estimator.gate_temperature == 0.5
    assert estimator.T_g == 0.5

    b = 2
    h_v = torch.randn(b, 16)
    h_a = torch.randn(b, 16)
    u_v = torch.rand(b, 1)
    u_a = torch.rand(b, 1)

    out = estimator(h_v=h_v, h_a=h_a, uncertainty_v=u_v, uncertainty_a=u_a, T_g=2.0)
    w_sum = out["w_v"] + out["w_a"]
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-6)

    with pytest.raises(ValueError, match="Temperature T_g must be positive"):
        estimator(h_v=h_v, h_a=h_a, uncertainty_v=u_v, uncertainty_a=u_a, T_g=-1.0)


def test_fusion_classifier_modes_and_validation() -> None:
    """FusionClassifier supports reliability, concat, equal_weight, learned_fixed and raises ValueError."""
    from src.models.fusion_classifier import FusionClassifier

    b, embed_dim, sync_dim = 3, 32, 4
    z_v = torch.randn(b, embed_dim)
    z_a = torch.randn(b, embed_dim)
    w_v = torch.full((b, 1), 0.6)
    w_a = torch.full((b, 1), 0.4)
    sync = torch.randn(b, sync_dim)

    # 1. Reliability mode
    fc_rel = FusionClassifier(embedding_dim=embed_dim, sync_feature_dim=sync_dim, fusion_mode="reliability")
    out_rel = fc_rel(z_v, z_a, w_v=w_v, w_a=w_a, sync_features=sync)
    assert out_rel["prob"].shape == (b, 1)

    # Missing weights in reliability mode raises ValueError
    with pytest.raises(ValueError, match="Reliability mode requires w_v and w_a"):
        fc_rel(z_v, z_a, w_v=None, w_a=w_a)

    # Bad dimensionality raises ValueError
    with pytest.raises(ValueError, match="Expected 2D tensors"):
        fc_rel(z_v.unsqueeze(1), z_a, w_v=w_v, w_a=w_a)

    # 2. Concat mode (unweighted, no explicit w_v/w_a appended)
    fc_concat = FusionClassifier(embedding_dim=embed_dim, sync_feature_dim=sync_dim, fusion_mode="concat")
    out_concat = fc_concat(z_v, z_a, sync_features=sync)
    assert out_concat["prob"].shape == (b, 1)

    # 3. Equal weight mode
    fc_eq = FusionClassifier(embedding_dim=embed_dim, sync_feature_dim=sync_dim, fusion_mode="equal_weight")
    out_eq = fc_eq(z_v, z_a, sync_features=sync)
    assert torch.allclose(out_eq["w_v"], torch.full((b, 1), 0.5))
    assert torch.allclose(out_eq["w_a"], torch.full((b, 1), 0.5))

    # 4. Learned fixed mode
    fc_fixed = FusionClassifier(embedding_dim=embed_dim, sync_feature_dim=sync_dim, fusion_mode="learned_fixed")
    out_fixed = fc_fixed(z_v, z_a, sync_features=sync)
    assert torch.allclose(out_fixed["w_v"] + out_fixed["w_a"], torch.ones(b, 1), atol=1e-6)


def test_sync_features_dispersion_calculation() -> None:
    """SyncFeatureExtractor computes dispersion as standard deviation across offsets."""
    from src.models.sync_features import SyncFeatureExtractor

    extractor = SyncFeatureExtractor(offsets=[-2, -1, 0, 1, 2], output_dim=4)
    b, t, d = 2, 8, 16
    proj_v = torch.randn(b, t, d)
    proj_a = torch.randn(b, t, d)

    sync = extractor(proj_v, proj_a)
    assert sync.shape == (b, 4)
    # 4th feature is dispersion >= 0
    dispersion = sync[:, 3]
    assert (dispersion >= 0.0).all()
    assert not torch.isnan(sync).any()


def test_losses_numerical_stability_and_schedules() -> None:
    """Verify reliability_loss clamp, contrastive real filtering, and balance loss schedule."""
    from src.train.losses import reliability_loss, contrastive_loss, CombinedLoss

    b = 4
    weights = torch.softmax(torch.randn(b, 2), dim=-1)
    # Extreme logits that would produce extreme losses
    logit_v = torch.ones(b, 1) * 100.0
    logit_a = torch.ones(b, 1) * -100.0
    target = torch.zeros(b, 1)

    loss = reliability_loss(weights, logit_v, logit_a, target, tau_r=0.01)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Contrastive loss with empty real batch (all target=1 fakes)
    proj_v = torch.randn(b, 8, 16)
    proj_a = torch.randn(b, 8, 16)
    target_all_fake = torch.ones(b, 1)
    con_loss = contrastive_loss(proj_v, proj_a, target=target_all_fake)
    assert con_loss.item() == 0.0

    # Contrastive loss with real samples
    target_half_real = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    con_loss_real = contrastive_loss(proj_v, proj_a, target=target_half_real)
    assert con_loss_real.item() > 0.0

    # CombinedLoss balance schedule auto-disable
    comb_loss = CombinedLoss(eta=0.5, balance_warmup_epochs=2, balance_schedule="step")
    assert comb_loss._get_effective_eta(epoch=1) == 0.5
    assert comb_loss._get_effective_eta(epoch=2) == 0.5
    assert comb_loss._get_effective_eta(epoch=3) == 0.0  # Auto-disabled!

    comb_linear = CombinedLoss(eta=0.5, balance_warmup_epochs=2, balance_schedule="linear")
    assert comb_linear._get_effective_eta(epoch=1) == 0.5
    assert comb_linear._get_effective_eta(epoch=2) == 0.25
    assert comb_linear._get_effective_eta(epoch=3) == 0.0

