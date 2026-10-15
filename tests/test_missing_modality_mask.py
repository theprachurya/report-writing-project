"""Verifies missing-modality masking degrades safely rather than fabricating confidence.

Status: implemented — tests that when visual, audio, or both modalities are missing
or masked, the reliability gate and full end-to-end model degrade safely, producing
valid bounded probabilities in [0, 1] without NaN/Inf or hallucinated high confidence.
See AGENTS.md for build order.
"""

from __future__ import annotations

import pytest
import torch

from src.models.full_model import FullModel
from src.models.fusion_classifier import FusionClassifier
from src.models.reliability_estimator import ReliabilityEstimator


def test_missing_modality_full_model_visual_missing() -> None:
    """When visual input is missing, model runs safely and outputs valid probabilities."""
    model = FullModel()
    model.eval()

    b = 2
    # 16 face crops of 160x160 RGB
    frames = torch.randn(b, 16, 3, 160, 160)
    # Log-Mel spectrogram (80 mels, 200 time steps)
    mel = torch.randn(b, 1, 80, 200)

    visual_missing = torch.ones(b, 1)  # Visual is missing
    audio_missing = torch.zeros(b, 1)  # Audio is present

    with torch.no_grad():
        out = model(
            frames=frames,
            mel_spec=mel,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

    p_f = out["p_f"]
    p_v = out["p_v"]
    p_a = out["p_a"]
    w_v = out["w_v"]
    w_a = out["w_a"]

    # No NaN or Inf
    for tensor_name, tensor in [
        ("p_f", p_f), ("p_v", p_v), ("p_a", p_a), ("w_v", w_v), ("w_a", w_a)
    ]:
        assert not torch.isnan(tensor).any(), f"NaN found in {tensor_name}"
        assert not torch.isinf(tensor).any(), f"Inf found in {tensor_name}"

    # Probabilities in [0, 1]
    assert (p_f >= 0.0).all() and (p_f <= 1.0).all()
    assert (p_v >= 0.0).all() and (p_v <= 1.0).all()
    assert (p_a >= 0.0).all() and (p_a <= 1.0).all()

    # Weights sum to 1
    assert torch.allclose(w_v + w_a, torch.ones_like(w_v), atol=1e-5)
    assert torch.equal(w_v, torch.zeros_like(w_v))
    assert torch.equal(w_a, torch.ones_like(w_a))


def test_missing_modality_full_model_audio_missing() -> None:
    """When audio input is missing, model runs safely and outputs valid probabilities."""
    model = FullModel()
    model.eval()

    b = 2
    frames = torch.randn(b, 16, 3, 160, 160)
    mel = torch.randn(b, 1, 80, 200)

    visual_missing = torch.zeros(b, 1)  # Visual is present
    audio_missing = torch.ones(b, 1)   # Audio is missing

    with torch.no_grad():
        out = model(
            frames=frames,
            mel_spec=mel,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

    p_f = out["p_f"]
    w_v = out["w_v"]
    w_a = out["w_a"]

    assert not torch.isnan(p_f).any()
    assert not torch.isinf(p_f).any()
    assert (p_f >= 0.0).all() and (p_f <= 1.0).all()
    assert torch.allclose(w_v + w_a, torch.ones_like(w_v), atol=1e-5)
    assert torch.equal(w_v, torch.ones_like(w_v))
    assert torch.equal(w_a, torch.zeros_like(w_a))


def test_missing_modality_full_model_both_missing() -> None:
    """When both modalities are missing, model degrades gracefully without crashing."""
    model = FullModel()
    model.eval()

    b = 2
    frames = torch.zeros(b, 16, 3, 160, 160)
    mel = torch.zeros(b, 1, 80, 200)

    visual_missing = torch.ones(b, 1)
    audio_missing = torch.ones(b, 1)

    with torch.no_grad():
        out = model(
            frames=frames,
            mel_spec=mel,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

    p_f = out["p_f"]
    assert not torch.isnan(p_f).any()
    assert not torch.isinf(p_f).any()
    assert (p_f >= 0.0).all() and (p_f <= 1.0).all()


def test_missingness_gate_behavior() -> None:
    """Reliability estimator produces finite normalized weights under missing flags."""
    estimator = ReliabilityEstimator(
        visual_feature_dim=64,
        audio_feature_dim=64,
        hidden_dim=32,
    )
    estimator.eval()

    b = 4
    h_v = torch.randn(b, 64)
    h_a = torch.randn(b, 64)
    u_v = torch.rand(b, 1)
    u_a = torch.rand(b, 1)

    # Test cases: normal, visual missing, audio missing, both missing
    cases = [
        (torch.zeros(b, 1), torch.zeros(b, 1)),
        (torch.ones(b, 1), torch.zeros(b, 1)),
        (torch.zeros(b, 1), torch.ones(b, 1)),
        (torch.ones(b, 1), torch.ones(b, 1)),
    ]

    for vm, am in cases:
        out = estimator(
            h_v=h_v,
            h_a=h_a,
            uncertainty_v=u_v,
            uncertainty_a=u_a,
            visual_missing=vm,
            audio_missing=am,
        )
        assert not torch.isnan(out["w_v"]).any()
        assert not torch.isnan(out["w_a"]).any()
        assert torch.allclose(out["w_v"] + out["w_a"], torch.ones(b, 1), atol=1e-6)


def test_fusion_classifier_with_zeroed_inputs() -> None:
    """Fusion classifier produces bounded probabilities even when one modality is zeroed."""
    classifier = FusionClassifier(
        embedding_dim=64,
        sync_feature_dim=4,
        hidden_dim=64,
        fusion_mode="reliability",
    )
    classifier.eval()

    b = 4
    # Case 1: zeroed visual, active audio
    z_v_zero = torch.zeros(b, 64)
    z_a_active = torch.randn(b, 64)
    w_v = torch.zeros(b, 1)
    w_a = torch.ones(b, 1)
    sync = torch.zeros(b, 4)

    out = classifier(z_v_bar=z_v_zero, z_a_bar=z_a_active, w_v=w_v, w_a=w_a, sync_features=sync)
    p_f = out["prob"]
    assert not torch.isnan(p_f).any()
    assert (p_f >= 0.0).all() and (p_f <= 1.0).all()

    # Case 2: active visual, zeroed audio
    z_v_active = torch.randn(b, 64)
    z_a_zero = torch.zeros(b, 64)
    w_v = torch.ones(b, 1)
    w_a = torch.zeros(b, 1)

    out2 = classifier(z_v_bar=z_v_active, z_a_bar=z_a_zero, w_v=w_v, w_a=w_a, sync_features=sync)
    p_f2 = out2["prob"]
    assert not torch.isnan(p_f2).any()
    assert (p_f2 >= 0.0).all() and (p_f2 <= 1.0).all()
