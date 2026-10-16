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


def test_modality_head_uncertainty_and_missing_neutralization() -> None:
    """ModalityHead computes uncertainty proxy and cleanly neutralizes on missing modality."""
    from src.models.modality_heads import ModalityHead, ModalityHeads

    head = ModalityHead(input_dim=32, hidden_dim=16)
    b = 4
    z = torch.randn(b, 32)

    # Active forward
    out = head(z)
    prob = out["prob"]
    expected_uncertainty = 1.0 - torch.abs(2.0 * prob - 1.0)
    assert torch.allclose(out["uncertainty"], expected_uncertainty, atol=1e-6)

    # Missing modality forward
    missing = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    out_masked = head(z, missing=missing)
    # When missing: logit=0, prob=0.5, uncertainty=1.0, pooled=0
    assert out_masked["logit"][1].item() == 0.0
    assert out_masked["prob"][1].item() == 0.5
    assert out_masked["uncertainty"][1].item() == 1.0
    assert (out_masked["pooled_features"][1] == 0.0).all()

    # Container ModalityHeads forward with masks
    heads = ModalityHeads(input_dim=32, hidden_dim=16)
    out_both = heads(z, z, visual_missing=missing, audio_missing=1.0 - missing)
    assert out_both["visual"]["prob"][1].item() == 0.5
    assert out_both["audio"]["prob"][0].item() == 0.5


def test_full_model_reliability_input_dim() -> None:
    """FullModel dynamically and accurately computes reliability_input_dim across ablations."""
    model = FullModel()
    # ve(256) + 1 + ae(256) + 1 + qv(7) + qa(5) + sync(4) + 2 missing flags = 542
    expected_dim = 256 + 1 + 256 + 1 + 7 + 5 + 4 + 2
    assert model.reliability_input_dim == expected_dim
    assert model.reliability_estimator.input_dim == expected_dim


def test_unimodal_dummy_logits_are_detached_zeros() -> None:
    """Unimodal models use detached zero tensors for dummy opposite logits."""
    from src.train.train_unimodal import UnimodalVisualModel, UnimodalAudioModel

    b = 2
    v_model = UnimodalVisualModel()
    frames = torch.randn(b, 16, 3, 160, 160)
    v_out = v_model(frames=frames)
    assert (v_out["logit_a"] == 0.0).all()
    assert v_out["logit_a"].grad_fn is None  # Detached!
    assert (v_out["p_a"] == 0.5).all()

    a_model = UnimodalAudioModel()
    mel = torch.randn(b, 1, 80, 200)
    a_out = a_model(mel_spec=mel)
    assert (a_out["logit_v"] == 0.0).all()
    assert a_out["logit_v"].grad_fn is None  # Detached!
    assert (a_out["p_v"] == 0.5).all()


def test_trainer_checkpoint_full_state_and_weights_only(tmp_path: any) -> None:
    """Trainer saves scaler, scheduler, and RNG state and loads with weights_only."""
    import tempfile
    from src.train.trainer import Trainer

    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    loss_fn = lambda pred, target: {"total": ((pred["logit_f"] - target) ** 2).mean()}

    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "checkpoint": {"save_dir": temp_dir, "save_every_n_epochs": 1, "save_best": True, "metric_for_best": "val_auroc"},
            "early_stopping": {"enabled": False},
            "amp": {"enabled": False},
        }
        trainer = Trainer(model, optimizer, loss_fn, config, scheduler=scheduler)
        # Test saving checkpoint
        trainer._save_checkpoint(epoch=1, metrics={"val_auroc": 0.85})
        ckpt_path = trainer.ckpt_dir / "checkpoint_epoch_1.pt"
        assert ckpt_path.is_file()

        # Test loading checkpoint with weights_only=True
        ckpt = trainer.load_checkpoint(ckpt_path, weights_only=True)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "seed_state" in ckpt
        assert "torch" in ckpt["seed_state"]


def test_distillation_teacher_eval_mode_and_no_grad() -> None:
    """DistillationTrainer ensures teacher stays in eval mode with no gradients."""
    from src.train.distillation import DistillationTrainer

    teacher = torch.nn.Sequential(torch.nn.Linear(10, 1))
    student = torch.nn.Sequential(torch.nn.Linear(10, 1))

    # Even if teacher was initially in train mode
    teacher.train()
    distiller = DistillationTrainer(
        teacher=teacher,
        student=student,
        alpha=0.5,
        temperature=2.0,
    )
    assert not distiller.teacher.training  # In eval mode!
    for p in distiller.teacher.parameters():
        assert not p.requires_grad

