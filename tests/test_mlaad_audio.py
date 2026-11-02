"""Verifies the audio-only MLAAD manifest and source-group bootstrap helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluate.mlaad_audio import (
    bootstrap_grouped_auc,
    build_mlaad_audio_records,
    predict_audio_head,
)


def test_build_mlaad_audio_records_pairs_source_audio(tmp_path: Path) -> None:
    """Metadata produces one matched real/spoof pair per source utterance."""
    fake_dir = tmp_path / "fake" / "de" / "ExampleTTS"
    original_dir = tmp_path / "original" / "de"
    fake_dir.mkdir(parents=True)
    original_dir.mkdir(parents=True)
    fake_audio = fake_dir / "sample_fake.wav"
    real_audio = original_dir / "sample.wav"
    fake_audio.write_bytes(b"fixture")
    real_audio.write_bytes(b"fixture")
    (fake_dir / "meta.csv").write_text(
        "path|original_file|language|is_original_language|duration|training_data|model_name|architecture|transcript|reference_speaker\n"
        "./fake/de/ExampleTTS/sample_fake.wav|de_DE/book/sample.wav|de|True|2.0|test|ExampleTTS|TTS|hello|speaker\n",
        encoding="utf-8",
    )

    records = build_mlaad_audio_records(tmp_path, "de")

    assert len(records) == 2
    assert {record.target for record in records} == {0, 1}
    assert len({record.source_group for record in records}) == 1
    assert {record.file_path for record in records} == {fake_audio, real_audio}
    assert {record.generator for record in records if record.target == 1} == {"ExampleTTS"}
    assert {record.generator for record in records if record.target == 0} == {"bona_fide"}


def test_build_mlaad_records_keeps_spoof_without_local_source_audio(tmp_path: Path) -> None:
    """Spoof rows remain usable when the tiny release omits their source recording."""
    fake_dir = tmp_path / "fake" / "de" / "ExampleTTS"
    original_dir = tmp_path / "original" / "de"
    fake_dir.mkdir(parents=True)
    original_dir.mkdir(parents=True)
    fake_audio = fake_dir / "sample_fake.wav"
    real_audio = original_dir / "unrelated.wav"
    fake_audio.write_bytes(b"fixture")
    real_audio.write_bytes(b"fixture")
    (fake_dir / "meta.csv").write_text(
        "path|original_file|language|is_original_language|duration|training_data|model_name|architecture|transcript|reference_speaker\n"
        "./fake/de/ExampleTTS/sample_fake.wav|de_DE/book/missing.wav|de|True|2.0|test|ExampleTTS|TTS|hello|speaker\n",
        encoding="utf-8",
    )

    records = build_mlaad_audio_records(tmp_path, "de")

    assert len(records) == 2
    fake_record = next(record for record in records if record.target == 1)
    real_record = next(record for record in records if record.target == 0)
    assert fake_record.file_path == fake_audio
    assert real_record.file_path == real_audio
    assert fake_record.source_group != real_record.source_group


def test_grouped_bootstrap_resamples_source_utterances() -> None:
    """A perfectly separable fixture keeps AUROC at one under group resampling."""
    result = bootstrap_grouped_auc(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array(["source-a", "source-a", "source-b", "source-b"]),
        n_iterations=30,
        seed=7,
    )
    assert result["lower"] == 1.0
    assert result["upper"] == 1.0
    assert result["valid_replicates"] == 30


def test_audio_prediction_supports_unimodal_checkpoint_structure() -> None:
    """The audio-only baseline exposes its branch as encoder/head."""
    class MeanLogitHead(nn.Module):
        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logit": features.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)}

    class AudioBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Identity()
            self.head = MeanLogitHead()

    loader = DataLoader(
        [
            {
                "mel_spec": torch.ones(1, 2, 2),
                "target": torch.tensor(0.0),
                "source_group": "source-a",
                "generator": "ExampleTTS",
            },
            {
                "mel_spec": torch.full((1, 2, 2), 2.0),
                "target": torch.tensor(1.0),
                "source_group": "source-b",
                "generator": "ExampleTTS",
            },
        ],
        batch_size=2,
    )

    output = predict_audio_head(AudioBaseline(), loader, torch.device("cpu"))

    np.testing.assert_allclose(output["logits"], [1.0, 2.0])
    np.testing.assert_array_equal(output["targets"], [0.0, 1.0])
