"""Build leakage-safe train/validation loaders from committed split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.manifest_dataset import ManifestAVDataset, dataset_roots_from_config


def _split_path(manifest_dir: str | Path, split: str) -> Path:
    """Resolve a current or legacy filename without generating splits in training."""
    directory = Path(manifest_dir)
    candidates = [directory / f"{split}_manifest.csv", directory / f"{split}.csv"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No {split} split manifest in {directory}. Run the data audit first; "
        "training never creates or changes data splits."
    )


def build_train_val_loaders(
    data_config: Any,
    train_config: Any,
    *,
    manifest_dir: str | Path = "manifests/splits",
    batch_size: int | None = None,
    seed: int = 42,
    allowed_datasets: set[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Construct train and validation loaders backed only by existing manifests."""
    roots = dataset_roots_from_config(data_config)
    if allowed_datasets is None:
        allowed_datasets = set(roots)
    missing_roots = {name.lower() for name in allowed_datasets} - set(roots)
    if missing_roots:
        raise RuntimeError(
            f"No local dataset root is configured for {sorted(missing_roots)}. "
            "Set the corresponding *_ROOT environment variable or edit configs/data_config.yaml."
        )

    cache_dir = Path(data_config.cache.cache_dir) / "features"
    aug_cfg = train_config.augmentation
    if not isinstance(aug_cfg, dict):
        aug_cfg = vars(aug_cfg)
    dropout_cfg = aug_cfg.get("modality_dropout", {})
    dropout_prob = (
        float(dropout_cfg.get("prob", 0.0)) if dropout_cfg.get("enabled", False) else 0.0
    )

    train_set = ManifestAVDataset(
        _split_path(manifest_dir, "train"), roots, cache_dir, data_config,
        training=True, modality_dropout_prob=dropout_prob,
        augmentation_config=aug_cfg,
        allowed_datasets=allowed_datasets,
    )
    val_set = ManifestAVDataset(
        _split_path(manifest_dir, "val"), roots, cache_dir, data_config,
        training=False,
        allowed_datasets=allowed_datasets,
    )

    workers = max(0, int(train_config.num_workers))
    batch = int(batch_size or train_config.batch_size)
    use_cuda = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = True
    balance_cfg = train_config.class_balance
    if not isinstance(balance_cfg, dict):
        balance_cfg = vars(balance_cfg)
    if balance_cfg.get("balanced_sampling", False):
        labels = train_set.rows["clip_fake"].astype(int).to_numpy().copy()
        labels_tensor = torch.as_tensor(labels)
        class_counts = torch.bincount(labels_tensor, minlength=2).clamp_min(1)
        weights = (1.0 / class_counts[labels_tensor]).double()
        num_samples = int(balance_cfg.get("samples_per_epoch") or len(weights))
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True, generator=generator
        )
        shuffle = False

    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=shuffle, sampler=sampler,
        num_workers=workers, pin_memory=use_cuda, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=use_cuda,
    )
    return train_loader, val_loader
