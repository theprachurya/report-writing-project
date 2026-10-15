"""Manifest-backed feature datasets for licensed local AV data."""

from src.data.manifest_dataset import ManifestAVDataset, dataset_roots_from_config

__all__ = ["ManifestAVDataset", "dataset_roots_from_config"]
