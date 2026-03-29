"""Tests unitaires pour src.data.dataloader."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import TrainingConfig
from src.data.dataloader import (
    compute_sample_weights,
    create_all_loaders,
    create_eval_loader,
    create_train_loader,
)


class TestComputeSampleWeights:
    """Tests pour la fonction compute_sample_weights."""

    def test_balanced_classes(self) -> None:
        """Verifie que les poids sont egaux pour des classes equilibrees."""
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        weights = compute_sample_weights(targets)
        assert weights.shape == (6,)
        assert torch.allclose(weights, weights[0].expand_as(weights))

    def test_imbalanced_classes(self) -> None:
        """Verifie que la classe rare a un poids plus eleve."""
        # Classe 0: 4 images, Classe 1: 1 image
        targets = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_sample_weights(targets)
        # Poids classe 0 = 1/4, poids classe 1 = 1/1
        assert weights[4] > weights[0]  # Classe rare plus ponderee
        assert torch.isclose(weights[0], torch.tensor(0.25))
        assert torch.isclose(weights[4], torch.tensor(1.0))


class TestCreateTrainLoader:
    """Tests pour la factory du DataLoader d'entrainement."""

    def test_creates_loader(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie la creation du DataLoader train avec sampler."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=2, num_workers=0)
        loader = create_train_loader(config, manifest_path=manifest, data_dir=data_dir)
        assert loader is not None
        assert loader.batch_size == 2
        # Le sampler doit etre un WeightedRandomSampler
        assert loader.sampler is not None

    def test_batch_content(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie le contenu d'un batch (tensors de bonnes dimensions)."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=2, num_workers=0, image_size=224)
        loader = create_train_loader(config, manifest_path=manifest, data_dir=data_dir)
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert labels.dtype == torch.long


class TestCreateEvalLoader:
    """Tests pour la factory du DataLoader d'evaluation."""

    def test_creates_val_loader(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie la creation du DataLoader val sans sampler pondere."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=2, num_workers=0)
        loader = create_eval_loader("val", config, manifest_path=manifest, data_dir=data_dir)
        assert loader is not None
        assert loader.batch_size == 2

    def test_creates_test_loader(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie la creation du DataLoader test."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=3, num_workers=0)
        loader = create_eval_loader("test", config, manifest_path=manifest, data_dir=data_dir)
        batch_images, batch_labels = next(iter(loader))
        assert batch_images.shape[0] == 3
        assert batch_labels.shape[0] == 3


class TestCreateAllLoaders:
    """Tests pour la factory combinee des trois DataLoaders."""

    def test_creates_three_loaders(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que les trois loaders sont crees avec un label_map coherent."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=2, num_workers=0)
        train_loader, val_loader, test_loader = create_all_loaders(
            config, manifest_path=manifest, data_dir=data_dir
        )
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_shared_label_map(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que les trois datasets partagent le meme label_map."""
        manifest, data_dir, _ = tmp_dataset
        config = TrainingConfig(batch_size=2, num_workers=0)
        train_ld, val_ld, test_ld = create_all_loaders(
            config, manifest_path=manifest, data_dir=data_dir
        )
        train_map = train_ld.dataset.label_map  # type: ignore[union-attr]
        val_map = val_ld.dataset.label_map  # type: ignore[union-attr]
        test_map = test_ld.dataset.label_map  # type: ignore[union-attr]
        assert train_map == val_map == test_map
