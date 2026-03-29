"""Factory de DataLoaders pour l'entrainement et l'evaluation.

Cree les DataLoaders PyTorch a partir du manifest CSV, avec
WeightedRandomSampler pour le split train afin de compenser
le desequilibre des classes (52 a 900 images par classe).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import PROCESSED_DIR, TrainingConfig
from src.data.dataset import (
    MushroomDataset,
    build_label_map,
    get_eval_transforms,
    get_train_transforms,
    load_manifest,
)

DATA_DIR = PROCESSED_DIR.parent
DEFAULT_MANIFEST = DATA_DIR / "split_manifest.csv"


def compute_sample_weights(targets: torch.Tensor) -> torch.Tensor:
    """Calcule les poids par echantillon pour le WeightedRandomSampler.

    Chaque echantillon recoit un poids inversement proportionnel
    a la frequence de sa classe, de sorte que les classes rares
    soient surechantillonnees.

    Args:
        targets: Tensor 1D des indices de classes pour chaque echantillon.

    Returns:
        Tensor 1D de poids (float64), un par echantillon.
    """
    class_counts = torch.bincount(targets)
    # Poids par classe = 1 / nb_echantillons_de_la_classe
    class_weights = 1.0 / class_counts.float()
    # Poids par echantillon
    return class_weights[targets]


def create_train_loader(
    config: TrainingConfig,
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
    label_map: dict[str, int] | None = None,
) -> DataLoader:
    """Cree le DataLoader d'entrainement avec WeightedRandomSampler.

    Le sampler compense le desequilibre des classes en surechantillonnant
    les classes rares. Le nombre d'echantillons par epoch est egal
    au nombre total d'images dans le split train.

    Args:
        config: Configuration d'entrainement (batch_size, num_workers, etc.).
        manifest_path: Chemin vers le manifest CSV. Par defaut split_manifest.csv.
        data_dir: Repertoire des images. Par defaut data/processed/.
        label_map: Mapping classe->index partage entre splits. Si None,
            construit automatiquement.

    Returns:
        DataLoader pret pour l'entrainement.
    """
    manifest = manifest_path or DEFAULT_MANIFEST
    images_dir = data_dir or PROCESSED_DIR

    transform = get_train_transforms(config.image_size)
    dataset = MushroomDataset(
        manifest_path=manifest,
        split="train",
        data_dir=images_dir,
        transform=transform,
        label_map=label_map,
    )

    sample_weights = compute_sample_weights(dataset.targets)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_eval_loader(
    split: str,
    config: TrainingConfig,
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
    label_map: dict[str, int] | None = None,
) -> DataLoader:
    """Cree un DataLoader d'evaluation (val ou test) sans augmentation.

    Pas de sampler pondere : l'evaluation se fait sur la distribution
    naturelle des classes.

    Args:
        split: Nom du split ('val' ou 'test').
        config: Configuration (batch_size, num_workers).
        manifest_path: Chemin vers le manifest CSV.
        data_dir: Repertoire des images.
        label_map: Mapping classe->index partage.

    Returns:
        DataLoader pret pour l'evaluation.
    """
    manifest = manifest_path or DEFAULT_MANIFEST
    images_dir = data_dir or PROCESSED_DIR

    transform = get_eval_transforms(config.image_size)
    dataset = MushroomDataset(
        manifest_path=manifest,
        split=split,
        data_dir=images_dir,
        transform=transform,
        label_map=label_map,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def create_all_loaders(
    config: TrainingConfig,
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cree les trois DataLoaders (train, val, test) avec un label_map partage.

    Le label_map est construit a partir du split train et reutilise
    pour val et test, garantissant la coherence des indices de classes.

    Args:
        config: Configuration d'entrainement.
        manifest_path: Chemin vers le manifest CSV.
        data_dir: Repertoire des images.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    manifest = manifest_path or DEFAULT_MANIFEST
    images_dir = data_dir or PROCESSED_DIR

    # Construire le label_map depuis le train pour le partager
    train_paths, train_labels = load_manifest(manifest, "train")
    label_map = build_label_map(train_labels)

    train_loader = create_train_loader(config, manifest, images_dir, label_map=label_map)
    val_loader = create_eval_loader("val", config, manifest, images_dir, label_map=label_map)
    test_loader = create_eval_loader("test", config, manifest, images_dir, label_map=label_map)

    return train_loader, val_loader, test_loader
