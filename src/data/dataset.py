"""Dataset PyTorch pour la classification de champignons.

Lit le manifest CSV produit par data/data_split.py et charge les images
depuis data/raw/Mushrooms_images/. Les transforms sont configurables selon le split
(augmentation pour train, simple resize+crop pour val/test).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Statistiques ImageNet pour la normalisation ResNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Retourne les transforms d'augmentation pour le split train.

    Pipeline renforcé conçu pour améliorer la robustesse, notamment
    sur les classes rares (sous-représentées), en introduisant de la
    variabilité géométrique, photométrique et d'occlusion :

    - ``RandomResizedCrop(scale=(0.7, 1.0))`` : zoom/cadrage aléatoire
      pour varier l'échelle et le cadrage du sujet.
    - ``RandomHorizontalFlip()`` : retournement horizontal (p=0.5).
    - ``RandomAffine(degrees=15, translate=(0.1, 0.1))`` : rotation
      aléatoire [-15°, +15°] et translations jusqu'à 10% de chaque axe.
    - ``ColorJitter(0.3, 0.3, 0.3, 0.1)`` : variations de luminosité,
      contraste, saturation et teinte pour simuler des conditions de
      prise de vue différentes.
    - ``Normalize(ImageNet)`` : normalisation standard ResNet.
    - ``RandomErasing(p=0.25)`` : masquage aléatoire d'une région
      (appliqué sur tensor, après ``Normalize``) pour forcer le modèle
      à s'appuyer sur plusieurs zones de l'image.

    Args:
        image_size: Taille cible des images en pixels (carré).

    Returns:
        Pipeline de transforms composé.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Retourne les transforms pour les splits val et test.

    Applique uniquement un redimensionnement et un crop central,
    sans augmentation aleatoire, pour une evaluation deterministe.

    Args:
        image_size: Taille cible des images en pixels (carre).

    Returns:
        Pipeline de transforms compose.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_manifest(
    manifest_path: Path,
    split: str,
) -> tuple[list[str], list[str]]:
    """Charge les entrees du manifest CSV pour un split donne.

    Args:
        manifest_path: Chemin vers le fichier split_manifest.csv.
        split: Nom du split a charger ('train', 'val' ou 'test').

    Returns:
        Tuple (chemins_relatifs, labels) pour le split demande.

    Raises:
        FileNotFoundError: Si le manifest n'existe pas.
        ValueError: Si le split demande n'existe pas dans le manifest.
    """
    if not manifest_path.exists():
        msg = f"Manifest introuvable : {manifest_path}"
        raise FileNotFoundError(msg)

    paths: list[str] = []
    labels: list[str] = []
    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                paths.append(row["path"])
                labels.append(row["label"])

    if not paths:
        msg = f"Aucune entree trouvee pour le split '{split}' dans {manifest_path}"
        raise ValueError(msg)

    return paths, labels


def build_label_map(labels: list[str]) -> dict[str, int]:
    """Construit le mapping nom_de_classe -> index numerique.

    Les classes sont triees par ordre alphabetique pour garantir
    un mapping deterministe independant de l'ordre des donnees.

    Args:
        labels: Liste des noms de classes (peut contenir des doublons).

    Returns:
        Dictionnaire {nom_classe: index} trie alphabetiquement.
    """
    unique = sorted(set(labels))
    return {name: idx for idx, name in enumerate(unique)}


class MushroomDataset(Dataset):  # type: ignore[misc]
    """Dataset PyTorch pour les images de champignons.

    Charge les images depuis data/raw/Mushrooms_images/ en suivant le manifest CSV.
    Applique les transforms specifiees (augmentation ou evaluation).

    Attributes:
        paths: Liste des chemins relatifs des images.
        labels: Liste des noms de classes.
        label_map: Mapping classe -> index numerique.
        targets: Tensor des indices de classes (pour WeightedRandomSampler).
        transform: Pipeline de transforms a appliquer.
        data_dir: Repertoire racine contenant les images.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        data_dir: Path,
        transform: transforms.Compose | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        """Initialise le dataset depuis un manifest CSV.

        Args:
            manifest_path: Chemin vers split_manifest.csv.
            split: Nom du split ('train', 'val', 'test').
            data_dir: Repertoire contenant les images (data/raw/Mushrooms_images/).
            transform: Transforms a appliquer. Si None, utilise les
                transforms d'evaluation par defaut.
            label_map: Mapping classe->index. Si None, construit
                automatiquement depuis les labels du split.
        """
        self.paths, self.labels = load_manifest(manifest_path, split)
        self.data_dir = data_dir
        self.transform = transform or get_eval_transforms()

        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = build_label_map(self.labels)

        self.targets = torch.tensor([self.label_map[lbl] for lbl in self.labels], dtype=torch.long)

    def __len__(self) -> int:
        """Retourne le nombre d'images dans le dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Charge et retourne une image et son label.

        Args:
            idx: Index de l'image dans le dataset.

        Returns:
            Tuple (image_tensor, label_index).
        """
        img_path = self.data_dir / self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.label_map[self.labels[idx]]
        return image, label

    @property
    def class_names(self) -> list[str]:
        """Retourne la liste ordonnee des noms de classes."""
        return sorted(self.label_map.keys())

    @property
    def num_classes(self) -> int:
        """Retourne le nombre de classes."""
        return len(self.label_map)
