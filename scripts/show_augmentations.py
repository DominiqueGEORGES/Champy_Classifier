"""Génère un PNG de validation visuelle des augmentations du pipeline train.

Prend une image du dataset (aléatoire par défaut ou fournie via CLI)
et applique ``get_train_transforms`` dix fois, pour produire une grille
de 3x4 avec l'image originale suivie de 10 variations augmentées.

Usage :
    python scripts/show_augmentations.py
    python scripts/show_augmentations.py --image path/vers/image.jpg
    python scripts/show_augmentations.py --output out.png --seed 42

Le fichier de sortie est écrit par défaut dans ``models/artifacts/
augmentations_preview.png``, emplacement lu par la page Streamlit
``03_augmentation.py``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ajoute la racine du projet au sys.path pour permettre l'import de src.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_train_transforms  # noqa: E402

N_AUGMENTATIONS = 10
GRID_ROWS = 3
GRID_COLS = 4
DEFAULT_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = _PROJECT_ROOT / "models" / "artifacts" / "augmentations_preview.png"


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse la normalisation ImageNet pour l'affichage.

    Args:
        tensor: Image normalisée, shape (C, H, W).

    Returns:
        Image numpy (H, W, C), valeurs [0, 255], dtype uint8.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def pick_random_image(data_dir: Path) -> Path:
    """Choisit une image aléatoire dans le répertoire ``data/processed/``.

    Scanne récursivement pour trouver les fichiers ``.jpg`` / ``.jpeg`` /
    ``.png`` et retourne un chemin au hasard parmi ceux trouvés.

    Args:
        data_dir: Répertoire racine contenant les images (data/processed/).

    Returns:
        Chemin absolu vers une image existante.

    Raises:
        FileNotFoundError: Si ``data_dir`` ne contient aucune image.
    """
    if not data_dir.exists():
        msg = (
            f"Répertoire introuvable : {data_dir}. "
            f"Lancez 'dvc pull' pour récupérer les données, ou passez --image."
        )
        raise FileNotFoundError(msg)

    candidates: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        candidates.extend(data_dir.rglob(ext))

    if not candidates:
        msg = (
            f"Aucune image trouvée sous {data_dir}. "
            f"Lancez 'dvc pull' pour récupérer les données, ou passez --image."
        )
        raise FileNotFoundError(msg)

    return random.choice(candidates)


def build_grid(
    original: Image.Image,
    augmented: list[np.ndarray],
    output: Path,
    source_label: str,
) -> None:
    """Construit la grille matplotlib et l'écrit dans un fichier PNG.

    Args:
        original: Image originale (PIL) à afficher en haut à gauche.
        augmented: Liste de ``N_AUGMENTATIONS`` images augmentées (numpy uint8).
        output: Chemin du PNG de sortie.
        source_label: Libellé de l'image source (affiché en titre).
    """
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 3, GRID_ROWS * 3))
    axes_flat = axes.flatten()

    # Cellule 0 : image originale
    axes_flat[0].imshow(original)
    axes_flat[0].set_title("Originale", fontsize=10, fontweight="bold", color="#1f77b4")
    axes_flat[0].axis("off")

    # Cellules 1 à N : augmentations
    for i, img in enumerate(augmented, start=1):
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(f"Augmentation #{i}", fontsize=9)
        axes_flat[i].axis("off")

    # Cellules restantes : vides
    for i in range(N_AUGMENTATIONS + 1, GRID_ROWS * GRID_COLS):
        axes_flat[i].axis("off")

    fig.suptitle(
        f"Pipeline d'augmentation train - source : {source_label}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI.

    Returns:
        Namespace argparse avec les attributs ``image``, ``output``, ``seed``.
    """
    parser = argparse.ArgumentParser(
        description="Génère une grille PNG montrant 10 augmentations d'une image.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image source (défaut : aléatoire dans le split train).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Fichier PNG de sortie (défaut : {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine pour la reproductibilité des augmentations.",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI : génère et sauvegarde la grille PNG."""
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.image is not None:
        if not args.image.exists():
            msg = f"Image introuvable : {args.image}"
            raise FileNotFoundError(msg)
        image_path = args.image
    else:
        image_path = pick_random_image(DEFAULT_PROCESSED_DIR)

    original = Image.open(image_path).convert("RGB")
    transform = get_train_transforms(224)

    augmented: list[np.ndarray] = []
    for _ in range(N_AUGMENTATIONS):
        tensor = transform(original)
        augmented.append(denormalize(tensor))

    source_label = image_path.parent.name + "/" + image_path.name
    build_grid(original, augmented, args.output, source_label)
    print(f"Grille écrite : {args.output}")


if __name__ == "__main__":
    main()
