"""Helpers partages pour les pages Streamlit liees aux donnees.

Fournit des fonctions de chargement pour les rapports JSON,
le manifest CSV, et le scan de repertoires d'images.
Principe : zero valeur hardcodee, tout est lu dynamiquement.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, PROCESSED_DIR


def load_json(path: Path) -> dict[str, Any]:
    """Charge un fichier JSON et retourne son contenu.

    Args:
        path: Chemin absolu vers le fichier JSON.

    Returns:
        Dictionnaire du contenu JSON.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not path.exists():
        msg = f"Fichier introuvable : {path}"
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_raw_stats() -> dict[str, Any]:
    """Charge le rapport d'etat des lieux des donnees brutes.

    Returns:
        Contenu de data/raw_stats.json.
    """
    return load_json(DATA_DIR / "raw_stats.json")


def load_cleaning_report() -> dict[str, Any]:
    """Charge le rapport de nettoyage (avant/apres exclusions).

    Returns:
        Contenu de data/cleaning_report.json.
    """
    return load_json(DATA_DIR / "cleaning_report.json")


def load_excluded() -> list[dict[str, Any]]:
    """Charge la liste des fichiers exclus.

    Returns:
        Liste de dictionnaires avec les champs 'path', 'reason', etc.
    """
    path = DATA_DIR / "excluded.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_split_stats() -> dict[str, Any]:
    """Charge les statistiques du split (par classe, par split).

    Returns:
        Contenu de data/split_stats.json.
    """
    return load_json(DATA_DIR / "split_stats.json")


def load_manifest(split: str | None = None) -> list[dict[str, str]]:
    """Charge le manifest CSV du split.

    Args:
        split: Si specifie, filtre sur ce split ('train', 'val', 'test').
            Si None, retourne toutes les entrees.

    Returns:
        Liste de dictionnaires avec cles 'split', 'path', 'label'.
    """
    manifest_path = DATA_DIR / "split_manifest.csv"
    if not manifest_path.exists():
        msg = f"Manifest introuvable : {manifest_path}"
        raise FileNotFoundError(msg)
    rows: list[dict[str, str]] = []
    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is None or row["split"] == split:
                rows.append(dict(row))
    return rows


def scan_classes(data_dir: Path | None = None) -> dict[str, int]:
    """Scanne un repertoire d'images et compte les fichiers par classe.

    Args:
        data_dir: Repertoire contenant les sous-dossiers par classe.
            Par defaut data/processed/.

    Returns:
        Dictionnaire {nom_classe: nombre_images} trie par nom.
    """
    root = data_dir or PROCESSED_DIR
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.iterdir() if f.is_file())
            counts[cls_dir.name] = n
    return counts


def get_random_images(
    class_name: str,
    n: int = 4,
    data_dir: Path | None = None,
) -> list[Path]:
    """Retourne n images aleatoires d'une classe donnee.

    Args:
        class_name: Nom de la classe (sous-dossier).
        n: Nombre d'images a retourner.
        data_dir: Repertoire des images. Par defaut data/processed/.

    Returns:
        Liste de chemins vers les images selectionnees.
    """
    root = data_dir or PROCESSED_DIR
    cls_dir = root / class_name
    if not cls_dir.exists():
        return []
    images = [
        f
        for f in cls_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        return []
    return random.sample(images, min(n, len(images)))
