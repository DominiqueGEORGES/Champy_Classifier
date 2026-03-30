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

from src.config import DATA_DIR, RAW_DIR


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


def scan_classes() -> dict[str, int]:
    """Retourne le nombre d'images par classe depuis raw_stats.json.

    Lit les class_counts depuis le rapport de statistiques
    genere par data/curate.py. Ne scanne pas le disque.

    Returns:
        Dictionnaire {nom_classe: nombre_images} trie par nom.
    """
    try:
        stats = load_raw_stats()
        return dict(sorted(stats.get("class_counts", {}).items()))
    except FileNotFoundError:
        return {}


def get_random_images(
    class_name: str,
    n: int = 4,
) -> list[Path]:
    """Retourne n images aleatoires d'une classe donnee.

    Lit le manifest de curation pour trouver les images de la classe,
    puis selectionne n images au hasard parmi celles existantes sur disque.

    Args:
        class_name: Nom scientifique de l'espece.
        n: Nombre d'images a retourner.

    Returns:
        Liste de chemins vers les images selectionnees.
    """
    raw_images_dir = RAW_DIR / "Mushrooms_images"
    curated_path = DATA_DIR / "curated_manifest.csv"

    if not curated_path.exists():
        return []

    # Lire le manifest pour trouver les fichiers de cette classe
    class_files: list[Path] = []
    with open(curated_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["species"] == class_name:
                img_path = raw_images_dir / row["image_lien"]
                if img_path.exists():
                    class_files.append(img_path)

    if not class_files:
        return []
    return random.sample(class_files, min(n, len(class_files)))
