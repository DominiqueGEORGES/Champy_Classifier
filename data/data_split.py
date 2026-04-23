"""Split stratifie reproductible du dataset.

Lit le manifest de curation (data/curated_manifest.csv) produit par
data/curate.py et effectue un split train/val/test stratifie.
Produit un manifest CSV et un fichier JSON de statistiques.
Aucune copie d'image n'est effectuee.

Usage:
    python data/data_split.py
    python data/data_split.py --seed 42 --train-ratio 0.70 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Chemins (relatifs a la racine du projet)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CURATED_MANIFEST = DATA_DIR / "curated_manifest.csv"
CURATED_MANIFEST_FILTERED = DATA_DIR / "curated_manifest_filtered.csv"
MANIFEST_PATH = DATA_DIR / "split_manifest.csv"
STATS_PATH = DATA_DIR / "split_stats.json"


def resolve_curated_manifest() -> Path:
    """Résout le manifest de curation à utiliser.

    Priorise ``curated_manifest_filtered.csv`` (produit par le filtre CLIP)
    s'il existe, sinon retombe sur ``curated_manifest.csv`` (sortie brute
    de ``data/curate.py``).

    Returns:
        Chemin absolu du manifest à lire.
    """
    if CURATED_MANIFEST_FILTERED.exists():
        return CURATED_MANIFEST_FILTERED
    return CURATED_MANIFEST


def load_curated(path: Path) -> tuple[list[str], list[str]]:
    """Charge le manifest de curation et retourne les chemins et labels.

    Le manifest contient les colonnes 'image_lien' (nom de fichier)
    et 'species' (nom scientifique de l'espece).

    Args:
        path: Chemin vers curated_manifest.csv.

    Returns:
        Tuple (image_liens, species) - listes paralleles.

    Raises:
        FileNotFoundError: Si le manifest n'existe pas.
    """
    if not path.exists():
        msg = f"Manifest de curation introuvable : {path}. Lancez d'abord data/curate.py"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    return df["image_lien"].tolist(), df["species"].tolist()


def split_data(
    paths: list[str],
    labels: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Effectue un split stratifie en train/val/test.

    Utilise deux appels successifs a train_test_split de scikit-learn
    pour garantir la stratification par classe.

    Args:
        paths: Liste des noms de fichiers (image_lien).
        labels: Liste des especes correspondantes.
        train_ratio: Proportion du split train (defaut 0.70).
        val_ratio: Proportion du split validation (defaut 0.15).
        seed: Graine pour la reproductibilite.

    Returns:
        Dictionnaire avec cles 'train', 'val', 'test', chacune contenant
        une liste de tuples (image_lien, species).

    Raises:
        ValueError: Si les ratios ne laissent pas assez pour le test.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0.01:
        msg = f"Ratios invalides : train={train_ratio}, val={val_ratio}, test={test_ratio:.2f}"
        raise ValueError(msg)

    # Premier split : train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    paths_train, paths_valtest, labels_train, labels_valtest = train_test_split(
        paths,
        labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Deuxieme split : val vs test
    val_fraction_of_valtest = val_ratio / val_test_ratio
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_valtest,
        labels_valtest,
        test_size=1.0 - val_fraction_of_valtest,
        random_state=seed,
        stratify=labels_valtest,
    )

    return {
        "train": list(zip(paths_train, labels_train, strict=False)),
        "val": list(zip(paths_val, labels_val, strict=False)),
        "test": list(zip(paths_test, labels_test, strict=False)),
    }


def write_manifest(splits: dict[str, list[tuple[str, str]]], path: Path) -> None:
    """Ecrit le manifest de split au format CSV : split,path,label.

    Le champ 'path' contient le nom du fichier image (image_lien),
    a resoudre relativement a data/raw/Mushrooms_images/.

    Args:
        splits: Dictionnaire des splits (train/val/test).
        path: Chemin de sortie du fichier CSV.
    """
    lines = ["split,path,label"]
    for split_name in ("train", "val", "test"):
        for img_file, label in sorted(splits[split_name]):
            lines.append(f"{split_name},{img_file},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_stats(
    splits: dict[str, list[tuple[str, str]]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    """Calcule les statistiques par classe et par split.

    Args:
        splits: Dictionnaire des splits.
        seed: Graine utilisee.
        train_ratio: Ratio train utilise.
        val_ratio: Ratio validation utilise.

    Returns:
        Dictionnaire JSON-serialisable avec totaux, ratios et details par classe.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
    stats: dict[str, Any] = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total": sum(len(v) for v in splits.values()),
        "splits": {},
        "per_class": {},
    }

    class_split_counts: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        entries = splits[split_name]
        stats["splits"][split_name] = len(entries)
        label_counts = Counter(label for _, label in entries)
        for cls, cnt in label_counts.items():
            class_split_counts.setdefault(cls, {})[split_name] = cnt

    for cls in sorted(class_split_counts.keys()):
        counts = class_split_counts[cls]
        total_cls = sum(counts.values())
        stats["per_class"][cls] = {
            "total": total_cls,
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
        }

    return stats


def main() -> None:
    """Point d'entree principal du script de split."""
    parser = argparse.ArgumentParser(description="Split stratifie du dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Seed : {args.seed}")
    print(
        f"Ratios : train={args.train_ratio}, val={args.val_ratio}, "
        f"test={round(1 - args.train_ratio - args.val_ratio, 2)}"
    )

    # Charger le manifest de curation (filtré CLIP si dispo)
    curated_path = resolve_curated_manifest()
    paths, labels = load_curated(curated_path)
    print(f"Manifest source : {curated_path.name}")
    print(f"Images curatees : {len(paths)}")

    # Split
    splits = split_data(paths, labels, args.train_ratio, args.val_ratio, args.seed)
    print(
        f"Train : {len(splits['train'])}, Val : {len(splits['val'])}, Test : {len(splits['test'])}"
    )

    # Ecrire le manifest CSV
    write_manifest(splits, MANIFEST_PATH)
    print(f"Manifest : {MANIFEST_PATH}")

    # Ecrire les stats JSON
    stats = compute_stats(splits, args.seed, args.train_ratio, args.val_ratio)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats : {STATS_PATH}")


if __name__ == "__main__":
    main()
