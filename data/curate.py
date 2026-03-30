"""Pipeline de curation des donnees depuis les sources brutes.

Reconstruit le dataset a partir des CSV d'observations et du referentiel
des 30 especes cibles, sans dependre du notebook 0 ni du filtre ResNet50.

Pipeline :
    1. Charger observations_mushroom.csv (647K observations)
    2. Croiser avec champignons_france_top30.csv (30 especes)
    3. Filtrer par confiance GBIF >= 92
    4. Deduplication image_lien (garder premiere occurrence)
    5. Retirer les images avec conflit d'especes
    6. Verifier l'existence des fichiers sur disque
    7. Generer le manifest (image_lien -> espece) et les stats

Usage:
    python data/curate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "Mushrooms_images"


def load_observations() -> pd.DataFrame:
    """Charge le fichier observations_mushroom.csv.

    Returns:
        DataFrame avec les colonnes image_lien, gbif_info/species,
        gbif_info/confidence, etc.
    """
    path = DATA_DIR / "observations_mushroom.csv"
    logger.info(f"Chargement de {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  {len(df):,} observations, {df['label'].nunique():,} especes")
    return df


def load_top30() -> set[str]:
    """Charge le referentiel des 30 especes cibles.

    Returns:
        Ensemble des noms scientifiques des 30 especes.
    """
    path = DATA_DIR / "champignons_france_top30.csv"
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    species = set(df["Nom scientifique"])
    logger.info(f"Referentiel : {len(species)} especes cibles")
    return species


def curate() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute le pipeline de curation complet.

    Returns:
        Tuple (dataframe_curate, rapport) ou le dataframe contient
        les colonnes 'image_lien' et 'species' pour chaque image retenue,
        et le rapport contient les statistiques de chaque etape.
    """
    report: dict[str, Any] = {"steps": []}

    # -- 1. Charger les observations --
    obs = load_observations()
    report["total_observations"] = len(obs)

    # -- 2. Croiser avec le top 30 --
    target_species = load_top30()
    merged = obs[obs["gbif_info/species"].isin(target_species)].copy()
    logger.info(f"Apres filtre top 30 : {len(merged):,} observations")
    report["steps"].append({"name": "filtre_top30", "count": len(merged)})

    # -- 3. Filtre confiance GBIF >= 92 --
    merged = merged[merged["gbif_info/confidence"] >= 92].copy()
    logger.info(f"Apres filtre confiance >= 92 : {len(merged):,}")
    report["steps"].append({"name": "filtre_confidence_92", "count": len(merged)})

    # -- 4. Deduplication image_lien --
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="image_lien", keep="first")
    n_dedup = n_before - len(merged)
    logger.info(f"Apres dedup image_lien : {len(merged):,} (retire {n_dedup})")
    report["steps"].append({"name": "dedup_image_lien", "count": len(merged), "removed": n_dedup})

    # -- 5. Retirer les conflits d'especes --
    # Detecter les images qui ont ete assignees a plusieurs especes
    species_per_image = obs.groupby("image_lien")["gbif_info/species"].nunique()
    conflict_images = set(species_per_image[species_per_image > 1].index)
    conflict_in_our_set = conflict_images & set(merged["image_lien"])
    merged = merged[~merged["image_lien"].isin(conflict_in_our_set)]
    logger.info(f"Apres retrait conflits : {len(merged):,} (retire {len(conflict_in_our_set)})")
    report["steps"].append(
        {
            "name": "retrait_conflits_especes",
            "count": len(merged),
            "removed": len(conflict_in_our_set),
        }
    )

    # -- 6. Verifier l'existence des fichiers --
    merged["file_exists"] = merged["image_lien"].apply(lambda x: (RAW_DIR / str(x)).exists())
    n_missing = (~merged["file_exists"]).sum()
    merged = merged[merged["file_exists"]].copy()
    logger.info(f"Apres verif fichiers : {len(merged):,} (manquants : {n_missing})")
    report["steps"].append(
        {
            "name": "verif_fichiers_disque",
            "count": len(merged),
            "missing": int(n_missing),
        }
    )

    # -- 7. Simplifier le dataframe --
    result = merged[["image_lien", "gbif_info/species"]].copy()
    result = result.rename(columns={"gbif_info/species": "species"})
    result = result.sort_values(["species", "image_lien"]).reset_index(drop=True)

    # -- Stats finales --
    dist = result["species"].value_counts()
    report["final"] = {
        "total_images": len(result),
        "num_classes": result["species"].nunique(),
        "class_counts": dist.to_dict(),
        "min_class": {"name": dist.idxmin(), "count": int(dist.min())},
        "max_class": {"name": dist.idxmax(), "count": int(dist.max())},
        "imbalance_ratio": round(float(dist.max() / dist.min()), 1),
    }
    report["policy"] = (
        "Pipeline de curation from scratch depuis raw/. "
        "Pas de filtre ResNet50 ImageNet (non reproductible). "
        "Pas d'augmentation statique (PyTorch gere au training). "
        "Source : observations_mushroom.csv + champignons_france_top30.csv."
    )

    logger.info(
        f"Pipeline termine : {len(result):,} images, {result['species'].nunique()} classes"
    )
    return result, report


def save_curated_manifest(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde le manifest de curation au format CSV.

    Args:
        df: DataFrame avec colonnes 'image_lien' et 'species'.
        output_path: Chemin de sortie du fichier CSV.
    """
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Manifest de curation sauvegarde : {output_path} ({len(df):,} lignes)")


def main() -> None:
    """Point d'entree principal du script de curation."""
    result, report = curate()

    # Sauvegarder le manifest de curation
    save_curated_manifest(result, DATA_DIR / "curated_manifest.csv")

    # Sauvegarder le rapport
    report_path = DATA_DIR / "curation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Rapport de curation : {report_path}")

    # Sauvegarder raw_stats.json (compatible avec le Streamlit)
    from collections import Counter

    from PIL import Image

    # Scanner les dimensions et tailles des images retenues
    dims: Counter[tuple[int, int]] = Counter()
    sizes: list[int] = []
    corrupted: list[str] = []

    for _, row in result.iterrows():
        img_path = RAW_DIR / row["image_lien"]
        fsize = img_path.stat().st_size
        sizes.append(fsize)
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                dims[im.size] += 1
        except Exception:
            corrupted.append(row["image_lien"])

    dim_list = sorted(dims.items(), key=lambda x: -x[1])
    raw_stats: dict[str, Any] = {
        "source": "data/raw/Mushrooms_images (curated from CSV)",
        "total_images": len(result),
        "num_classes": result["species"].nunique(),
        "class_counts": report["final"]["class_counts"],
        "extensions": {".jpg": len(result)},
        "file_size_bytes": {
            "min": min(sizes),
            "max": max(sizes),
            "avg": sum(sizes) // len(sizes),
            "total": sum(sizes),
        },
        "dimensions": {f"{w}x{h}": cnt for (w, h), cnt in dim_list},
        "corrupted_count": len(corrupted),
        "corrupted_files": corrupted,
        "duplicate_groups_count": 0,
        "duplicate_total_images": 0,
        "duplicates": {},
    }

    with open(DATA_DIR / "raw_stats.json", "w", encoding="utf-8") as f:
        json.dump(raw_stats, f, indent=2, ensure_ascii=False)
    logger.info("raw_stats.json regenere")

    # Sauvegarder cleaning_report.json
    cleaning_report: dict[str, Any] = {
        "policy": report["policy"],
        "before": {
            "total_images": report["total_observations"],
            "description": "observations_mushroom.csv (toutes especes)",
        },
        "after": {
            "total_images": len(result),
            "num_classes": result["species"].nunique(),
            "class_counts": report["final"]["class_counts"],
        },
        "excluded_count": report["total_observations"] - len(result),
        "exclusion_reasons": {
            "hors_top30": report["total_observations"] - report["steps"][0]["count"],
            "dedup_image_lien": report["steps"][2]["removed"],
            "conflit_especes": report["steps"][3]["removed"],
            "fichier_manquant": report["steps"][4]["missing"],
        },
    }
    with open(DATA_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2, ensure_ascii=False)
    logger.info("cleaning_report.json regenere")

    # Plus besoin d'excluded.json (on part de raw, pas de processed)
    excluded_path = DATA_DIR / "excluded.json"
    if excluded_path.exists():
        excluded_path.unlink()
        logger.info("excluded.json supprime (plus necessaire avec le pipeline from raw)")


if __name__ == "__main__":
    main()
