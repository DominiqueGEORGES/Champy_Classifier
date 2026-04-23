"""Filtre de qualité OpenCLIP pour le dataset brut de champignons.

Le dataset ``curated_manifest.csv`` contient des faux positifs : images
qui ont un nom d'espèce valide dans les observations GBIF mais qui ne
représentent pas un champignon (photos de personnes, de textes, de
paysages sans champignon, etc.).

Ce script utilise OpenCLIP (ViT-B-32 pré-entraîné sur ``laion2b_s34b_b79k``,
~150 MB) pour scorer chaque image sur deux axes :

    - similarité max avec des prompts positifs (mushroom, fungus, etc.)
    - similarité max avec des prompts négatifs (person, text, landscape...)

    score_final = score_positive - score_negative

Une image qui ressemble clairement à un champignon aura un score élevé ;
une image parasite aura un score négatif ou proche de zéro.

Sortie :
    - ``data/quality_scores.csv`` : image_path, species, score_positive,
      score_negative, score_final, is_mushroom (bool).
    - ``data/quality_report.json`` : distribution globale, pass/fail par
      classe, quantiles, paramètres du run.

Usage :
    # Calibration sur un échantillon stratifié de 500 images
    python data/quality_filter.py --sample 500 --output-suffix _calibration

    # Application à tout le dataset avec seuil choisi
    python data/quality_filter.py --threshold 0.05

    # Sur GPU si dispo (défaut : auto)
    python data/quality_filter.py --device cuda

Dépendance : ``open_clip_torch`` (installé via pip, non requis par le
pipeline training/serving).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "Mushrooms_images"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

POSITIVE_PROMPTS = (
    "a photo of a mushroom",
    "a fungus growing in nature",
    "a close-up photograph of a mushroom",
)
NEGATIVE_PROMPTS = (
    "a photo of a person",
    "an indoor scene",
    "a landscape without mushrooms",
    "a photo of text",
)

DEFAULT_BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.0  # Placeholder, à calibrer empiriquement
CLEANING_REPORT_REASON = "clip_quality_filter"


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    """Résout le device effectif à utiliser.

    Args:
        requested: Demande CLI (``'auto'``, ``'cpu'``, ``'cuda'``).

    Returns:
        Nom de device PyTorch effectif (``'cpu'`` ou ``'cuda'``).
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA demandé mais indisponible, fallback sur CPU.")
        return "cpu"
    return requested


def load_clip_model(
    device: str,
) -> tuple[nn.Module, Any, Any]:
    """Charge le modèle OpenCLIP et ses dépendances associées.

    Args:
        device: Device PyTorch (``'cpu'`` ou ``'cuda'``).

    Returns:
        Tuple ``(model, preprocess, tokenizer)``.
    """
    import open_clip

    logger.info(f"Chargement OpenCLIP {MODEL_NAME} / {PRETRAINED} sur {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_prompts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Iterable[str],
    device: str,
) -> torch.Tensor:
    """Encode une liste de prompts texte en features L2-normalisées.

    Args:
        model: Modèle OpenCLIP chargé.
        tokenizer: Tokenizer OpenCLIP associé.
        prompts: Liste de prompts texte.
        device: Device cible.

    Returns:
        Tensor ``(N, embed_dim)`` avec les features texte normalisées.
    """
    tokens = tokenizer(list(prompts)).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_batch(
    model: nn.Module,
    preprocess: Any,
    image_paths: list[Path],
    text_pos: torch.Tensor,
    text_neg: torch.Tensor,
    device: str,
) -> list[tuple[float, float]]:
    """Score un batch d'images : retourne (score_pos_max, score_neg_max) par image.

    Args:
        model: Modèle OpenCLIP.
        preprocess: Transform image de OpenCLIP.
        image_paths: Chemins absolus des images du batch.
        text_pos: Features texte des prompts positifs ``(P, D)``.
        text_neg: Features texte des prompts négatifs ``(N, D)``.
        device: Device PyTorch.

    Returns:
        Liste de tuples ``(score_positive, score_negative)`` par image.
        Les images illisibles reçoivent ``(nan, nan)``.
    """
    batch_tensors: list[torch.Tensor] = []
    valid_indices: list[int] = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(preprocess(img))
            valid_indices.append(i)
        except (OSError, ValueError) as e:
            logger.warning(f"Image illisible {path.name} : {e}")

    results: list[tuple[float, float]] = [(float("nan"), float("nan"))] * len(image_paths)
    if not batch_tensors:
        return results

    batch = torch.stack(batch_tensors).to(device)
    image_features = model.encode_image(batch)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Similarité cosinus entre chaque image et chaque prompt
    sim_pos = image_features @ text_pos.T  # (B, P)
    sim_neg = image_features @ text_neg.T  # (B, N)

    max_pos = sim_pos.max(dim=-1).values.cpu().tolist()
    max_neg = sim_neg.max(dim=-1).values.cpu().tolist()
    for local_idx, global_idx in enumerate(valid_indices):
        results[global_idx] = (float(max_pos[local_idx]), float(max_neg[local_idx]))
    return results


def score_dataset(
    manifest: pd.DataFrame,
    raw_dir: Path,
    model: nn.Module,
    preprocess: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Score toutes les images du manifest et retourne un DataFrame.

    Args:
        manifest: DataFrame avec colonnes ``image_lien`` et ``species``.
        raw_dir: Répertoire racine des images brutes (``data/raw/Mushrooms_images``).
        model: Modèle OpenCLIP chargé.
        preprocess: Transform image OpenCLIP.
        tokenizer: Tokenizer OpenCLIP.
        device: Device PyTorch.
        batch_size: Taille de batch pour l'inférence.

    Returns:
        DataFrame avec colonnes ``image_lien``, ``species``, ``score_positive``,
        ``score_negative``, ``score_final``.
    """
    text_pos = encode_prompts(model, tokenizer, POSITIVE_PROMPTS, device)
    text_neg = encode_prompts(model, tokenizer, NEGATIVE_PROMPTS, device)

    scores_pos: list[float] = []
    scores_neg: list[float] = []
    n_total = len(manifest)
    t0 = time.time()
    for start in range(0, n_total, batch_size):
        batch_rows = manifest.iloc[start : start + batch_size]
        paths = [raw_dir / lien for lien in batch_rows["image_lien"]]
        batch_results = score_batch(model, preprocess, paths, text_pos, text_neg, device)
        for pos, neg in batch_results:
            scores_pos.append(pos)
            scores_neg.append(neg)
        done = start + len(batch_rows)
        if done % (batch_size * 10) == 0 or done == n_total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            logger.info(
                f"  {done:>6,}/{n_total:,} images ({rate:.1f} img/s, ETA {eta / 60:.1f} min)"
            )

    result = manifest[["image_lien", "species"]].copy()
    result["score_positive"] = scores_pos
    result["score_negative"] = scores_neg
    result["score_final"] = result["score_positive"] - result["score_negative"]
    return result


# ---------------------------------------------------------------------------
# Echantillonnage stratifié
# ---------------------------------------------------------------------------
def stratified_sample(
    manifest: pd.DataFrame,
    total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Prélève un échantillon stratifié par classe.

    Chaque classe contribue proportionnellement à sa taille dans
    le manifest, avec un minimum de ``max(1, total // n_classes // 2)``
    pour garantir une représentation des classes rares.

    Args:
        manifest: DataFrame complet.
        total: Taille approximative de l'échantillon.
        seed: Graine pour la reproductibilité.

    Returns:
        Sous-ensemble du manifest, mélangé.
    """
    n_classes = manifest["species"].nunique()
    min_per_class = max(1, total // n_classes // 2)
    samples: list[pd.DataFrame] = []
    for _species, group in manifest.groupby("species"):
        n_take = max(min_per_class, round(len(group) / len(manifest) * total))
        n_take = min(n_take, len(group))
        samples.append(group.sample(n=n_take, random_state=seed))
    sampled = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Echantillon stratifié : {len(sampled)} images sur {n_classes} classes")
    return sampled  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def build_report(
    scores: pd.DataFrame,
    threshold: float,
    device: str,
    batch_size: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Construit le rapport JSON synthétique du run de filtrage.

    Args:
        scores: DataFrame des scores.
        threshold: Seuil utilisé pour ``is_mushroom``.
        device: Device PyTorch utilisé.
        batch_size: Taille de batch.
        elapsed_seconds: Durée totale du run.

    Returns:
        Dictionnaire sérialisable pour ``quality_report.json``.
    """
    valid = scores.dropna(subset=["score_final"])
    passed = valid[valid["score_final"] >= threshold]
    failed = valid[valid["score_final"] < threshold]

    quantiles = valid["score_final"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

    per_class_total = scores.groupby("species").size()
    per_class_passed = passed.groupby("species").size()
    per_class: dict[str, dict[str, int]] = {}
    for species, total in per_class_total.items():
        kept = int(per_class_passed.get(species, 0))
        per_class[str(species)] = {
            "total": int(total),
            "kept": kept,
            "excluded": int(total) - kept,
        }

    return {
        "model": f"OpenCLIP {MODEL_NAME} / {PRETRAINED}",
        "device": device,
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "positive_prompts": list(POSITIVE_PROMPTS),
        "negative_prompts": list(NEGATIVE_PROMPTS),
        "threshold": threshold,
        "total_images": len(scores),
        "valid_images": len(valid),
        "unreadable_images": int(len(scores) - len(valid)),
        "passed": len(passed),
        "failed": len(failed),
        "score_final_stats": {
            "min": round(float(valid["score_final"].min()), 4),
            "max": round(float(valid["score_final"].max()), 4),
            "mean": round(float(valid["score_final"].mean()), 4),
            "std": round(float(valid["score_final"].std()), 4),
            "quantiles": {str(k): round(float(v), 4) for k, v in quantiles.items()},
        },
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def apply_filter(
    scores: pd.DataFrame,
    threshold: float,
    model_descriptor: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    """Sépare les images gardées des images exclues et prépare excluded.json.

    Args:
        scores: DataFrame des scores avec ``is_mushroom`` déjà calculé.
        threshold: Seuil utilisé (pour traçabilité dans excluded.json).
        model_descriptor: Descripteur du modèle (pour traçabilité).

    Returns:
        Tuple ``(filtered_manifest, excluded_entries, summary)`` où :
            - ``filtered_manifest`` a les colonnes ``image_lien`` et ``species``
            - ``excluded_entries`` est la liste JSON-sérialisable pour ``excluded.json``
            - ``summary`` = ``{"kept": int, "excluded": int, "unreadable": int}``
    """
    valid = scores.dropna(subset=["score_final"])
    unreadable = scores[scores["score_final"].isna()]
    passed = valid[valid["is_mushroom"]]
    failed = valid[~valid["is_mushroom"]]

    filtered_manifest = (
        passed[["image_lien", "species"]]
        .sort_values(["species", "image_lien"])
        .reset_index(drop=True)
    )

    excluded: list[dict[str, Any]] = []
    for _, row in failed.iterrows():
        excluded.append(
            {
                "image_lien": str(row["image_lien"]),
                "species": str(row["species"]),
                "reason": CLEANING_REPORT_REASON,
                "threshold": threshold,
                "model": model_descriptor,
                "score_positive": round(float(row["score_positive"]), 4),
                "score_negative": round(float(row["score_negative"]), 4),
                "score_final": round(float(row["score_final"]), 4),
            }
        )
    for _, row in unreadable.iterrows():
        excluded.append(
            {
                "image_lien": str(row["image_lien"]),
                "species": str(row["species"]),
                "reason": "unreadable_image",
                "threshold": threshold,
                "model": model_descriptor,
                "score_positive": None,
                "score_negative": None,
                "score_final": None,
            }
        )

    summary = {
        "kept": len(passed),
        "excluded": len(failed),
        "unreadable": len(unreadable),
    }
    return filtered_manifest, excluded, summary


def update_cleaning_report(
    cleaning_report_path: Path,
    summary: dict[str, int],
    threshold: float,
    model_descriptor: str,
    filtered_class_counts: dict[str, int],
) -> None:
    """Met à jour data/cleaning_report.json pour refléter le filtrage CLIP.

    Ajoute une nouvelle entrée ``clip_quality_filter`` dans ``exclusion_reasons``,
    met à jour ``after.total_images``, ``after.num_classes``, ``after.class_counts``
    et ``excluded_count``. Préserve les autres clés du rapport.

    Args:
        cleaning_report_path: Chemin vers data/cleaning_report.json.
        summary: Résultats du filtrage (``kept``, ``excluded``, ``unreadable``).
        threshold: Seuil appliqué (pour traçabilité).
        model_descriptor: Descripteur du modèle utilisé.
        filtered_class_counts: Distribution des classes après filtrage.
    """
    if cleaning_report_path.exists():
        with open(cleaning_report_path, encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {"policy": "", "before": {}, "after": {}, "exclusion_reasons": {}}

    before_after = int(report.get("after", {}).get("total_images", 0))
    new_after = summary["kept"]
    removed_by_clip = summary["excluded"] + summary["unreadable"]

    report["after"] = {
        "total_images": new_after,
        "num_classes": len(filtered_class_counts),
        "class_counts": dict(sorted(filtered_class_counts.items())),
    }
    reasons = report.setdefault("exclusion_reasons", {})
    reasons[CLEANING_REPORT_REASON] = removed_by_clip
    report["excluded_count"] = int(report.get("excluded_count", 0)) + removed_by_clip
    report.setdefault("clip_quality_filter", {})
    report["clip_quality_filter"] = {
        "model": model_descriptor,
        "threshold": threshold,
        "excluded_by_score": summary["excluded"],
        "excluded_unreadable": summary["unreadable"],
        "before_total": before_after,
        "after_total": new_after,
    }

    with open(cleaning_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI.

    Returns:
        Namespace argparse.
    """
    parser = argparse.ArgumentParser(
        description="Filtre de qualité OpenCLIP pour le dataset brut de champignons.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIR / "curated_manifest.csv",
        help="Chemin du manifest source (défaut : data/curated_manifest.csv).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Répertoire des images (défaut : {RAW_DIR}).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Si fourni, scorer un échantillon stratifié de N images (mode calibration).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Seuil sur score_final pour is_mushroom (défaut : {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Taille de batch (défaut : {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device PyTorch (défaut : auto).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffixe à ajouter aux noms de fichiers de sortie (ex: '_calibration').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pour l'échantillonnage stratifié (défaut : 42).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Après scoring, écrit curated_manifest_filtered.csv + excluded.json "
            "et met à jour cleaning_report.json. Non compatible avec --sample."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI : charge, score, sauvegarde."""
    args = parse_args()

    if not args.manifest.exists():
        msg = f"Manifest introuvable : {args.manifest}"
        raise FileNotFoundError(msg)
    if not args.raw_dir.exists():
        msg = f"Répertoire d'images introuvable : {args.raw_dir}"
        raise FileNotFoundError(msg)

    manifest = pd.read_csv(args.manifest)
    logger.info(
        f"Manifest chargé : {len(manifest):,} images, {manifest['species'].nunique()} classes"
    )

    if args.sample is not None:
        manifest = stratified_sample(manifest, args.sample, seed=args.seed)

    device = resolve_device(args.device)
    model, preprocess, tokenizer = load_clip_model(device)

    t0 = time.time()
    scores = score_dataset(
        manifest=manifest,
        raw_dir=args.raw_dir,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    scores["is_mushroom"] = scores["score_final"] >= args.threshold

    scores_path = DATA_DIR / f"quality_scores{args.output_suffix}.csv"
    report_path = DATA_DIR / f"quality_report{args.output_suffix}.json"

    scores.to_csv(scores_path, index=False, encoding="utf-8")
    logger.info(f"Scores sauvegardés : {scores_path}")

    report = build_report(scores, args.threshold, device, args.batch_size, elapsed)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Rapport sauvegardé : {report_path}")

    logger.info(
        f"Terminé en {elapsed / 60:.1f} min : "
        f"{report['passed']:,} images gardées / {report['failed']:,} exclues "
        f"(seuil = {args.threshold})"
    )

    # ---------------------------------------------------------------------
    # Mode --apply : écrit le manifest filtré + excluded.json + cleaning_report
    # ---------------------------------------------------------------------
    if args.apply:
        if args.sample is not None:
            msg = "--apply ne peut pas être combiné avec --sample (résultat partiel)."
            raise ValueError(msg)

        model_descriptor = f"OpenCLIP {MODEL_NAME} / {PRETRAINED}"
        filtered_manifest, excluded, summary = apply_filter(
            scores, args.threshold, model_descriptor
        )

        filtered_path = DATA_DIR / "curated_manifest_filtered.csv"
        filtered_manifest.to_csv(filtered_path, index=False, encoding="utf-8")
        logger.info(f"Manifest filtré : {filtered_path} ({len(filtered_manifest):,} images)")

        excluded_path = DATA_DIR / "excluded.json"
        with open(excluded_path, "w", encoding="utf-8") as f:
            json.dump(excluded, f, indent=2, ensure_ascii=False)
        logger.info(f"Exclusions : {excluded_path} ({len(excluded):,} entrées)")

        filtered_class_counts = filtered_manifest["species"].value_counts().to_dict()
        update_cleaning_report(
            DATA_DIR / "cleaning_report.json",
            summary,
            args.threshold,
            model_descriptor,
            filtered_class_counts,
        )
        logger.info(
            f"cleaning_report.json mis à jour : kept={summary['kept']:,}, "
            f"excluded={summary['excluded']:,}, unreadable={summary['unreadable']}"
        )
        logger.info(
            "Prochaine étape : relancer 'python data/data_split.py' pour régénérer "
            "split_manifest.csv à partir du manifest filtré."
        )


if __name__ == "__main__":
    sys.exit(main())
