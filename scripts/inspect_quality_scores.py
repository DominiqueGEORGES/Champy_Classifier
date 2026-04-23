"""Visualise la distribution des scores OpenCLIP pour calibrer le seuil.

Lit le CSV produit par ``data/quality_filter.py`` et génère :

    1. Un histogramme de ``score_final`` avec les quantiles Q5/Q25/Q50/Q75/Q95.
    2. Trois panels d'images (4x5 = 20 images chacun) :
        - ``top20``         : les scores les plus élevés (clairement champignons)
        - ``bottom20``      : les scores les plus bas (probablement parasites)
        - ``borderline20``  : les scores autour du seuil candidat

L'objectif est de choisir visuellement un seuil qui élimine les faux
positifs sans sacrifier de vrais champignons.

Usage :
    python scripts/inspect_quality_scores.py
    python scripts/inspect_quality_scores.py --scores data/quality_scores_calibration.csv
    python scripts/inspect_quality_scores.py --threshold 0.05 --output-dir models/artifacts/quality_calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = PROJECT_ROOT / "data" / "quality_scores_calibration.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Mushrooms_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "artifacts" / "quality_calibration"

PANEL_ROWS = 4
PANEL_COLS = 5
PANEL_SIZE = PANEL_ROWS * PANEL_COLS  # 20


def plot_histogram(
    scores: pd.DataFrame,
    threshold: float,
    output: Path,
) -> None:
    """Trace l'histogramme de ``score_final`` avec quantiles et seuil.

    Args:
        scores: DataFrame avec colonne ``score_final``.
        threshold: Seuil candidat à matérialiser en trait rouge.
        output: Chemin du PNG de sortie.
    """
    valid = scores.dropna(subset=["score_final"])
    values = valid["score_final"].to_numpy()
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Seuil = {threshold}")

    labels = ["Q5", "Q25", "Q50 (médiane)", "Q75", "Q95"]
    colors = ["#9e9e9e", "#607d8b", "#000000", "#607d8b", "#9e9e9e"]
    for q, label, color in zip(quantiles, labels, colors, strict=True):
        ax.axvline(q, color=color, linestyle=":", alpha=0.6, linewidth=1)
        ax.text(
            q, ax.get_ylim()[1] * 0.95, f"{label}\n{q:.3f}", fontsize=8, ha="center", color=color
        )

    ax.set_xlabel("score_final = max(positifs) - max(négatifs)")
    ax.set_ylabel("Nombre d'images")
    ax.set_title(
        f"Distribution des scores OpenCLIP ({len(values)} images valides)",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_image_panel(
    rows: pd.DataFrame,
    raw_dir: Path,
    title: str,
    output: Path,
    score_color: str = "black",
) -> None:
    """Trace un panel 4x5 d'images avec leur score et leur espèce.

    Args:
        rows: DataFrame ``PANEL_SIZE`` lignes, triées dans l'ordre voulu.
        raw_dir: Répertoire des images brutes.
        title: Titre du panel.
        output: PNG de sortie.
        score_color: Couleur du score dans le sous-titre (vert, rouge, orange).
    """
    fig, axes = plt.subplots(PANEL_ROWS, PANEL_COLS, figsize=(PANEL_COLS * 3, PANEL_ROWS * 3))
    axes_flat = axes.flatten()
    rows = rows.reset_index(drop=True)

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= len(rows):
            continue
        row = rows.iloc[i]
        img_path = raw_dir / str(row["image_lien"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except (OSError, ValueError):
            ax.text(0.5, 0.5, "image\nillisible", ha="center", va="center")
            continue
        species_short = str(row["species"])[:24]
        ax.set_title(
            f"{species_short}\nscore = {row['score_final']:.3f}",
            fontsize=8,
            color=score_color,
        )

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110, bbox_inches="tight")
    plt.close(fig)


def select_panels(
    scores: pd.DataFrame,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Sélectionne les 3 panels (top, bottom, borderline) depuis les scores.

    Args:
        scores: DataFrame complet des scores.
        threshold: Seuil candidat autour duquel sélectionner les borderlines.

    Returns:
        Dictionnaire ``{'top20': df, 'bottom20': df, 'borderline20': df}``.
    """
    valid = scores.dropna(subset=["score_final"]).sort_values("score_final")
    bottom = valid.head(PANEL_SIZE)
    top = valid.tail(PANEL_SIZE).iloc[::-1]  # ordre décroissant
    # Borderline : PANEL_SIZE images dont score_final est le plus proche du seuil
    borderline = (
        valid.assign(delta=(valid["score_final"] - threshold).abs())
        .nsmallest(PANEL_SIZE, "delta")
        .sort_values("score_final")
    )
    return {"top20": top, "bottom20": bottom, "borderline20": borderline}


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI.

    Returns:
        Namespace argparse.
    """
    parser = argparse.ArgumentParser(
        description="Visualise la distribution des scores OpenCLIP et génère 3 panels d'exemples.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help=f"CSV des scores (défaut : {DEFAULT_SCORES}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Répertoire des images brutes (défaut : {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Seuil candidat pour l'histogramme et le panel borderline (défaut : 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Répertoire de sortie pour les PNG (défaut : {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI : histogramme + 3 panels."""
    args = parse_args()

    if not args.scores.exists():
        msg = f"Fichier de scores introuvable : {args.scores}"
        raise FileNotFoundError(msg)

    scores = pd.read_csv(args.scores)
    print(f"Scores chargés : {len(scores)} lignes depuis {args.scores.name}")

    hist_path = args.output_dir / "histogram.png"
    plot_histogram(scores, args.threshold, hist_path)
    print(f"Histogramme écrit : {hist_path}")

    panels = select_panels(scores, args.threshold)
    for name, df in panels.items():
        color = {"top20": "#2e7d32", "bottom20": "#c62828", "borderline20": "#ef6c00"}[name]
        title = {
            "top20": "Top 20 - scores les plus élevés (clairement champignons)",
            "bottom20": "Bottom 20 - scores les plus bas (probablement parasites)",
            "borderline20": f"Borderline 20 - scores autour de {args.threshold}",
        }[name]
        out = args.output_dir / f"{name}.png"
        plot_image_panel(df, args.raw_dir, title, out, score_color=color)
        print(f"Panel {name} écrit : {out}")

    # Résumé texte
    valid = scores.dropna(subset=["score_final"])
    print("\n=== Résumé ===")
    print(f"  Total valides : {len(valid)}")
    print(f"  min / max     : {valid['score_final'].min():.3f} / {valid['score_final'].max():.3f}")
    print(f"  médiane       : {valid['score_final'].median():.3f}")
    print(
        f"  Q5 / Q95      : {valid['score_final'].quantile(0.05):.3f} / {valid['score_final'].quantile(0.95):.3f}"
    )
    for thr in (-0.05, 0.0, 0.05, 0.1, 0.15):
        n_passed = int((valid["score_final"] >= thr).sum())
        pct = n_passed / len(valid) * 100
        print(f"  seuil = {thr:+.2f}  -> {n_passed:>4}/{len(valid)} gardés ({pct:.1f}%)")


if __name__ == "__main__":
    main()
