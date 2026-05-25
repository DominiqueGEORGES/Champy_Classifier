"""Génère un rapport Evidently de drift sur les prédictions récentes.

Compare les prédictions stockées dans le SQLite (PredictionStore, Bloc M2)
sur une fenêtre glissante (défaut 24h) avec la baseline calculée par
``baseline_snapshot.py``. Produit un fichier HTML auto-suffisant dans
``monitoring/reports/drift_YYYYMMDD_HHMM.html`` qui peut être ouvert
directement dans un navigateur ou embarqué dans la page Streamlit
``11_drift.py``.

Le rapport HTML est augmenté d'un bandeau d'en-tête (date FR, fenêtre
analysée, tailles des datasets) injecté juste après ``<body>`` pour
que le fichier reste explicite quand on le télécharge ou le partage
hors de Streamlit.

Le rapport couvre deux dimensions :

- **Drift de classes** : la distribution des espèces prédites s'écarte-
  t-elle de la baseline ? (test du chi-2 sur la catégorie
  ``predicted_class``).
- **Drift de confiance** : la distribution des scores de confiance
  top-1 dérive-t-elle ? (test KS sur la colonne numérique
  ``confidence``).

Usage :
    python monitoring/run_drift_report.py
    python monitoring/run_drift_report.py --hours 6
    python monitoring/run_drift_report.py --baseline monitoring/baseline_reference.json \
        --output-dir monitoring/reports
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# =====================================================================
# Imports tiers
# =====================================================================
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger

# =====================================================================
# Setup chemin projet
# =====================================================================

# Module de monitoring : on a besoin du PredictionStore pour relire les
# prédictions de production. ``sys.path.insert`` permet de lancer le
# script depuis n'importe où.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# =====================================================================
# Imports projet
# =====================================================================

from src.serving_bentoml.storage import PredictionStore  # noqa: E402

# =====================================================================
# Constantes
# =====================================================================

DEFAULT_BASELINE = REPO_ROOT / "monitoring" / "baseline_reference.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "monitoring" / "reports"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "runtime" / "predictions.db"
DEFAULT_HOURS = 24

# Mapping mois français pour formatage lisible (évite la dépendance à
# la locale système qui peut ne pas être installée dans le container).
MOIS_FR = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}


# =====================================================================
# Fonctions utilitaires
# =====================================================================


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec ``hours``, ``baseline``, ``output_dir``, ``db``.
    """
    parser = argparse.ArgumentParser(
        description="Génère un rapport Evidently de drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Fenêtre glissante en heures sur les prédictions stockées.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Chemin du JSON baseline produit par baseline_snapshot.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des rapports HTML.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Chemin de la base SQLite (PredictionStore).",
    )
    return parser.parse_args()


def _format_date_fr(ts: datetime) -> str:
    """Formate un timestamp en français lisible.

    Args:
        ts: Timestamp à formater.

    Returns:
        Chaîne du type ``21 mai 2026 à 23h12``.
    """
    return f"{ts.day} {MOIS_FR[ts.month]} {ts.year} à {ts:%Hh%M}"


def _format_int_fr(n: int) -> str:
    """Formate un entier avec espace insécable comme séparateur de milliers.

    Args:
        n: Entier à formater.

    Returns:
        Chaîne du type ``2&nbsp;872`` (HTML-safe pour insertion directe).
    """
    return f"{n:,}".replace(",", "&nbsp;")


def build_header_html(
    generated_at: datetime,
    hours: int,
    n_baseline: int,
    n_baseline_classes: int,
    n_current: int,
    n_current_classes: int,
    baseline_model: str,
) -> str:
    """Construit le bandeau HTML d'en-tête à injecter dans le rapport.

    Le bandeau est self-contained (styles inline) pour rester lisible
    quand le fichier est ouvert hors Streamlit. Palette projet (vert
    forêt #1F4E3D pour l'accent principal, ambre #B85C00 pour le
    soulignement, crème #FAFAF5 pour le fond).

    Args:
        generated_at: Horodatage de génération.
        hours: Fenêtre temporelle analysée.
        n_baseline: Nombre de lignes dans la baseline de référence.
        n_baseline_classes: Nombre de classes distinctes en baseline.
        n_current: Nombre de prédictions de production analysées.
        n_current_classes: Nombre de classes distinctes en production.
        baseline_model: Nom du modèle ayant produit la baseline.

    Returns:
        Bloc HTML à injecter juste après ``<body>``.
    """
    date_fr = _format_date_fr(generated_at)
    iso_ts = generated_at.strftime("%Y-%m-%dT%H:%M:%S")

    return f"""
<div data-champy-header="1" data-generated-at="{iso_ts}" style="
    font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
    background: #FAFAF5;
    border-left: 4px solid #1F4E3D;
    padding: 24px 32px;
    margin: 0 0 24px 0;
    color: #1a1a1a;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
">
  <div style="
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #B85C00;
      margin-bottom: 8px;
  ">
    Rapport MLOps
  </div>
  <h1 style="
      margin: 0 0 8px 0;
      font-size: 26px;
      font-weight: 800;
      color: #1F4E3D;
      letter-spacing: -0.01em;
  ">
    Détection de drift &mdash; Champy Classifier
  </h1>
  <p style="
      margin: 0 0 20px 0;
      font-size: 14px;
      color: #555;
      line-height: 1.5;
      max-width: 720px;
  ">
    Comparaison entre la baseline du test set et les prédictions de
    production récentes. Le verdict de drift se lit dans les tableaux
    et graphiques générés par Evidently AI ci-dessous.
  </p>
  <table style="
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      max-width: 720px;
  ">
    <tr>
      <td style="padding: 6px 12px 6px 0; color: #777; width: 32%; vertical-align: top;">
        Généré le
      </td>
      <td style="padding: 6px 0; font-weight: 600; vertical-align: top;">
        {date_fr}
      </td>
    </tr>
    <tr>
      <td style="padding: 6px 12px 6px 0; color: #777; vertical-align: top;">
        Fenêtre temporelle analysée
      </td>
      <td style="padding: 6px 0; font-weight: 600; vertical-align: top;">
        Dernières {hours} heures glissantes
      </td>
    </tr>
    <tr>
      <td style="padding: 6px 12px 6px 0; color: #777; vertical-align: top;">
        Baseline (référence)
      </td>
      <td style="padding: 6px 0; font-weight: 600; vertical-align: top;">
        {_format_int_fr(n_baseline)} prédictions sur {n_baseline_classes} classes
      </td>
    </tr>
    <tr>
      <td style="padding: 6px 12px 6px 0; color: #777; vertical-align: top;">
        Production (current)
      </td>
      <td style="padding: 6px 0; font-weight: 600; vertical-align: top;">
        {_format_int_fr(n_current)} prédictions sur {n_current_classes} classes
      </td>
    </tr>
    <tr>
      <td style="padding: 6px 12px 6px 0; color: #777; vertical-align: top;">
        Modèle de la baseline
      </td>
      <td style="padding: 6px 0; font-weight: 600; vertical-align: top;">
        <code style="background: #eef2ee; padding: 2px 6px; border-radius: 3px; font-size: 13px;">{baseline_model}</code>
      </td>
    </tr>
  </table>
</div>
"""


def inject_header(html_path: Path, header_html: str) -> None:
    """Injecte le bandeau HTML d'en-tête juste après ``<body>``.

    Lit le fichier généré par Evidently, insère le bloc d'en-tête au
    tout début du body, puis ré-écrit le fichier. Idempotent : si un
    bandeau existe déjà (data-champy-header="1"), on ne fait rien.

    Args:
        html_path: Chemin du rapport HTML à modifier in-place.
        header_html: Bloc HTML à injecter.
    """
    with open(html_path, encoding="utf-8") as f:
        content = f.read()

    if 'data-champy-header="1"' in content:
        logger.warning("Bandeau d'en-tête déjà présent, pas de ré-injection.")
        return

    # Tentative d'injection juste après <body> (cas général Evidently)
    if "<body>" in content:
        content = content.replace("<body>", "<body>\n" + header_html, 1)
    elif "<body " in content:
        # Cas où <body> a des attributs (style, class, etc.)
        idx_end = content.index(">", content.index("<body ")) + 1
        content = content[:idx_end] + "\n" + header_html + content[idx_end:]
    else:
        logger.warning("Balise <body> introuvable, injection au début du fichier.")
        content = header_html + content

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


# =====================================================================
# Helpers de chargement des données
# =====================================================================


def baseline_to_dataframe(baseline: dict[str, Any]) -> pd.DataFrame:
    """Reconstruit un DataFrame de référence à partir de la baseline JSON.

    Pour chaque classe, on matérialise ``count`` lignes avec
    ``confidence_mean`` comme valeur de la colonne ``confidence``. Cette
    matérialisation perd l'information de variance intra-classe mais
    reste suffisante pour Evidently (il compare les distributions
    globales des deux dataframes, pas les pré-calculs).

    Args:
        baseline: Dictionnaire issu de ``baseline_reference.json``.

    Returns:
        DataFrame avec colonnes ``predicted_class`` et ``confidence``.
    """
    rows: list[dict[str, object]] = []
    for species, stats in baseline.get("per_class", {}).items():
        n = int(stats.get("count", 0))
        conf = float(stats.get("confidence_mean", 0.0))
        for _ in range(n):
            rows.append({"predicted_class": species, "confidence": conf})
    return pd.DataFrame(rows)


async def _load_current(db_path: Path, hours: int) -> pd.DataFrame:
    """Lit les prédictions récentes depuis le PredictionStore SQLite.

    Args:
        db_path: Chemin du fichier SQLite.
        hours: Fenêtre glissante.

    Returns:
        DataFrame avec colonnes ``predicted_class`` et ``confidence``,
        éventuellement vide si aucune prédiction récente.
    """
    store = PredictionStore(db_path)
    await store.init()
    try:
        recent = await store.get_recent(hours=hours, limit=100_000)
    finally:
        await store.close()
    if not recent:
        return pd.DataFrame(columns=["predicted_class", "confidence"])
    return pd.DataFrame(
        [{"predicted_class": r.predicted_class, "confidence": r.confidence} for r in recent]
    )


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Construit le rapport Evidently et le génère sur les deux datasets.

    Le rapport comporte un ``DataDriftPreset`` (drift par colonne) et un
    ``DataSummaryPreset`` (statistiques descriptives référence vs
    current) pour donner les chiffres bruts qui justifient le verdict
    de drift.

    Args:
        reference: DataFrame de référence (issue de la baseline).
        current: DataFrame des prédictions de production récentes.

    Returns:
        ``Report`` Evidently dont ``run`` a déjà été exécuté.
    """
    data_definition = DataDefinition(
        categorical_columns=["predicted_class"],
        numerical_columns=["confidence"],
    )
    ref_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    cur_ds = Dataset.from_pandas(current, data_definition=data_definition)
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset(),
        ]
    )
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    return snapshot


# =====================================================================
# Point d'entrée
# =====================================================================


def main() -> int:
    """Point d'entrée CLI.

    Returns:
        Code de sortie : 0 si rapport généré, 1 sinon (données absentes
        ou erreur).
    """
    args = parse_args()
    if not args.baseline.exists():
        logger.error(
            f"Baseline introuvable : {args.baseline}. "
            "Lancer 'python monitoring/baseline_snapshot.py' au préalable."
        )
        return 1

    logger.info(f"Chargement baseline : {args.baseline}")
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    reference = baseline_to_dataframe(baseline)

    n_baseline = len(reference)
    n_baseline_classes = int(reference["predicted_class"].nunique())
    baseline_meta = baseline.get("metadata", {})
    baseline_model = Path(baseline_meta.get("model_path", "inconnu")).name

    logger.info(f"Référence : {n_baseline} lignes, {n_baseline_classes} classes")

    logger.info(f"Lecture prédictions récentes ({args.hours}h) depuis {args.db}")
    current = asyncio.run(_load_current(args.db, args.hours))
    if current.empty:
        logger.warning(
            f"Aucune prédiction sur les {args.hours} dernières heures. "
            "Le rapport ne peut pas être généré - alimenter d'abord la "
            "base via /predict (BentoML)."
        )
        return 1

    n_current = len(current)
    n_current_classes = int(current["predicted_class"].nunique())
    logger.info(f"Current : {n_current} prédictions, {n_current_classes} classes")

    snapshot = build_report(reference, current)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y%m%d_%H%M")
    output_path = args.output_dir / f"drift_{timestamp}.html"
    snapshot.save_html(str(output_path))
    logger.success(f"Rapport généré : {output_path}")

    # =================================================================
    # Injection du bandeau d'en-tête (date FR + métadonnées)
    # =================================================================
    header_html = build_header_html(
        generated_at=generated_at,
        hours=args.hours,
        n_baseline=n_baseline,
        n_baseline_classes=n_baseline_classes,
        n_current=n_current,
        n_current_classes=n_current_classes,
        baseline_model=baseline_model,
    )
    inject_header(output_path, header_html)
    logger.info("Bandeau d'en-tête injecté dans le rapport.")

    logger.info(f"Ouvrir dans le navigateur : {output_path.resolve().as_uri()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
