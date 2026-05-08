"""Genere un rapport Evidently de drift sur les predictions recentes.

Compare les predictions stockees dans le SQLite (PredictionStore, Bloc M2)
sur une fenetre glissante (defaut 24h) avec la baseline calculee par
``baseline_snapshot.py``. Produit un fichier HTML auto-suffisant dans
``monitoring/reports/drift_YYYYMMDD_HHMM.html`` qui peut etre ouvert
directement dans un navigateur ou embarque dans la page Streamlit
``11_drift.py``.

Le rapport couvre deux dimensions :

- **Drift de classes** : la distribution des especes predites s'ecarte-
  t-elle de la baseline ? (test du chi-2 sur la categorie
  ``predicted_class``).
- **Drift de confiance** : la distribution des scores de confiance
  top-1 derive-t-elle ? (test KS sur la colonne numerique
  ``confidence``).

Usage :
    python monitoring/run_drift_report.py
    python monitoring/run_drift_report.py --hours 6
    python monitoring/run_drift_report.py --baseline monitoring/baseline_reference.json \
        --output-dir monitoring/reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger

# Module de monitoring : on a besoin du PredictionStore pour relire les
# predictions de production. ``sys.path.insert`` permet de lancer le
# script depuis n'importe ou.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.serving_bentoml.storage import PredictionStore  # noqa: E402

DEFAULT_BASELINE = REPO_ROOT / "monitoring" / "baseline_reference.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "monitoring" / "reports"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "runtime" / "predictions.db"
DEFAULT_HOURS = 24


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec ``hours``, ``baseline``, ``output_dir``, ``db``.
    """
    parser = argparse.ArgumentParser(
        description="Genere un rapport Evidently de drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Fenetre glissante en heures sur les predictions stockees.",
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
        help="Repertoire de sortie des rapports HTML.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Chemin de la base SQLite (PredictionStore).",
    )
    return parser.parse_args()


def baseline_to_dataframe(baseline: dict[str, Any]) -> pd.DataFrame:
    """Reconstruit un DataFrame de reference a partir de la baseline JSON.

    Pour chaque classe, on materialise ``count`` lignes avec
    ``confidence_mean`` comme valeur de la colonne ``confidence``. Cette
    materialisation perd l'information de variance intra-classe mais
    reste suffisante pour Evidently (il compare les distributions
    globales des deux dataframes, pas les pre-calculs).

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
    """Lit les predictions recentes depuis le PredictionStore SQLite.

    Args:
        db_path: Chemin du fichier SQLite.
        hours: Fenetre glissante.

    Returns:
        DataFrame avec colonnes ``predicted_class`` et ``confidence``,
        eventuellement vide si aucune prediction recente.
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
    """Construit le rapport Evidently et le genere sur les deux datasets.

    Le rapport comporte un ``DataDriftPreset`` (drift par colonne) et un
    ``DataSummaryPreset`` (statistiques descriptives reference vs
    current) pour donner les chiffres bruts qui justifient le verdict
    de drift.

    Args:
        reference: DataFrame de reference (issue de la baseline).
        current: DataFrame des predictions de production recentes.

    Returns:
        ``Report`` Evidently dont ``run`` a deja ete execute.
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


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si rapport genere, 1 sinon (donnees absentes
        ou erreur).
    """
    args = parse_args()
    if not args.baseline.exists():
        logger.error(
            f"Baseline introuvable : {args.baseline}. "
            "Lancer 'python monitoring/baseline_snapshot.py' au prealable."
        )
        return 1

    logger.info(f"Chargement baseline : {args.baseline}")
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    reference = baseline_to_dataframe(baseline)
    logger.info(
        f"Reference : {len(reference)} lignes, {reference['predicted_class'].nunique()} classes"
    )

    logger.info(f"Lecture predictions recentes ({args.hours}h) depuis {args.db}")
    current = asyncio.run(_load_current(args.db, args.hours))
    if current.empty:
        logger.warning(
            f"Aucune prediction sur les {args.hours} dernieres heures. "
            "Le rapport ne peut pas etre genere - alimenter d'abord la base "
            "via /predict (BentoML)."
        )
        return 1
    logger.info(
        f"Current : {len(current)} predictions, {current['predicted_class'].nunique()} classes"
    )

    snapshot = build_report(reference, current)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = args.output_dir / f"drift_{timestamp}.html"
    snapshot.save_html(str(output_path))
    logger.success(f"Rapport genere : {output_path}")
    logger.info(f"Ouvrir dans le navigateur : {output_path.resolve().as_uri()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
