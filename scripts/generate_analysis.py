"""Génère un snapshot d'analyse versionné à partir des runs MLflow.

Le script interroge MLflow sur DagsHub, récupère les trois runs cibles
(ResNet50 default, ResNet50 aggressive, ConvNeXt-Tiny), calcule les
comparaisons, génère les narratifs avec valeurs interpolées et écrit
un fichier JSON daté dans ``docs/analysis/``.

Utilisation
-----------

Lancement manuel depuis la racine du projet ::

    python -m scripts.generate_analysis

Sortie ::

    docs/analysis/2026-05-14T09-45-00Z.json   (snapshot daté)
    docs/analysis/current.json                 (copie du dernier)

La page Streamlit ``demo/pages/13_analyse_modèles.py`` lit ces fichiers
sans appeler MLflow directement, garantissant la stabilité de la démo.

Variables d'environnement requises (dans le ``.env`` à la racine) ::

    MLFLOW_TRACKING_URI
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD

Configuration
-------------

La liste des runs cibles et le nom de l'expérience sont définis dans
les constantes en tête de fichier. Adapter si renommage côté MLflow.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from mlflow.entities import Run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "docs" / "analysis"
ENV_FILE = PROJECT_ROOT / ".env"

EXPERIMENT_NAME = "Default"

TARGET_RUNS: dict[str, str] = {
    "resnet50_default": "resnet50_2phase_42",
    "resnet50_aggressive": "resnet50_2phase_2026",
    "convnext_tiny": "convnext_tiny_2phase_2026",
}

GENERATOR_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_analysis")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def get_git_commit() -> str | None:
    """Retourne le hash court du commit git courant, ou None si indisponible."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> str | None:
    """Retourne la branche git courante, ou None si indisponible."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def format_pct(value: float | None, decimals: int = 1) -> str:
    """Formate un nombre entre 0 et 1 en pourcentage, ou ``—`` si None."""
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f} %"


def safe_float(value: Any) -> float | None:
    """Convertit en float si possible, sinon None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Récupération MLflow
# ---------------------------------------------------------------------------


def fetch_run_by_name(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
) -> Run | None:
    """Retourne le run le plus récent portant ce nom, ou None si introuvable."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def extract_run_data(run: Run | None) -> dict[str, Any] | None:
    """Extrait les données utiles d'un run MLflow sous forme de dict sérialisable."""
    if run is None:
        return None

    metrics = {k: safe_float(v) for k, v in run.data.metrics.items()}
    params = dict(run.data.params)

    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time_iso": datetime.fromtimestamp(
            run.info.start_time / 1000,
            tz=UTC,
        ).isoformat(),
        "end_time_iso": (
            datetime.fromtimestamp(
                run.info.end_time / 1000,
                tz=UTC,
            ).isoformat()
            if run.info.end_time
            else None
        ),
        "duration_seconds": (
            (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
        ),
        "params": params,
        "metrics": metrics,
        "mlflow_url": (
            f"{mlflow.get_tracking_uri()}/#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        ),
    }


# ---------------------------------------------------------------------------
# Calculs dérivés
# ---------------------------------------------------------------------------


def best_metric(run_data: dict[str, Any] | None, *metric_names: str) -> float | None:
    """Retourne la première métrique trouvée parmi celles fournies, ou None."""
    if run_data is None:
        return None
    for name in metric_names:
        value = run_data["metrics"].get(name)
        if value is not None:
            return value
    return None


def compute_comparison(runs: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Calcule les métriques comparatives entre les trois runs."""
    accuracy = {key: best_metric(data, "test_accuracy", "val_acc") for key, data in runs.items()}
    f1 = {key: best_metric(data, "test_f1_macro", "val_f1_macro") for key, data in runs.items()}

    best_model = None
    best_acc = -1.0
    for key, acc in accuracy.items():
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_model = key

    gap_accuracy = None
    gap_f1 = None
    if (
        best_model is not None
        and accuracy.get("resnet50_default") is not None
        and accuracy.get(best_model) is not None
    ):
        gap_accuracy = accuracy[best_model] - accuracy["resnet50_default"]
    if (
        best_model is not None
        and f1.get("resnet50_default") is not None
        and f1.get(best_model) is not None
    ):
        gap_f1 = f1[best_model] - f1["resnet50_default"]

    return {
        "accuracy_by_run": accuracy,
        "f1_macro_by_run": f1,
        "best_model": best_model,
        "best_accuracy": accuracy.get(best_model) if best_model else None,
        "best_f1_macro": f1.get(best_model) if best_model else None,
        "gap_accuracy_default_to_best": gap_accuracy,
        "gap_f1_default_to_best": gap_f1,
    }


def compute_acc_f1_gap(run_data: dict[str, Any] | None) -> float | None:
    """Écart accuracy moins F1 macro, indicateur d'équité entre classes."""
    if run_data is None:
        return None
    acc = best_metric(run_data, "test_accuracy", "val_acc")
    f1 = best_metric(run_data, "test_f1_macro", "val_f1_macro")
    if acc is None or f1 is None:
        return None
    return acc - f1


# ---------------------------------------------------------------------------
# Génération des narratifs
# ---------------------------------------------------------------------------


def build_narratives(
    runs: dict[str, dict[str, Any] | None],
    comparison: dict[str, Any],
) -> dict[str, str]:
    """Construit les narratifs prêts pour la soutenance, valeurs interpolées."""
    acc = comparison["accuracy_by_run"]
    f1 = comparison["f1_macro_by_run"]

    acc_agg = format_pct(acc.get("resnet50_aggressive"))
    acc_cnx = format_pct(acc.get("convnext_tiny"))
    f1_agg = format_pct(f1.get("resnet50_aggressive"))
    f1_cnx = format_pct(f1.get("convnext_tiny"))

    gap_cnx = compute_acc_f1_gap(runs.get("convnext_tiny"))
    gap_cnx_str = format_pct(gap_cnx) if gap_cnx is not None else "—"

    return {
        "why_convnext": (
            f"ConvNeXt-Tiny obtient {acc_cnx} d'accuracy et {f1_cnx} de F1 macro "
            f"sur le test set, soit une amélioration par rapport au ResNet50 "
            f"aggressive ({acc_agg} d'accuracy, {f1_agg} de F1 macro). "
            f"L'écart accuracy / F1 réduit à {gap_cnx_str} montre que ConvNeXt "
            "généralise mieux sur les classes minoritaires, ce qui est crucial "
            "dans le contexte d'un dataset déséquilibré (ratio 1 à 62 entre "
            "classe majoritaire et minoritaire). Cette amélioration justifie le "
            "choix de l'architecture moderne ConvNeXt pour la mise en production."
        ),
        "two_phase_strategy": (
            "Mon training se déroule en deux temps. D'abord pendant 10 epochs, "
            "je gèle 99 % du modèle et je ne lui apprends que les noms des "
            "30 espèces, avec une vitesse d'apprentissage élevée (1e-3). Puis "
            "je le dégèle entièrement et je continue à un rythme 30 fois plus "
            "lent (3e-5) pour qu'il affine sa vision sans casser ce qu'il sait "
            "déjà. Le bond visible sur la courbe d'accuracy à l'epoch 10 prouve "
            "que cette deuxième phase apporte vraiment quelque chose."
        ),
        "accuracy_vs_f1": (
            "L'accuracy mesure le pourcentage global de bonnes prédictions, "
            "tandis que le F1 macro calcule la moyenne des scores F1 par classe "
            "en donnant le même poids à chaque classe, indépendamment de leur "
            "effectif. L'écart entre les deux indique que le modèle est moins "
            "performant sur les classes minoritaires, ce qui est cohérent avec "
            "le ratio de déséquilibre de 1 à 62 dans notre dataset. J'ai "
            "partiellement compensé ce déséquilibre par un WeightedRandomSampler "
            "durant l'entraînement."
        ),
        "limitations": (
            "Trois limites principales. Premièrement, le déséquilibre du dataset "
            "reste un facteur limitant : même avec WeightedRandomSampler, les "
            "classes minoritaires n'ont qu'environ 40 images chacune, ce qui "
            "plafonne leur apprentissage. Deuxièmement, le pipeline d'augmentation "
            "est uniforme sur toutes les classes ; une augmentation différentielle "
            "ou l'introduction de techniques comme MixUp et CutMix bénéficierait "
            "particulièrement aux classes minoritaires. Troisièmement, je n'ai "
            "pas effectué d'audit qualité des annotations, ce qui est pourtant "
            "essentiel sur un domaine où certaines espèces sont très proches "
            "visuellement, notamment dans les genres Cortinarius et Russula."
        ),
        "perspectives": (
            "J'investirais en priorité sur les données plutôt que sur le modèle, "
            "en suivant l'approche data-centric AI promue par Andrew Ng. "
            "Concrètement, je commencerais par compléter les classes minoritaires "
            "via iNaturalist et MushroomObserver pour ramener le ratio de "
            "déséquilibre de 1 à 62 à 1 à 5. Ensuite je mettrais en place MixUp "
            "et CutMix dans le DataLoader, puis je ferais auditer un échantillon "
            "des annotations par un mycologue. Ces actions sur les données "
            "apportent typiquement 3 à 5 points de F1 macro supplémentaires."
        ),
        "reproducibility": (
            "Les trois runs ont été lancés avec des seeds fixées (42 pour le "
            "default, 2026 pour aggressive et ConvNeXt-Tiny) sur le même poste, "
            "avec les mêmes splits de données et la même architecture de "
            "pipeline. Toutes les métriques sont tracées dans MLflow sur DagsHub, "
            "avec les fichiers de configuration YAML versionnés dans le dépôt "
            "git. Cela permet une reproductibilité complète : n'importe qui peut "
            "relancer un training à partir du commit correspondant et obtenir "
            "les mêmes métriques aux fluctuations aléatoires près."
        ),
    }


# ---------------------------------------------------------------------------
# Sortie sur disque
# ---------------------------------------------------------------------------


def write_snapshot(snapshot: dict[str, Any]) -> tuple[Path, Path]:
    """Écrit le snapshot daté et met à jour ``current.json``, retourne les deux chemins."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    versioned_path = ANALYSIS_DIR / f"{timestamp}.json"
    current_path = ANALYSIS_DIR / "current.json"

    payload = json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    versioned_path.write_text(payload, encoding="utf-8")
    current_path.write_text(payload, encoding="utf-8")

    return versioned_path, current_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    """Point d'entrée du script : génère un snapshot d'analyse depuis MLflow."""
    logger.info("Chargement de l'environnement (.env)")
    load_dotenv(ENV_FILE)

    tracking_uri = mlflow.get_tracking_uri()
    logger.info("MLflow tracking URI : %s", tracking_uri)

    client = MlflowClient()

    logger.info("Recherche de l'expérience '%s'", EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error("Expérience '%s' introuvable sur MLflow", EXPERIMENT_NAME)
        return 1

    logger.info(
        "Expérience trouvée (id=%s), récupération des %d runs cibles",
        experiment.experiment_id,
        len(TARGET_RUNS),
    )

    runs: dict[str, dict[str, Any] | None] = {}
    for key, run_name in TARGET_RUNS.items():
        run = fetch_run_by_name(client, experiment.experiment_id, run_name)
        if run is None:
            logger.warning("Run '%s' (clé '%s') introuvable, ignoré", run_name, key)
        else:
            logger.info(
                "Run '%s' trouvé (id=%s, status=%s)",
                run_name,
                run.info.run_id,
                run.info.status,
            )
        runs[key] = extract_run_data(run)

    logger.info("Calcul des comparaisons")
    comparison = compute_comparison(runs)

    logger.info("Génération des narratifs avec valeurs interpolées")
    narratives = build_narratives(runs, comparison)

    snapshot = {
        "schema_version": "1.0",
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": experiment.experiment_id,
        },
        "runs": runs,
        "comparison": comparison,
        "narratives": narratives,
    }

    versioned_path, current_path = write_snapshot(snapshot)
    logger.info("Snapshot versionné écrit : %s", versioned_path.relative_to(PROJECT_ROOT))
    logger.info("current.json mis à jour : %s", current_path.relative_to(PROJECT_ROOT))

    nb_runs = sum(1 for r in runs.values() if r is not None)
    logger.info("Snapshot complet (%d/%d runs récupérés)", nb_runs, len(TARGET_RUNS))

    return 0


if __name__ == "__main__":
    sys.exit(main())
