"""DAG d'orchestration Champy : régénération de l'analyse depuis MLflow.

Ce DAG enchaîne quatre tâches autour du pipeline analysis-as-code :

1. ``check_mlflow_health``      : vérifie que le MLflow configuré (local) répond
2. ``regenerate_analysis``      : exécute ``scripts.generate_analysis``
3. ``list_generated_snapshots`` : inventorie les fichiers JSON présents
4. ``notify_completion``        : log final récapitulatif

Si MLflow ne répond pas, la régénération est court-circuitée pour éviter
de produire un snapshot corrompu ou vide.

Déclenchement
-------------

- **Manuel** : depuis l'UI Airflow (bouton Trigger DAG)
- **Programmé** : aucun schedule par défaut (``schedule=None``)

Pour activer un déclenchement quotidien, remplacer ``schedule=None`` par
``schedule="@daily"`` ou par une expression cron.

Évolutions prévues
------------------

- Ajout d'une tâche ``trigger_training`` qui lance un entraînement
  distant via SSH sur le poste GPU (XPS2)
- Ajout d'une tâche ``import_to_bentoml`` qui pousse le meilleur modèle
  dans le BentoML Model Store
- Ajout d'une tâche ``commit_snapshot`` qui versionne automatiquement
  le snapshot généré dans git
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "Champy Team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Le PROJECT_ROOT est monté à /opt/champy dans le conteneur Airflow
# (cf. docker-compose.airflow.yml, section volumes du service airflow).
PROJECT_ROOT_IN_CONTAINER = Path("/opt/champy")
ANALYSIS_DIR_IN_CONTAINER = PROJECT_ROOT_IN_CONTAINER / "docs" / "analysis"


@dag(
    dag_id="champy_train_pipeline",
    description="Orchestration de la régénération de l'analyse Champy depuis MLflow",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 5, 14),
    schedule=None,
    catchup=False,
    tags=["champy", "analysis", "mlflow"],
)
def champy_train_pipeline() -> None:
    """Pipeline d'orchestration de la régénération d'analyse Champy."""

    @task
    def check_mlflow_health() -> dict:
        """Vérifie que le MLflow configuré (MLFLOW_TRACKING_URI) répond.

        Le client MLflow lit l'URI et les identifiants éventuels dans
        l'environnement, sans aucune URL codée en dur : il valide donc le
        MLflow local du projet (et resterait correct si l'environnement
        pointait vers un MLflow distant protégé par identifiants).
        """
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI non définie dans l'environnement.")

        # search_experiments interroge l'API MLflow standard : elle échoue si le
        # serveur est injoignable ou si une authentification requise n'aboutit pas.
        try:
            MlflowClient().search_experiments(max_results=1)
        except Exception as exc:
            raise RuntimeError(
                f"MLflow injoignable à {tracking_uri} : {exc}",
            ) from exc

        print(f"MLflow OK : {tracking_uri}")
        return {"tracking_uri": tracking_uri, "status": "healthy"}

    # La régénération elle-même utilise un BashOperator pour rester proche
    # de la commande qu'un développeur taperait à la main. Le BashOperator
    # hérite par défaut de toutes les variables d'environnement du conteneur
    # Airflow, dont MLFLOW_TRACKING_URI (qui pointe vers le MLflow local),
    # transmise sans configuration explicite.
    regenerate_analysis = BashOperator(
        task_id="regenerate_analysis",
        bash_command=(
            "cd /opt/champy && PYTHONPATH=/opt/champy python -m scripts.generate_analysis"
        ),
    )

    @task
    def list_generated_snapshots() -> list[dict]:
        """Inventorie les snapshots présents et retourne leurs métadonnées."""
        if not ANALYSIS_DIR_IN_CONTAINER.exists():
            raise FileNotFoundError(
                f"Le dossier {ANALYSIS_DIR_IN_CONTAINER} est introuvable. "
                "La régénération n'a probablement pas abouti.",
            )

        snapshots = []
        for path in sorted(ANALYSIS_DIR_IN_CONTAINER.glob("*.json")):
            stat = path.stat()
            snapshots.append(
                {
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )

        if not snapshots:
            raise FileNotFoundError(
                f"Aucun snapshot trouvé dans {ANALYSIS_DIR_IN_CONTAINER}.",
            )

        print(f"Snapshots disponibles ({len(snapshots)}) :")
        for snap in snapshots:
            print(
                f"  - {snap['name']} "
                f"({snap['size_bytes']} octets, modifié {snap['modified_iso']})",
            )

        return snapshots

    @task
    def notify_completion(
        mlflow_status: dict,
        snapshots: list[dict],
    ) -> str:
        """Log final récapitulatif et retour pour XCom."""
        latest_versioned = [s for s in snapshots if s["name"] != "current.json"]
        latest_name = latest_versioned[-1]["name"] if latest_versioned else "?"

        summary = (
            f"Pipeline Champy terminé avec succès.\n"
            f"  - MLflow : OK ({mlflow_status['tracking_uri']})\n"
            f"  - Snapshots totaux : {len(snapshots)}\n"
            f"  - Dernier snapshot : {latest_name}"
        )
        print(summary)

        return summary

    # Chaînage explicite : MLflow OK avant régénération, régénération
    # OK avant inventaire, inventaire OK avant notification.
    mlflow_status = check_mlflow_health()
    mlflow_status >> regenerate_analysis  # type: ignore[operator]
    snapshots = list_generated_snapshots()
    regenerate_analysis >> snapshots  # type: ignore[operator]
    notify_completion(mlflow_status, snapshots)


champy_train_pipeline()
