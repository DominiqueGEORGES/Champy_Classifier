"""DAG de test pour valider l'installation Airflow Champy.

Trois tâches simples qui vérifient pas-à-pas que l'environnement est
correctement câblé :

1. ``say_hello``        : exécution Python basique
2. ``check_environment`` : accès aux variables MLflow et au système de fichiers
3. ``check_champy_volume`` : accès au volume ``/opt/champy`` monté depuis le projet

Lancement manuel uniquement (``schedule=None``), idéal pour valider sans
contraintes temporelles. À supprimer (ou désactiver) une fois la stack
en production.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "Dominique GEORGES",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="hello_world",
    description="DAG de test pour valider l'installation Airflow Champy",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 5, 14),
    schedule=None,
    catchup=False,
    tags=["champy", "test", "smoke"],
)
def hello_world_dag() -> None:
    """Pipeline de validation minimaliste."""

    @task
    def say_hello() -> str:
        """Affiche un message de bienvenue et retourne la date courante."""
        now = datetime.now().isoformat()
        message = f"Bonjour depuis Airflow Champy ! Heure actuelle : {now}"
        print(message)
        return now

    @task
    def check_environment(execution_time: str) -> dict[str, str]:
        """Vérifie que les variables d'environnement Champy sont accessibles."""
        keys = [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
            "DAGSHUB_USER",
            "DAGSHUB_TOKEN",
        ]
        report = {}
        for key in keys:
            value = os.environ.get(key)
            if value is None:
                report[key] = "ABSENTE"
            elif "PASSWORD" in key or "TOKEN" in key:
                report[key] = f"présente ({len(value)} caractères, masquée)"
            else:
                report[key] = f"présente : {value}"

        print("Vérification des variables MLflow / DagsHub depuis Airflow")
        print(f"Heure d'exécution : {execution_time}")
        for key, status in report.items():
            print(f"  - {key} : {status}")

        missing = [k for k, v in report.items() if v == "ABSENTE"]
        if missing:
            raise ValueError(
                f"Variables d'environnement manquantes : {', '.join(missing)}. "
                "Vérifier que docker-compose.airflow.yml les transmet bien."
            )
        return report

    @task
    def check_champy_volume() -> dict[str, bool]:
        """Vérifie que les volumes Champy sont bien montés dans le conteneur."""
        paths_to_check = {
            "/opt/champy/scripts": "scripts (read-only)",
            "/opt/champy/scripts/generate_analysis.py": "script generate_analysis",
            "/opt/champy/configs": "configs (read-only)",
            "/opt/champy/docs": "docs (read-write pour les analyses)",
            "/opt/champy/.env": ".env du projet (read-only)",
        }

        results = {}
        for path_str, label in paths_to_check.items():
            exists = Path(path_str).exists()
            results[path_str] = exists
            status = "✓ trouvé" if exists else "✗ MANQUANT"
            print(f"  {status} : {path_str} ({label})")

        missing = [p for p, ok in results.items() if not ok]
        if missing:
            raise FileNotFoundError(
                f"Volumes attendus introuvables : {', '.join(missing)}. "
                "Vérifier les sections 'volumes' de docker-compose.airflow.yml."
            )

        print("Tous les volumes Champy sont correctement montés.")
        return results

    now_str = say_hello()
    env_report = check_environment(now_str)
    volume_report = check_champy_volume()

    # Dépendance explicite : volumes vérifiés après les variables d'env
    env_report >> volume_report  # type: ignore[operator]


hello_world_dag()
