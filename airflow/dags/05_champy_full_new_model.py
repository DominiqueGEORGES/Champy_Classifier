"""DAG Airflow : promotion du candidat (full new model).

Bascule 100% du trafic vers le candidat (`api_v2`) et le marque comme modele de
production dans le registre MLflow. Declenche a la main : c'est la reponse "on
adopte le nouveau modele" a l'alerte du canari (03). L'ancien champion est
archive dans le registre.

Suite (non branchee) : apres adoption, une rotation propre rechargerait le
nouveau modele dans le service `api` et remettrait `api_v2` en attente du
prochain candidat. Pour cette V1, on se limite a la bascule de trafic et a la
mise a jour du registre.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

MODELE = "champy-classifier"  # nom du modele dans le registre MLflow


@dag(
    dag_id="full_new_model",
    dag_display_name="05_champy_full_new_model",
    schedule=None,  # declenche a la main apres decision humaine
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "deploiement", "promotion"],
)
def full_new_model():

    @task
    def promouvoir() -> None:
        """Candidat a 100% du trafic, puis passage en Production dans le registre."""
        from champy_canary import alerter_discord, appliquer_poids
        from mlflow.tracking import MlflowClient

        # 1. tout le trafic sur le candidat
        appliquer_poids(100)

        # 2. marque le candidat comme modele de production.
        # NOTE : les stages MLflow (Staging/Production) sont deprecies depuis la
        # 2.9 ; la cible serait les alias (set_registered_model_alias). On garde
        # les stages tant que le reste du projet les utilise, pour ne pas casser
        # la coherence du registre.
        client = MlflowClient()  # lit MLFLOW_TRACKING_URI dans l'environnement
        versions = client.get_latest_versions(MODELE, stages=["Staging"])
        if not versions:
            raise RuntimeError(f"aucun candidat en Staging pour {MODELE} : rien a promouvoir")
        candidat = versions[0]
        client.transition_model_version_stage(
            MODELE,
            candidat.version,
            "Production",
            archive_existing_versions=True,  # archive l'ancien Production
        )

        alerter_discord(
            f"Modèle promu : la version {candidat.version} passe en Production "
            "et reçoit 100% du trafic."
        )

    promouvoir()


full_new_model()
