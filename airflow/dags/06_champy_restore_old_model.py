"""DAG Airflow : retour arriere (restore old model).

Coupe le candidat (`api_v2`) et renvoie 100% du trafic vers le champion (`api`),
le modele de production. Declenche a la main : c'est la reponse "on rejette le
candidat" a l'alerte du canari (03). Le candidat n'ayant pas ete promu, le
registre n'est pas modifie ; on se contente de le sortir du roulement.

Si le candidat avait deja ete promu par erreur, il faudrait en plus le faire
repasser de Production a Archived et restaurer l'ancien en Production ; ce cas
n'arrive pas dans le flux canari, ou le rejet precede toujours la promotion.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="restore_old_model",
    dag_display_name="06_champy_restore_old_model",
    schedule=None,  # declenche a la main apres decision humaine
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "deploiement", "rollback"],
)
def restore_old_model():

    @task
    def revenir_en_arriere() -> None:
        """Candidat coupe, champion a 100% du trafic."""
        from champy_canary import alerter_discord, appliquer_poids

        appliquer_poids(0)  # candidat coupe (down), champion seul
        alerter_discord(
            "Retour arrière : le candidat est coupé, le champion reçoit "
            "100% du trafic. Le modèle de production est inchangé."
        )

    revenir_en_arriere()


restore_old_model()
