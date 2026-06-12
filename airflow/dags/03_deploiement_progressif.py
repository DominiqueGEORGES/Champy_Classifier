"""DAG Airflow : deploiement canari (canary) du modele candidat.

Declenche par le DAG 02 a l'issue d'un reentrainement REUSSI (dont le F1 sur le
jeu de test a franchi le seuil ; cette porte de qualite vit dans le 02, pas ici).
Ce DAG bascule 10% du trafic vers le candidat (`api_v2`), laisse 90% sur le
champion (`api`), puis alerte un humain sur Telegram pour qu'il decide de la suite.

La decision de PROMOUVOIR (100%) ou de REVENIR EN ARRIERE (0%) n'est pas
automatique : elle se prend a la main en declenchant le DAG 05 (full new model)
ou le DAG 06 (restore old model). Le human-in-the-loop est assume : laisser une
machine promouvoir seule sur quelques minutes de trafic serait imprudent.

----------------------------------------------------------------------------
SUITE CONCUE, NON BRANCHEE (V2) : promotion automatique apres observation.
----------------------------------------------------------------------------
Automatiser la decision suppose trois choses, volontairement laissees de cote
pour cette V1 :

1. Une fenetre d'observation entre la bascule a 10% et la mesure. Comparer
   immediatement reviendrait a lire des series Prometheus vides (le candidat
   vient de recevoir ses premieres requetes) : taux d'erreur a 0, donc promotion
   a tous les coups. Il faut une attente de 15 a 30 min (TimeDeltaSensor).

2. Des metriques par version. L'API expose aujourd'hui
   `bentoml_service_request_total{http_response_code=...}` sans label de version.
   Comparer champion et candidat demande soit un label `version` ajoute au
   middleware, soit deux jobs Prometheus distincts.

3. Une porte metier, deja couverte en amont : un modele peut repondre 200 avec
   de mauvaises predictions. La qualite (F1 sur le jeu de test) se juge dans le
   02 ; le 03 ne valide que la sante runtime (5xx, latence).

Le squelette de cette logique est conserve en commentaire en bas de fichier,
comme trace de conception.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

# Les imports lourds (docker, requests) vivent dans le module champy_canary et
# sont charges a l'interieur des taches : le DAG s'affiche dans l'interface meme
# si ces bibliotheques manquent ; l'erreur eventuelle n'arrive qu'a l'execution.

PART_CANARI = 10  # pourcentage de trafic envoye au candidat pendant le canari


@dag(
    dag_id="deploiement_progressif",
    dag_display_name="03_deploiement_progressif",
    schedule=None,  # declenche par le 02, jamais sur planning
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "deploiement", "canary"],
)
def deploiement_progressif():

    @task
    def regler_trafic_canari() -> None:
        """Bascule PART_CANARI % du trafic vers le candidat, le reste au champion."""
        from champy_canary import appliquer_poids

        appliquer_poids(PART_CANARI)

    @task
    def alerter_humain() -> None:
        """Previent l'equipe que le canari est actif et attend une decision."""
        from champy_canary import alerter_telegram

        alerter_telegram(
            f"Canari actif : le candidat reçoit {PART_CANARI}% du trafic.\n"
            "Le modèle a passé la porte de qualité du réentraînement.\n"
            "Décision attendue : promouvoir (DAG 05) ou revenir en arrière (DAG 06)."
        )

    regler_trafic_canari() >> alerter_humain()


deploiement_progressif()


# ===========================================================================
# V2 (concue, non branchee) : decision automatique apres fenetre d'observation.
# Conservee en trace de conception ; voir le docstring du module pour le detail
# des trois prerequis manquants (fenetre, metriques par version, seuils).
#
#   from datetime import timedelta
#   from airflow.sensors.time_delta import TimeDeltaSensor
#   from champy_canary import CHAMPION, CHALLENGER
#
#   @task
#   def comparer() -> bool:
#       # taux_5xx() interroge Prometheus sur les vraies metriques BentoML,
#       # filtrees par version, sur une fenetre posterieure a l'observation.
#       candidat = taux_5xx(CHALLENGER)
#       champion = taux_5xx(CHAMPION)
#       # le candidat tient s'il ne degrade pas le taux d'erreur de plus de 20%
#       return candidat <= max(champion * 1.2, 0.01)
#
#   @task.branch
#   def decider(candidat_ok: bool) -> str:
#       # promouvoir  -> declenche la logique du DAG 05 (candidat a 100%)
#       # marche_arriere -> declenche la logique du DAG 06 (champion a 100%)
#       return "promouvoir" if candidat_ok else "marche_arriere"
#
#   observer = TimeDeltaSensor(task_id="observer", delta=timedelta(minutes=20))
#   verdict = comparer()
#   regler_trafic_canari() >> observer >> verdict
#   decider(verdict) >> [promouvoir(), marche_arriere()]
# ===========================================================================
