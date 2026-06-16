"""DAG watchdog qualité Champy : déclenche le réentraînement sur dégradation.

Interroge Prometheus à intervalle régulier. Si la confiance moyenne des
prédictions récentes franchit le seuil de dégradation ET qu'aucun réentraînement
n'a eu lieu récemment, déclenche le DAG ``champy_reentrainement``.

C'est le pont manquant entre la supervision (Prometheus) et l'action (Airflow) :
Airflow ne lit pas Prometheus tout seul, ce DAG va chercher l'information.

Deux garde-fous contre les boucles de réentraînement :

1. Fenêtre glissante : sans trafic récent, ``rate(...)`` vaut NaN, donc aucun
   signal, donc aucun déclenchement. On ne réentraîne pas un modèle qui ne sert
   pas, et la métrique ne peut de toute façon pas se rafraîchir sans prédictions.
2. Cooldown : aucun nouveau déclenchement tant qu'un réentraînement date de moins
   de ``COOLDOWN_HOURS``, le temps que le nouveau modèle accumule du trafic et
   prouve sa valeur.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.models import DagRun
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils import timezone
from airflow.utils.session import create_session

# Airflow 3.x : si l'import du TriggerDagRunOperator echoue, utiliser
#   from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

logger = logging.getLogger(__name__)

# =====================================================================
# A ADAPTER a ta metrique reelle (ou externaliser en Variables Airflow)
# =====================================================================

# URL de Prometheus joignable DEPUIS le conteneur Airflow (prefixe de route inclus).
PROMETHEUS_URL = "http://prometheus:9090/prometheus"

# Confiance moyenne des predictions sur une fenetre glissante (sum/count via rate).
# La fenetre est volontaire : sans trafic recent, le resultat est NaN, traite comme
# "pas de signal", ce qui evite de boucler sur un modele qui ne sert pas.
# Seul signal de qualite disponible en continu : ni accuracy ni score de drift ne
# sont exposes en gauge (le drift Evidently est calcule a la demande).
#
# POUR RE-TESTER LE CABLAGE hors charge : remplacer temporairement par le ratio
# global "champy_prediction_confidence_sum / champy_prediction_confidence_count"
# (jamais NaN) et mettre DEGRADED_THRESHOLD au-dessus de la confiance courante.
QUALITY_QUERY = (
    "rate(champy_prediction_confidence_sum[6h]) / rate(champy_prediction_confidence_count[6h])"
)

# Degrade si la confiance moyenne tombe SOUS le seuil. Confiance observee ~0,72 ;
# 0,60 laisse une vraie marge avant de declencher.
DEGRADED_THRESHOLD = 0.60
# DEGRADED_THRESHOLD = 0.99

# Delai minimal entre deux reentrainements, pour laisser le nouveau modele faire
# ses preuves avant de rejuger (anti-boucle).
COOLDOWN_HOURS = 12
# COOLDOWN_HOURS = 0

# DAG d'action a declencher et mode d'entrainement souhaite.
TARGET_DAG_ID = "champy_reentrainement"
TRAINING_COMMAND = "train --config configs/training/convnext.yaml"


# =====================================================================
# Lecture de la qualite (Prometheus)
# =====================================================================


def _query_prometheus(query: str) -> float | None:
    """Interroge l'API instantanée de Prometheus et renvoie la valeur scalaire.

    Args:
        query: Requête PromQL renvoyant une série à valeur unique.

    Returns:
        La valeur numérique, ou None si Prometheus est indisponible, si la réponse
        est vide, mal formée, ou vaut NaN (0/0, absence de trafic).
    """
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Prometheus injoignable ou reponse invalide : %s", exc)
        return None

    results = payload.get("data", {}).get("result", [])
    if not results:
        logger.warning("Requete Prometheus sans resultat : %s", query)
        return None
    try:
        # result[0]["value"] = [timestamp, "valeur"]
        value = float(results[0]["value"][1])
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Format de reponse Prometheus inattendu : %s", exc)
        return None

    if math.isnan(value):
        # 0/0 : aucune prediction dans la fenetre, donc aucun signal exploitable.
        logger.warning("Valeur NaN (pas de trafic recent) pour : %s", query)
        return None
    return value


def _is_degraded() -> bool:
    """Détermine si la qualité du modèle est dégradée selon Prometheus.

    En cas d'indisponibilité de Prometheus ou d'absence de trafic, renvoie False :
    on ne déclenche pas un réentraînement sur une panne de supervision ni sur un
    modèle qui ne sert pas.

    Returns:
        True si la qualité mesurée est dégradée.
    """
    value = _query_prometheus(QUALITY_QUERY)
    if value is None:
        return False

    degraded = value <= DEGRADED_THRESHOLD
    logger.info("Qualite mesuree=%s seuil=%s degrade=%s", value, DEGRADED_THRESHOLD, degraded)
    return degraded


# =====================================================================
# Garde-fou anti-boucle (cooldown)
# =====================================================================


def _last_retraining_start() -> datetime | None:
    """Renvoie l'horodatage de démarrage du dernier réentraînement, ou None.

    Lit le dernier ``DagRun`` du DAG cible directement dans la base de métadonnées
    Airflow (pas d'appel API ni d'authentification).

    Returns:
        Le ``start_date`` du run le plus récent ayant démarré, ou None s'il
        n'existe aucun run.
    """
    with create_session() as session:
        run = (
            session.query(DagRun)
            .filter(DagRun.dag_id == TARGET_DAG_ID, DagRun.start_date.isnot(None))
            .order_by(DagRun.start_date.desc())
            .first()
        )
        return run.start_date if run else None


def _in_cooldown() -> bool:
    """Indique si un réentraînement a eu lieu trop récemment pour en relancer un.

    Returns:
        True si le dernier réentraînement date de moins de ``COOLDOWN_HOURS``.
    """
    last = _last_retraining_start()
    if last is None:
        return False

    elapsed = timezone.utcnow() - last
    in_cooldown = elapsed < timedelta(hours=COOLDOWN_HOURS)
    logger.info(
        "Dernier reentrainement il y a %s (cooldown=%sh, actif=%s)",
        elapsed,
        COOLDOWN_HOURS,
        in_cooldown,
    )
    return in_cooldown


def _should_retrain() -> bool:
    """Décide s'il faut déclencher un réentraînement.

    Combine les deux garde-fous : on ne déclenche que si la qualité est dégradée
    ET qu'aucun réentraînement n'est en période de cooldown.

    Returns:
        True si un réentraînement doit être déclenché.
    """
    if _in_cooldown():
        logger.info("Cooldown actif : pas de declenchement.")
        return False
    return _is_degraded()


def _alerter_reentrainement() -> None:
    """Previent sur Discord qu'un reentrainement automatique vient d'etre declenche."""
    from champy_canary import alerter_discord

    alerter_discord(
        f"Qualité dégradée : la confiance moyenne est passée sous {DEGRADED_THRESHOLD}. "
        "Un réentraînement automatique a été déclenché."
    )


# =====================================================================
# Definition du DAG
# =====================================================================

with DAG(
    dag_id="champy_qualite_watchdog",
    dag_display_name="04_champy_qualite_watchdog",
    description="Surveille Prometheus et declenche le reentrainement si la qualite chute.",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "mlops", "monitoring"],
) as dag:
    verifier_qualite = ShortCircuitOperator(
        task_id="verifier_qualite",
        python_callable=_should_retrain,
    )

    declencher_reentrainement = TriggerDagRunOperator(
        task_id="declencher_reentrainement",
        trigger_dag_id=TARGET_DAG_ID,
        conf={"training_command": TRAINING_COMMAND},
        wait_for_completion=False,
        reset_dag_run=True,
    )

    notifier = PythonOperator(
        task_id="notifier_discord",
        python_callable=_alerter_reentrainement,
    )

    verifier_qualite >> declencher_reentrainement >> notifier
