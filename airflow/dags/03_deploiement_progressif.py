"""DAG Airflow : deploiement progressif (canary) pilote.

Met le candidat en service a faible part (10%), compare les deux versions via
Prometheus, puis promeut le candidat a 100% ou fait marche arriere vers l'ancien.
Tout est decide sans intervention manuelle.

MLflow : aucune URL n'est codee en dur. Le client lit MLFLOW_TRACKING_URI dans
l'environnement du conteneur Airflow, qui doit pointer vers le MLflow LOCAL du projet
(et non DagsHub). Basculer local/distant se fait donc a un seul endroit.

Note : les imports lourds (requests, docker, mlflow) sont faits a l'interieur des
taches. Ainsi le DAG se charge et affiche son graphe dans l'interface Airflow meme si
ces bibliotheques ne sont pas installees ; l'eventuelle erreur n'arrive qu'a l'execution.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

# --- Reperes de la stack (a aligner sur les noms reels du projet) ---
NGINX_CONF = "/etc/nginx/conf.d/upstream_champy.conf"  # fichier de conf monte cote NGINX
NGINX_CONTAINER = "champy_nginx"  # conteneur a recharger
PROM_URL = "http://champy_prometheus:9090/api/v1/query"  # API de Prometheus
MODELE = "champy-classifier"  # nom du modele dans le registre


def _ligne_server(nom: str, poids: int) -> str:
    # NGINX refuse un poids nul : on coupe alors le serveur avec la directive 'down'
    if poids <= 0:
        return f"    server {nom}:8000 down;"
    return f"    server {nom}:8000 weight={poids};"


def _appliquer_poids(part_candidat: int) -> None:
    """Reecrit les poids NGINX (candidat = part_candidat %) puis recharge a chaud."""
    import docker

    bloc = (
        "upstream champy_api {\n"
        f"{_ligne_server('api_v1', 100 - part_candidat)}\n"
        f"{_ligne_server('api_v2', part_candidat)}\n"
        "}\n"
    )
    Path(NGINX_CONF).write_text(bloc, encoding="utf-8")
    # reload gracieux : ne coupe pas les requetes en cours
    docker.from_env().containers.get(NGINX_CONTAINER).exec_run("nginx -s reload")


def _taux_erreur(version: str) -> float:
    """Taux de reponses en erreur (5xx) d'une version, lu dans Prometheus."""
    import requests

    promql = (
        f'sum(rate(api_requests_total{{version="{version}",status=~"5.."}}[5m]))'
        f' / sum(rate(api_requests_total{{version="{version}"}}[5m]))'
    )
    reponse = requests.get(PROM_URL, params={"query": promql}, timeout=10).json()
    points = reponse["data"]["result"]
    return float(points[0]["value"][1]) if points else 0.0


@dag(
    dag_id="deploiement_progressif",
    dag_display_name="03_deploiement_progressif",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "deploiement"],
)
def deploiement_progressif():

    @task
    def regler_trafic_10() -> None:
        _appliquer_poids(10)  # candidat a 10%, ancien a 90%

    @task
    def comparer() -> bool:
        candidat = _taux_erreur("v2")
        ancien = _taux_erreur("v1")
        # le candidat tient s'il ne degrade pas le taux d'erreur de plus de 20%
        return candidat <= max(ancien * 1.2, 0.01)

    @task.branch
    def decider(candidat_ok: bool) -> str:
        return "promouvoir" if candidat_ok else "marche_arriere"

    @task
    def promouvoir() -> None:
        from mlflow.tracking import MlflowClient

        _appliquer_poids(100)  # le candidat prend tout le trafic, l'ancien est coupe
        # MlflowClient() sans argument lit MLFLOW_TRACKING_URI dans l'environnement :
        # il pointe donc vers le MLflow local configure au niveau du conteneur Airflow.
        client = MlflowClient()
        candidate = client.get_latest_versions(MODELE, stages=["Staging"])[0]
        client.transition_model_version_stage(MODELE, candidate.version, "Production")

    @task
    def marche_arriere() -> None:
        _appliquer_poids(0)  # candidat coupe : tout le trafic repart sur l'ancien (v1)

    verdict = comparer()
    regler_trafic_10() >> verdict
    decider(verdict) >> [promouvoir(), marche_arriere()]


deploiement_progressif()
