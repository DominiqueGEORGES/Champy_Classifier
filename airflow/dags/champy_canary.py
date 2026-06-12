"""Boite a outils partagee par les DAG de deploiement Champy (03, 05, 06).

Centralise la mecanique de bascule du trafic entre le modele en production
(service `api`, le champion) et le modele candidat (service `api_v2`, le
challenger), via la reecriture d'un upstream NGINX recharge a chaud.

Architecture cible (voir docker-compose, profil `canary`, et configs/nginx) :
- `api`     : instance servant le modele de production (Production dans MLflow).
- `api_v2`  : instance servant le modele candidat (Staging dans MLflow),
              demarree uniquement sous le profil Compose `canary`.
- NGINX     : un upstream nomme `champy_api` pondere les deux backends ; le
              fichier d'upstream est monte en partage entre Airflow et NGINX,
              ce qui permet a un DAG de le reecrire puis de recharger NGINX.

Aucun secret ni URL distante n'est code en dur : tout passe par l'environnement
du conteneur Airflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# --- Reperes de la stack (alignes sur les noms reels du projet) ---
NGINX_CONF = "/etc/nginx/conf.d/upstream_champy.conf"  # monte en partage Airflow <-> NGINX
NGINX_CONTAINER = "champy_nginx"
CHAMPION = "api"  # service servant le modele de production
CHALLENGER = "api_v2"  # service servant le modele candidat (profil canary)
UPSTREAM = "champy_api"  # nom de l'upstream NGINX pondere


def _ligne_server(service: str, poids: int) -> str:
    """Ligne `server` d'un backend NGINX ; 'down' si le poids est nul.

    NGINX refuse un poids de zero : pour retirer un backend du roulement on
    utilise la directive `down` plutot qu'un `weight=0`.
    """
    if poids <= 0:
        return f"    server {service}:8000 down;"
    return f"    server {service}:8000 weight={poids};"


def appliquer_poids(part_candidat: int) -> None:
    """Reecrit l'upstream NGINX (candidat = part_candidat %) puis recharge a chaud.

    part_candidat = 10  -> champion 90 / candidat 10 (canari)
    part_candidat = 100 -> candidat seul (promotion)
    part_candidat = 0   -> champion seul (retour arriere)
    """
    import docker

    bloc = (
        f"upstream {UPSTREAM} {{\n"
        f"{_ligne_server(CHAMPION, 100 - part_candidat)}\n"
        f"{_ligne_server(CHALLENGER, part_candidat)}\n"
        "}\n"
    )
    Path(NGINX_CONF).write_text(bloc, encoding="utf-8")
    log.info(
        "upstream NGINX reecrit : champion=%d%% candidat=%d%%",
        100 - part_candidat,
        part_candidat,
    )

    # reload gracieux : NGINX termine les requetes en cours avant de basculer
    client = docker.from_env()
    code, sortie = client.containers.get(NGINX_CONTAINER).exec_run("nginx -s reload")
    if code != 0:
        raise RuntimeError(f"echec du reload NGINX (code {code}) : {sortie!r}")
    log.info("NGINX recharge avec succes")


def alerter_discord(message: str) -> None:
    """Envoie un message au canal Discord du projet via webhook.

    L'URL du webhook est lue dans l'environnement (DISCORD_WEBHOOK_URL). Si elle
    manque, on journalise sans faire echouer la tache : une alerte non partie ne
    doit pas casser un deploiement.
    """
    import requests

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        log.warning("Discord non configure (DISCORD_WEBHOOK_URL absent) : alerte ignoree")
        return

    reponse = requests.post(webhook, json={"content": message}, timeout=10)
    reponse.raise_for_status()
    log.info("alerte Discord envoyee")
