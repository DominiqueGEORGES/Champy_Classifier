"""Helpers partagés pour les appels à l'API FastAPI et Prometheus.

Fournit des fonctions pour interroger les endpoints de l'API
(predict, health, metrics) et les métriques Prometheus.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

DEFAULT_API_URL = os.environ.get("CHAMPY_API_URL", "http://localhost:8010")
DEFAULT_PROMETHEUS_URL = os.environ.get("CHAMPY_PROMETHEUS_URL", "http://localhost:9090")
DEFAULT_GRAFANA_URL = os.environ.get("CHAMPY_GRAFANA_URL", "http://localhost:3010")


def get_api_url() -> str:
    """Retourne l'URL de l'API depuis la variable d'environnement ou le défaut.

    Priorité : ``CHAMPY_API_URL`` (env) > ``http://localhost:8010`` (défaut).
    Le NUC3 étant un hôte partagé, le port 8010 a été choisi avec un offset
    +10 par rapport au standard 8000 (occupé par un autre projet). Voir
    PLAYBOOK.md pour le rationnel de mapping des ports.

    Returns:
        URL de base de l'API FastAPI.
    """
    return DEFAULT_API_URL


def get_prometheus_url() -> str:
    """Retourne l'URL de Prometheus depuis l'environnement ou le défaut.

    Priorité : ``CHAMPY_PROMETHEUS_URL`` (env) > ``http://localhost:9090``
    (défaut). Voir PLAYBOOK.md pour le mapping des ports sur hôte partagé.

    Returns:
        URL de base de Prometheus.
    """
    return DEFAULT_PROMETHEUS_URL


def get_grafana_url() -> str:
    """Retourne l'URL de Grafana depuis l'environnement ou le défaut.

    Priorité : ``CHAMPY_GRAFANA_URL`` (env) > ``http://localhost:3010``
    (défaut). Le port 3000 étant occupé par un autre projet sur le NUC3,
    Grafana est mappé sur 3010 (offset +10). Voir PLAYBOOK.md.

    Returns:
        URL de base de Grafana.
    """
    return DEFAULT_GRAFANA_URL


@st.cache_data(ttl=30)
def get_health() -> dict[str, Any] | None:
    """Interroge l'endpoint /health de l'API.

    Returns:
        Dictionnaire de la réponse ou None si l'API est indisponible.
    """
    import httpx

    try:
        response = httpx.get(f"{get_api_url()}/health", timeout=5)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


def predict_image(image_bytes: bytes, top_n: int = 5) -> dict[str, Any] | None:
    """Envoie une image à l'endpoint /predict de l'API.

    Args:
        image_bytes: Contenu brut de l'image (JPEG/PNG).
        top_n: Nombre de prédictions demandées.

    Returns:
        Dictionnaire de la réponse ou None si l'API est indisponible.
    """
    import httpx

    try:
        response = httpx.post(
            f"{get_api_url()}/predict",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            params={"top_n": top_n},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_model_info() -> dict[str, Any] | None:
    """Interroge l'endpoint /model/info de l'API.

    Returns:
        Dictionnaire des métadonnées du modèle ou None.
    """
    import httpx

    try:
        response = httpx.get(f"{get_api_url()}/model/info", timeout=5)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_data(ttl=15)
def get_prometheus_metrics() -> str | None:
    """Récupère les métriques brutes depuis l'endpoint /metrics de l'API.

    Returns:
        Texte Prometheus ou None si indisponible.
    """
    import httpx

    try:
        response = httpx.get(f"{get_api_url()}/metrics", timeout=5)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


@st.cache_data(ttl=30)
def query_prometheus(query: str) -> list[dict[str, Any]]:
    """Exécute une requête PromQL sur le serveur Prometheus.

    Args:
        query: Expression PromQL (ex: 'champy_predictions_total').

    Returns:
        Liste de résultats Prometheus ou liste vide si indisponible.
    """
    import httpx

    try:
        response = httpx.get(
            f"{get_prometheus_url()}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("result", [])  # type: ignore[no-any-return]
    except Exception:
        return []
