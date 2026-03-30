"""Helpers partages pour les appels a l'API FastAPI et Prometheus.

Fournit des fonctions pour interroger les endpoints de l'API
(predict, health, metrics) et les metriques Prometheus.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"


def _get_api_url() -> str:
    """Retourne l'URL de l'API depuis la config ou le defaut.

    Returns:
        URL de base de l'API FastAPI.
    """
    try:
        from src.config import get_serving_settings

        settings = get_serving_settings()
        return f"http://{settings.api_host}:{settings.api_port}"
    except Exception:
        return DEFAULT_API_URL


@st.cache_data(ttl=30)
def get_health() -> dict[str, Any] | None:
    """Interroge l'endpoint /health de l'API.

    Returns:
        Dictionnaire de la reponse ou None si l'API est indisponible.
    """
    import httpx

    try:
        response = httpx.get(f"{_get_api_url()}/health", timeout=5)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


def predict_image(image_bytes: bytes, top_n: int = 5) -> dict[str, Any] | None:
    """Envoie une image a l'endpoint /predict de l'API.

    Args:
        image_bytes: Contenu brut de l'image (JPEG/PNG).
        top_n: Nombre de predictions demandees.

    Returns:
        Dictionnaire de la reponse ou None si l'API est indisponible.
    """
    import httpx

    try:
        response = httpx.post(
            f"{_get_api_url()}/predict",
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
        Dictionnaire des metadonnees du modele ou None.
    """
    import httpx

    try:
        response = httpx.get(f"{_get_api_url()}/model/info", timeout=5)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_data(ttl=15)
def get_prometheus_metrics() -> str | None:
    """Recupere les metriques brutes depuis l'endpoint /metrics de l'API.

    Returns:
        Texte Prometheus ou None si indisponible.
    """
    import httpx

    try:
        response = httpx.get(f"{_get_api_url()}/metrics", timeout=5)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


@st.cache_data(ttl=30)
def query_prometheus(query: str) -> list[dict[str, Any]]:
    """Execute une requete PromQL sur le serveur Prometheus.

    Args:
        query: Expression PromQL (ex: 'champy_predictions_total').

    Returns:
        Liste de resultats Prometheus ou liste vide si indisponible.
    """
    import httpx

    try:
        response = httpx.get(
            f"{DEFAULT_PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("result", [])  # type: ignore[no-any-return]
    except Exception:
        return []
