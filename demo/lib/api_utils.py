"""Helpers partagés pour les appels à l'API de serving et à Prometheus.

Fournit des fonctions pour interroger les endpoints du service de serving
(predict, health, model/info, metrics) et les métriques Prometheus.

Note migration FastAPI -> BentoML (2026-05-22) :
Le service de serving expose désormais BentoML (port host 8010, interne 8000).
Les endpoints applicatifs `/health` et `/model/info` sont en **POST** chez
BentoML 1.4 par défaut (les `@bentoml.api` sont POST sauf override). On utilise
donc httpx.post() pour ces endpoints. Le endpoint `/predict` reste en POST
(envoi d'une image), `/metrics` reste en GET (natif Prometheus).
Pour les probes Kubernetes, BentoML expose `/healthz` et `/readyz` en GET
nativement -> utilisés pour le healthcheck Docker uniquement.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from loguru import logger

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
        URL de base de l'API.
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

    BentoML expose `/health` en POST (méthode par défaut des `@bentoml.api`).
    On envoie donc un POST sans payload.

    Returns:
        Dictionnaire de la réponse ou None si l'API est indisponible.
    """
    import httpx

    try:
        response = httpx.post(f"{get_api_url()}/health", timeout=5)
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
            # files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            # params={"top_n": top_n},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"top_n": str(top_n)},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


def explain_image(image_bytes: bytes, target_class_id: int = -1) -> dict[str, Any] | None:
    """Appelle POST /explain pour obtenir une visualisation Grad-CAM.

    Args:
        image_bytes: Contenu brut de l'image.
        target_class_id: Classe pour laquelle expliquer la décision.
            Si -1 (défaut), utilise la classe prédite top-1.

    Returns:
        Dict avec ``target_class_id``, ``target_class_name``, ``original_b64``,
        ``heatmap_b64``, ``overlay_b64``, ou None en cas d'échec.
    """
    try:
        response = httpx.post(
            f"{get_api_url()}/explain",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"target_class_id": str(target_class_id)},
            timeout=30,  # Grad-CAM peut prendre 2-5s (init paresseuse au 1er appel)
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        import traceback

        print(f"Echec /explain : {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None


def get_recent_predictions(hours: int = 24, limit: int = 1000) -> list[dict[str, Any]] | None:
    """Appelle POST /predictions/recent pour récupérer l'historique des prédictions.

    Args:
        hours: Fenêtre temporelle en heures (défaut 24).
        limit: Nombre maximum de prédictions à retourner (défaut 1000).

    Returns:
        Liste de dicts ``{timestamp, predicted_class, confidence, ...}``, ou
        None en cas d'échec HTTP.
    """
    try:
        response = httpx.post(
            f"{get_api_url()}/predictions/recent",
            json={"hours": hours, "limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error(f"Echec /predictions/recent : {exc}")
        return None


def get_model_registry() -> dict[str, Any] | None:
    """Appelle POST /model/registry pour l'inventaire des modèles.

    Returns:
        Dict avec ``models``, ``checkpoint``, ``onnx_validation``,
        ``num_classes``, ``class_names``, ou None en cas d'échec.
    """
    try:
        response = httpx.post(f"{get_api_url()}/model/registry", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error(f"Echec /model/registry : {exc}")
        return None


@st.cache_data(ttl=30)
def get_model_info() -> dict[str, Any] | None:
    """Interroge l'endpoint /model/info de l'API.

    BentoML expose `/model/info` en POST (méthode par défaut). On envoie un
    POST sans payload.

    Returns:
        Dictionnaire des métadonnées du modèle ou None.
    """
    import httpx

    try:
        response = httpx.post(f"{get_api_url()}/model/info", timeout=5)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_data(ttl=15)
def get_prometheus_metrics() -> str | None:
    """Récupère les métriques brutes depuis l'endpoint /metrics de l'API.

    BentoML expose `/metrics` en GET (endpoint natif Prometheus, inchangé
    par rapport à FastAPI).

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
