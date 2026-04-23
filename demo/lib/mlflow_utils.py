"""Helpers partagés pour l'acces a MLflow depuis les pages Streamlit.

Fournit des fonctions cachées pour récupérer les runs, métriques,
artefacts et versions du modèle depuis le serveur MLflow DagsHub.
Principe : zéro hardcoded, tout est lu dynamiquement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def _get_tracking_uri() -> str:
    """Retourne l'URI de tracking MLflow depuis la config.

    Returns:
        URI du serveur MLflow.
    """
    try:
        from src.config import get_mlflow_settings

        return get_mlflow_settings().mlflow_tracking_uri
    except Exception:
        return "https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow"


@st.cache_data(ttl=120)
def search_runs(
    max_results: int = 100,
    order_by: str = "start_time DESC",
) -> list[dict[str, Any]]:
    """Recherche les runs MLflow et retourne les résultats.

    Args:
        max_results: Nombre maximum de runs à retourner.
        order_by: Critère de tri (ex: 'metrics.val_acc DESC').

    Returns:
        Liste de dictionnaires avec les infos de chaque run.
    """
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    runs = mlflow.search_runs(
        order_by=[order_by],
        max_results=max_results,
    )
    return runs.to_dict("records") if not runs.empty else []


@st.cache_data(ttl=120)
def get_best_run(metric: str = "val_acc") -> dict[str, Any] | None:
    """Récupère le meilleur run selon une métrique donnée.

    Args:
        metric: Nom de la métrique à maximiser.

    Returns:
        Dictionnaire du meilleur run, ou None si aucun run.
    """
    runs = search_runs(max_results=1, order_by=f"metrics.{metric} DESC")
    return runs[0] if runs else None


@st.cache_data(ttl=300)
def get_run_metrics(run_id: str) -> dict[str, Any]:
    """Récupère toutes les métriques d'un run spécifique.

    Args:
        run_id: Identifiant du run MLflow.

    Returns:
        Dictionnaire {nom_metrique: valeur}.
    """
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    run = mlflow.get_run(run_id)
    return dict(run.data.metrics)


@st.cache_data(ttl=300)
def get_run_params(run_id: str) -> dict[str, str]:
    """Récupère les hyperparamètres d'un run spécifique.

    Args:
        run_id: Identifiant du run MLflow.

    Returns:
        Dictionnaire {nom_param: valeur}.
    """
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    run = mlflow.get_run(run_id)
    return dict(run.data.params)


@st.cache_data(ttl=300)
def get_metric_history(run_id: str, metric_name: str) -> list[dict[str, Any]]:
    """Récupère l'historique d'une métrique par epoch.

    Args:
        run_id: Identifiant du run MLflow.
        metric_name: Nom de la métrique (ex: 'val_loss').

    Returns:
        Liste de dictionnaires avec 'step' et 'value'.
    """
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric_name)
    return [{"step": m.step, "value": m.value} for m in history]


def load_local_metrics() -> dict[str, Any] | None:
    """Charge les métriques depuis le fichier JSON local (fallback).

    Utilise models/artifacts/metrics.json si MLflow n'est pas disponible.

    Returns:
        Dictionnaire de métriques ou None si le fichier n'existe pas.
    """
    import json

    metrics_path = (
        Path(__file__).resolve().parent.parent.parent / "models" / "artifacts" / "metrics.json"
    )
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]
