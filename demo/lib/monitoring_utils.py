"""Helpers Streamlit pour la page de monitoring.

Charge les seuils d'alerting depuis ``configs/monitoring/thresholds.yml``,
agrege les metriques cles depuis Prometheus (RPS, latence p50/p95/p99,
taux d'erreur, total predictions), et fournit la logique de decision
``OK / warning / critical`` pour chaque metrique.

Aucune valeur (URL, seuil, palette) n'est ecrite en dur dans la page :
elles sont toutes resolues a chaque appel pour rester coherent avec la
contrainte ``zero hardcoded`` du Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from demo.lib.api_utils import query_prometheus

# Le fichier de seuils vit a cote de ``configs/`` pour rester avec le
# reste de la configuration runtime du projet (Prometheus scrape config,
# Grafana provisioning, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_PATH = _PROJECT_ROOT / "configs" / "monitoring" / "thresholds.yml"

# Niveaux d'alerte ordonnes du moins grave au plus grave : utile pour
# l'affichage (couleur) et pour determiner l'etat global du systeme.
LEVEL_OK = "ok"
LEVEL_WARNING = "warning"
LEVEL_CRITICAL = "critical"
LEVEL_UNKNOWN = "unknown"

LEVEL_COLORS = {
    LEVEL_OK: "#1f8a3e",
    LEVEL_WARNING: "#d4a017",
    LEVEL_CRITICAL: "#c0392b",
    LEVEL_UNKNOWN: "#7f8c8d",
}

LEVEL_LABELS = {
    LEVEL_OK: "OK",
    LEVEL_WARNING: "WARNING",
    LEVEL_CRITICAL: "CRITICAL",
    LEVEL_UNKNOWN: "INDISPONIBLE",
}


@dataclass(frozen=True)
class MetricStatus:
    """Resultat de l'evaluation d'une metrique contre ses seuils.

    Attributes:
        name: Nom lisible de la metrique (pour le label affiche).
        value: Valeur courante (None si la metrique est indisponible).
        unit: Unite affichee (ex: ``ms``, ``%``).
        level: ``ok`` / ``warning`` / ``critical`` / ``unknown``.
        message: Message court explicitant le niveau choisi.
    """

    name: str
    value: float | None
    unit: str
    level: str
    message: str


@st.cache_data(ttl=60)
def load_thresholds() -> dict[str, Any]:
    """Charge les seuils depuis ``configs/monitoring/thresholds.yml``.

    Le cache TTL=60s evite de relire le YAML a chaque rendu Streamlit
    sans rendre le rechargement instantane (un edit du YAML est visible
    apres au plus 1 minute).

    Returns:
        Dictionnaire des seuils (vide si le fichier est absent).
    """
    if not THRESHOLDS_PATH.exists():
        return {}
    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data  # type: ignore[no-any-return]


def evaluate_metric(
    name: str,
    value: float | None,
    config: dict[str, Any],
    unit: str = "",
    fmt: str = "{:.2f}",
) -> MetricStatus:
    """Evalue une valeur courante contre la configuration de seuils.

    La direction de la metrique determine si on compare a une borne basse
    ou haute :

    - ``higher_is_worse`` : on compare a ``warning_above`` / ``critical_above``.
    - ``lower_is_worse`` : on compare a ``warning_below`` / ``critical_below``.

    Args:
        name: Libelle de la metrique pour l'affichage.
        value: Valeur courante, ``None`` si indisponible.
        config: Section yaml correspondante (cf. ``thresholds.yml``).
        unit: Unite affichee dans le message (ex: ``ms``, ``%``).
        fmt: Format Python pour la valeur (ex: ``"{:.0f}"``).

    Returns:
        Statut evalue.
    """
    if value is None or not config:
        return MetricStatus(
            name=name,
            value=None,
            unit=unit,
            level=LEVEL_UNKNOWN,
            message="Indisponible (Prometheus hors ligne ou pas de donnees).",
        )

    direction = config.get("direction", "higher_is_worse")
    formatted = fmt.format(value)
    if direction == "higher_is_worse":
        critical = config.get("critical_above")
        warning = config.get("warning_above")
        if critical is not None and value >= critical:
            return MetricStatus(
                name=name,
                value=value,
                unit=unit,
                level=LEVEL_CRITICAL,
                message=f"{formatted}{unit} >= seuil critical {critical}{unit}",
            )
        if warning is not None and value >= warning:
            return MetricStatus(
                name=name,
                value=value,
                unit=unit,
                level=LEVEL_WARNING,
                message=f"{formatted}{unit} >= seuil warning {warning}{unit}",
            )
        return MetricStatus(
            name=name,
            value=value,
            unit=unit,
            level=LEVEL_OK,
            message=f"{formatted}{unit} sous les seuils.",
        )
    # lower_is_worse : on alerte quand la valeur tombe sous les seuils
    critical = config.get("critical_below")
    warning = config.get("warning_below")
    if critical is not None and value <= critical:
        return MetricStatus(
            name=name,
            value=value,
            unit=unit,
            level=LEVEL_CRITICAL,
            message=f"{formatted}{unit} <= seuil critical {critical}{unit}",
        )
    if warning is not None and value <= warning:
        return MetricStatus(
            name=name,
            value=value,
            unit=unit,
            level=LEVEL_WARNING,
            message=f"{formatted}{unit} <= seuil warning {warning}{unit}",
        )
    return MetricStatus(
        name=name,
        value=value,
        unit=unit,
        level=LEVEL_OK,
        message=f"{formatted}{unit} au-dessus des seuils.",
    )


def _first_value(results: list[dict[str, Any]]) -> float | None:
    """Extrait la premiere valeur d'un resultat instant Prometheus.

    Les ``histogram_quantile`` retournent NaN quand le bucket
    correspondant n'a recu aucune observation sur la fenetre - on
    propage ``None`` plutot que NaN pour declencher un affichage
    ``-`` cote UI plutot que ``NaN s``.

    Args:
        results: Liste retournee par ``query_prometheus``.

    Returns:
        Float, ou ``None`` si la liste est vide, la valeur n'est pas
        parsable, ou est NaN.
    """
    if not results:
        return None
    try:
        v = float(results[0].get("value", [0, 0])[1])
    except (TypeError, ValueError, IndexError):
        return None
    if math.isnan(v):
        return None
    return v


def fetch_live_metrics() -> dict[str, float | None]:
    """Interroge Prometheus pour les metriques cles affichees Section 1.

    Les requetes utilisent une fenetre de ``5m`` pour le rate (assez
    long pour absorber la granularite scrape=15s, assez court pour
    refleter les variations recentes) et calculent les percentiles
    standard sur ``champy_prediction_latency_seconds``.

    Returns:
        Dictionnaire ``{rps, p50, p95, p99, error_rate, total_predictions,
        confidence_avg}`` ou les valeurs sont des float ou ``None``.
    """
    rps = _first_value(query_prometheus("sum(rate(champy_requests_total[5m]))"))
    p50 = _first_value(
        query_prometheus(
            "histogram_quantile(0.50, sum(rate(champy_prediction_latency_seconds_bucket[5m])) by (le))"
        )
    )
    p95 = _first_value(
        query_prometheus(
            "histogram_quantile(0.95, sum(rate(champy_prediction_latency_seconds_bucket[5m])) by (le))"
        )
    )
    p99 = _first_value(
        query_prometheus(
            "histogram_quantile(0.99, sum(rate(champy_prediction_latency_seconds_bucket[5m])) by (le))"
        )
    )
    # Le taux d'erreur est ratio errors / requests. Si Prometheus n'a
    # jamais vu de ``champy_http_errors_total`` (cas frequent au boot,
    # aucune erreur encore), la metrique n'existe pas et la requete
    # retourne un tableau vide. On force un 0 via ``or on() vector(0)``
    # pour que le ratio soit calculable (et non None).
    error_rate = _first_value(
        query_prometheus(
            "(sum(rate(champy_http_errors_total[5m])) or on() vector(0)) / "
            "clamp_min(sum(rate(champy_requests_total[5m])), 0.001)"
        )
    )
    total = _first_value(query_prometheus("sum(champy_predictions_total)"))
    confidence_avg = _first_value(
        query_prometheus(
            "champy_prediction_confidence_sum / "
            "clamp_min(champy_prediction_confidence_count, 1)"
        )
    )
    return {
        "rps": rps,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "error_rate": error_rate,
        "total_predictions": total,
        "confidence_avg": confidence_avg,
    }


def evaluate_alerts(
    metrics: dict[str, float | None],
    thresholds: dict[str, Any],
) -> list[MetricStatus]:
    """Calcule les 3 alertes visuelles de la Section 4.

    Args:
        metrics: Dictionnaire renvoye par ``fetch_live_metrics``.
        thresholds: Dictionnaire renvoye par ``load_thresholds``.

    Returns:
        Liste de 3 ``MetricStatus`` (confidence, latence, error rate).
    """
    return [
        evaluate_metric(
            "Confiance moyenne",
            metrics["confidence_avg"],
            thresholds.get("confidence", {}),
            unit="",
            fmt="{:.1%}",
        ),
        evaluate_metric(
            "Latence p95",
            metrics["p95"],
            thresholds.get("latency_p95_seconds", {}),
            unit="s",
            fmt="{:.3f}",
        ),
        evaluate_metric(
            "Taux d'erreur",
            metrics["error_rate"],
            thresholds.get("error_rate", {}),
            unit="",
            fmt="{:.2%}",
        ),
    ]
