"""Metriques Prometheus pour l'API FastAPI.

Expose des compteurs et histogrammes pour le monitoring :
latence d'inference, nombre de predictions, distribution
des classes predites, et confiance moyenne.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Summary

# -- Compteur de predictions par espece --
PREDICTIONS_TOTAL = Counter(
    "champy_predictions_total",
    "Nombre total de predictions par espece",
    ["species"],
)

# -- Histogramme de latence d'inference --
PREDICTION_LATENCY = Histogram(
    "champy_prediction_latency_seconds",
    "Latence de l'inference (secondes)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# -- Confiance moyenne des predictions --
PREDICTION_CONFIDENCE = Summary(
    "champy_prediction_confidence",
    "Confiance de la prediction top-1",
)

# -- Compteur d'erreurs HTTP --
HTTP_ERRORS = Counter(
    "champy_http_errors_total",
    "Nombre total d'erreurs HTTP par code",
    ["status_code"],
)

# -- Compteur de requetes par endpoint --
REQUESTS_TOTAL = Counter(
    "champy_requests_total",
    "Nombre total de requetes par endpoint et methode",
    ["method", "endpoint"],
)
