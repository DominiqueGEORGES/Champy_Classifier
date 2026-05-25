"""Génération dynamique des règles d'alerte Prometheus.

Source de vérité : ``configs/alerts/thresholds.yml`` qui ne contient
que les seuils numériques et les durées de chaque règle. Le présent
module construit programmatiquement ``configs/alerts/champy_alerts.yml``
à partir de ces seuils et d'un template figé des expressions PromQL.

Utilisation :
    # En CLI
    python -m monitoring.alert_generator

    # Depuis Python (par exemple depuis la page Streamlit)
    from monitoring.alert_generator import regenerate_alerts
    regenerate_alerts()

Ce module n'utilise PAS de moteur de templates externe (Jinja2 ou
autre) pour rester sans dépendance ajoutée. Le YAML est construit via
``yaml.safe_dump`` à partir d'un dict Python, ce qui garantit l'absence
de YAML malformé.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import logging
from pathlib import Path
from typing import Any

# =====================================================================
# Imports tiers
# =====================================================================
import yaml

# =====================================================================
# Constantes
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "configs" / "alerts" / "thresholds.yml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "configs" / "alerts" / "champy_alerts.yml"

logger = logging.getLogger(__name__)

# Valeurs par défaut, utilisées si thresholds.yml manque ou pour le
# bouton "Reset" de la page Streamlit.
DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "api_down": {
        "for": "1m",
        "severity": "critical",
    },
    "high_latency_p95": {
        "threshold_seconds": 0.5,
        "for": "5m",
        "severity": "warning",
    },
    "high_latency_mean": {
        "threshold_seconds": 0.2,
        "for": "5m",
        "severity": "warning",
    },
    "low_mean_confidence": {
        "threshold": 0.5,
        "for": "10m",
        "severity": "warning",
    },
    "no_recent_predictions": {
        "for": "10m",
        "severity": "warning",
    },
}

# Métadonnées d'affichage (utilisées par la page Streamlit pour les
# labels et tooltips). Garde le module 100% YAML-only.
THRESHOLD_METADATA: dict[str, dict[str, Any]] = {
    "api_down": {
        "label": "API hors ligne (APIDown)",
        "description": "L'API ne répond plus aux scrapes Prometheus.",
        "tunable": ["for"],
    },
    "high_latency_p95": {
        "label": "Latence p95 élevée (HighLatencyP95)",
        "description": "p95 de la latence des prédictions au-delà du seuil.",
        "tunable": ["threshold_seconds", "for"],
        "threshold_range": (0.05, 5.0, 0.05),
    },
    "high_latency_mean": {
        "label": "Latence moyenne élevée (HighLatencyMean)",
        "description": "Moyenne de la latence des prédictions au-delà du seuil.",
        "tunable": ["threshold_seconds", "for"],
        "threshold_range": (0.05, 5.0, 0.05),
    },
    "low_mean_confidence": {
        "label": "Confiance moyenne basse (LowMeanConfidence)",
        "description": "Confiance moyenne en deçà du seuil, possible drift.",
        "tunable": ["threshold", "for"],
        "threshold_range": (0.1, 0.9, 0.05),
    },
    "no_recent_predictions": {
        "label": "Aucune prédiction récente (NoRecentPredictions)",
        "description": "Requêtes /predict reçues mais aucune prédiction loggée.",
        "tunable": ["for"],
    },
}


# =====================================================================
# Lecture / écriture des seuils
# =====================================================================


def load_thresholds(
    path: Path = DEFAULT_THRESHOLDS_PATH,
) -> dict[str, dict[str, Any]]:
    """Charge les seuils depuis le fichier YAML, retombe sur les défauts.

    Args:
        path: Chemin du fichier de seuils.

    Returns:
        Dictionnaire ``{alert_name: {threshold: value, for: duration, ...}}``.
    """
    if not path.exists():
        logger.warning(f"{path} introuvable, utilisation des défauts.")
        return DEFAULT_THRESHOLDS

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Fusion avec les défauts pour gérer un fichier partiel
    merged = {**DEFAULT_THRESHOLDS}
    for key, value in data.items():
        if key in merged:
            merged[key] = {**merged[key], **value}
    return merged


def save_thresholds(
    thresholds: dict[str, dict[str, Any]],
    path: Path = DEFAULT_THRESHOLDS_PATH,
) -> None:
    """Sauvegarde les seuils dans le YAML.

    Args:
        thresholds: Dict des seuils à sauver.
        path: Chemin de destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Seuils des alertes Champy.\n"
            "# Ne pas editer a la main si la page Streamlit Alertes est utilisee.\n"
            "# Genere automatiquement via monitoring.alert_generator.save_thresholds().\n\n"
        )
        yaml.safe_dump(
            thresholds,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
        )
    logger.info(f"Seuils sauvegardes dans {path}")


# =====================================================================
# Construction des règles Prometheus
# =====================================================================


def build_alerts_dict(
    thresholds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construit le dict Prometheus à partir des seuils.

    Args:
        thresholds: Dict des seuils chargés depuis thresholds.yml.

    Returns:
        Dict YAML-serializable avec la structure ``{groups: [...]}``.
    """
    t = thresholds  # alias court

    rules = [
        {
            "alert": "APIDown",
            "expr": 'up{job="champy-api"} == 0',
            "for": t["api_down"]["for"],
            "labels": {
                "severity": t["api_down"]["severity"],
                "project": "champy",
            },
            "annotations": {
                "summary": "API Champy hors ligne",
                "description": (
                    "L'endpoint /metrics de l'API n'est plus scrape par "
                    "Prometheus depuis plus de "
                    f"{t['api_down']['for']}. Verifier sur le NUC3 : "
                    "'docker compose ps api'."
                ),
            },
        },
        {
            "alert": "HighLatencyP95",
            "expr": (
                "histogram_quantile(0.95, sum(rate("
                "champy_prediction_latency_seconds_bucket[5m])) by (le)) "
                f"> {t['high_latency_p95']['threshold_seconds']}"
            ),
            "for": t["high_latency_p95"]["for"],
            "labels": {
                "severity": t["high_latency_p95"]["severity"],
                "project": "champy",
            },
            "annotations": {
                "summary": "Latence p95 elevee sur les predictions",
                "description": (
                    f"La latence p95 des predictions depasse "
                    f"{t['high_latency_p95']['threshold_seconds']}s "
                    f"sur les 5 dernieres minutes (seuil ajustable)."
                ),
            },
        },
        {
            "alert": "HighLatencyMean",
            "expr": (
                "(rate(champy_prediction_latency_seconds_sum[5m]) / "
                "rate(champy_prediction_latency_seconds_count[5m])) > "
                f"{t['high_latency_mean']['threshold_seconds']}"
            ),
            "for": t["high_latency_mean"]["for"],
            "labels": {
                "severity": t["high_latency_mean"]["severity"],
                "project": "champy",
            },
            "annotations": {
                "summary": "Latence moyenne elevee sur les predictions",
                "description": (
                    f"La latence moyenne des predictions depasse "
                    f"{t['high_latency_mean']['threshold_seconds']}s "
                    f"sur les 5 dernieres minutes (seuil ajustable)."
                ),
            },
        },
        {
            "alert": "LowMeanConfidence",
            "expr": (
                "(rate(champy_prediction_confidence_sum[10m]) / "
                "rate(champy_prediction_confidence_count[10m])) < "
                f"{t['low_mean_confidence']['threshold']}"
            ),
            "for": t["low_mean_confidence"]["for"],
            "labels": {
                "severity": t["low_mean_confidence"]["severity"],
                "project": "champy",
            },
            "annotations": {
                "summary": "Confiance moyenne basse - possible drift",
                "description": (
                    f"La confiance moyenne des predictions est tombee "
                    f"sous {t['low_mean_confidence']['threshold']} sur "
                    f"les 10 dernieres minutes. Generer un rapport drift "
                    f"Evidently pour investiguer (seuil ajustable)."
                ),
            },
        },
        {
            "alert": "NoRecentPredictions",
            "expr": (
                "rate(champy_predictions_total[10m]) == 0 and on() "
                'rate(champy_requests_total{endpoint="/predict"}[10m]) > 0'
            ),
            "for": t["no_recent_predictions"]["for"],
            "labels": {
                "severity": t["no_recent_predictions"]["severity"],
                "project": "champy",
            },
            "annotations": {
                "summary": ("Requetes /predict recues mais aucune prediction loggee"),
                "description": (
                    "Le compteur champy_requests_total{endpoint=/predict} "
                    "progresse mais champy_predictions_total stagne. "
                    "Probable erreur cote inference (modele non charge ou "
                    "exception silencieuse)."
                ),
            },
        },
    ]

    return {
        "groups": [
            {
                "name": "champy_api",
                "interval": "30s",
                "rules": rules,
            }
        ]
    }


def regenerate_alerts(
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Régénère le fichier ``champy_alerts.yml`` à partir des seuils.

    Args:
        thresholds_path: Chemin du YAML des seuils.
        output_path: Chemin du YAML d'alertes Prometheus à produire.

    Returns:
        Le chemin du fichier généré.
    """
    thresholds = load_thresholds(thresholds_path)
    alerts = build_alerts_dict(thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "# Regles d'alerte Prometheus pour Champy Classifier.\n"
            "# GENERATED FILE - ne pas editer a la main.\n"
            "# Source : monitoring/alert_generator.py + configs/alerts/thresholds.yml\n\n"
        )
        yaml.safe_dump(
            alerts,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
            width=200,
        )

    logger.info(f"Regles d'alerte regenerees dans {output_path}")
    return output_path


# =====================================================================
# Point d'entrée CLI
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = regenerate_alerts()
    print(f"Generated: {path}")
