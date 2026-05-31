"""Synthèse lisible de la détection de dérive (drift) pour le portfolio Champy.

Transforme le résumé chiffré d'un rapport Evidently en verdict clair,
compréhensible par un public non technique. Un garde-fou d'échantillon minimal
évite de confondre une vraie dérive avec un artefact dû à un trop petit nombre
de prédictions (cas typique d'une démo : 31 prédictions comparées à plusieurs
milliers).

Utilisé par la page Streamlit ``demo/pages/11_drift.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

# --- Constantes ---

MIN_CURRENT_ROWS = 200
DEFAULT_DRIFT_THRESHOLD = 0.5

DriftStatus = Literal["no_data", "insufficient", "stable", "drift"]


# --- Modèle de résultat ---


@dataclass(frozen=True)
class DriftVerdict:
    """Verdict de dérive prêt à afficher, en langage clair.

    Attributes:
        status: 'no_data' (aucune prédiction), 'insufficient' (échantillon trop
            petit pour conclure), 'stable' (pas de dérive), ou 'drift' (dérive).
        n_current: Nombre de prédictions analysées en production.
        n_reference: Nombre de prédictions du jeu de référence (baseline).
        drift_share: Part de colonnes en dérive selon Evidently (0 à 1), ou None.
        drift_threshold: Seuil de déclenchement de la dérive.
        min_current_rows: Nombre minimal de prédictions requis pour conclure.
        headline: Titre court du verdict.
        detail: Explication en clair pour un public non technique.
        confidence_trend: Phrase sur l'évolution de la confiance, ou None.
    """

    status: DriftStatus
    n_current: int
    n_reference: int
    drift_share: float | None
    drift_threshold: float
    min_current_rows: int
    headline: str
    detail: str
    confidence_trend: str | None


# --- Fonctions publiques ---


def build_drift_verdict(
    *,
    n_current: int,
    n_reference: int,
    dataset_drift: bool | None = None,
    drift_share: float | None = None,
    confidence_current_mean: float | None = None,
    confidence_reference_mean: float | None = None,
    confidence_current_std: float | None = None,
    confidence_reference_std: float | None = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    min_current_rows: int = MIN_CURRENT_ROWS,
) -> DriftVerdict:
    """Construit un verdict de dérive lisible à partir d'un résumé chiffré.

    Le garde-fou d'échantillon prime sur tout : en dessous de ``min_current_rows``
    prédictions en production, aucun verdict de dérive n'est rendu, car le résultat
    statistique ne serait pas fiable.

    Args:
        n_current: Nombre de prédictions de production analysées.
        n_reference: Nombre de prédictions du jeu de référence.
        dataset_drift: Verdict global d'Evidently (True si dérive), ou None.
        drift_share: Part de colonnes en dérive (0 à 1), ou None.
        confidence_current_mean: Confiance moyenne en production, ou None.
        confidence_reference_mean: Confiance moyenne en référence, ou None.
        confidence_current_std: Écart-type de la confiance en production, ou None.
        confidence_reference_std: Écart-type de la confiance en référence, ou None.
        drift_threshold: Seuil au-delà duquel la dérive est déclarée.
        min_current_rows: Nombre minimal de prédictions pour pouvoir conclure.

    Returns:
        Un ``DriftVerdict`` prêt à être affiché.
    """
    confidence_trend = _describe_confidence_trend(
        confidence_current_mean,
        confidence_reference_mean,
        confidence_current_std,
        confidence_reference_std,
    )

    drift_detected = _is_drift_detected(dataset_drift, drift_share, drift_threshold)

    if n_current == 0:
        status: DriftStatus = "no_data"
        headline = "Suivi de dérive en attente de données"
        detail = (
            "Aucune prédiction de production n'est encore enregistrée. Le suivi "
            "de dérive s'activera dès les premières prédictions, et rendra un "
            f"verdict fiable au-delà de {min_current_rows}."
        )
    elif n_current < min_current_rows:
        status = "insufficient"
        headline = "Détection de dérive : non concluante"
        detail = (
            f"Seulement {n_current} prédictions analysées en production, contre "
            f"{n_reference} en référence. C'est trop peu pour conclure de façon "
            "fiable. Le verdict sera disponible une fois quelques centaines de "
            f"prédictions accumulées (seuil fixé à {min_current_rows})."
        )
    elif drift_detected:
        status = "drift"
        headline = "Dérive détectée"
        detail = (
            f"Sur {n_current} prédictions analysées, la distribution s'écarte "
            "nettement de la référence. Un réentraînement du modèle est à envisager."
        )
    else:
        status = "stable"
        headline = "Aucune dérive significative"
        detail = (
            f"Sur {n_current} prédictions analysées, le comportement du modèle reste "
            "conforme à la référence. Aucune action requise."
        )

    logger.info(
        "Verdict de derive : status={} n_current={} n_reference={} drift_share={}",
        status,
        n_current,
        n_reference,
        drift_share,
    )

    return DriftVerdict(
        status=status,
        n_current=n_current,
        n_reference=n_reference,
        drift_share=drift_share,
        drift_threshold=drift_threshold,
        min_current_rows=min_current_rows,
        headline=headline,
        detail=detail,
        confidence_trend=confidence_trend,
    )


# --- Fonctions privées ---


def _is_drift_detected(
    dataset_drift: bool | None,
    drift_share: float | None,
    drift_threshold: float,
) -> bool:
    """Détermine si une dérive est déclarée, à partir des indicateurs disponibles.

    Args:
        dataset_drift: Verdict global d'Evidently, prioritaire s'il est fourni.
        drift_share: Part de colonnes en dérive, utilisée à défaut.
        drift_threshold: Seuil de déclenchement appliqué à ``drift_share``.

    Returns:
        True si une dérive est déclarée, False sinon.
    """
    if dataset_drift is not None:
        return dataset_drift
    if drift_share is not None:
        return drift_share >= drift_threshold
    return False


def _describe_confidence_trend(
    current_mean: float | None,
    reference_mean: float | None,
    current_std: float | None,
    reference_std: float | None,
) -> str | None:
    """Décrit en clair l'évolution de la confiance du modèle, si disponible.

    Args:
        current_mean: Confiance moyenne en production.
        reference_mean: Confiance moyenne en référence.
        current_std: Écart-type de la confiance en production.
        reference_std: Écart-type de la confiance en référence.

    Returns:
        Une phrase explicative, ou None si les moyennes ne sont pas disponibles.
    """
    if current_mean is None or reference_mean is None:
        return None

    direction = "plus basse" if current_mean < reference_mean else "comparable ou plus haute"
    phrase = (
        f"Confiance moyenne en production : {_format_fr(current_mean)}, "
        f"contre {_format_fr(reference_mean)} en référence. Elle est {direction}"
    )

    if current_std is not None and reference_std is not None:
        dispersion = "plus dispersée" if current_std > reference_std else "aussi stable"
        phrase += (
            f" et {dispersion} (écart-type {_format_fr(current_std)} "
            f"contre {_format_fr(reference_std)})"
        )

    phrase += ", ce qui est cohérent avec des images de terrain plus variées."
    return phrase


def _format_fr(value: float, decimals: int = 2) -> str:
    """Formate un nombre avec une virgule décimale française.

    Args:
        value: Valeur à formater.
        decimals: Nombre de décimales.

    Returns:
        La valeur formatée, virgule en séparateur décimal.
    """
    return f"{value:.{decimals}f}".replace(".", ",")
