"""Helpers de formatage des timestamps pour les pages Streamlit.

Les timestamps sont stockes en UTC dans la base SQLite (bonne pratique
MLOps : independants du fuseau du serveur, comparaisons sans ambiguite).
La conversion en heure locale Europe/Paris (CET/CEST automatique selon
la saison) est realisee a l'affichage via ce module.

Usage type :

    from demo.lib.datetime_utils import format_local_time

    # Depuis un timestamp ISO 8601 stocke en SQLite
    affichage = format_local_time("2026-05-19T19:24:17.575759+00:00")
    # Resultat : "19/05/2026 21:24:17 (CEST)"

    # Format compact
    compact = format_local_time(ts, fmt="compact")
    # Resultat : "20/05 09:30"

    # Pour un dataframe pandas
    df["timestamp_local"] = df["timestamp"].apply(format_local_time)
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

# Fuseau de reference pour l'affichage utilisateur
LOCAL_TZ = ZoneInfo("Europe/Paris")

# Formats disponibles via le parametre fmt de format_local_time()
_FORMATS = {
    "full": "%d/%m/%Y %H:%M:%S (%Z)",
    "datetime": "%d/%m/%Y %H:%M:%S",
    "compact": "%d/%m %H:%M",
    "time": "%H:%M:%S",
    "date": "%d/%m/%Y",
    "iso": "%Y-%m-%dT%H:%M:%S%z",
}


def format_local_time(
    timestamp: str | datetime | None,
    fmt: str = "full",
) -> str:
    """Convertit un timestamp UTC en heure locale Europe/Paris formatee.

    Gere automatiquement le passage CET (hiver) / CEST (ete).

    Args:
        timestamp: Timestamp a formater. Accepte trois types :
            - Chaine ISO 8601 (ex: "2026-05-19T19:24:17.575759+00:00")
            - Objet datetime (avec ou sans timezone)
            - None (retourne une chaine vide)
        fmt: Cle du format souhaite, parmi :
            - "full"     : "19/05/2026 21:24:17 (CEST)"      [defaut]
            - "datetime" : "19/05/2026 21:24:17"
            - "compact"  : "19/05 21:24"
            - "time"     : "21:24:17"
            - "date"     : "19/05/2026"
            - "iso"      : "2026-05-19T21:24:17+0200"

    Returns:
        Chaine formatee en heure locale, ou chaine vide si timestamp est None.

    Raises:
        ValueError: Si le format demande n'existe pas, ou si le timestamp
                    string n'est pas un ISO 8601 valide.
    """
    if timestamp is None:
        return ""

    if fmt not in _FORMATS:
        raise ValueError(f"Format inconnu : '{fmt}'. Valeurs autorisees : {list(_FORMATS.keys())}")

    # Normalisation en datetime
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise TypeError(
            f"Type non supporte : {type(timestamp).__name__}. "
            f"Attendu : str ISO 8601, datetime, ou None."
        )

    # Si le datetime est naif, on suppose UTC (defaut du store SQLite)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    # Conversion en local Europe/Paris (CET/CEST auto)
    dt_local = dt.astimezone(LOCAL_TZ)

    return dt_local.strftime(_FORMATS[fmt])


def now_local_str(fmt: str = "full") -> str:
    """Retourne l'heure courante formatee en heure locale Europe/Paris.

    Args:
        fmt: Cle du format souhaite (voir `format_local_time`).

    Returns:
        Chaine de l'heure locale courante.
    """
    return format_local_time(datetime.now(ZoneInfo("UTC")), fmt=fmt)
