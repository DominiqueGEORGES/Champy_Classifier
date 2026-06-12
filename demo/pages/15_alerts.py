"""Page Streamlit : gestion dynamique des seuils d'alerte Prometheus.

Permet de modifier en live les seuils des règles d'alerte sans
redémarrer Prometheus. Le flux est :

    1. Lecture des seuils actuels (``configs/alerts/thresholds.yml``)
    2. Affichage avec sliders / inputs
    3. Bouton "Appliquer" :
        a. Sauve les nouveaux seuils dans thresholds.yml
        b. Régénère champy_alerts.yml via ``alert_generator``
        c. POST sur ``/-/reload`` de Prometheus pour hot reload
    4. Vérification que Prometheus a bien rechargé (statut)

Prérequis :
    - Prometheus démarré avec ``--web.enable-lifecycle`` (sinon le
      POST /-/reload renvoie 405).
    - Le dossier ``configs/alerts/`` doit être mounté en RW dans le
      container demo (sinon écriture impossible).
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# =====================================================================
# Setup chemin projet
# =====================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =====================================================================
# Imports tiers
# =====================================================================

import httpx
import streamlit as st

# =====================================================================
# Imports projet
# =====================================================================
from demo import auth
from monitoring.alert_generator import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLDS_PATH,
    THRESHOLD_METADATA,
    load_thresholds,
    regenerate_alerts,
    save_thresholds,
)

# =====================================================================
# Constantes
# =====================================================================

# URL d'admin de Prometheus pour le hot reload. En interne via le
# réseau docker-compose (nom de service + port interne), incluant le
# route prefix configuré pour servir Prometheus sous /prometheus.
PROMETHEUS_RELOAD_URL = os.environ.get(
    "CHAMPY_PROMETHEUS_RELOAD_URL",
    "http://prometheus:9090/prometheus/-/reload",
)

# Durées proposées dans les selectbox "for"
DURATION_OPTIONS = ["30s", "1m", "2m", "5m", "10m", "15m", "30m", "1h"]

# =====================================================================
# URL de l'API des alertes Prometheus (etat pending / firing). Meme
# service et meme route prefix que l'URL de reload.
# =====================================================================

PROMETHEUS_ALERTS_URL = os.environ.get(
    "CHAMPY_PROMETHEUS_ALERTS_URL",
    "http://prometheus:9090/prometheus/api/v1/alerts",
)

# =====================================================================
# Authentification (lit access_policy.yaml)
# =====================================================================

auth.setup_page()

# =====================================================================
# Configuration de la page
# =====================================================================

st.set_page_config(page_title="14 - Alertes", layout="wide")
st.title(":rotating_light: Gestion dynamique des alertes")

st.markdown(
    """
Modifie les seuils des règles d'alerte Prometheus en live. La sauvegarde
régénère `configs/alerts/champy_alerts.yml` et déclenche un **hot reload**
de Prometheus (aucun redémarrage du container).

> :information_source: Les noms d'alerte et les expressions PromQL sont
> figés (par sécurité). Seuls les seuils numériques et les durées
> peuvent être ajustés depuis cette interface.
"""
)
st.divider()


# =====================================================================
# Chargement des seuils courants
# =====================================================================

try:
    current_thresholds = load_thresholds()
except Exception as exc:
    st.error(f"Impossible de charger les seuils actuels : {exc}")
    st.stop()


# =====================================================================
# Section 1 : édition des seuils
# =====================================================================

st.header("1. Seuils actuels")

# On travaille sur une copie pour ne modifier l'état qu'au "Appliquer"
new_thresholds = deepcopy(current_thresholds)

for alert_name, meta in THRESHOLD_METADATA.items():
    with st.expander(meta["label"], expanded=True):
        st.caption(meta["description"])
        cols = st.columns(3)

        current = current_thresholds.get(alert_name, {})
        tunable = meta.get("tunable", [])

        # Seuil numérique (si tunable)
        if "threshold_seconds" in tunable:
            r_min, r_max, r_step = meta["threshold_range"]
            new_thresholds[alert_name]["threshold_seconds"] = cols[0].slider(
                "Seuil (secondes)",
                min_value=float(r_min),
                max_value=float(r_max),
                value=float(current.get("threshold_seconds", r_min)),
                step=float(r_step),
                key=f"{alert_name}_threshold_seconds",
            )
        elif "threshold" in tunable:
            r_min, r_max, r_step = meta["threshold_range"]
            new_thresholds[alert_name]["threshold"] = cols[0].slider(
                "Seuil (0-1)",
                min_value=float(r_min),
                max_value=float(r_max),
                value=float(current.get("threshold", r_min)),
                step=float(r_step),
                key=f"{alert_name}_threshold",
            )
        else:
            cols[0].markdown("_Pas de seuil numérique pour cette règle_")

        # Durée (toujours tunable)
        if "for" in tunable:
            current_for = current.get("for", "1m")
            new_thresholds[alert_name]["for"] = cols[1].selectbox(
                "Durée minimale (for)",
                options=DURATION_OPTIONS,
                index=(
                    DURATION_OPTIONS.index(current_for) if current_for in DURATION_OPTIONS else 1
                ),
                key=f"{alert_name}_for",
                help=(
                    "Durée pendant laquelle la condition doit être vraie "
                    "avant que l'alerte ne se déclenche."
                ),
            )

        # Severity (figée pour l'instant, affichée en read-only)
        cols[2].markdown(f"**Severity** : `{current.get('severity', '?')}`")


st.divider()


# =====================================================================
# Section 2 : actions
# =====================================================================

st.header("2. Appliquer les changements")

col_apply, col_reset, _ = st.columns([1, 1, 3])

apply_clicked = col_apply.button(
    ":white_check_mark: Appliquer",
    type="primary",
    use_container_width=True,
)
reset_clicked = col_reset.button(
    ":arrows_counterclockwise: Reset aux defauts",
    use_container_width=True,
)


def _fetch_active_alerts() -> tuple[list[dict], str | None]:
    """Récupère les alertes en cours depuis l'API Prometheus.

    Returns:
        Tuple ``(alertes, erreur)``. ``alertes`` est la liste des alertes
        actives (état ``pending`` ou ``firing``). ``erreur`` vaut ``None``
        en cas de succès, sinon un message destiné à l'utilisateur.
    """
    try:
        response = httpx.get(PROMETHEUS_ALERTS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
        alerts = payload.get("data", {}).get("alerts", [])
        # Tri : les alertes déclenchées (firing) en premier.
        alerts.sort(key=lambda a: a.get("state") != "firing")
        return alerts, None
    except Exception as exc:
        return [], f"Impossible de lire les alertes Prometheus : {exc}"


def _format_active_since(raw: str) -> str:
    """Convertit un horodatage ISO 8601 UTC en heure locale lisible.

    Args:
        raw: Horodatage renvoyé par Prometheus (ex. ``2026-06-04T19:34:17.39Z``).

    Returns:
        Heure locale au format ``HH:MM:SS``, ou la chaine brute si le
        parsing échoue.
    """
    if not raw:
        return ""
    try:
        cleaned = raw.replace("Z", "+00:00")
        if "." in cleaned:
            base, frac_tz = cleaned.split(".", 1)
            fraction, offset = frac_tz, ""
            for index, char in enumerate(frac_tz):
                if char in "+-":
                    fraction, offset = frac_tz[:index], frac_tz[index:]
                    break
            cleaned = f"{base}.{fraction[:6]}{offset}"
        moment = datetime.fromisoformat(cleaned)
        return moment.astimezone().strftime("%H:%M:%S")
    except Exception:
        return raw


def _reload_prometheus() -> tuple[bool, str]:
    """Appelle l'endpoint /-/reload de Prometheus.

    Returns:
        Tuple ``(succes, message)`` pour affichage utilisateur.
    """
    try:
        response = httpx.post(PROMETHEUS_RELOAD_URL, timeout=10)
        if response.status_code == 200:
            return True, "Prometheus a recharge la configuration avec succes."
        if response.status_code == 405:
            return False, (
                "Prometheus refuse le reload (HTTP 405). "
                "Verifier que le flag --web.enable-lifecycle est present "
                "dans la command du container prometheus."
            )
        return False, (f"Prometheus a repondu HTTP {response.status_code} : {response.text[:200]}")
    except Exception as exc:
        return False, f"Erreur reseau lors de l'appel reload : {exc}"


if apply_clicked:
    with st.spinner("Sauvegarde des seuils et regeneration des regles..."):
        try:
            save_thresholds(new_thresholds)
            output_path = regenerate_alerts()
            st.success(
                f"Seuils sauves dans `{DEFAULT_THRESHOLDS_PATH.name}`, "
                f"regles regenerees dans `{output_path.name}`."
            )
        except PermissionError as exc:
            st.error(
                "Impossible d'ecrire dans configs/alerts/ : "
                f"{exc}.\nVerifier que le mount est en RW dans "
                "docker-compose.yml."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Erreur durant la regeneration : {exc}")
            st.stop()

    with st.spinner("Hot reload de Prometheus..."):
        success, msg = _reload_prometheus()
        if success:
            st.success(msg)
            st.balloons()
        else:
            st.error(msg)

if reset_clicked:
    with st.spinner("Reset aux valeurs par defaut..."):
        try:
            save_thresholds(DEFAULT_THRESHOLDS)
            regenerate_alerts()
            success, msg = _reload_prometheus()

            # Invalider le session_state des widgets pour qu'ils
            # reprennent les valeurs par defaut au prochain rendu.
            # Sans ca, les sliders garderaient leurs anciennes valeurs
            # malgre la modification du fichier source.
            for alert_name in DEFAULT_THRESHOLDS:
                for suffix in ["threshold_seconds", "threshold", "for"]:
                    key = f"{alert_name}_{suffix}"
                    if key in st.session_state:
                        del st.session_state[key]

            if success:
                st.success("Reset effectue et applique sur Prometheus.")
                st.rerun()
            else:
                st.warning(f"Fichiers reinitialises mais reload Prometheus echoue : {msg}")
        except Exception as exc:
            st.error(f"Erreur durant le reset : {exc}")


# =====================================================================
# Section 3 : Alertes actives (live)
# =====================================================================

st.divider()

st.header("3. Alertes actives (live)")
st.caption(
    "Lecture directe de Prometheus. Après un changement de seuil, l'alerte "
    "apparaît d'abord en `pending`, puis passe en `firing` une fois la durée "
    "minimale écoulée. Les alertes `firing` sont celles transmises à "
    "Alertmanager, et donc à Discord."
)


@st.fragment(run_every="5s")
def _render_active_alerts() -> None:
    """Affiche les alertes actives, rafraîchies automatiquement toutes les 5 s."""
    alerts, error = _fetch_active_alerts()

    st.caption(f"Dernière actualisation : {datetime.now():%H:%M:%S}")

    if error:
        st.error(error)
        return

    if not alerts:
        st.success("Aucune alerte active pour le moment.")
        return

    for alert in alerts:
        labels = alert.get("labels", {})
        name = labels.get("alertname", "inconnue")
        state = alert.get("state", "?")
        severity = labels.get("severity", "?")
        since = _format_active_since(alert.get("activeAt", ""))

        ligne = f"**{name}** (severity `{severity}`)"
        if since:
            ligne += f", active depuis {since}"

        if state == "firing":
            st.error(f"FIRING : {ligne}")
        else:
            st.warning(f"PENDING : {ligne}")


_render_active_alerts()

# =====================================================================
# Section 4 : état des fichiers générés
# =====================================================================

st.header("4. Etat des fichiers")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("thresholds.yml")
    if DEFAULT_THRESHOLDS_PATH.exists():
        mtime = datetime.fromtimestamp(DEFAULT_THRESHOLDS_PATH.stat().st_mtime)
        st.caption(f"Modifie le {mtime:%Y-%m-%d %H:%M:%S}")
        with open(DEFAULT_THRESHOLDS_PATH, encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    else:
        st.info("Fichier inexistant, sera cree au premier 'Appliquer'.")

with col_b:
    st.subheader("champy_alerts.yml (genere)")
    if DEFAULT_OUTPUT_PATH.exists():
        mtime = datetime.fromtimestamp(DEFAULT_OUTPUT_PATH.stat().st_mtime)
        st.caption(f"Modifie le {mtime:%Y-%m-%d %H:%M:%S}")
        with open(DEFAULT_OUTPUT_PATH, encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    else:
        st.info("Fichier inexistant, sera cree au premier 'Appliquer'.")

st.divider()
