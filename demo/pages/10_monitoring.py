"""Page Streamlit : monitoring complet (Bloc M4).

Quatre sections :

1. Metriques live depuis Prometheus (RPS, latence p50/p95/p99, taux
   d'erreur, total predictions). Refresh manuel + cache TTL=15s.
2. Dashboards Grafana embarques en iframe (6 dashboards : performance,
   predictions, system health, containers, hote, eco) avec fallback vers des liens si l'iframe
   ne charge pas (Grafana down ou auth bloquante).
3. Top-10 especes predites : barchart Plotly construit depuis Prometheus
   (cumul global) et tendance temporelle depuis le PredictionStore SQLite
   (Bloc M2, fenetre glissante 24h).
4. Alerting visuel : 3 indicateurs vert/jaune/rouge sur la confiance, la
   latence p95 et le taux d'erreur. Seuils lus depuis
   ``configs/monitoring/thresholds.yml`` (zero hardcoded).

La page est resiliente : Prometheus down -> messages explicites, pas de
crash. Grafana down -> fallback vers des liens. SQLite vide -> placeholder.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from demo import auth

auth.setup_page(min_role="user")  # ou "guest" pour pages publiques


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="10 - Monitoring", layout="wide")
st.title(":bar_chart: Monitoring")

# Imports en mode defensif : si l'un des helpers casse, on degrade
# proprement plutot que d'afficher une page blanche.
try:
    from demo.lib.api_utils import (
        get_grafana_url,
        get_prometheus_url,
        query_prometheus,
    )
    from demo.lib.monitoring_utils import (
        LEVEL_COLORS,
        LEVEL_LABELS,
        evaluate_alerts,
        fetch_live_metrics,
        load_thresholds,
    )
except Exception as e:
    st.error(f"Impossible de charger les helpers : {e}")
    st.stop()

PROMETHEUS_URL = get_prometheus_url()
GRAFANA_URL = get_grafana_url()
THRESHOLDS = load_thresholds()

st.caption(
    f"Prometheus : `{PROMETHEUS_URL}`  |  Grafana : `{GRAFANA_URL}`  |  "
    f"Last load : {datetime.now(ZoneInfo('Europe/Paris')).strftime('%d/%m/%Y %H:%M:%S (%Z)')}"
)
if st.button("Rafraichir maintenant", type="primary"):
    st.cache_data.clear()
    st.rerun()

# =====================================================================
# Section 1 : Metriques live depuis Prometheus
# =====================================================================
st.header("1. Metriques live (Prometheus)")

metrics = fetch_live_metrics()
prometheus_alive = any(v is not None for v in metrics.values())

if not prometheus_alive:
    st.warning(
        f"Prometheus n'est pas joignable sur `{PROMETHEUS_URL}`. "
        "Verifier que le container `prometheus` tourne (docker compose ps) "
        "et qu'il scrape bien l'API (cible `champy-api` dans la config)."
    )
else:
    cols = st.columns(5)
    cols[0].metric(
        "Total predictions",
        f"{int(metrics['total_predictions']):,}".replace(",", " ")
        if metrics["total_predictions"] is not None
        else "-",
    )
    cols[1].metric(
        "RPS (5min)",
        f"{metrics['rps']:.2f}" if metrics["rps"] is not None else "-",
    )
    cols[2].metric(
        "Latence p50",
        f"{metrics['p50'] * 1000:.0f} ms" if metrics["p50"] is not None else "-",
    )
    cols[3].metric(
        "Latence p95",
        f"{metrics['p95'] * 1000:.0f} ms" if metrics["p95"] is not None else "-",
    )
    cols[4].metric(
        "Latence p99",
        f"{metrics['p99'] * 1000:.0f} ms" if metrics["p99"] is not None else "-",
    )

    col_err, col_conf = st.columns(2)
    col_err.metric(
        "Taux d'erreur (5min)",
        f"{metrics['error_rate']:.2%}" if metrics["error_rate"] is not None else "-",
    )
    col_conf.metric(
        "Confiance moyenne",
        f"{metrics['confidence_avg']:.2%}" if metrics["confidence_avg"] is not None else "-",
    )

st.divider()

# =====================================================================
# Section 2 : Dashboards Grafana embarques
# =====================================================================
st.header("2. Dashboards Grafana")

DASHBOARDS = [
    ("Performance API", "champy-api-performance"),
    ("Prédictions", "champy-predictions"),
    ("Ressources containers", "champy-containers"),
    ("Hôte serveur", "champy-host"),
    ("Impact écologique", "champy-eco-impact"),
]

# Detection rapide : Grafana repond-il sur /api/health ? Si non, on
# bascule en fallback liens-only sans tenter d'embed (sinon iframe vide
# pendant 30s avant de timeout cote browser).
import httpx


@st.cache_data(ttl=15)
def _grafana_alive(url: str) -> bool:
    """Verifie que Grafana repond sur ``/api/health`` en moins de 3s.

    Args:
        url: URL de base de Grafana.

    Returns:
        True si Grafana repond 2xx.
    """
    try:
        resp = httpx.get(f"{url}/api/health", timeout=3)
        return resp.status_code < 400
    except Exception:
        return False


grafana_ok = _grafana_alive(GRAFANA_URL)

# L'iframe pointe sur l'URL "client" de Grafana. Quand le Streamlit tourne
# dans le compose, le helper renvoie `http://grafana:3000`, ce qui ne
# fonctionne PAS pour un iframe (le navigateur du client ne resout pas
# le DNS interne du compose). On tape donc une URL externe :
# - via env var CHAMPY_GRAFANA_URL_EXTERNAL si on est en compose,
# - sinon on retombe sur GRAFANA_URL (mode local natif).

GRAFANA_EXTERNAL_URL = os.environ.get("CHAMPY_GRAFANA_URL_EXTERNAL", GRAFANA_URL)
parsed = urlparse(GRAFANA_EXTERNAL_URL)
if parsed.hostname in {"grafana", "host.docker.internal"}:
    # Heuristique : un nom de service interne ne resoudra pas cote browser.
    GRAFANA_EXTERNAL_URL = f"http://localhost:{parsed.port or 3010}"

if not grafana_ok:
    st.warning(
        f"Grafana n'est pas joignable sur `{GRAFANA_URL}`. "
        "Verifier que le container `grafana` tourne (docker compose ps) "
        "et que l'auth anonyme + ALLOW_EMBEDDING sont actives "
        "(voir docker-compose.yml)."
    )
    st.markdown("**Liens directs vers les dashboards** (a ouvrir dans un nouvel onglet) :")
    for label, uid in DASHBOARDS:
        st.markdown(f"- [{label}]({GRAFANA_EXTERNAL_URL}/d/{uid})")
else:
    tabs = st.tabs([label for label, _ in DASHBOARDS])
    for tab, (_label, uid) in zip(tabs, DASHBOARDS, strict=True):
        with tab:
            iframe_url = f"{GRAFANA_EXTERNAL_URL}/d/{uid}?orgId=1&kiosk&theme=light&refresh=30s"
            st.caption(f"[Ouvrir dans un nouvel onglet]({iframe_url})")
            st.components.v1.iframe(iframe_url, height=720, scrolling=True)

st.divider()

# =====================================================================
# Section 3 : Top-10 especes predites + tendance SQLite
# =====================================================================
st.header("3. Top-10 especes predites")

import pandas as pd
import plotly.express as px

col_prom, col_sql = st.columns(2)

# 3a - Cumul global depuis Prometheus
with col_prom:
    st.subheader("Cumul (Prometheus)")
    rows = []
    for r in query_prometheus("topk(10, sum by (species) (champy_predictions_total))"):
        species = r.get("metric", {}).get("species", "?")
        try:
            value = int(float(r.get("value", [0, 0])[1]))
        except (TypeError, ValueError):
            continue
        rows.append({"Espece": species, "Predictions": value})
    if rows:
        df_prom = pd.DataFrame(rows).sort_values("Predictions", ascending=True)
        fig = px.bar(
            df_prom,
            x="Predictions",
            y="Espece",
            orientation="h",
            color="Predictions",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune metrique champy_predictions_total dans Prometheus.")

# 3b - Tendance temporelle 24h depuis le PredictionStore SQLite (Bloc M2).
# Si le store n'est pas accessible (compose sans BentoML, fichier
# manquant), on degrade silencieusement.
with col_sql:
    st.subheader("Tendance 24h (via API)")
    from demo.lib.api_utils import get_recent_predictions

    records = get_recent_predictions(hours=24, limit=10000)
    if records is None:
        st.warning(
            "L'API n'a pas répondu pour `/predictions/recent`. "
            "Vérifier que le service `champy_api` est sain."
        )
    elif not records:
        st.info("Aucune prédiction sur les 24 dernières heures.")
    else:
        df_sql = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(r["timestamp"]).tz_convert("Europe/Paris"),
                    "species": r["predicted_class"],
                }
                for r in records
            ]
        )
        # Agregation horaire par classe (top-5 visibles).
        top5_species = df_sql["species"].value_counts().head(5).index.tolist()
        df_top = df_sql[df_sql["species"].isin(top5_species)].copy()
        df_top["hour"] = df_top["timestamp"].dt.floor("h")
        trend = df_top.groupby(["hour", "species"]).size().reset_index(name="count")
        fig = px.line(
            trend,
            x="hour",
            y="count",
            color="species",
            markers=True,
        )
        fig.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(records)} prédictions sur 24h, top-5 espèces affichées.")
st.divider()

# =====================================================================
# Section 4 : Alerting visuel
# =====================================================================
st.header("4. Alerting visuel")

if not THRESHOLDS:
    st.warning(
        "Fichier `configs/monitoring/thresholds.yml` introuvable ou vide. "
        "Les seuils ne peuvent pas etre evalues."
    )
elif not prometheus_alive:
    st.warning("Prometheus indisponible : alertes non evaluees.")
else:
    statuses = evaluate_alerts(metrics, THRESHOLDS)
    cols = st.columns(len(statuses))
    for col, status in zip(cols, statuses, strict=True):
        color = LEVEL_COLORS[status.level]
        label = LEVEL_LABELS[status.level]
        value_str = (
            "-"
            if status.value is None
            else (
                f"{status.value:.1%}"
                if status.unit == "" and status.name in {"Confiance moyenne", "Taux d'erreur"}
                else f"{status.value:.3f}{status.unit}"
            )
        )
        col.markdown(
            f"""<div style="padding:1rem;border-radius:8px;background:{color}22;
            border-left:6px solid {color};">
            <div style="font-size:0.8rem;text-transform:uppercase;
            letter-spacing:0.05em;color:{color};font-weight:600;">{label}</div>
            <div style="font-size:1.1rem;margin-top:0.25rem;">{status.name}</div>
            <div style="font-size:1.6rem;font-weight:700;margin-top:0.25rem;">
            {value_str}</div>
            <div style="font-size:0.75rem;color:#666;margin-top:0.5rem;">
            {status.message}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with st.expander("Configuration des seuils"):
        st.code(
            THRESHOLDS_PATH_STR := str(
                _PROJECT_ROOT / "configs" / "monitoring" / "thresholds.yml"
            ),
            language="text",
        )
        st.json(THRESHOLDS)
