"""Page Streamlit : monitoring via Prometheus et Grafana.

Affiche les métriques de monitoring (latence, prédictions, confiance)
depuis Prometheus via requêtes PromQL, avec lien vers Grafana.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="10 - Monitoring", layout="wide")
st.title(":bar_chart: Monitoring")

try:
    from demo.lib.api_utils import (
        get_grafana_url,
        get_prometheus_metrics,
        get_prometheus_url,
        query_prometheus,
    )
except Exception as e:
    st.error(f"Impossible de charger les helpers : {e}")
    st.stop()

# =====================================================================
# Section 1 : Liens vers les outils
# =====================================================================
st.header("Outils de monitoring")

prometheus_url = get_prometheus_url()
grafana_url = get_grafana_url()

col1, col2 = st.columns(2)
col1.markdown(f"""
**Prometheus** : [{prometheus_url}]({prometheus_url})
- Requêtes PromQL
- Alertes
- Targets et scrape status
""")
col2.markdown(f"""
**Grafana** : [{grafana_url}]({grafana_url})
- Dashboards pré-configurés
- Login : admin / (voir .env GRAFANA_PASSWORD)
""")

st.divider()

# =====================================================================
# Section 2 : Métriques en temps reel
# =====================================================================
st.header("Métriques en temps reel")

if st.button("Rafraîchir", type="primary"):
    st.cache_data.clear()

# Prédictions totales
predictions = query_prometheus("champy_predictions_total")
if predictions:
    st.subheader("Prédictions par espèce")
    import pandas as pd

    rows = []
    for result in predictions:
        species = result.get("metric", {}).get("species", "?")
        value = float(result.get("value", [0, 0])[1])
        rows.append({"Espèce": species, "Prédictions": int(value)})

    if rows:
        df = pd.DataFrame(rows).sort_values("Prédictions", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Aucune donnée de prédiction disponible (Prometheus hors ligne ou aucune requête).")

# Latence
st.subheader("Latence d'inférence")
latency = query_prometheus(
    "histogram_quantile(0.95, rate(champy_prediction_latency_seconds_bucket[5m]))"
)
if latency:
    for result in latency:
        value = float(result.get("value", [0, 0])[1])
        st.metric("Latence p95", f"{value * 1000:.0f} ms")
else:
    st.info("Données de latence non disponibles.")

# Confiance moyenne
confidence = query_prometheus(
    "champy_prediction_confidence_sum / champy_prediction_confidence_count"
)
if confidence:
    for result in confidence:
        value = float(result.get("value", [0, 0])[1])
        st.metric("Confiance moyenne", f"{value:.1%}")

st.divider()

# =====================================================================
# Section 3 : Métriques brutes
# =====================================================================
with st.expander("Métriques Prometheus brutes"):
    raw = get_prometheus_metrics()
    if raw:
        st.code(raw, language="text")
    else:
        st.info("API hors ligne.")
