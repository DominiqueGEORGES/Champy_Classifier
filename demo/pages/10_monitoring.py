"""Page Streamlit : monitoring via Prometheus et Grafana.

Affiche les metriques de monitoring (latence, predictions, confiance)
depuis Prometheus via requetes PromQL, avec lien vers Grafana.
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
    from demo.lib.api_utils import get_prometheus_metrics, query_prometheus
except Exception as e:
    st.error(f"Impossible de charger les helpers : {e}")
    st.stop()

# =====================================================================
# Section 1 : Liens vers les outils
# =====================================================================
st.header("Outils de monitoring")

col1, col2 = st.columns(2)
col1.markdown("""
**Prometheus** : [http://localhost:9090](http://localhost:9090)
- Requetes PromQL
- Alertes
- Targets et scrape status
""")
col2.markdown("""
**Grafana** : [http://localhost:3000](http://localhost:3000)
- Dashboards pre-configures
- Login : admin / (voir .env GRAFANA_PASSWORD)
""")

st.divider()

# =====================================================================
# Section 2 : Metriques en temps reel
# =====================================================================
st.header("Metriques en temps reel")

if st.button("Rafraichir", type="primary"):
    st.cache_data.clear()

# Predictions totales
predictions = query_prometheus("champy_predictions_total")
if predictions:
    st.subheader("Predictions par espece")
    import pandas as pd

    rows = []
    for result in predictions:
        species = result.get("metric", {}).get("species", "?")
        value = float(result.get("value", [0, 0])[1])
        rows.append({"Espece": species, "Predictions": int(value)})

    if rows:
        df = pd.DataFrame(rows).sort_values("Predictions", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Aucune donnee de prediction disponible (Prometheus hors ligne ou aucune requete).")

# Latence
st.subheader("Latence d'inference")
latency = query_prometheus(
    "histogram_quantile(0.95, rate(champy_prediction_latency_seconds_bucket[5m]))"
)
if latency:
    for result in latency:
        value = float(result.get("value", [0, 0])[1])
        st.metric("Latence p95", f"{value * 1000:.0f} ms")
else:
    st.info("Donnees de latence non disponibles.")

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
# Section 3 : Metriques brutes
# =====================================================================
with st.expander("Metriques Prometheus brutes"):
    raw = get_prometheus_metrics()
    if raw:
        st.code(raw, language="text")
    else:
        st.info("API hors ligne.")
