"""Page Streamlit : statut et documentation de l'API FastAPI.

Affiche l'etat de sante de l'API, les metadonnees du modele,
un lien vers la documentation Swagger, et les metriques brutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="09 - API", layout="wide")
st.title(":electric_plug: API FastAPI")

try:
    from demo.lib.api_utils import get_health, get_model_info, get_prometheus_metrics
except Exception as e:
    st.error(f"Impossible de charger les helpers API : {e}")
    st.stop()

# =====================================================================
# Section 1 : Statut de l'API
# =====================================================================
st.header("Statut")

health = get_health()
if health is not None:
    status = health.get("status", "inconnu")
    if status == "healthy":
        st.success(f"API en ligne - statut : {status}")
    elif status == "no_model":
        st.warning(f"API en ligne mais sans modele : {status}")
    else:
        st.error(f"API en erreur : {status}")

    col1, col2 = st.columns(2)
    col1.metric("Modele charge", "Oui" if health.get("model_loaded") else "Non")
    col2.metric("Version", health.get("model_version", "?"))
else:
    st.error("API indisponible. Verifiez que le serveur est demarre.")

st.divider()

# =====================================================================
# Section 2 : Documentation Swagger
# =====================================================================
st.header("Documentation")

api_url = "http://localhost:8000"
st.markdown(f"""
- **Swagger UI** : [{api_url}/docs]({api_url}/docs)
- **ReDoc** : [{api_url}/redoc]({api_url}/redoc)
- **OpenAPI JSON** : [{api_url}/openapi.json]({api_url}/openapi.json)

**Endpoints disponibles** :
| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | /predict | Prediction top-5 depuis une image |
| GET | /health | Etat de sante du service |
| GET | /metrics | Metriques Prometheus |
| GET | /model/info | Metadonnees du modele |
""")

st.divider()

# =====================================================================
# Section 3 : Informations du modele
# =====================================================================
st.header("Informations du modele")

model_info = get_model_info()
if model_info is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Classes", model_info.get("num_classes", "?"))
    col2.metric("Input shape", str(model_info.get("input_shape", "?")))
    col3.metric("Version", model_info.get("model_version", "?"))

    with st.expander("Liste des classes"):
        for i, name in enumerate(model_info.get("class_names", [])):
            st.write(f"{i:2d}. {name}")
else:
    st.info("Informations du modele non disponibles (API hors ligne ou modele non charge).")

st.divider()

# =====================================================================
# Section 4 : Metriques brutes
# =====================================================================
st.header("Metriques Prometheus (brutes)")

if st.button("Rafraichir les metriques"):
    st.cache_data.clear()

metrics_text = get_prometheus_metrics()
if metrics_text:
    # Filtrer les lignes champy_
    champy_lines = [line for line in metrics_text.split("\n") if line.startswith("champy_")]
    if champy_lines:
        st.code("\n".join(champy_lines), language="text")
    else:
        st.info("Aucune metrique champy_ disponible (aucune prediction effectuee).")
else:
    st.info("Metriques non disponibles (API hors ligne).")
