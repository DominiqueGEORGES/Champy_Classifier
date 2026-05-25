"""Page Streamlit : statut et documentation de l'API FastAPI.

Affiche :
    - L'état de santé live de l'API (endpoint `/health`)
    - Les métadonnées du modèle (endpoint `/model/info`)
    - La documentation des endpoints avec liens vers Swagger UI / ReDoc
    - Les métriques Prometheus brutes filtrées sur `champy_*`

L'API est interrogée en interne (réseau Docker) par les pages Streamlit.
Pour l'accès externe (Swagger UI cliquable), la variable d'environnement
`CHAMPY_API_PUBLIC_URL` doit pointer vers l'URL publique de l'API.
Si elle est absente, la documentation reste textuelle.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import os
import sys
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

import streamlit as st

# =====================================================================
# Imports projet
# =====================================================================
from demo import auth

# =====================================================================
# Authentification (lit access_policy.yaml)
# =====================================================================

auth.setup_page()

# =====================================================================
# Configuration de la page
# =====================================================================

st.set_page_config(page_title="09 - API", layout="wide")
st.title(":electric_plug: API FastAPI")

# =====================================================================
# Chargement des helpers
# =====================================================================

try:
    from demo.lib.api_utils import (
        get_health,
        get_model_info,
        get_prometheus_metrics,
    )
except Exception as exc:
    st.error(f"Impossible de charger les helpers API : {exc}")
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
        st.warning(f"API en ligne mais sans modèle : {status}")
    else:
        st.error(f"API en erreur : {status}")

    col1, col2 = st.columns(2)
    col1.metric("Modèle chargé", "Oui" if health.get("model_loaded") else "Non")
    col2.metric("Version", health.get("model_version", "?"))
else:
    st.error("API indisponible. Vérifiez que le serveur est démarré.")

st.divider()


# =====================================================================
# Section 2 : Documentation des endpoints
# =====================================================================

st.header("Documentation")

# URL publique de l'API (configurée via CHAMPY_API_PUBLIC_URL).
# Si absente, on reste sur de la documentation textuelle sans liens
# cliquables (les URLs internes Docker ne sont pas exploitables depuis
# le navigateur de l'utilisateur).
api_public_url = os.environ.get("CHAMPY_API_PUBLIC_URL", "").rstrip("/")

st.markdown(
    "L'API REST FastAPI est interrogée en interne par les pages "
    "Streamlit (notamment **Prédiction**). Elle est également exposée "
    "via Cloudflare Tunnel pour permettre l'inspection directe de la "
    "documentation interactive."
)

st.markdown(
    """
**Endpoints disponibles**

| Méthode | Endpoint        | Description                                       |
|---------|-----------------|---------------------------------------------------|
| POST    | `/predict`      | Prédiction top-5 depuis une image (multipart)     |
| GET     | `/health`       | État de santé du service et version du modèle     |
| GET     | `/metrics`      | Métriques Prometheus au format `text/plain`       |
| GET     | `/model/info`   | Métadonnées du modèle (classes, input shape...)   |
"""
)

if api_public_url:
    st.markdown("**Documentation interactive (cliquables)**")
    st.markdown(
        f"""
- :open_book: **Swagger UI** (testez les endpoints en live) : [{api_public_url}/]({api_public_url}/)
- :card_file_box: **OpenAPI JSON** (schéma machine, utilisable pour générer des clients) : [{api_public_url}/docs.json]({api_public_url}/docs.json)
"""
    )
    st.caption(
        "Accès protégé par Cloudflare Access (One-Time PIN par email). "
        "Si tu es déjà authentifié sur le portfolio, l'accès est immédiat."
    )
else:
    st.info(
        "Variable `CHAMPY_API_PUBLIC_URL` non définie : la documentation "
        "interactive (Swagger UI, ReDoc) n'est accessible qu'en interne. "
        "Pour activer les liens cliquables, exposer l'API via Cloudflare "
        "Tunnel et définir `CHAMPY_API_PUBLIC_URL=https://...` dans le "
        "`docker-compose.yml`."
    )

st.divider()


# =====================================================================
# Section 3 : Informations du modèle
# =====================================================================

st.header("Informations du modèle")

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
    st.info("Informations du modèle non disponibles (API hors ligne ou modèle non chargé).")

st.divider()


# =====================================================================
# Section 4 : Métriques Prometheus brutes
# =====================================================================

st.header("Métriques Prometheus (brutes)")
st.caption(
    "Métriques exposées par l'API au format Prometheus, filtrées sur "
    "le préfixe `champy_*`. Utile pour vérifier la collecte avant "
    "qu'elles soient agrégées dans Grafana."
)

if st.button("Rafraîchir les métriques"):
    st.cache_data.clear()

metrics_text = get_prometheus_metrics()
if metrics_text:
    champy_lines = [line for line in metrics_text.split("\n") if line.startswith("champy_")]
    if champy_lines:
        st.code("\n".join(champy_lines), language="text")
    else:
        st.info(
            "Aucune métrique `champy_*` disponible (aucune prédiction effectuée pour l'instant)."
        )
else:
    st.info("Métriques non disponibles (API hors ligne).")
