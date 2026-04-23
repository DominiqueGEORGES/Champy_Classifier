"""Page Streamlit : détection de drift avec Evidently.

Permet de générer un rapport Evidently on-demand en comparant
la distribution des prédictions récentes avec une référence.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="11 - Drift", layout="wide")
st.title(":warning: Détection de drift")

st.markdown("""
La détection de drift surveille si la distribution des données
en production s'écarte de la distribution d'entraînement.

**Types de drift monitorés** :
- **Data drift** : les images soumises changent de distribution
- **Prédiction drift** : la répartition des classes prédites change
- **Confiance drift** : le score de confiance moyen évolue

**Implémentation** : Evidently AI génère des rapports HTML
comparant les données actuelles à une référence.
""")

st.divider()

# =====================================================================
# Section 1 : Statut Evidently
# =====================================================================
st.header("Rapport Evidently")

st.info(
    "La génération de rapports Evidently sera disponible une fois que "
    "l'API aura accumulé suffisamment de prédictions en production. "
    "Le rapport compare la distribution des prédictions récentes avec "
    "la distribution du split test (référence)."
)

# Vérifier si un rapport existe déjà
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "artifacts"
report_path = REPORTS_DIR / "drift_report.html"

if report_path.exists():
    st.success("Un rapport de drift existe.")
    with open(report_path, encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.info("Aucun rapport de drift généré pour l'instant.")

st.divider()

# =====================================================================
# Section 2 : Indicateurs proxy
# =====================================================================
st.header("Indicateurs proxy (depuis Prometheus)")

try:
    from demo.lib.api_utils import query_prometheus

    # Distribution des classes prédites
    predictions = query_prometheus("champy_predictions_total")
    if predictions:
        import pandas as pd

        rows = []
        total = 0.0
        for result in predictions:
            species = result.get("metric", {}).get("species", "?")
            value = float(result.get("value", [0, 0])[1])
            rows.append({"Espèce": species, "Prédictions": int(value)})
            total += value

        if rows and total > 0:
            df = pd.DataFrame(rows)
            df["Proportion"] = df["Prédictions"] / total
            st.dataframe(
                df.sort_values("Prédictions", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Si une classe domine anormalement les prédictions, "
                "cela peut indiquer un drift dans les données soumises."
            )
    else:
        st.info("Aucune donnée de prédiction (Prometheus hors ligne ou aucune requête).")

    # Confiance moyenne
    confidence = query_prometheus(
        "champy_prediction_confidence_sum / champy_prediction_confidence_count"
    )
    if confidence:
        for result in confidence:
            value = float(result.get("value", [0, 0])[1])
            st.metric("Confiance moyenne globale", f"{value:.1%}")
            if value < 0.5:
                st.warning("Confiance moyenne basse - vérifier la qualité des images soumises.")

except Exception as e:
    st.warning(f"Métriques Prometheus non disponibles : {e}")
