"""Page Streamlit : detection de drift avec Evidently.

Permet de generer un rapport Evidently on-demand en comparant
la distribution des predictions recentes avec une reference.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="11 - Drift", layout="wide")
st.title(":warning: Detection de drift")

st.markdown("""
La detection de drift surveille si la distribution des donnees
en production s'ecarte de la distribution d'entrainement.

**Types de drift monitores** :
- **Data drift** : les images soumises changent de distribution
- **Prediction drift** : la repartition des classes predites change
- **Confiance drift** : le score de confiance moyen evolue

**Implementation** : Evidently AI genere des rapports HTML
comparant les donnees actuelles a une reference.
""")

st.divider()

# =====================================================================
# Section 1 : Statut Evidently
# =====================================================================
st.header("Rapport Evidently")

st.info(
    "La generation de rapports Evidently sera disponible une fois que "
    "l'API aura accumule suffisamment de predictions en production. "
    "Le rapport compare la distribution des predictions recentes avec "
    "la distribution du split test (reference)."
)

# Verifier si un rapport existe deja
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "artifacts"
report_path = REPORTS_DIR / "drift_report.html"

if report_path.exists():
    st.success("Un rapport de drift existe.")
    with open(report_path, encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.info("Aucun rapport de drift genere pour l'instant.")

st.divider()

# =====================================================================
# Section 2 : Indicateurs proxy
# =====================================================================
st.header("Indicateurs proxy (depuis Prometheus)")

try:
    from demo.lib.api_utils import query_prometheus

    # Distribution des classes predites
    predictions = query_prometheus("champy_predictions_total")
    if predictions:
        import pandas as pd

        rows = []
        total = 0.0
        for result in predictions:
            species = result.get("metric", {}).get("species", "?")
            value = float(result.get("value", [0, 0])[1])
            rows.append({"Espece": species, "Predictions": int(value)})
            total += value

        if rows and total > 0:
            df = pd.DataFrame(rows)
            df["Proportion"] = df["Predictions"] / total
            st.dataframe(
                df.sort_values("Predictions", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Si une classe domine anormalement les predictions, "
                "cela peut indiquer un drift dans les donnees soumises."
            )
    else:
        st.info("Aucune donnee de prediction (Prometheus hors ligne ou aucune requete).")

    # Confiance moyenne
    confidence = query_prometheus(
        "champy_prediction_confidence_sum / champy_prediction_confidence_count"
    )
    if confidence:
        for result in confidence:
            value = float(result.get("value", [0, 0])[1])
            st.metric("Confiance moyenne globale", f"{value:.1%}")
            if value < 0.5:
                st.warning("Confiance moyenne basse - verifier la qualite des images soumises.")

except Exception as e:
    st.warning(f"Metriques Prometheus non disponibles : {e}")
