"""Page Streamlit : évaluation du modèle.

Affiche la confusion matrix, le F1 par classe, et le rapport
de classification depuis les artefacts locaux ou MLflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="06 - Évaluation", layout="wide")
st.title(":bar_chart: Évaluation du modèle")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "artifacts"

# --- Charger les métriques ---
metrics = None
try:
    from demo.lib.mlflow_utils import load_local_metrics

    metrics = load_local_metrics()
except Exception:
    pass

if metrics is None:
    st.warning("Aucune métrique disponible. Lancez un entraînement d'abord.")
    st.stop()

# =====================================================================
# Section 1 : Métriques globales
# =====================================================================
st.header("Métriques globales")

col1, col2 = st.columns(2)
col1.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
col2.metric("Test F1 macro", f"{metrics.get('f1_macro', 0):.1%}")

st.divider()

# =====================================================================
# Section 2 : Confusion matrix
# =====================================================================
st.header("Matrice de confusion")

cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
if cm_path.exists():
    st.image(
        str(cm_path),
        caption="Matrice de confusion (normalisation par classe)",
        use_container_width=True,
    )
else:
    st.info(
        "Image de la matrice de confusion non disponible (models/artifacts/confusion_matrix.png)."
    )

st.divider()

# =====================================================================
# Section 3 : F1 par classe
# =====================================================================
st.header("F1-score par classe")

report = metrics.get("report", {})
if report:
    import pandas as pd
    import plotly.express as px

    # Extraire les métriques par classe (exclure les moyennes)
    class_metrics = {
        cls: vals
        for cls, vals in report.items()
        if isinstance(vals, dict) and cls not in ("accuracy", "macro avg", "weighted avg")
    }

    if class_metrics:
        rows = []
        for cls, vals in sorted(class_metrics.items()):
            rows.append(
                {
                    "Espèce": cls,
                    "Precision": vals.get("precision", 0),
                    "Recall": vals.get("recall", 0),
                    "F1-score": vals.get("f1-score", 0),
                    "Support": vals.get("support", 0),
                }
            )

        df_class = pd.DataFrame(rows)

        # Graphique F1 par classe
        fig = px.bar(
            df_class.sort_values("F1-score"),
            x="F1-score",
            y="Espèce",
            orientation="h",
            title="F1-score par espèce",
            color="F1-score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
        )
        fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        # Tableau détaillé
        st.subheader("Détail par classe")
        st.dataframe(
            df_class.sort_values("F1-score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        # Identifier les classes faibles
        weak = df_class[df_class["F1-score"] < 0.7].sort_values("F1-score")
        if not weak.empty:
            st.subheader("Classes à améliorer (F1 < 70%)")
            st.dataframe(weak, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun détail par classe dans le rapport.")
else:
    st.info("Rapport de classification non disponible.")

st.divider()

# =====================================================================
# Section 4 : Courbes d'apprentissage
# =====================================================================
st.header("Courbes d'apprentissage")

curves_path = ARTIFACTS_DIR / "learning_curves.png"
if curves_path.exists():
    st.image(
        str(curves_path), caption="Évolution loss et métriques par epoch", use_container_width=True
    )
else:
    st.info("Image des courbes non disponible (models/artifacts/learning_curves.png).")
