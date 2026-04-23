"""Page Streamlit : prédiction interactive.

Upload d'une image de champignon, appel à l'API /predict,
affichage du top-5 avec barres de confiance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="08 - Prédiction", layout="wide")
st.title(":crystal_ball: Prédiction")

st.markdown("""
Uploadez une photo de champignon pour obtenir une prédiction
du modèle. L'image est envoyée à l'API FastAPI qui effectue
l'inférence via ONNX Runtime.
""")

# --- Upload ---
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Image de champignon au format JPEG ou PNG",
)

if uploaded_file is not None:
    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.image(uploaded_file, caption="Image uploadée", use_container_width=True)

    with col_results:
        image_bytes = uploaded_file.getvalue()

        try:
            from demo.lib.api_utils import predict_image

            with st.spinner("Inférence en cours..."):
                result = predict_image(image_bytes, top_n=5)

            if result is None:
                st.error(
                    "API indisponible. Vérifiez que le serveur est démarré "
                    "(uvicorn src.serving.app:app --port 8010)."
                )
            elif "predictions" in result:
                predictions = result["predictions"]
                st.subheader("Résultats")

                # Top-1 en grand
                top1 = predictions[0]
                st.metric(
                    f":trophy: {top1['species']}",
                    f"{top1['confidence']:.1%}",
                )

                st.divider()

                # Barres de confiance pour le top-5
                st.subheader("Top-5 prédictions")
                import pandas as pd
                import plotly.express as px

                df_pred = pd.DataFrame(predictions)
                df_pred["confidence_pct"] = df_pred["confidence"] * 100

                fig = px.bar(
                    df_pred,
                    x="confidence_pct",
                    y="species",
                    orientation="h",
                    title="Confiance par espèce",
                    labels={"confidence_pct": "Confiance (%)", "species": "Espèce"},
                    color="confidence_pct",
                    color_continuous_scale="Greens",
                    range_color=[0, 100],
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Version du modèle
                st.caption(f"Modèle version : {result.get('model_version', '?')}")
            else:
                st.error(f"Réponse inattendue de l'API : {result}")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.info("Uploadez une image pour lancer une prédiction.")
