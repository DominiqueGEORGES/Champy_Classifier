"""Page Streamlit : exploration des données brutes.

Lit data/raw_stats.json et affiche la distribution des classes,
les formats, dimensions, et une galerie d'exemples aléatoires.
Zéro valeur hardcodée - tout est lu dynamiquement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from demo import auth

auth.setup_page(min_role="user")  # ou "guest" pour pages publiques

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="01 - Données brutes", layout="wide")
st.title(":file_folder: Données brutes")

try:
    from demo.lib.data_utils import get_random_images, load_raw_stats

    stats = load_raw_stats()
except Exception as e:
    st.error(f"Impossible de charger les statistiques : {e}")
    st.info("Lancez le script d'analyse des données pour générer data/raw_stats.json.")
    st.stop()

# --- Métriques globales ---
st.header("Vue d'ensemble")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total images", f"{stats['total_images']:,}")
col2.metric("Classes", stats["num_classes"])
col3.metric("Corrompues", stats["corrupted_count"])
col4.metric("Doublons", f"{stats['duplicate_groups_count']} paires")

st.divider()

# --- Distribution des classes ---
st.header("Distribution par classe")

class_counts = stats["class_counts"]
if class_counts:
    import pandas as pd
    import plotly.express as px

    df_classes = pd.DataFrame(
        sorted(class_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Espèce", "Nombre d'images"],
    )

    fig = px.bar(
        df_classes,
        x="Nombre d'images",
        y="Espèce",
        orientation="h",
        title="Nombre d'images par espèce (data/processed/)",
        color="Nombre d'images",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Stats de distribution
    counts = list(class_counts.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Min par classe", min(counts))
    col2.metric("Max par classe", max(counts))
    col3.metric("Ratio max/min", f"{max(counts) / min(counts):.2f}")
else:
    st.warning("Aucune donnée de distribution disponible.")

st.divider()

# --- Formats et dimensions ---
st.header("Formats et dimensions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Extensions")
    extensions = stats.get("extensions", {})
    if extensions:
        for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
            st.write(f"**{ext}** : {count:,} images")
    else:
        st.info("Aucune information sur les extensions.")

with col2:
    st.subheader("Dimensions (top 10)")
    dimensions = stats.get("dimensions", {})
    if dimensions:
        import pandas as pd

        df_dims = pd.DataFrame(
            list(dimensions.items())[:10],
            columns=["Dimension", "Nombre"],
        )
        st.dataframe(df_dims, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune information sur les dimensions.")

# Tailles de fichiers
file_sizes = stats.get("file_size_bytes", {})
if file_sizes:
    st.subheader("Taille des fichiers")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min", f"{file_sizes.get('min', 0) / 1024:.1f} KB")
    col2.metric("Moyenne", f"{file_sizes.get('avg', 0) / 1024:.1f} KB")
    col3.metric("Max", f"{file_sizes.get('max', 0) / 1024:.1f} KB")

st.divider()

# --- Galerie d'exemples ---
st.header("Galerie d'exemples")

selected_class = st.selectbox(
    "Choisir une espèce",
    options=sorted(class_counts.keys()) if class_counts else [],
)

if selected_class:
    n_images = st.slider("Nombre d'images", min_value=1, max_value=8, value=4)
    images = get_random_images(selected_class, n=n_images)

    if images:
        cols = st.columns(min(n_images, 4))
        for i, img_path in enumerate(images):
            col = cols[i % len(cols)]
            col.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.warning(f"Aucune image trouvée pour {selected_class}.")
