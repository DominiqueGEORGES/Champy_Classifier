"""Page Streamlit : nettoyage et exclusions.

Lit data/cleaning_report.json et data/excluded.json pour afficher
le bilan avant/après, les raisons d'exclusion, et le détail
des fichiers exclus. Zéro valeur hardcodée.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="02 - Nettoyage", layout="wide")
st.title(":broom: Nettoyage des données")

try:
    from demo.lib.data_utils import load_cleaning_report, load_excluded

    report = load_cleaning_report()
    excluded = load_excluded()
except Exception as e:
    st.error(f"Impossible de charger les rapports de nettoyage : {e}")
    st.stop()

# --- Politique de nettoyage ---
st.info(f"**Politique** : {report.get('policy', 'Non définie')}")

st.divider()

# --- Métriques avant/après ---
st.header("Avant / Après")

before = report.get("before", {})
after = report.get("after", {})

col1, col2, col3 = st.columns(3)
col1.metric(
    "Images avant",
    f"{before.get('total_images', 0):,}",
)
col2.metric(
    "Images après",
    f"{after.get('total_images', 0):,}",
    delta=f"-{before.get('total_images', 0) - after.get('total_images', 0):,}",
    delta_color="inverse",
)
col3.metric("Exclues", f"{report.get('excluded_count', 0):,}")

st.divider()

# --- Raisons d'exclusion ---
st.header("Raisons d'exclusion")

reasons = report.get("exclusion_reasons", {})
if reasons:
    import pandas as pd

    df_reasons = pd.DataFrame(
        sorted(reasons.items(), key=lambda x: -x[1]),
        columns=["Raison", "Nombre"],
    )
    st.dataframe(df_reasons, use_container_width=True, hide_index=True)

    import plotly.express as px

    fig = px.pie(
        df_reasons,
        values="Nombre",
        names="Raison",
        title="Répartition des exclusions par raison",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aucune exclusion enregistrée.")

st.divider()

# --- Distribution par classe après nettoyage ---
st.header("Distribution par classe après nettoyage")

after_counts = after.get("class_counts", {})
if after_counts:
    import pandas as pd
    import plotly.express as px

    df_after = pd.DataFrame(
        sorted(after_counts.items(), key=lambda x: x[1]),
        columns=["Espèce", "Nombre d'images"],
    )

    fig = px.bar(
        df_after,
        x="Nombre d'images",
        y="Espèce",
        orientation="h",
        title="Images retenues par classe (originaux uniquement)",
        color="Nombre d'images",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    counts_list = list(after_counts.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Min par classe", min(counts_list))
    col2.metric("Max par classe", max(counts_list))
    col3.metric("Ratio max/min", f"{max(counts_list) / min(counts_list):.1f}x")

st.divider()

# --- Détail des exclusions ---
st.header("Détail des fichiers exclus")

if excluded:
    import pandas as pd

    # Filtre par raison
    all_reasons = sorted({e.get("reason", "inconnu") for e in excluded})
    selected_reason = st.selectbox("Filtrer par raison", ["Toutes", *all_reasons])

    filtered = excluded
    if selected_reason != "Toutes":
        filtered = [e for e in excluded if e.get("reason") == selected_reason]

    st.write(f"**{len(filtered)}** fichiers affiches sur {len(excluded)} total")

    df_excluded = pd.DataFrame(filtered)
    # Limiter l'affichage à 200 lignes pour la performance
    st.dataframe(
        df_excluded.head(200),
        use_container_width=True,
        hide_index=True,
    )
    if len(filtered) > 200:
        st.caption(f"Affichage limite a 200 lignes sur {len(filtered)}.")
else:
    st.info("Aucun fichier exclu.")
