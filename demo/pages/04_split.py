"""Page Streamlit : split stratifie train/val/test.

Lit data/split_stats.json et data/split_manifest.csv pour afficher
la distribution par classe par split, la verification de la
stratification, et le desequilibre naturel du dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="04 - Split", layout="wide")
st.title(":scissors: Split stratifie")

try:
    from demo.lib.data_utils import load_split_stats

    stats = load_split_stats()
except Exception as e:
    st.error(f"Impossible de charger les statistiques de split : {e}")
    st.stop()

# --- Parametres du split ---
st.header("Parametres")

col1, col2, col3, col4 = st.columns(4)
ratios = stats.get("ratios", {})
col1.metric("Ratio train", f"{ratios.get('train', 0):.0%}")
col2.metric("Ratio val", f"{ratios.get('val', 0):.0%}")
col3.metric("Ratio test", f"{ratios.get('test', 0):.0%}")
col4.metric("Seed", stats.get("seed", "?"))

st.divider()

# --- Tailles des splits ---
st.header("Taille des splits")

splits = stats.get("splits", {})
total = stats.get("total", 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train", f"{splits.get('train', 0):,}")
col2.metric("Validation", f"{splits.get('val', 0):,}")
col3.metric("Test", f"{splits.get('test', 0):,}")
col4.metric("Total", f"{total:,}")

st.divider()

# --- Distribution par classe par split ---
st.header("Distribution par classe et par split")

per_class = stats.get("per_class", {})
if per_class:
    import pandas as pd
    import plotly.express as px

    # Construire un DataFrame long pour le graphique
    rows = []
    for cls, counts in sorted(per_class.items()):
        for split_name in ("train", "val", "test"):
            rows.append(
                {
                    "Espece": cls,
                    "Split": split_name,
                    "Nombre": counts.get(split_name, 0),
                }
            )

    df_split = pd.DataFrame(rows)

    fig = px.bar(
        df_split,
        x="Nombre",
        y="Espece",
        color="Split",
        orientation="h",
        title="Images par classe et par split",
        barmode="stack",
        color_discrete_map={"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"},
    )
    fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Verification de la stratification ---
    st.header("Verification de la stratification")

    strat_rows = []
    for cls, counts in sorted(per_class.items()):
        cls_total = counts.get("total", 1)
        strat_rows.append(
            {
                "Espece": cls,
                "Total": cls_total,
                "Train %": round(counts.get("train", 0) / cls_total * 100, 1),
                "Val %": round(counts.get("val", 0) / cls_total * 100, 1),
                "Test %": round(counts.get("test", 0) / cls_total * 100, 1),
            }
        )

    df_strat = pd.DataFrame(strat_rows)
    st.dataframe(df_strat, use_container_width=True, hide_index=True)

    # Stats de stratification
    train_pcts = df_strat["Train %"]
    val_pcts = df_strat["Val %"]
    test_pcts = df_strat["Test %"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Train % (min-max)", f"{train_pcts.min():.1f}% - {train_pcts.max():.1f}%")
    col2.metric("Val % (min-max)", f"{val_pcts.min():.1f}% - {val_pcts.max():.1f}%")
    col3.metric("Test % (min-max)", f"{test_pcts.min():.1f}% - {test_pcts.max():.1f}%")

    st.divider()

    # --- Desequilibre naturel ---
    st.header("Desequilibre naturel du dataset")

    totals = {cls: counts["total"] for cls, counts in per_class.items()}
    sorted_totals = sorted(totals.items(), key=lambda x: x[1])

    min_cls, min_count = sorted_totals[0]
    max_cls, max_count = sorted_totals[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Classe la plus petite", f"{min_cls}", f"{min_count} images")
    col2.metric("Classe la plus grande", f"{max_cls}", f"{max_count} images")
    col3.metric("Ratio desequilibre", f"{max_count / min_count:.1f}x")

    st.markdown("""
    **Strategie d'equilibrage** : `WeightedRandomSampler` au training.
    Chaque classe a un poids inversement proportionnel a sa taille,
    de sorte que les classes rares sont surechantillonnees a chaque epoch.
    """)
else:
    st.warning("Aucune donnee de distribution par classe disponible.")
