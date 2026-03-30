"""Page Streamlit : suivi de l'entrainement.

Affiche les courbes d'apprentissage, les hyperparametres et les metriques
depuis MLflow. Fallback sur le fichier JSON local si MLflow est indisponible.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="05 - Entrainement", layout="wide")
st.title(":chart_with_upwards_trend: Entrainement")

# --- Tenter de charger depuis MLflow, sinon fallback local ---
mlflow_available = False
runs = []

try:
    from demo.lib.mlflow_utils import get_metric_history, search_runs

    runs = search_runs(max_results=10, order_by="start_time DESC")
    if runs:
        mlflow_available = True
except Exception as e:
    st.warning(f"MLflow non disponible : {e}")

# --- Fallback : metriques locales ---
local_metrics = None
if not mlflow_available:
    try:
        from demo.lib.mlflow_utils import load_local_metrics

        local_metrics = load_local_metrics()
        if local_metrics:
            st.info("Metriques chargees depuis le fichier local (models/artifacts/metrics.json)")
    except Exception:
        pass

if not mlflow_available and local_metrics is None:
    st.error("Aucune source de metriques disponible (ni MLflow ni fichier local).")
    st.stop()

# =====================================================================
# Section 1 : Liste des runs
# =====================================================================
if mlflow_available and runs:
    st.header("Historique des runs")

    import pandas as pd

    df_runs = pd.DataFrame(runs)
    # Colonnes utiles
    display_cols = [
        c
        for c in df_runs.columns
        if c.startswith(("metrics.", "params.", "run_id", "start_time", "status"))
    ]
    if display_cols:
        st.dataframe(df_runs[display_cols].head(10), use_container_width=True, hide_index=True)

    st.divider()

# =====================================================================
# Section 2 : Courbes d'apprentissage
# =====================================================================
st.header("Courbes d'apprentissage")

if mlflow_available and runs:
    run_id = runs[0].get("run_id", "")
    st.caption(f"Run : {run_id}")

    import plotly.graph_objects as go

    try:
        train_loss_hist = get_metric_history(run_id, "train_loss")
        val_loss_hist = get_metric_history(run_id, "val_loss")

        if train_loss_hist and val_loss_hist:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[m["step"] for m in train_loss_hist],
                    y=[m["value"] for m in train_loss_hist],
                    name="Train loss",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[m["step"] for m in val_loss_hist],
                    y=[m["value"] for m in val_loss_hist],
                    name="Val loss",
                )
            )
            fig.update_layout(
                title="Evolution de la loss", xaxis_title="Epoch", yaxis_title="Loss"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible de charger l'historique des metriques : {e}")

    try:
        val_acc_hist = get_metric_history(run_id, "val_acc")
        val_f1_hist = get_metric_history(run_id, "val_f1_macro")

        if val_acc_hist:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=[m["step"] for m in val_acc_hist],
                    y=[m["value"] for m in val_acc_hist],
                    name="Val accuracy",
                )
            )
            if val_f1_hist:
                fig2.add_trace(
                    go.Scatter(
                        x=[m["step"] for m in val_f1_hist],
                        y=[m["value"] for m in val_f1_hist],
                        name="Val F1 macro",
                    )
                )
            fig2.update_layout(
                title="Evolution des metriques", xaxis_title="Epoch", yaxis_title="Score"
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible de charger les metriques val : {e}")

elif local_metrics and "history" in local_metrics:
    import plotly.graph_objects as go

    history = local_metrics["history"]
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig = go.Figure()
    if "train_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train loss"))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val loss"))
    fig.update_layout(title="Evolution de la loss", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    if "val_acc" in history:
        fig2.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val accuracy"))
    if "val_f1" in history:
        fig2.add_trace(go.Scatter(x=epochs, y=history["val_f1"], name="Val F1 macro"))
    fig2.update_layout(title="Evolution des metriques", xaxis_title="Epoch", yaxis_title="Score")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =====================================================================
# Section 3 : Hyperparametres
# =====================================================================
st.header("Hyperparametres")

if mlflow_available and runs:
    try:
        from demo.lib.mlflow_utils import get_run_params

        params = get_run_params(runs[0]["run_id"])
        if params:
            import pandas as pd

            df_params = pd.DataFrame(
                sorted(params.items()),
                columns=["Parametre", "Valeur"],
            )
            st.dataframe(df_params, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Impossible de charger les hyperparametres : {e}")
elif local_metrics and "report" in local_metrics:
    st.json(local_metrics.get("report", {}))

st.divider()

# =====================================================================
# Section 4 : Metriques finales
# =====================================================================
st.header("Metriques finales")

if mlflow_available and runs:
    metrics = {
        k.replace("metrics.", ""): v for k, v in runs[0].items() if k.startswith("metrics.")
    }
    if metrics:
        cols = st.columns(4)
        for i, (name, value) in enumerate(sorted(metrics.items())):
            col = cols[i % 4]
            if isinstance(value, float):
                col.metric(name, f"{value:.4f}")
            else:
                col.metric(name, str(value))
elif local_metrics:
    col1, col2 = st.columns(2)
    col1.metric("Test accuracy", f"{local_metrics.get('accuracy', 0):.1%}")
    col2.metric("Test F1 macro", f"{local_metrics.get('f1_macro', 0):.1%}")
