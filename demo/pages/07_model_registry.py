"""Page Streamlit : model registry et export ONNX.

Affiche les versions du modèle, les métadonnées du checkpoint PyTorch
et la validation du modèle ONNX. Conformément au pattern MLOps strict,
toutes les données proviennent de l'API BentoML via /model/registry :
aucun accès direct au filesystem côté Streamlit, aucun chargement de
torch ou de onnx en local.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from demo import auth

auth.setup_page(min_role="user")

import pandas as pd
import streamlit as st
from demo.lib.api_utils import get_model_registry

st.set_page_config(page_title="07 - Model Registry", layout="wide")
st.title(":package: Model Registry & Export ONNX")

# =====================================================================
# Récupération centralisée via API (un seul appel HTTP pour toute la page)
# =====================================================================
with st.spinner("Récupération de l'inventaire des modèles via l'API..."):
    registry = get_model_registry()

if registry is None:
    st.error(
        "L'API n'a pas répondu pour `/model/registry`. "
        "Vérifier que le service `champy_api` est sain."
    )
    st.stop()

models = registry.get("models", [])
checkpoint = registry.get("checkpoint")
onnx_validation = registry.get("onnx_validation")
num_classes = registry.get("num_classes")
class_names = registry.get("class_names", [])

pt_size = next((m["size_mb"] for m in models if m["filename"] == "best_model.pt"), None)
onnx_size = next((m["size_mb"] for m in models if m["filename"] == "best_model.onnx"), None)

# =====================================================================
# Section 1 : Fichiers modèle disponibles
# =====================================================================
st.header("Modèles disponibles")

if models:
    df_models = pd.DataFrame(
        [
            {
                "Fichier": m["filename"],
                "Format": m["format"],
                "Taille (MB)": m["size_mb"],
            }
            for m in models
        ]
    )
    st.dataframe(df_models, use_container_width=True, hide_index=True)
else:
    st.warning("Aucun modèle trouvé.")

st.divider()

# =====================================================================
# Section 2 : Checkpoint PyTorch
# =====================================================================
st.header("Checkpoint PyTorch")

if checkpoint and pt_size is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Epoch", checkpoint.get("epoch") or "?")
    col2.metric("Best val_loss", f"{checkpoint.get('best_score', 0):.4f}")
    col3.metric("Taille", f"{pt_size:.1f} MB")
else:
    st.info("Pas de checkpoint PyTorch disponible (`models/best_model.pt`).")

st.divider()

# =====================================================================
# Section 3 : Modèle ONNX
# =====================================================================
st.header("Modèle ONNX")

if onnx_size is not None:
    col1, col2 = st.columns(2)
    col1.metric("Taille ONNX", f"{onnx_size:.1f} MB")
    if num_classes:
        col2.metric("Classes", num_classes)
        with st.expander("Liste des classes"):
            for i, name in enumerate(class_names):
                st.write(f"{i:2d}. {name}")
    else:
        col2.info("Liste des classes non disponible")

    if onnx_validation:
        if onnx_validation.get("valid"):
            st.success("Validation ONNX : modèle valide")
            st.subheader("Architecture ONNX")
            inp_shape = onnx_validation.get("input_shape")
            out_shape = onnx_validation.get("output_shape")
            if inp_shape:
                st.write(f"**Entrée** : shape `{inp_shape}`")
            if out_shape:
                st.write(f"**Sortie** : shape `{out_shape}`")
        else:
            st.error(
                f"Validation ONNX échouée : {onnx_validation.get('error', 'erreur inconnue')}"
            )
else:
    st.info(
        "Pas de modèle ONNX (`models/best_model.onnx`). Lancez `python -m src.models.export_onnx`."
    )

st.divider()

# =====================================================================
# Section 4 : Benchmark PyTorch vs ONNX
# =====================================================================
st.header("Benchmark inférence")

if onnx_size is not None and pt_size is not None:
    st.markdown(
        """
    Le modèle ONNX doit produire des sorties identiques au modèle PyTorch.
    La comparaison est effectuée lors de l'export (voir logs de `export_onnx.py`).

    **Avantages ONNX en production** :
    - Pas de dépendance PyTorch (~2 GB) — seul ONNX Runtime (~50 MB) suffit
    - Inférence CPU optimisée (quantification possible)
    - Portable (Python, C++, C#, Java, JavaScript)
        """
    )

    col1, col2 = st.columns(2)
    col1.metric("PyTorch (.pt)", f"{pt_size:.1f} MB")
    col2.metric("ONNX (.onnx)", f"{onnx_size:.1f} MB")

    ratio = pt_size / max(onnx_size, 0.001)
    st.metric("Ratio compression", f"{ratio:.1f}x")
