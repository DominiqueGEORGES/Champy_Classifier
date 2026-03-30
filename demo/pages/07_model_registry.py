"""Page Streamlit : model registry et export ONNX.

Affiche les versions du modele, le benchmark PyTorch vs ONNX,
et les metadonnees du modele exporte.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="07 - Model Registry", layout="wide")
st.title(":package: Model Registry & Export ONNX")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# =====================================================================
# Section 1 : Fichiers modele disponibles
# =====================================================================
st.header("Modeles disponibles")

model_files = []
if MODELS_DIR.exists():
    for ext in ("*.pt", "*.onnx"):
        for f in MODELS_DIR.glob(ext):
            model_files.append(
                {
                    "Fichier": f.name,
                    "Format": f.suffix.upper(),
                    "Taille (MB)": round(f.stat().st_size / 1024 / 1024, 1),
                }
            )

if model_files:
    import pandas as pd

    st.dataframe(pd.DataFrame(model_files), use_container_width=True, hide_index=True)
else:
    st.warning("Aucun modele trouve dans models/.")

st.divider()

# =====================================================================
# Section 2 : Checkpoint PyTorch
# =====================================================================
st.header("Checkpoint PyTorch")

checkpoint_path = MODELS_DIR / "best_model.pt"
if checkpoint_path.exists():
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Epoch", ckpt.get("epoch", "?"))
        col2.metric("Best val_loss", f"{ckpt.get('best_score', 0):.4f}")
        col3.metric("Taille", f"{checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        st.warning(f"Impossible de lire le checkpoint : {e}")
else:
    st.info("Pas de checkpoint PyTorch (models/best_model.pt).")

st.divider()

# =====================================================================
# Section 3 : Modele ONNX
# =====================================================================
st.header("Modele ONNX")

onnx_path = MODELS_DIR / "best_model.onnx"
class_names_path = MODELS_DIR / "class_names.json"

if onnx_path.exists():
    col1, col2 = st.columns(2)
    col1.metric("Taille ONNX", f"{onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    if class_names_path.exists():
        with open(class_names_path, encoding="utf-8") as f:
            class_names = json.load(f)
        col2.metric("Classes", len(class_names))

        with st.expander("Liste des classes"):
            for i, name in enumerate(class_names):
                st.write(f"{i:2d}. {name}")
    else:
        col2.info("class_names.json non disponible")

    # Validation ONNX
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        st.success("Validation ONNX : modele valide")

        # Afficher les entrees/sorties
        st.subheader("Architecture ONNX")
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            st.write(f"**Entree** : `{inp.name}` - shape {shape}")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            st.write(f"**Sortie** : `{out.name}` - shape {shape}")
    except ImportError:
        st.info("Module onnx non installe, validation non disponible.")
    except Exception as e:
        st.error(f"Validation ONNX echouee : {e}")
else:
    st.info(
        "Pas de modele ONNX (models/best_model.onnx). Lancez `python -m src.models.export_onnx`."
    )

st.divider()

# =====================================================================
# Section 4 : Benchmark PyTorch vs ONNX
# =====================================================================
st.header("Benchmark inference")

if onnx_path.exists() and checkpoint_path.exists():
    st.markdown("""
    Le modele ONNX doit produire des sorties identiques au modele PyTorch.
    La comparaison est effectuee lors de l'export (voir logs de `export_onnx.py`).

    **Avantages ONNX en production** :
    - Pas de dependance PyTorch (~2 GB) - seul ONNX Runtime (~50 MB) suffit
    - Inference CPU optimisee (quantification possible)
    - Portable (Python, C++, C#, Java, JavaScript)
    """)

    col1, col2 = st.columns(2)
    col1.metric("PyTorch (.pt)", f"{checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    col2.metric("ONNX (.onnx)", f"{onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    ratio = checkpoint_path.stat().st_size / max(onnx_path.stat().st_size, 1)
    st.metric("Ratio compression", f"{ratio:.1f}x")
