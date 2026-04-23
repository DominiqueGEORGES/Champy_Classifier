"""Page Streamlit : visualisation des augmentations PyTorch.

Prend une image au hasard du dataset, applique les transforms
PyTorch en live et affiche l'original et les versions augmentées.
Montre le pipeline d'augmentation configuré.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="03 - Augmentation", layout="wide")
st.title(":twisted_rightwards_arrows: Augmentation des données")

try:
    import numpy as np
    import torch
    from demo.lib.data_utils import get_random_images, load_raw_stats
    from PIL import Image

    from src.data.dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_eval_transforms,
        get_train_transforms,
    )
except Exception as e:
    st.error(f"Impossible de charger les modules nécessaires : {e}")
    st.stop()


def denormalize(tensor: torch.Tensor) -> np.ndarray:  # type: ignore[type-arg]
    """Inverse la normalisation ImageNet pour l'affichage.

    Args:
        tensor: Image normalisée (C, H, W).

    Returns:
        Image en numpy (H, W, C), valeurs [0, 255], dtype uint8.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# --- Description du pipeline ---
st.header("Pipeline d'augmentation")

st.markdown("""
**Train** (augmentation aléatoire à chaque epoch, pipeline renforcé pour les classes rares) :
1. `RandomResizedCrop(224, scale=(0.7, 1.0))` - zoom/cadrage aléatoire (70-100%)
2. `RandomHorizontalFlip()` - retournement horizontal (p=0.5)
3. `RandomAffine(degrees=15, translate=(0.1, 0.1))` - rotation + translation
4. `ColorJitter(0.3, 0.3, 0.3, 0.1)` - luminosité, contraste, saturation, teinte
5. `ToTensor()` + `Normalize(ImageNet)` - normalisation standard
6. `RandomErasing(p=0.25)` - masquage aléatoire d'une région (sur tensor)

**Val / Test** (déterministe, sans augmentation) :
1. `Resize(256)` -> `CenterCrop(224)` -> `ToTensor()` -> `Normalize(ImageNet)`
""")

# Pipeline réel introspecté depuis src.data.dataset (zero hardcoded).
with st.expander("Pipeline réel (introspection de `get_train_transforms()`)"):
    st.code(repr(get_train_transforms(224)), language="text")

st.divider()

# --- Sélection d'une image ---
st.header("Démonstration en live")

try:
    stats = load_raw_stats()
    class_names = sorted(stats.get("class_counts", {}).keys())
except Exception:
    class_names = []

if not class_names:
    st.warning("Aucune classe disponible. Vérifiez data/raw_stats.json.")
    st.stop()

selected_class = st.selectbox("Choisir une espèce", class_names)
n_augmented = st.slider("Nombre de versions augmentées", min_value=2, max_value=8, value=4)

if st.button("Générer les augmentations", type="primary"):
    # Charger une image aléatoire
    images = get_random_images(selected_class, n=1)
    if not images:
        st.warning(f"Aucune image trouvée pour {selected_class}.")
        st.stop()

    img_path = images[0]
    original = Image.open(img_path).convert("RGB")

    st.subheader(f"Image originale : {img_path.name}")
    st.image(
        original, caption=f"{selected_class} - {original.size[0]}x{original.size[1]}", width=300
    )

    st.divider()

    # Appliquer les transforms
    train_transform = get_train_transforms(224)
    eval_transform = get_eval_transforms(224)

    # Version eval (déterministe)
    st.subheader("Version évaluation (déterministe)")
    eval_tensor = eval_transform(original)
    eval_img = denormalize(eval_tensor)
    st.image(eval_img, caption="Resize(256) + CenterCrop(224) + Normalize", width=300)

    st.divider()

    # Versions augmentées (aléatoires)
    st.subheader(f"{n_augmented} versions augmentées (aléatoires)")
    cols = st.columns(min(n_augmented, 4))
    for i in range(n_augmented):
        col = cols[i % len(cols)]
        augmented_tensor = train_transform(original)
        augmented_img = denormalize(augmented_tensor)
        col.image(augmented_img, caption=f"Augmentation #{i + 1}", use_container_width=True)

    st.caption(
        "Chaque version est différente grâce aux transforms aléatoires. "
        "Le modèle voit des variations à chaque epoch, ce qui améliore la généralisation."
    )
