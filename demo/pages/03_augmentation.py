"""Page Streamlit : visualisation des augmentations PyTorch.

Prend une image au hasard du dataset, applique les transforms
PyTorch en live et affiche l'original et les versions augmentees.
Montre le pipeline d'augmentation configure.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="03 - Augmentation", layout="wide")
st.title(":twisted_rightwards_arrows: Augmentation des donnees")

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
    st.error(f"Impossible de charger les modules necessaires : {e}")
    st.stop()


def denormalize(tensor: torch.Tensor) -> np.ndarray:  # type: ignore[type-arg]
    """Inverse la normalisation ImageNet pour l'affichage.

    Args:
        tensor: Image normalisee (C, H, W).

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
**Train** (augmentation aleatoire a chaque epoch) :
1. `Resize(256)` - redimensionne le cote le plus petit a 256px
2. `RandomCrop(224)` - crop aleatoire 224x224
3. `RandomHorizontalFlip()` - retournement horizontal (p=0.5)
4. `RandomRotation(15)` - rotation aleatoire [-15, +15] degres
5. `ColorJitter(0.2, 0.2, 0.2)` - variations de luminosite, contraste, saturation
6. `ToTensor()` + `Normalize(ImageNet)` - normalisation standard

**Val / Test** (deterministe, sans augmentation) :
1. `Resize(256)` -> `CenterCrop(224)` -> `ToTensor()` -> `Normalize(ImageNet)`
""")

st.divider()

# --- Selection d'une image ---
st.header("Demonstration en live")

try:
    stats = load_raw_stats()
    class_names = sorted(stats.get("class_counts", {}).keys())
except Exception:
    class_names = []

if not class_names:
    st.warning("Aucune classe disponible. Verifiez data/raw_stats.json.")
    st.stop()

selected_class = st.selectbox("Choisir une espece", class_names)
n_augmented = st.slider("Nombre de versions augmentees", min_value=2, max_value=8, value=4)

if st.button("Generer les augmentations", type="primary"):
    # Charger une image aleatoire
    images = get_random_images(selected_class, n=1)
    if not images:
        st.warning(f"Aucune image trouvee pour {selected_class}.")
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

    # Version eval (deterministe)
    st.subheader("Version evaluation (deterministe)")
    eval_tensor = eval_transform(original)
    eval_img = denormalize(eval_tensor)
    st.image(eval_img, caption="Resize(256) + CenterCrop(224) + Normalize", width=300)

    st.divider()

    # Versions augmentees (aleatoires)
    st.subheader(f"{n_augmented} versions augmentees (aleatoires)")
    cols = st.columns(min(n_augmented, 4))
    for i in range(n_augmented):
        col = cols[i % len(cols)]
        augmented_tensor = train_transform(original)
        augmented_img = denormalize(augmented_tensor)
        col.image(augmented_img, caption=f"Augmentation #{i + 1}", use_container_width=True)

    st.caption(
        "Chaque version est differente grace aux transforms aleatoires. "
        "Le modele voit des variations a chaque epoch, ce qui ameliore la generalisation."
    )
