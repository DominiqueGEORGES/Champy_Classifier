"""Page Streamlit : prediction interactive avec Grad-CAM.

Trois sources d'image au choix (onglets) :
    - Upload (drag and drop d'un fichier local)
    - Browse dans un dossier d'exemples locaux (data/sample/, data/unseen/, etc.)
    - URL distante (telechargement direct ou extraction og:image d'une page)

L'image courante est conservee dans st.session_state pour ne reagir qu'aux
actions explicites de l'utilisateur (nouvel upload, clic sur "Utiliser cet
exemple", clic sur "Telecharger"). Cela evite que le selectbox de l'onglet
"Exemples locaux" n'ecrase un upload, puisque chaque `with tab_X:` est
re-execute a chaque rerun de Streamlit.

L'inference et le Grad-CAM ne sont declenches que sur clic explicite du
bouton "Lancer la prediction" (pattern UX clair : action utilisateur ->
appel API). Conformement au pattern MLOps strict, la prediction et le
calcul Grad-CAM sont entierement delegues a l'API BentoML (endpoints
POST /predict et POST /explain). La page ne charge aucun modele en local,
ne realise aucune inference et ne persiste aucune donnee.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import base64
import hashlib
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

# =====================================================================
# Configuration du sys.path (avant les imports projet)
# =====================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =====================================================================
# Imports tiers
# =====================================================================

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =====================================================================
# Imports projet
# =====================================================================
from demo import auth
from demo.lib.api_utils import explain_image, predict_image
from PIL import Image as PILImage

# =====================================================================
# Configuration de la page (DOIT etre la premiere commande Streamlit)
# =====================================================================

st.set_page_config(page_title="08 - Prediction", layout="wide")

# =====================================================================
# Authentification (lit access_policy.yaml automatiquement)
# =====================================================================

auth.setup_page()

# =====================================================================
# Constantes
# =====================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES_IN_BROWSE = 100

SAMPLE_DIRECTORIES: list[tuple[str, Path]] = [
    ("Echantillon curé (data/sample)", Path("data/sample")),
    ("Images jamais vues (data/unseen)", Path("data/unseen")),
    ("Dataset brut (data/raw/Mushrooms_images)", Path("data/raw/Mushrooms_images")),
]

# Cles session_state utilisees par la page
SS_IMAGE = "pred_selected_image"
SS_SOURCE = "pred_selected_source"
SS_UPLOAD_HASH = "pred_last_upload_hash"
SS_LAST_RESULT = "pred_last_result"
SS_LAST_EXPLAIN = "pred_last_explain"


# =====================================================================
# Helpers
# =====================================================================


def _hash_bytes(data: bytes) -> str:
    """Hash MD5 pour detecter les changements d'image."""
    return hashlib.md5(data).hexdigest()


def _set_selected_image(image_bytes: bytes, source: str) -> None:
    """Enregistre l'image active et invalide les caches de resultats."""
    st.session_state[SS_IMAGE] = image_bytes
    st.session_state[SS_SOURCE] = source
    # Nouvelle image -> invalider les resultats precedents
    st.session_state[SS_LAST_RESULT] = None
    st.session_state[SS_LAST_EXPLAIN] = None


@st.cache_data(ttl=300, show_spinner="Indexation des images...")
def list_images_in_directory(
    directory_str: str, max_images: int = MAX_IMAGES_IN_BROWSE
) -> list[str]:
    """Liste les images d'un dossier avec early-break."""
    directory = Path(directory_str)
    if not directory.exists() or not directory.is_dir():
        return []
    images: list[str] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(str(p))
            if len(images) >= max_images:
                break
    images.sort()
    return images


def _extract_image_url_from_html(html_content: str) -> str | None:
    """Cherche une URL d'image dans une page HTML (Open Graph en priorite)."""
    patterns = [
        r'<meta\s+[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
        r'<meta\s+[^>]*name=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _download_image_or_extract(url: str) -> bytes | None:
    """Telecharge une image directe ou extrait l'URL og:image d'une page."""
    headers = {"User-Agent": "Champy/1.0 (Mozilla/5.0)"}
    try:
        with st.spinner("Téléchargement..."):
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if content_type.startswith("image/"):
                return response.content

            if content_type.startswith("text/html"):
                image_url = _extract_image_url_from_html(response.text)
                if not image_url:
                    st.error(
                        "Aucune image principale trouvée dans la page. "
                        "Essayez de coller directement l'URL de l'image "
                        "(clic droit > 'Copier l'adresse de l'image')."
                    )
                    return None
                st.info(f"Image extraite de la page : {image_url}")
                image_response = requests.get(image_url, timeout=15, headers=headers)
                image_response.raise_for_status()
                image_ct = image_response.headers.get("Content-Type", "").lower()
                if not image_ct.startswith("image/"):
                    st.error(f"L'URL extraite ne renvoie pas une image : {image_ct}")
                    return None
                return image_response.content

            st.error(f"Type de contenu non supporté : {content_type}")
            return None

    except requests.exceptions.RequestException as exc:
        st.error(f"Échec du téléchargement : {exc}")
        return None
    except Exception as exc:
        st.error(f"Erreur inattendue : {exc}")
        return None


# =====================================================================
# Initialisation session_state
# =====================================================================

for key in (SS_IMAGE, SS_SOURCE, SS_UPLOAD_HASH, SS_LAST_RESULT, SS_LAST_EXPLAIN):
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================================
# Titre et description
# =====================================================================

st.title(":crystal_ball: Prédiction")

st.markdown(
    """
Choisissez une image dans l'un des trois onglets ci-dessous, puis cliquez
sur **Lancer la prédiction**. La case Grad-CAM permet d'obtenir en plus
une carte de chaleur des zones de l'image qui ont influencé le modèle.
"""
)


# =====================================================================
# Section 1 : Selection de la source d'image
# =====================================================================

st.subheader("1. Sélection de l'image")

tab_upload, tab_browse, tab_url = st.tabs(
    [
        ":outbox_tray: Upload",
        ":file_folder: Exemples locaux",
        ":globe_with_meridians: URL",
    ]
)

# --- Onglet Upload : declenche quand un NOUVEAU fichier est uploade ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Glissez-déposez ou cliquez pour sélectionner une image de champignon.",
        key="pred_uploader",
    )
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        new_hash = _hash_bytes(data)
        # Ne declencher la mise a jour que si c'est un nouveau fichier
        if new_hash != st.session_state[SS_UPLOAD_HASH]:
            st.session_state[SS_UPLOAD_HASH] = new_hash
            _set_selected_image(data, f"Upload : {uploaded_file.name}")
            st.success(f"Image uploadée : `{uploaded_file.name}`")

# --- Onglet Exemples : bouton explicite pour activer une image ---
with tab_browse:
    available_dirs = [
        (label, path) for label, path in SAMPLE_DIRECTORIES if path.exists() and path.is_dir()
    ]
    if not available_dirs:
        st.warning(
            "Aucun dossier d'exemples disponible. Créez par exemple "
            "`data/sample/` et placez-y quelques images."
        )
    else:
        dir_labels = [label for label, _ in available_dirs]
        dir_choice = st.selectbox("Dossier d'images", dir_labels, key="browse_dir")
        dir_path = dict(available_dirs)[dir_choice]

        image_paths = [Path(p) for p in list_images_in_directory(str(dir_path))]
        if not image_paths:
            st.info(f"Dossier vide ou aucune image reconnue dans {dir_path}.")
        else:
            if len(image_paths) >= MAX_IMAGES_IN_BROWSE:
                st.caption(
                    f"Affichage limité aux {MAX_IMAGES_IN_BROWSE} premières images du dossier."
                )

            image_labels = [p.name for p in image_paths]
            chosen_label = st.selectbox("Image", image_labels, key="browse_image")
            chosen_path = image_paths[image_labels.index(chosen_label)]

            if st.button(
                "Utiliser cet exemple",
                key="use_browse_button",
                type="primary",
            ):
                _set_selected_image(
                    chosen_path.read_bytes(),
                    f"Exemple : {chosen_label}",
                )
                st.success(f"Image sélectionnée : `{chosen_label}`")

# --- Onglet URL : bouton explicite pour telecharger ---
with tab_url:
    url = st.text_input(
        "URL de l'image ou de la page web",
        placeholder="https://commons.wikimedia.org/wiki/File:... ou URL directe",
        help=(
            "Collez l'URL d'une image, OU l'URL d'une page web contenant une "
            "image (Wikipedia, Wikimedia, iNaturalist, etc.)."
        ),
        key="pred_url_input",
    )

    if url:
        if not (url.startswith("http://") or url.startswith("https://")):
            st.error("L'URL doit commencer par http:// ou https://")
        elif st.button(
            "Télécharger et utiliser",
            key="download_button",
            type="primary",
        ):
            downloaded = _download_image_or_extract(url)
            if downloaded:
                _set_selected_image(downloaded, "URL distante")
                st.success("Image téléchargée.")


# =====================================================================
# Section 2 : Image active et bouton de lancement
# =====================================================================

image_bytes: bytes | None = st.session_state[SS_IMAGE]
source_label: str = st.session_state[SS_SOURCE] or ""

if image_bytes is None:
    st.info(
        "Sélectionnez une image dans l'un des trois onglets ci-dessus, puis "
        "cliquez sur **Lancer la prédiction**."
    )
    st.stop()

st.divider()
st.subheader("2. Image active")

col_preview, col_action = st.columns([1, 2])
with col_preview:
    st.image(image_bytes, caption=f"Source : {source_label}", width=280)

with col_action:
    st.markdown(f"**Source** : {source_label}")
    st.markdown(f"**Taille** : {len(image_bytes) / 1024:.1f} Ko")
    st.markdown("")

    show_gradcam = st.checkbox(
        "Inclure l'explication Grad-CAM",
        value=True,
        help=(
            "Visualise les zones de l'image qui ont le plus influencé la "
            "prédiction. Calcul plus lent (~500 ms à 5 s selon initialisation)."
        ),
    )

    run_prediction = st.button(
        "🚀 Lancer la prédiction",
        type="primary",
        use_container_width=True,
    )

# Si pas de clic et pas de resultat en cache, on s'arrete ici
if not run_prediction and st.session_state[SS_LAST_RESULT] is None:
    st.info("Cliquez sur **Lancer la prédiction** pour interroger le modèle.")
    st.stop()


# =====================================================================
# Section 3 : Inference (uniquement si bouton cliqué ou résultat en cache)
# =====================================================================

if run_prediction:
    try:
        with st.spinner("Inférence via l'API BentoML..."):
            result = predict_image(image_bytes, top_n=5)
        if result is None or "predictions" not in result:
            st.error(
                "API indisponible ou réponse inattendue. Vérifie que le "
                "service `api` est démarré et accessible."
            )
            st.stop()
        st.session_state[SS_LAST_RESULT] = result
    except Exception as exc:
        st.error(f"Erreur lors de la prédiction : {exc}")
        st.stop()

result = st.session_state[SS_LAST_RESULT]
predictions: list[dict[str, Any]] = result["predictions"]
top1 = predictions[0]

st.divider()
st.subheader("3. Résultats")

col_top, col_chart = st.columns([1, 2])
with col_top:
    st.metric(f":trophy: {top1['species']}", f"{top1['confidence']:.1%}")
    st.caption(f"Modèle : `{result.get('model_version', '?')}`")

with col_chart:
    df_pred = pd.DataFrame(predictions)
    df_pred["confidence_pct"] = df_pred["confidence"] * 100
    fig = px.bar(
        df_pred,
        x="confidence_pct",
        y="species",
        orientation="h",
        title="Top-5 — Confiance par espèce",
        labels={"confidence_pct": "Confiance (%)", "species": "Espèce"},
        color="confidence_pct",
        color_continuous_scale="Greens",
        range_color=[0, 100],
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# Section 4 : Grad-CAM (si demandé)
# =====================================================================

if show_gradcam:
    st.divider()
    st.subheader("4. Explication Grad-CAM")

    if run_prediction:
        with st.spinner("Calcul Grad-CAM via l'API..."):
            explain_result = explain_image(image_bytes, target_class_id=-1)
        if explain_result is None:
            st.error(
                "Grad-CAM indisponible : l'API n'a pas répondu. Vérifie que "
                "le modèle PyTorch (`models/best_model.pt`) est bien monté "
                "dans le container `champy_api`."
            )
        else:
            st.session_state[SS_LAST_EXPLAIN] = explain_result

    explain_result = st.session_state[SS_LAST_EXPLAIN]
    if explain_result is not None:

        def _decode_b64(b64_str: str) -> PILImage.Image:
            """Décode une chaîne base64 en image PIL (helper Grad-CAM)."""
            return PILImage.open(BytesIO(base64.b64decode(b64_str)))

        st.caption(
            f"Classe expliquée : **{explain_result['target_class_name']}** "
            f"(id={explain_result['target_class_id']})"
        )

        col_orig, col_heatmap, col_overlay = st.columns(3)
        with col_orig:
            st.image(
                _decode_b64(explain_result["original_b64"]),
                caption="Image redimensionnée (224x224)",
                use_container_width=True,
            )
        with col_heatmap:
            st.image(
                _decode_b64(explain_result["heatmap_b64"]),
                caption="Heatmap brute",
                use_container_width=True,
            )
        with col_overlay:
            st.image(
                _decode_b64(explain_result["overlay_b64"]),
                caption="Superposition (zones rouges/jaunes = influentes)",
                use_container_width=True,
            )
