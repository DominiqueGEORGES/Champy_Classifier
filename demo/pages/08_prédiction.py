"""Page Streamlit : prediction interactive avec Grad-CAM.

Trois sources d'image au choix (onglets) :
    - Upload (drag and drop d'un fichier local)
    - Galerie d'exemples locaux (data/sample/, data/unseen/, etc.) : un clic sur
      une vignette selectionne l'image ET lance la prediction (choix visuel direct)
    - URL distante (telechargement direct ou extraction og:image d'une page)

L'image courante est conservee dans st.session_state pour ne reagir qu'aux
actions explicites de l'utilisateur (nouvel upload, clic sur une vignette,
clic sur "Telecharger"). Cela evite que la galerie de l'onglet "Exemples
locaux" n'ecrase un upload, puisque chaque `with tab_X:` est re-execute a
chaque rerun de Streamlit.

L'inference et le Grad-CAM ne sont declenches que sur action explicite :
soit un clic sur une vignette (auto-run), soit le bouton "Lancer la
prediction" (upload / URL). Conformement au pattern MLOps strict, la
prediction et le calcul Grad-CAM sont entierement delegues a l'API BentoML
(endpoints POST /predict et POST /explain). La page ne charge aucun modele
en local, ne realise aucune inference et ne persiste aucune donnee.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import base64
import hashlib
import re
import sys
import tempfile
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
from PIL import ImageOps
from streamlit_image_select import image_select

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
MAX_IMAGES_IN_BROWSE = 60

SAMPLE_DIRECTORIES: list[tuple[str, Path]] = [
    ("Echantillon curé (data/sample)", Path("data/sample")),
    ("Images jamais vues (data/unseen)", Path("data/unseen")),
    ("Dataset brut (data/raw/Mushrooms_images)", Path("data/raw/Mushrooms_images")),
]

# Referentiel complet des observations : colonne image_lien (nom de fichier) ->
# label (espece). Couvre toutes les images, pas seulement le split du modele.
# Sert a afficher l'espece reelle de chaque image dans la galerie.
LABELS_CSV = Path("data/observations_mushroom.csv")

# Cles session_state utilisees par la page
SS_IMAGE = "pred_selected_image"
SS_SOURCE = "pred_selected_source"
SS_UPLOAD_HASH = "pred_last_upload_hash"
SS_LAST_RESULT = "pred_last_result"
SS_LAST_EXPLAIN = "pred_last_explain"
SS_AUTORUN = "pred_autorun"
SS_TRUE_LABEL = "pred_true_label"


# =====================================================================
# Helpers
# =====================================================================


def _hash_bytes(data: bytes) -> str:
    """Hash MD5 pour detecter les changements d'image."""
    return hashlib.md5(data).hexdigest()


def _set_selected_image(
    image_bytes: bytes,
    source: str,
    autorun: bool = False,
    true_label: str | None = None,
) -> None:
    """Enregistre l'image active et invalide les caches de resultats.

    Si autorun est vrai, la prediction est lancee dans la foulee, sans bouton.
    true_label est l'espece reelle connue (exemples du dataset) : elle sert a
    colorer le verdict (bon / boff / cata) ; None pour un upload ou une URL.
    """
    st.session_state[SS_IMAGE] = image_bytes
    st.session_state[SS_SOURCE] = source
    st.session_state[SS_TRUE_LABEL] = true_label
    # Nouvelle image -> invalider les resultats precedents
    st.session_state[SS_LAST_RESULT] = None
    st.session_state[SS_LAST_EXPLAIN] = None
    if autorun:
        st.session_state[SS_AUTORUN] = True


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


@st.cache_data(show_spinner="Chargement des espèces...")
def load_label_map() -> dict[str, str]:
    """Charge le mapping nom_de_fichier -> espece depuis le referentiel complet."""
    if not LABELS_CSV.exists():
        return {}
    try:
        frame = pd.read_csv(LABELS_CSV, usecols=["image_lien", "label"])
    except (OSError, ValueError):
        return {}
    return dict(zip(frame["image_lien"], frame["label"], strict=False))


def _placeholder_path() -> Path:
    """Cree (une seule fois) une tuile neutre pour la 1ere case de la galerie.

    Sa presence en tete evite qu'une vraie image soit pre-selectionnee au
    chargement : toutes les images deviennent cliquables, y compris la premiere.
    """
    path = Path(tempfile.gettempdir()) / "champy_gallery_placeholder.png"
    if not path.exists():
        PILImage.new("RGB", (200, 200), (55, 60, 68)).save(path)
    return path


@st.cache_data(show_spinner=False)
def load_known_classes() -> set[str]:
    """Les 30 especes connues du modele = labels distincts du split d'entrainement."""
    manifest = Path("data/split_manifest.csv")
    if not manifest.exists():
        return set()
    try:
        frame = pd.read_csv(manifest, usecols=["label"])
    except (OSError, ValueError):
        return set()
    return set(frame["label"].dropna().unique())


@st.cache_data(show_spinner=False)
def bordered_thumbnail(image_path: str, in_known: bool) -> str:
    """Vignette avec cadre vert (espece connue) ou rouge neon (hors 30 classes).

    image_select ne permet pas de colorer les cadres conditionnellement ; on
    dessine donc la bordure directement dans l'image via PIL, puis on sert le
    fichier genere (mis en cache en /tmp).
    """
    color = (46, 204, 113) if in_known else (255, 35, 80)
    try:
        img = PILImage.open(image_path).convert("RGB")
    except (OSError, ValueError):
        return image_path
    # Recadrage centre au carre : image_select affiche les vignettes dans des
    # cases uniformes et rogne les ratios non conformes, ce qui couperait deux
    # cotes du cadre. Un carre garantit une bordure visible sur les 4 cotes.
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((204, 204))
    bordered = ImageOps.expand(img, border=8, fill=color)
    out_dir = Path(tempfile.gettempdir()) / "champy_thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{'in' if in_known else 'out'}_{Path(image_path).name}.png"
    bordered.save(out)
    return str(out)


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
if SS_AUTORUN not in st.session_state:
    st.session_state[SS_AUTORUN] = False


# =====================================================================
# Titre et description
# =====================================================================

st.title(":crystal_ball: Prédiction")

st.markdown(
    """
Cliquez une image dans l'onglet **Exemples locaux** : elle est sélectionnée et
la prédiction se lance aussitôt. Un upload ou une URL déclenche la prédiction de
la même façon, dès que l'image est chargée. Pour les exemples du dataset, le
verdict est coloré selon que l'espèce réelle est bien retrouvée (vert), présente
dans le top-5 (orange) ou manquée (rouge). La case **Grad-CAM** de la barre
latérale ajoute une carte de chaleur des zones qui ont influencé le modèle.
"""
)

# Reglage Grad-CAM dans la barre laterale : choisi une fois, respecte a chaque
# clic sur une vignette (lu en debut de script, avant tout declenchement).
show_gradcam = st.sidebar.checkbox(
    "Inclure l'explication Grad-CAM",
    value=True,
    key="pred_show_gradcam",
    help=(
        "Visualise les zones de l'image qui ont le plus influencé la prédiction. "
        "Calcul plus lent (~500 ms à 5 s selon initialisation)."
    ),
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
            _set_selected_image(data, f"Upload : {uploaded_file.name}", autorun=True)
            st.success(f"Image uploadée : `{uploaded_file.name}`")

# --- Onglet Exemples : galerie de vignettes, un clic = selection + prediction ---
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
                    f"Affichage limité aux {MAX_IMAGES_IN_BROWSE} premières images. "
                    "Cliquez une vignette pour lancer directement la prédiction."
                )
            else:
                st.caption("Cliquez une vignette pour lancer directement la prédiction.")

            # Galerie cliquable. Une tuile "Choisir" est placee en tete : ainsi
            # aucune vraie image n'est pre-selectionnee au chargement, et tout
            # clic sur une image (y compris la premiere) declenche la prediction.
            # La legende de chaque image est son espece reelle (manifest), pour
            # comparer d'un coup d'oeil avec la prediction du modele.
            label_map = load_label_map()
            known = load_known_classes()
            placeholder = str(_placeholder_path())
            gallery_images = [placeholder]
            gallery_captions = ["Choisir une image"]
            origin_of: dict[str, Path] = {}  # vignette bordee -> image originale
            for p in image_paths:
                species = label_map.get(p.name)
                in_known = bool(species and species in known)
                thumb = bordered_thumbnail(str(p), in_known)
                gallery_images.append(thumb)
                gallery_captions.append(("🟢 " if in_known else "🔴 ") + (species or p.name))
                origin_of[thumb] = p
            clicked = image_select(
                label="Cadre vert = espèce connue du modèle · cadre rouge = hors des 30 classes",
                images=gallery_images,
                captions=gallery_captions,
                index=0,
                return_value="original",
                use_container_width=False,
                key=f"imgsel_{dir_choice}",
            )
            prev_key = f"browse_prev_{dir_choice}"
            if clicked and clicked != placeholder and clicked != st.session_state.get(prev_key):
                st.session_state[prev_key] = clicked
                chosen = origin_of.get(clicked)
                if chosen is not None:
                    species = label_map.get(chosen.name)
                    _set_selected_image(
                        chosen.read_bytes(),
                        f"Exemple : {species or chosen.name} ({chosen.name})",
                        autorun=True,
                        true_label=species,
                    )

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
                _set_selected_image(downloaded, "URL distante", autorun=True)
                st.success("Image téléchargée.")


# =====================================================================
# Section 2 : Image active et bouton de lancement
# =====================================================================

image_bytes: bytes | None = st.session_state[SS_IMAGE]
source_label: str = st.session_state[SS_SOURCE] or ""

# Auto-run arme par un clic sur une vignette (consomme une seule fois).
autorun = bool(st.session_state.pop(SS_AUTORUN, False))

if image_bytes is None:
    st.info(
        "Cliquez une image dans **Exemples locaux** (la prédiction se lance "
        "aussitôt), ou utilisez **Upload** / **URL**."
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

# La prediction part automatiquement a chaque nouvelle selection (autorun).
run_prediction = autorun

# Aucune nouvelle selection et aucun resultat en cache : on attend.
if not run_prediction and st.session_state[SS_LAST_RESULT] is None:
    st.info("Cliquez une image pour lancer la prédiction.")
    st.stop()


# =====================================================================
# Section 3 : Inference (uniquement si declenche ou résultat en cache)
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
top1_species = top1["species"]
species_top5 = [p["species"] for p in predictions]
true_label = st.session_state.get(SS_TRUE_LABEL)
known = load_known_classes()

st.divider()
st.subheader("3. Résultats")

# Verdict colore. On distingue trois situations :
#  - espece hors des 30 classes : le modele ne peut pas la connaitre (erreur attendue) ;
#  - espece connue : vert (top-1 correct), orange (dans le top-5), rouge (manque) ;
#  - espece reelle inconnue (upload / URL) : on affiche juste la prediction.
if true_label and true_label not in known:
    st.info(
        f"🔵 **Hors des 30 classes du modèle.** Espèce réelle : *{true_label}*. "
        f"Le modèle ne l'a jamais apprise, il a forcément répondu autre chose "
        f"(*{top1_species}*, {top1['confidence']:.1%}). L'erreur est **attendue**, "
        "ce n'est pas un échec du modèle."
    )
    if run_prediction:
        st.toast(f"🔵 Hors 30 classes : {true_label}", icon="🔵")
elif true_label:
    if top1_species == true_label:
        verdict_icon = "✅"
        st.success(
            f"✅ **Bien reconnu.** Réel : *{true_label}*, "
            f"prédit : *{top1_species}* ({top1['confidence']:.1%})"
        )
    elif true_label in species_top5:
        verdict_icon = "⚠️"
        rang = species_top5.index(true_label) + 1
        st.warning(
            f"⚠️ **Presque.** Réel : *{true_label}* (rang {rang} du top-5), "
            f"prédit : *{top1_species}* ({top1['confidence']:.1%})"
        )
    else:
        verdict_icon = "❌"
        st.error(
            f"❌ **Manqué.** Réel : *{true_label}* (dans les 30 classes), "
            f"prédit : *{top1_species}* ({top1['confidence']:.1%})"
        )
    if run_prediction:
        st.toast(f"{verdict_icon} {true_label} → {top1_species}", icon=verdict_icon)
else:
    st.info(f"Prédiction : **{top1_species}** ({top1['confidence']:.1%})")
    if run_prediction:
        st.toast(f"Prédiction : {top1_species}", icon="🔮")

is_ood = bool(true_label and true_label not in known)
if is_ood:
    st.markdown(
        """
        <style>
        @keyframes ood-pulse {
            0%, 100% { box-shadow: 0 0 6px 1px rgba(255, 35, 80, 0.45); }
            50%      { box-shadow: 0 0 22px 6px rgba(255, 35, 80, 0.95); }
        }
        .st-key-ood_chart_frame {
            border: 3px solid #ff2350;
            border-radius: 12px;
            padding: 10px;
            animation: ood-pulse 2.4s ease-in-out infinite;
        }
        .ood-banner {
            background: rgba(255, 35, 80, 0.12);
            color: #ff2350;
            font-weight: 700;
            text-align: center;
            padding: 6px 10px;
            border-radius: 6px;
            margin: 4px 0 10px 0;
            animation: ood-pulse 2.4s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

col_top, col_chart = st.columns([1, 2])
with col_top:
    st.metric(f":trophy: {top1_species}", f"{top1['confidence']:.1%}")
    st.caption(f"Modèle : `{result.get('model_version', '?')}`")

with col_chart:
    if is_ood:
        st.markdown(
            '<div class="ood-banner">⚠ Détection NON FIABLE : espèce hors des 30 classes</div>',
            unsafe_allow_html=True,
        )
        try:
            chart_box = st.container(key="ood_chart_frame")
        except TypeError:
            chart_box = st.container()
    else:
        chart_box = st.container()
    with chart_box:
        df_pred = pd.DataFrame(predictions)
        df_pred["confidence_pct"] = df_pred["confidence"] * 100
        fig = px.bar(
            df_pred,
            x="confidence_pct",
            y="species",
            orientation="h",
            title=(
                "Top-5 : détection NON FIABLE (hors des 30 classes)"
                if is_ood
                else "Top-5 : confiance par espèce"
            ),
            labels={"confidence_pct": "Confiance (%)", "species": "Espèce"},
            color="confidence_pct",
            color_continuous_scale="Reds" if is_ood else "Greens",
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
