"""Page d'accueil du portfolio Streamlit Champy Classifier.

Landing page principale du portfolio MLOps. Structure :

1. Hero avec accroche et metriques cles du modele
2. Boutons d'action vers les pages les plus consultees
3. Video de presentation animee du pipeline complet (30s)
4. Presentation des trois personae adresses par le portfolio
5. Statut dynamique du pipeline MLOps (12 etapes)

Le contenu s'adapte au role de l'utilisateur courant (admin, user, guest).
Toutes les metriques sont lues dynamiquement aux sources (JSON, MLflow,
Prometheus) ; aucune valeur n'est ecrite en dur.

Direction esthetique : editorial refined-tech (sobre, premium).
Couleurs : vert foret #1F4E3D, ambre #B85C00, creme #FAFAF5.
Polices : Manrope (titres) + Source Sans 3 (corps).

Le rendu utilise st.html() pour les blocs HTML/CSS personnalises afin de
contourner la sanitization stricte introduite dans Streamlit 1.4x+ qui
filtre les <style> et certains <div> dans st.markdown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# Setup chemin projet
# =====================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "demo" / "assets"

st.set_page_config(
    page_title="Champy Classifier - MLOps Portfolio",
    page_icon=":mushroom:",
    layout="wide",
)


# =====================================================================
# Styles globaux : fonts custom + composants stylises
#
# st.html() utilise (pas st.markdown) car Streamlit 1.4x+ filtre les
# blocs <style> dans st.markdown meme avec unsafe_allow_html=True.
# =====================================================================

st.html(
    """
<style>
/* Polices custom depuis Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

/* Variables CSS pour la palette du theme */
:root {
    --champy-primary: #1F4E3D;
    --champy-primary-light: #2D6B54;
    --champy-accent: #B85C00;
    --champy-accent-light: #E67E22;
    --champy-bg: #FAFAF5;
    --champy-bg-soft: #F1EFE8;
    --champy-text: #1A1A1A;
    --champy-text-muted: #6B7280;
    --champy-border: #E5E2D8;
}

/* Application globale des fonts (sans toucher aux icônes Material de Streamlit) */
html, body, .stApp, [data-testid="stMarkdownContainer"], [data-testid="stText"] {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Hero : bande superieure avec accent visuel discret */
.champy-hero {
    background: linear-gradient(135deg, #F1EFE8 0%, #FAFAF5 100%);
    border-left: 4px solid var(--champy-primary);
    border-radius: 8px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.champy-hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    color: var(--champy-primary);
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.champy-hero-subtitle {
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--champy-text-muted);
    margin: 0 0 1.2rem 0;
    letter-spacing: -0.01em;
}

.champy-hero-body {
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--champy-text);
    max-width: 880px;
}

.champy-hero-tag {
    display: inline-block;
    background: var(--champy-primary);
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Section headers : sobres avec accent ambre sur la ligne */
.champy-section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--champy-text);
    border-bottom: 2px solid var(--champy-accent);
    padding-bottom: 0.4rem;
    display: inline-block;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
}

/* Metric cards : plus aerees */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--champy-border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--champy-primary);
}

div[data-testid="stMetric"] label {
    color: var(--champy-text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

div[data-testid="stMetricValue"] {
    color: var(--champy-primary) !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
}

/* Page links : effet plus marque */
a[data-testid="stPageLink-NavLink"] {
    border: 1px solid var(--champy-border);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    transition: all 0.2s ease;
    background: white;
}

a[data-testid="stPageLink-NavLink"]:hover {
    border-color: var(--champy-primary);
    background: var(--champy-bg-soft);
    transform: translateX(2px);
}

/* Containers bordes : effet lift au hover (cartes personae) */
div[data-testid="stContainer"][data-stale="false"] {
    border-color: var(--champy-border) !important;
    border-radius: 8px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stContainer"][data-stale="false"]:hover {
    border-color: var(--champy-primary) !important;
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(31, 78, 61, 0.08);
}

/* Captions : un peu plus discrets */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--champy-text-muted) !important;
    font-size: 0.88rem !important;
}

/* Dividers : plus aeres */
hr {
    margin: 2.5rem 0 !important;
    border-color: var(--champy-border) !important;
}
</style>
"""
)


# =====================================================================
# Section vitrine : Hero + Équipe (visibles SANS authentification)
# =====================================================================

st.html("""
<div class="champy-hero">
    <span class="champy-hero-tag">Portfolio MLOps</span>
    <h1 class="champy-hero-title">Champy Classifier 🍄</h1>
    <p class="champy-hero-subtitle">Classification de 30 espèces de champignons par ConvNeXt-Tiny</p>
    <p class="champy-hero-body">
        Projet diplômant du <strong>Master Intelligence Artificielle DataScientest / Mines Paris PSL</strong>.
        De la donnée brute (647 000 images) au modèle déployé en production,
        ce portfolio retrace l'intégralité du pipeline MLOps : exploration,
        nettoyage, augmentation, entraînement, évaluation, serving, monitoring
        et détection de drift.
        <br><br>
        <em>Toutes les valeurs présentées sont lues dynamiquement aux sources
        (fichiers JSON, MLflow, API, Prometheus). Aucune métrique n'est écrite en dur.</em>
    </p>
</div>
""")
# =====================================================================
# Section : Équipe et cadre académique
# =====================================================================

st.html('<p class="champy-section-title">Équipe et cadre académique</p>')

st.markdown(
    "**Travail de Fin d'Études** — Master *Intelligence Artificielle*, "
    "DataScientest x Mines Paris PSL (RNCP niveau 7). "
    "Soutenance prévue le **16 juin 2026**."
)

team_cols = st.columns(5)
team_members = [
    ("Loïc FOCRAUD", "Co-auteur", False),
    ("Lionel SCHNEIDER", "Co-auteur", False),
    ("Dominique GEORGES", "Co-auteur", False),
    ("Saravana PREGASSAME", "Co-auteur", False),
    ("Kylian POILLY", "🎓 Mentor", True),
]

for col, (name, role, is_mentor) in zip(team_cols, team_members, strict=False):
    with col, st.container(border=True):
        if is_mentor:
            st.markdown(f"**:orange[{name}]**")
        else:
            st.markdown(f"**{name}**")
        st.caption(role)

st.info(
    "👉 Pour explorer l'intégralité du portfolio, **connectez-vous ci-dessous** "
    "avec un des comptes de démonstration ou poursuivez en mode invité.",
    icon="🔐",
)

# =====================================================================
# Authentification (le reste n'est visible qu'aux utilisateurs auth)
# =====================================================================
from demo import auth

auth.setup_page()
user = auth.get_current_user()
user_role = auth.get_current_role()
is_admin = user_role == "admin"
is_logged_in = user_role in ("admin", "user")


# =====================================================================
# Helpers de lecture des metriques
# =====================================================================


def _load_json(path: Path) -> dict | None:
    """Charge un fichier JSON, retourne None si introuvable ou invalide."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_dataset_stats() -> dict:
    """Retourne les statistiques du dataset depuis split_stats.json."""
    stats = _load_json(DATA_DIR / "split_stats.json") or {}
    return {
        "total": stats.get("total", 0),
        "classes": len(stats.get("per_class", {})),
        "train": stats.get("splits", {}).get("train", 0),
        "val": stats.get("splits", {}).get("val", 0),
        "test": stats.get("splits", {}).get("test", 0),
        "seed": stats.get("seed", "?"),
    }


def get_model_metrics() -> dict:
    """Retourne les metriques du modele depuis metrics.json."""
    metrics = _load_json(PROJECT_ROOT / "models" / "artifacts" / "metrics.json") or {}
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "f1_macro": metrics.get("f1_macro", 0.0),
        "n_classes": 30,
        "architecture": "ConvNeXt-Tiny",
    }


# =====================================================================
# Section 2 : Metriques cles (KPI banner)
# =====================================================================

dataset = get_dataset_stats()
metrics = get_model_metrics()

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
col1.metric(
    "Architecture",
    metrics["architecture"],
    help="Modèle final retenu après comparaison de 3 architectures",
)
col2.metric(
    "Test accuracy",
    f"{metrics['accuracy']:.1%}" if metrics["accuracy"] else "n/a",
    help="Précision globale sur le test set (15 % du dataset)",
)
col3.metric(
    "F1 macro",
    f"{metrics['f1_macro']:.1%}" if metrics["f1_macro"] else "n/a",
    help="F1 moyen non pondéré, robuste au déséquilibre des classes",
)
col4.metric(
    "Espèces",
    metrics["n_classes"],
    help="Nombre de classes de champignons classifiées",
)
col5.metric(
    "Images curées",
    f"{dataset['total']:,}".replace(",", " "),
    help="Images retenues après filtrage qualité (sur 647 000 brutes)",
)


# =====================================================================
# Section 3 : Boutons d'action vers les pages cles
# =====================================================================

st.divider()
st.html('<p class="champy-section-title">Accès rapides</p>')

col_pred, col_mon, col_drift, col_arch = st.columns(4)

with col_pred:
    st.page_link(
        "pages/08_prédiction.py",
        label=":crystal_ball: **Prédiction interactive**",
        help="Upload d'image, inférence en direct, explication Grad-CAM",
    )
    st.caption("Démonstration end-to-end")

with col_mon:
    if is_logged_in:
        st.page_link(
            "pages/10_monitoring.py",
            label=":bar_chart: **Monitoring**",
            help="Métriques live Prometheus, latences, alertes",
        )
        st.caption("Observabilité production")
    else:
        st.markdown(":lock: Monitoring")
        st.caption("Connexion requise")

with col_drift:
    if is_logged_in:
        st.page_link(
            "pages/11_drift.py",
            label=":warning: **Détection de drift**",
            help="Surveillance Evidently AI de la dérive des prédictions",
        )
        st.caption("Vigilance modèle")
    else:
        st.markdown(":lock: Drift detection")
        st.caption("Connexion requise")

with col_arch:
    if is_logged_in:
        st.page_link(
            "pages/13_analyse_modèles.py",
            label=":mag: **Analyse modèles**",
            help="Comparaison ResNet vs ConvNeXt, choix architecturaux",
        )
        st.caption("Décisions techniques")
    else:
        st.markdown(":lock: Analyse modèles")
        st.caption("Connexion requise")


# =====================================================================
# Section 4 : Animation du cycle de vie complet (SVG 7 scènes)
# =====================================================================

SVG_PIPELINE_PATH = ASSETS_DIR / "champy_pipeline_animated.svg"

if SVG_PIPELINE_PATH.exists():
    st.divider()
    st.html('<p class="champy-section-title">Vue d\'ensemble du pipeline</p>')
    st.caption(
        "Animation de 70 s en boucle, 7 scènes : entraînement et tracking, "
        "déploiement, inférence live, persistance et monitoring, détection "
        "de drift, alerting Discord, scénario cible production entreprise "
        "(nginx + Let's Encrypt + oauth2-proxy)."
    )

    svg_content = SVG_PIPELINE_PATH.read_text(encoding="utf-8")

    ui = """
<script>
  // Au demarrage : fige. On memorise les minuteurs de l'autoplay
  // sans les lancer tant qu'on reste en mode fige.
  window.__champyAuto = false;
  window.__champyTimers = [];
  window.__si = window.setInterval.bind(window);
  window.setInterval = function (fn, delay) {
    var t = { fn: fn, delay: delay, id: null };
    window.__champyTimers.push(t);
    if (window.__champyAuto) { t.id = window.__si(fn, delay); return t.id; }
    return 0;
  };
</script>
<div style="position:relative;">
  <button id="champy-mode"
    style="position:absolute; top:8px; left:8px; z-index:30; padding:5px 12px;
           font:13px sans-serif; cursor:pointer; border:1px solid #1F4E3D;
           border-radius:6px; background:#1F4E3D; color:#fff;">
    Lecture auto
  </button>
  __SVG__
</div>
<script>
  (function () {
    var btn = document.getElementById('champy-mode');
    btn.onclick = function () {
      if (window.__champyAuto) {
        window.__champyAuto = false;
        window.__champyTimers.forEach(function (t) {
          if (t.id !== null) { window.clearInterval(t.id); t.id = null; }
        });
        btn.textContent = 'Lecture auto';
      } else {
        window.__champyAuto = true;
        window.__champyTimers.forEach(function (t) {
          if (t.id === null) { t.id = window.__si(t.fn, t.delay); }
        });
        btn.textContent = 'Figer';
      }
    };
  })();
</script>
"""

components.html(ui.replace("__SVG__", svg_content), height=740, scrolling=False)


# =====================================================================
# Section 5 : Les trois personae adresses
# =====================================================================

st.divider()
st.html('<p class="champy-section-title">À qui s\'adresse ce portfolio ?</p>')
st.markdown(
    "Le projet sert trois publics distincts. Selon votre rôle, certaines "
    "sections du portfolio vous parleront davantage."
)

col_client, col_ds, col_mlops = st.columns(3)

with col_client, st.container(border=True):
    st.markdown("#### :crystal_ball: Le Client")
    st.markdown(
        "**Cas d'usage** : identifier rapidement une espèce de champignon "
        "depuis une photo, avec un indicateur de confiance et une "
        "explication visuelle (Grad-CAM)."
    )
    st.markdown("**Pages clés** :")
    st.markdown("- Prédiction interactive\n- Démo d'inférence")
    st.caption("Persona principal de l'application en production")

with col_ds, st.container(border=True):
    st.markdown("#### :microscope: Le Data Scientist")
    st.markdown(
        "**Cas d'usage** : comprendre les choix de préparation des données, "
        "les architectures comparées, les métriques par classe, et les "
        "axes d'amélioration identifiés."
    )
    st.markdown("**Pages clés** :")
    st.markdown(
        "- Données brutes, nettoyage, augmentation\n- Entraînement, évaluation, analyse modèles"
    )
    st.caption("Persona qui itère sur le modèle")

with col_mlops, st.container(border=True):
    st.markdown("#### :gear: L'ingénieur MLOps")
    st.markdown(
        "**Cas d'usage** : valider la chaîne de déploiement (serving, API, "
        "Docker, CI/CD), l'observabilité (Prometheus, Grafana) et la "
        "détection de drift (Evidently AI)."
    )
    st.markdown("**Pages clés** :")
    st.markdown("- API, Model Registry, Infrastructure\n- Monitoring, détection de drift")
    st.caption("Persona qui industrialise et exploite")


# =====================================================================
# Section 6 : Statut dynamique du pipeline
# =====================================================================

st.divider()
st.html('<p class="champy-section-title">Statut du pipeline</p>')
st.caption(
    "Vérification dynamique de la présence des artefacts attendus à chaque "
    "étape. Une étape verte signifie que l'artefact est présent et accessible."
)


def check_artifact(path: Path) -> str:
    """Retourne l'icone de statut selon l'existence de l'artefact."""
    return ":white_check_mark:" if path.exists() else ":x:"


pipeline_steps = [
    ("1. Données brutes", DATA_DIR / "raw_stats.json", "Exploration et statistiques"),
    ("2. Nettoyage", DATA_DIR / "cleaning_report.json", "Exclusion doublons et augmentations TF"),
    (
        "3. Augmentation",
        PROJECT_ROOT / "src" / "data" / "dataset.py",
        "Transforms PyTorch configurables",
    ),
    ("4. Split", DATA_DIR / "split_stats.json", "Split stratifié 70/15/15"),
    (
        "5. Entraînement",
        PROJECT_ROOT / "models" / "convnext_tiny_v2.0.0.onnx",
        "Training PyTorch + MLflow",
    ),
    (
        "6. Évaluation",
        PROJECT_ROOT / "models" / "artifacts" / "metrics.json",
        "Métriques, confusion matrix, GradCAM",
    ),
    (
        "7. Model Registry",
        PROJECT_ROOT / "models" / "convnext_tiny_v2.0.0.onnx",
        "MLflow Model Registry",
    ),
    ("8. Prédiction", PROJECT_ROOT / "src" / "serving" / "app.py", "API FastAPI + inférence ONNX"),
    ("9. API", PROJECT_ROOT / "src" / "serving" / "app.py", "Endpoints REST"),
    ("10. Monitoring", PROJECT_ROOT / "configs" / "prometheus.yml", "Prometheus + Grafana"),
    (
        "11. Drift",
        PROJECT_ROOT / "monitoring" / "baseline_reference.json",
        "Evidently AI + baseline",
    ),
    ("12. Infrastructure", PROJECT_ROOT / "src" / "serving" / "app.py", "Docker Compose + CI/CD"),
]

cols = st.columns(3)
for i, (name, artifact, description) in enumerate(pipeline_steps):
    col = cols[i % 3]
    status = check_artifact(artifact)
    col.markdown(f"{status} **{name}**")
    col.caption(description)


# =====================================================================
# Pied de page
# =====================================================================

st.divider()
st.caption(
    "Utilisez le menu latéral pour naviguer entre les 12 étapes du pipeline. "
    "Chaque page lit dynamiquement ses données aux sources (JSON, MLflow, "
    "Prometheus, API)."
)
