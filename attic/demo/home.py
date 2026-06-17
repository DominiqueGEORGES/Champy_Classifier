"""Page d'accueil de la démo Champy Classifier.

Landing page présentant le projet à un visiteur (mentor, jury, recruteur) :
- Identité et cadre académique
- Équipe et mentor
- Objectif scientifique et résultats clés
- Stack technique MLOps
- Navigation vers les pages métier

Fichier d'entrée Streamlit (configuré dans le Dockerfile via
`streamlit run demo/Home.py`).
"""

from __future__ import annotations

import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Champy Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------

st.title("🍄 Champy Classifier")
st.subheader("Classification d'espèces de champignons par deep learning")

st.markdown(
    """
    Plateforme MLOps complète pour l'entraînement, la mise en production et le
    monitoring d'un classifieur d'images dédié à la reconnaissance d'espèces
    de champignons. Le projet adresse un cas d'usage à enjeu de santé publique
    (distinction entre espèces toxiques et comestibles visuellement proches)
    en s'appuyant sur les standards industriels de l'écosystème MLOps moderne.
    """
)

st.divider()


# ----------------------------------------------------------------------------
# Équipe
# ----------------------------------------------------------------------------

st.header("👥 Équipe")

team_cols = st.columns(4)
team_members = [
    ("Loïc FOCRAUD", "Co-auteur"),
    ("Lionel SCHNEIDER", "Co-auteur"),
    ("Dominique GEORGES", "Co-auteur"),
    ("Saravana PREGASSAME", "Co-auteur"),
]

for col, (name, role) in zip(team_cols, team_members, strict=False):
    with col, st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(role)

# Mentor mis en valeur séparément
st.markdown("")
mentor_col, _, _, _ = st.columns(4)
with mentor_col, st.container(border=True):
    st.markdown("**Kylian POILLY**")
    st.caption("🎓 Mentor de projet")

st.divider()


# ----------------------------------------------------------------------------
# Cadre académique
# ----------------------------------------------------------------------------

st.header("🎓 Cadre académique")

st.markdown(
    """
    **Travail de Fin d'Études** — Master *Intelligence Artificielle*
    proposé conjointement par **DataScientest** et **Mines Paris PSL**.

    - **Diplôme** : Master IA (RNCP niveau 7)
    - **Soutenance** : 16 juin 2026
    - **Promotion** : 2026
    """
)

st.divider()


# ----------------------------------------------------------------------------
# Objectif scientifique
# ----------------------------------------------------------------------------

st.header("🎯 Objectif scientifique")

st.markdown(
    """
    Entraîner un classifieur fiable capable de reconnaître **30 espèces de
    champignons** à partir d'une simple photographie, avec une attention
    particulière portée aux **espèces toxiques visuellement proches d'espèces
    comestibles** (couples tels que la galère marginée vs la pholiote
    changeante, ou l'amanite phalloïde vs certaines russules).

    Au-delà de la performance brute du modèle, le projet vise une mise en
    œuvre **conforme aux pratiques MLOps de production** : reproductibilité
    via le versioning des données et du code, traçabilité complète des
    expériences, orchestration des pipelines, monitoring en production et
    alerting sur dérive.
    """
)

st.divider()


# ----------------------------------------------------------------------------
# Résultats clés
# ----------------------------------------------------------------------------

st.header("📊 Résultats clés")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric(
        label="Modèle retenu",
        value="ConvNeXt-Tiny",
        help="Architecture moderne efficace, exportée en ONNX pour le serving.",
    )
with metric_cols[1]:
    st.metric(
        label="Accuracy (top-1)",
        value="90 %",
        help="Précision globale mesurée sur le jeu de test.",
    )
with metric_cols[2]:
    st.metric(
        label="F1-score macro",
        value="81 %",
        help="F1 moyenné sur les 30 classes, sans pondération.",
    )
with metric_cols[3]:
    st.metric(
        label="Espèces classifiées",
        value="30",
        help="Couvre les espèces communes et les confusions toxiques majeures.",
    )

st.divider()


# ----------------------------------------------------------------------------
# Stack technique
# ----------------------------------------------------------------------------

st.header("🛠️ Stack technique")

st.markdown(
    """
    Approche **MLOps de bout en bout**, sans dépendance à un cloud propriétaire :

    | Domaine | Outil | Rôle |
    |---|---|---|
    | Versioning données | **DVC** | Traçabilité des datasets, remote MinIO S3 |
    | Tracking expériences | **MLflow** | Runs, métriques, artefacts, registre de modèles |
    | Serving API | **BentoML** | Endpoints d'inférence et d'explicabilité (Grad-CAM) |
    | Orchestration | **Airflow** | Pipelines d'entraînement, validation, déploiement |
    | Métriques | **Prometheus** | Collecte time-series modèle + infrastructure |
    | Dashboards | **Grafana** | Monitoring temps réel, drift, performance |
    | Alerting | **Alertmanager** + Discord | Notifications d'incidents |
    | Stockage objet | **MinIO** | S3 auto-hébergé, artefacts et cache DVC |
    | Conteneurisation | **Docker Compose** | 11 services orchestrés |
    | Reverse-proxy | **nginx** | Hub d'accès unifié sur un seul domaine |
    | Sécurité | **Cloudflare Access** | Authentification Zero Trust |
    | CI/CD | **GitHub Actions** | Tests, build images, déploiement |
    """
)

st.divider()


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------

st.header("🧭 Naviguer dans la démo")

st.markdown(
    """
    Utilisez la barre latérale (à gauche) pour accéder aux différentes pages
    de la démonstration. Points d'entrée recommandés :

    - **🍄 Plateforme** — vue d'ensemble des services et accès direct à chaque outil
    - **🔬 Prédiction** — tester le modèle sur vos propres images, avec carte de chaleur Grad-CAM
    - **📊 Monitoring** — observer la santé du modèle en production
    - **📚 Registre des modèles** — consulter les versions disponibles
    - **🏗️ Infrastructure** — détail des services et de leur topologie
    """
)

st.caption(
    "Plateforme accessible publiquement via `champy.sbdg-ia.fr` "
    "et sécurisée par Cloudflare Access (Zero Trust)."
)
