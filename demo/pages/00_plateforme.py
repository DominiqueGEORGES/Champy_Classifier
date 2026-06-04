"""Page Plateforme : panneau d'accès unifié à tous les services de la stack Champy.

Affiche une carte par service avec :
- Icône, nom, description courte du rôle dans la stack
- Statut UP/DOWN en temps réel (ping /health via réseau Docker interne)
- Bouton d'ouverture (URL publique via Cloudflare Access)
- Identifiants de connexion (masqués par défaut, lus depuis l'environnement)

Tous les services sont accessibles via le hub nginx (port 8088) routé sur
champy.sbdg-ia.fr/<service>/, derrière Cloudflare Access (Zero Trust).
Une seule authentification globale donne accès à l'ensemble.

ATTENTION : l'affichage en clair des identifiants en surface UI est acceptable
pour la démo de soutenance (interface verrouillée par Cloudflare Access), mais
ne doit pas être conservé en production. Voir la bannière d'avertissement.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PUBLIC_BASE_URL = os.getenv(
    "CHAMPY_PUBLIC_BASE_URL",
    "http://localhost:8088",
)

# Délai max pour un ping de health (s). 1.5s suffit largement en réseau Docker
# interne ; au-delà on considère le service indisponible.
HEALTH_TIMEOUT_SECONDS = 1.5

# TTL du cache des statuts. 10s = compromis entre fraîcheur et charge.
STATUS_CACHE_TTL = 10


@dataclass(frozen=True)
class ServiceCard:
    """Description d'un service de la stack pour l'affichage en carte."""

    key: str  # identifiant interne, ex. "api"
    name: str  # nom affiché, ex. "BentoML API"
    icon: str  # emoji ou caractère
    description: str  # rôle dans la stack (1 phrase)
    internal_health: str  # URL de health en réseau Docker interne
    public_path: str  # path public, concaténé à PUBLIC_BASE_URL
    auth_user_env: str = ""  # nom de la variable d'env contenant l'utilisateur
    auth_pwd_env: str = ""  # nom de la variable d'env contenant le mot de passe
    auth_note: str = ""  # remarque éventuelle (ex. "pas d'authentification")


# Liste exhaustive des services exposés via le hub nginx.
# L'ordre détermine l'ordre d'affichage dans la grille.
SERVICES: list[ServiceCard] = [
    ServiceCard(
        key="streamlit",
        name="Streamlit",
        icon="🍄",
        description=(
            "Interface de démonstration : exploration des données, prédiction "
            "interactive, monitoring et registre de modèles."
        ),
        internal_health="http://demo:8501/_stcore/health",
        public_path="/",
        auth_note="Accès sécurisé par Cloudflare Access uniquement.",
    ),
    ServiceCard(
        key="api",
        name="BentoML API",
        icon="🚀",
        description=(
            "API d'inférence (predict, explain, model registry). Swagger UI "
            "interactif pour tester directement les endpoints."
        ),
        internal_health="http://api:8000/healthz",
        public_path="/api/",
        auth_note="Accès sécurisé par Cloudflare Access uniquement.",
    ),
    ServiceCard(
        key="mlflow",
        name="MLflow",
        icon="📊",
        description=(
            "Tracking des expériences d'entraînement, comparaison de runs, "
            "métriques et artefacts versionnés."
        ),
        internal_health="http://mlflow:5000/health",
        public_path="/mlflow/",
        auth_note="Accès sécurisé par Cloudflare Access uniquement.",
    ),
    ServiceCard(
        key="airflow",
        name="Airflow",
        icon="🌬️",
        description=(
            "Orchestration des pipelines : ingestion DVC, entraînement, "
            "validation, déploiement et tâches périodiques."
        ),
        internal_health="http://airflow:8080/airflow/health",
        public_path="/airflow/",
        auth_user_env="AIRFLOW_ADMIN_USERNAME",
        auth_pwd_env="AIRFLOW_ADMIN_PASSWORD",
    ),
    ServiceCard(
        key="grafana",
        name="Grafana",
        icon="📈",
        description=(
            "Dashboards de monitoring : performance du modèle, métriques "
            "infrastructure, drift de données, impact écologique."
        ),
        internal_health="http://grafana:3000/api/health",
        public_path="/grafana/",
        auth_user_env="GRAFANA_USER",
        auth_pwd_env="GRAFANA_PASSWORD",
    ),
    ServiceCard(
        key="prometheus",
        name="Prometheus",
        icon="🔥",
        description=(
            "Collecte et stockage time-series des métriques (modèle, API, "
            "containers, hôte) consommées par Grafana et Alertmanager."
        ),
        internal_health="http://prometheus:9090/prometheus/-/healthy",
        public_path="/prometheus/",
        auth_note="Accès sécurisé par Cloudflare Access uniquement.",
    ),
    ServiceCard(
        key="alertmanager",
        name="Alertmanager",
        icon="🚨",
        description=(
            "Routing, déduplication et inhibition des alertes Prometheus, "
            "avec notifications Discord pour les incidents critiques."
        ),
        internal_health="http://alertmanager:9093/alertmanager/-/healthy",
        public_path="/alertmanager/",
        auth_note="Accès sécurisé par Cloudflare Access uniquement.",
    ),
    ServiceCard(
        key="minio",
        name="MinIO",
        icon="💾",
        description=(
            "Stockage objet S3-compatible : artefacts MLflow, cache DVC, "
            "backups. Console web pour navigation des buckets."
        ),
        internal_health="http://minio:9000/minio/health/live",
        public_path="/minio/",
        auth_user_env="MINIO_ROOT_USER",
        auth_pwd_env="MINIO_ROOT_PASSWORD",
    ),
]


# ----------------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------------


async def _ping(service: ServiceCard, client: httpx.AsyncClient) -> tuple[str, bool]:
    """Ping un endpoint /health et retourne (clé, est_up)."""
    try:
        response = await client.get(
            service.internal_health,
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
        return service.key, response.status_code < 400
    except Exception:
        return service.key, False


async def _check_all_services() -> dict[str, bool]:
    """Pinge tous les services en parallèle et retourne un dict {key: is_up}."""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_ping(s, client) for s in SERVICES))
    return dict(results)


@st.cache_data(ttl=STATUS_CACHE_TTL, show_spinner=False)
def get_service_statuses() -> dict[str, bool]:
    """Cache court pour éviter de spammer les /health à chaque rerender."""
    return asyncio.run(_check_all_services())


# ----------------------------------------------------------------------------
# Lecture des identifiants
# ----------------------------------------------------------------------------


def get_credentials(service: ServiceCard) -> tuple[str, str] | None:
    """Retourne (utilisateur, mot_de_passe) ou None si pas d'auth applicable.

    Les variables d'environnement doivent être injectées dans le container
    Streamlit via le `docker-compose.yml`. Si une variable manque, on retourne
    une chaîne d'avertissement plutôt que de masquer silencieusement.
    """
    if not service.auth_user_env and not service.auth_pwd_env:
        return None
    user = os.getenv(service.auth_user_env, "(variable non définie)")
    pwd = os.getenv(service.auth_pwd_env, "(variable non définie)")
    return user, pwd


# ----------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Plateforme — Champy",
    page_icon="🍄",
    layout="wide",
)

st.title("🍄 Plateforme Champy Classifier")
st.markdown(
    """
    Panneau d'accès unifié à l'ensemble des services de la stack MLOps.

    Tous les services sont exposés via un point d'entrée unique
    (`champy.sbdg-ia.fr`), routés par un reverse-proxy nginx, et sécurisés
    par **Cloudflare Access** (Zero Trust). L'authentification est globale :
    une seule connexion donne accès à l'ensemble des outils ci-dessous.
    """
)

st.warning(
    "**Démo de soutenance** : les identifiants techniques sont affichés "
    "sur cette page pour faciliter la navigation du jury. En production, cet "
    "affichage doit être désactivé et l'authentification entièrement déléguée "
    "à Cloudflare Access ou à un fournisseur d'identité (SSO).",
    icon="⚠️",
)

# Boutons de contrôle (rafraîchissement manuel)
col_left, col_right = st.columns([3, 1])
with col_right:
    if st.button("🔄 Rafraîchir les statuts", use_container_width=True):
        get_service_statuses.clear()
        st.rerun()

statuses = get_service_statuses()
nb_up = sum(1 for is_up in statuses.values() if is_up)
nb_total = len(SERVICES)

with col_left:
    if nb_up == nb_total:
        st.success(f"✅ Tous les services sont opérationnels ({nb_up}/{nb_total}).")
    elif nb_up >= nb_total - 1:
        st.warning(f"⚠️ Un service indisponible ({nb_up}/{nb_total} opérationnels).")
    else:
        st.error(f"❌ Plusieurs services indisponibles ({nb_up}/{nb_total} opérationnels).")

st.divider()

# Grille 2 colonnes pour des cartes larges et lisibles
COLS = 2
for row_start in range(0, len(SERVICES), COLS):
    cols = st.columns(COLS)
    for idx, col in enumerate(cols):
        service_idx = row_start + idx
        if service_idx >= len(SERVICES):
            break
        service = SERVICES[service_idx]
        is_up = statuses.get(service.key, False)
        public_url = f"{PUBLIC_BASE_URL}{service.public_path}"
        creds = get_credentials(service)

        with col, st.container(border=True):
            badge = "🟢 UP" if is_up else "🔴 DOWN"
            st.markdown(f"### {service.icon} {service.name} &nbsp;&nbsp; *{badge}*")
            st.caption(service.description)
            st.link_button(
                "Ouvrir →",
                public_url,
                use_container_width=True,
            )

            # Bloc identifiants : repli pour éviter la surcharge visuelle.
            # st.code() expose un bouton "copier" natif au survol.
            with st.expander("🔑 Identifiants de connexion", expanded=False):
                if creds is None:
                    st.caption(service.auth_note or "Pas d'authentification applicative.")
                else:
                    user, pwd = creds
                    st.caption("Utilisateur")
                    st.code(user, language="text")
                    st.caption("Mot de passe")
                    st.code(pwd, language="text")
st.divider()

# Section pédagogique pour la défense : explique l'architecture
with st.expander("Architecture du hub d'accès", expanded=False):
    st.markdown(
        """
        **Flux d'une requête utilisateur :**

        ```
        Utilisateur
            │
            ▼
        Cloudflare Edge  ◄─── authentification Cloudflare Access (Zero Trust)
            │
            ▼
        Cloudflare Tunnel (cloudflared, sur NUC Ubuntu)
            │
            ▼
        nginx reverse-proxy (port 8088, sur NUC3 Windows)
            │
            ├──► Streamlit       (port 8501, path /)
            ├──► BentoML API     (port 8000, path /api/)
            ├──► MLflow          (port 5000, path /mlflow/)
            ├──► Airflow         (port 8080, path /airflow/)
            ├──► Grafana         (port 3000, path /grafana/)
            ├──► Prometheus      (port 9090, path /prometheus/)
            ├──► Alertmanager    (port 9093, path /alertmanager/)
            └──► MinIO Console   (port 9001, path /minio/)
        ```

        **Avantages de cette architecture :**

        - **Un seul certificat TLS** (géré par Cloudflare)
        - **Une seule politique d'authentification** (Cloudflare Access)
        - **Pas de ports exposés sur Internet** depuis le NUC3
        - **Sortie unique via le NUC Ubuntu** (cloudflared)
        - **Routage interne par nginx** sur le réseau Docker privé
        """
    )

st.caption(
    f"💡 Statut rafraîchi automatiquement toutes les {STATUS_CACHE_TTL}s. "
    "Utilise le bouton ci-dessus pour forcer un nouveau check immédiat."
)
