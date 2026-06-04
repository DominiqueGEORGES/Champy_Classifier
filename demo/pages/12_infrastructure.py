"""Page Streamlit : infrastructure et architecture.

Affiche :
    - Le schéma d'architecture du pipeline MLOps (Mermaid)
    - Le statut live des services via health checks HTTP
    - Les jobs CI/CD GitHub Actions
    - La stack technique complète avec justifications

Le rendu Mermaid est réalisé via la librairie mermaid.js (chargée depuis
CDN) injectée dans un composant HTML iframe Streamlit. Le diagramme est
zoomable et copiable côté navigateur.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import os
import sys
import time
from pathlib import Path

# =====================================================================
# Setup chemin projet
# =====================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =====================================================================
# Imports tiers
# =====================================================================

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# Imports projet
# =====================================================================
from demo import auth

# =====================================================================
# Authentification (lit access_policy.yaml)
# =====================================================================

auth.setup_page()

# =====================================================================
# Configuration de la page
# =====================================================================

st.set_page_config(page_title="12 - Infrastructure", layout="wide")
st.title(":building_construction: Infrastructure")


# =====================================================================
# Section 1 : Architecture (diagramme Mermaid)
# =====================================================================

st.header("Architecture du pipeline MLOps")

st.markdown(
    "Vue d'ensemble des trois environnements (développement, cloud, "
    "production), des flux de données et d'alerting, et des deux "
    "scénarios d'exposition (démo Cloudflare actuel vs cible prod "
    "entreprise nginx)."
)

MERMAID_DIAGRAM = """
flowchart LR
    subgraph DEV["💻 Dev (XPS 9520)"]
        direction TB
        TR["PyTorch<br/>ConvNeXt-Tiny"]
        EXP["Export ONNX"]
        TR --> EXP
    end

    subgraph CLOUD["☁️ DagsHub (alternative)"]
        direction TB
        DVC_REMOTE[("DVC remote")]
        MLFLOW_SRV["MLflow<br/>Tracking"]
        REPO["Git repo"]
    end

    subgraph PROD["🖥️ Production NUC3 (Docker Compose)"]
        direction TB
        API["API BentoML<br/>+ ONNX<br/>:8000"]
        MINIO[("MinIO<br/>:9010")]
        DEMO["Streamlit<br/>:8501"]
        AIRFLOW["Airflow<br/>:8081"]
        PROM["Prometheus<br/>:9090"]
        GRAFANA["Grafana<br/>:3000"]
        SQLITE[("SQLite<br/>predictions.db")]
    end

    subgraph ALERTS["🔔 Alerting"]
        direction TB
        ALERTM["Alertmanager<br/>:9093"]
        ADAPT["Adapter Discord<br/>:9094"]
        DISCORD["Discord webhook<br/>#champy-alerts"]
        ALERTM --> ADAPT --> DISCORD
    end

    subgraph EXPO["🌐 Exposition"]
        direction TB
        subgraph CURRENT["Démo (actuel)"]
            CF["Cloudflare<br/>Tunnel + Access<br/>(OTP email)"]
        end
        subgraph TARGET["Prod entreprise (cible)"]
            NGINX["nginx<br/>reverse-proxy<br/>+ rate limit"]
            LE["Let's Encrypt<br/>SSL auto"]
            AUTH["oauth2-proxy<br/>ou Authelia"]
        end
    end

    INTERNET((Internet))

    TR -.->|"params"| MLFLOW_SRV
    EXP -->|"dvc push"| MINIO
    EXP -.->|"dvc push (alt)"| DVC_REMOTE
    MINIO -->|"dvc pull"| API
    DVC_REMOTE -.->|"dvc pull (alt)"| API

    DEMO -->|"HTTP /predict"| API
    API -->|"log"| SQLITE
    API -->|"/metrics"| PROM
    PROM --> GRAFANA
    AIRFLOW -->|"drift Evidently"| SQLITE
    PROM -->|"alertes"| ALERTM

    INTERNET --> CF
    INTERNET -.-> NGINX
    LE -.-> NGINX
    AUTH -.-> NGINX
    CF -->|"HTTPS"| PROD
    NGINX -.->|"HTTPS"| PROD

    classDef devNode fill:#e8f4f8,stroke:#0288d1,stroke-width:2px,color:#01579b
    classDef cloudNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef prodNode fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#1b5e20
    classDef futureNode fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,stroke-dasharray:5 5,color:#1b5e20
    classDef alertNode fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f
    classDef expoNode fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#311b92
    classDef expoFutureNode fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,stroke-dasharray:5 5,color:#311b92
    classDef netNode fill:#fff,stroke:#555,stroke-width:2px,color:#000

    class TR,EXP devNode
    class DVC_REMOTE,MLFLOW_SRV,REPO cloudNode
    class API,DEMO,AIRFLOW,PROM,GRAFANA,SQLITE prodNode
    class BENTO futureNode
    class ALERTM,ADAPT,DISCORD alertNode
    class CF expoNode
    class NGINX,LE,AUTH expoFutureNode
    class INTERNET netNode
"""

_MERMAID_HTML = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 8px;
      background: transparent;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .mermaid {{
      display: flex;
      justify-content: center;
    }}
  </style>
</head>
<body>
  <div class="mermaid">
{MERMAID_DIAGRAM}
  </div>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'default',
      flowchart: {{
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis',
        nodeSpacing: 30,
        rankSpacing: 50
      }},
      themeVariables: {{
        fontSize: '13px',
        fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif'
      }}
    }});
  </script>
</body>
</html>
"""

components.html(_MERMAID_HTML, height=720, scrolling=True)

st.caption(
    "Le diagramme est interactif côté navigateur (zoom, déplacement). "
    "Codé en Mermaid pour une édition facile (cf. source de cette page)."
)

with st.expander("Légende des flèches et des couleurs"):
    st.markdown(
        """
**Flèches**

- `-->` : flux de données ou d'appels en production active
- `-.->` : flux occasionnel (push artefact, dépendance future)

**Couleurs des environnements**

- Bleu : poste de développement (XPS 9520, GPU NVIDIA RTX 3050 Ti)
- Orange : cloud DagsHub (gratuit, DVC remote + MLflow tracking)
- Vert plein : production locale active (NUC3, Docker Desktop)
- Vert pointillé : composant codé mais non activé dans le compose actuel
- Rose : chaîne d'alerting (Alertmanager → adaptateur Discord → webhook)
- Violet plein : scénario d'exposition **actuel** (Cloudflare Tunnel + Access)
- Violet pointillé : scénario d'exposition **cible production entreprise**
  (nginx + Let's Encrypt + oauth2-proxy / Authelia pour l'auth SSO)

**Pourquoi deux scénarios d'exposition**

Cloudflare Tunnel + Access est idéal pour une **démo de portfolio ou un MVP**
(SSO email OTP, zéro port ouvert, gratuit). En **production entreprise
classique**, on remplacerait par nginx (reverse-proxy + SSL termination)
avec Let's Encrypt pour le SSL et un oauth2-proxy ou Authelia pour
l'authentification SSO entreprise (LDAP, OIDC, Active Directory). Le code
applicatif et le compose Docker restent identiques : seule la couche
d'exposition change.
"""
    )

st.divider()


# =====================================================================
# Section 2 : Statut live des services (health checks HTTP)
# =====================================================================

st.header("Statut des services")
st.caption(
    "Health checks HTTP réalisés depuis le réseau Docker interne. "
    "Chaque service est interrogé sur son endpoint de santé avec un "
    "timeout de 2 s. La colonne URL pointe vers l'interface accessible "
    "depuis l'hôte ou vers le service externe pour MLflow."
)


def check_http_health(url: str, timeout: float = 2.0) -> tuple[str, str]:
    """Interroge un endpoint HTTP de santé et mesure la latence.

    Args:
        url: URL à interroger (interne au réseau Docker).
        timeout: Timeout maximal en secondes.

    Returns:
        Tuple ``(statut, latence_str)`` où :
            - statut est l'un de "OK", "HTTP <code>", "Timeout",
              "Injoignable" ou "Erreur (...)"
            - latence_str est la latence en ms formatée, ou "—"
    """
    try:
        start = time.perf_counter()
        response = requests.get(url, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000

        if 200 <= response.status_code < 400:
            return "OK", f"{latency_ms:.0f} ms"
        return f"HTTP {response.status_code}", f"{latency_ms:.0f} ms"
    except requests.exceptions.Timeout:
        return "Timeout", f">{int(timeout * 1000)} ms"
    except requests.exceptions.ConnectionError:
        return "Injoignable", "—"
    except Exception as exc:
        return f"Erreur ({type(exc).__name__})", "—"


# =====================================================================
# URLs publiques parametrables (defaut local)
# =====================================================================
_PUBLIC_BASE_URL = os.getenv("CHAMPY_PUBLIC_BASE_URL", "http://localhost:8088").rstrip("/")
_API_PUBLIC_URL = os.getenv("CHAMPY_API_PUBLIC_URL", "").rstrip("/") or (f"{_PUBLIC_BASE_URL}/api")


SERVICES_CONFIG: list[dict[str, str | None]] = [
    # {
    #     "name": "API FastAPI",
    #     "internal_url": "http://api:8000/healthz",
    #     "external_url": "https://champy-api.sbdg-ia.fr/docs",
    #     "kind": "internal",
    # },
    {
        "name": "API BentoML",
        "internal_url": "http://api:8000/healthz",
        "external_url": _API_PUBLIC_URL,
        "kind": "internal",
    },
    {
        "name": "Streamlit demo (cette page)",
        "internal_url": None,
        "external_url": _PUBLIC_BASE_URL,
        "kind": "self",
    },
    {
        "name": "Prometheus",
        "internal_url": "http://prometheus:9090/prometheus/-/healthy",
        "external_url": f"{_PUBLIC_BASE_URL}/prometheus",
        "kind": "internal",
    },
    {
        "name": "Grafana",
        "internal_url": "http://grafana:3000/api/health",
        "external_url": f"{_PUBLIC_BASE_URL}/grafana",
        "kind": "internal",
    },
    {
        "name": "MinIO (stockage objet)",
        "internal_url": "http://minio:9000/minio/health/live",
        "external_url": f"{_PUBLIC_BASE_URL}/minio/",
        "kind": "internal",
    },
    {
        "name": "Alertmanager",
        "internal_url": "http://alertmanager:9093/alertmanager/-/healthy",
        "external_url": f"{_PUBLIC_BASE_URL}/alertmanager/",
        "kind": "internal",
    },
    {
        "name": "Adaptateur Discord",
        "internal_url": "http://alertmanager-discord:9094/",
        "external_url": None,
        "kind": "internal",
    },
    {
        "name": "Airflow webserver",
        "internal_url": "http://airflow:8080/airflow/health",
        "external_url": f"{_PUBLIC_BASE_URL}/airflow/",
        "kind": "internal",
    },
    # {
    #     "name": "MLflow (DagsHub)",
    #     "internal_url": None,
    #     "external_url": (
    #         "https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow"
    #     ),
    #     "kind": "external",
    # },
    {
        "name": "MLflow (local)",
        "internal_url": "http://mlflow:5000/health",
        "external_url": f"{_PUBLIC_BASE_URL}/mlflow/",
        "kind": "internal",
    },
    {
        "name": "MLflow + DVC (DagsHub cloud, alternative)",
        "internal_url": None,
        "external_url": ("https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow"),
        "kind": "external",
    },
    {
        "name": "nginx (reverse-proxy)",
        "internal_url": "http://nginx:80/nginx-health",
        "external_url": None,
        "kind": "internal",
    },
]

rows: list[dict[str, str]] = []
for svc in SERVICES_CONFIG:
    name = svc["name"]
    internal_url = svc["internal_url"]
    external_url = svc["external_url"]
    kind = svc["kind"]

    if kind == "self":
        status_label = "OK"
        latency = "—"
        icon = "🟢"
    elif kind == "external":
        status_label = "Externe"
        latency = "—"
        icon = "🌐"
    else:
        status_label, latency = check_http_health(internal_url)
        if status_label == "OK":
            icon = "🟢"
        elif status_label.startswith("HTTP"):
            icon = "🟠"
        else:
            icon = "🔴"

    rows.append(
        {
            "Service": name,
            "Statut": f"{icon} {status_label}",
            "Latence": latency,
            "URL": external_url,  # None => cellule vide dans la LinkColumn
        }
    )

df_status = pd.DataFrame(rows)

st.dataframe(
    df_status,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Service": st.column_config.TextColumn("Service", width="medium"),
        "Statut": st.column_config.TextColumn("Statut", width="small"),
        "Latence": st.column_config.TextColumn("Latence", width="small"),
        "URL": st.column_config.LinkColumn(
            "URL d'accès",
            help="Cliquez pour ouvrir l'interface du service dans un nouvel onglet",
            display_text="Ouvrir",
        ),
    },
)

st.caption(
    "Endpoints interrogés : `/healthz` (BentoML), `/prometheus/-/healthy` "
    "(Prometheus en sous-chemin), `/api/health` (Grafana), `/-/healthy` "
    "(Alertmanager), `/health` (Airflow). L'adaptateur Discord ne "
    "publie pas d'UI : on vérifie qu'il accepte les connexions HTTP. "
    "MLflow est hébergé sur DagsHub donc en accès externe."
)

st.divider()


# =====================================================================
# Section 3 : CI/CD
# =====================================================================

st.header("CI/CD")
st.markdown(
    "Pipeline GitHub Actions déclenché sur chaque push et chaque pull "
    "request, avec gates de qualité non négociables avant merge."
)

st.markdown(
    """
| Job          | Description                                          |
|--------------|------------------------------------------------------|
| lint         | `ruff check` + `ruff format --check`                 |
| typecheck    | `mypy` en mode strict                                |
| docstrings   | `interrogate` avec exigence de 100 % de couverture   |
| test         | `pytest` + couverture de code (objectif ≥ 80 %)      |
| build        | `docker compose build` de tous les services          |
"""
)

st.markdown(
    "**Repository** : "
    "[DagsHub - LoicFocraud/Champy_Classifier]"
    "(https://dagshub.com/LoicFocraud/Champy_Classifier)"
)

st.divider()


# =====================================================================
# Section 4 : Stack technique
# =====================================================================

st.header("Stack technique")
st.markdown(
    "Choix techniques validés par l'équipe, avec justification courte pour chaque composant."
)

st.markdown(
    """
| Composant                | Techno                              | Justification                                           |
|--------------------------|-------------------------------------|---------------------------------------------------------|
| Framework ML             | PyTorch + torchvision               | Standard industrie, support GPU mature                  |
| Modèle de production     | ConvNeXt-Tiny                       | 90 % accuracy, 81 % F1 macro après comparaison         |
| Data versioning          | DVC (MinIO self-hosted, DagsHub en backup) | Souveraineté des données, bascule en une commande   |
| Experiment tracking      | MLflow (DagsHub)                    | Tracking centralisé, intégré DagsHub                    |
| Serving                  | BentoML 1.4 + ONNX Runtime          | Async, validation Pydantic, OpenAPI auto                |
| Demo UI                  | Streamlit (multi-page)              | Python natif, itération rapide                          |
| Orchestration            | Airflow                             | Scheduling drift checks, retrain automatique            |
| Monitoring               | Prometheus + Grafana                | Standard industrie, scraping `/metrics`                 |
| Détection de drift       | Evidently AI                        | Rapports HTML, intégration SQLite                       |
| Alerting (routing)       | Alertmanager                        | Dédoublonnage, group_by, inhibition, hot reload         |
| Alerting (canal)         | Discord webhook + adaptateur Go     | Embeds riches, canal privé, SSO Cloudflare Access      |
| Persistance prédictions  | SQLite (WAL)                        | Léger, sans serveur, adapté au volume actuel            |
| Exposition (actuel)      | Cloudflare Tunnel + Access          | SSO email OTP, zéro port ouvert, gratuit               |
| Exposition (cible prod)  | nginx + Let's Encrypt + oauth2-proxy| Standard prod entreprise, SSL auto, SSO LDAP/OIDC      |
| Containerisation         | Docker Compose                      | Suffisant pour l'échelle TFE                            |
| CI/CD                    | GitHub Actions                      | Intégré DagsHub, jobs parallélisables                   |
| Linting                  | Ruff                                | Rapide, tout-en-un (lint + format)                      |
| Type checking            | Mypy (strict)                       | Rigueur, détection précoce d'erreurs                    |
| Tests                    | Pytest + pytest-cov                 | Standard Python, écosystème mature                      |
| Config                   | Pydantic Settings + YAML            | Validation automatique des configs                      |
"""
)
