"""Page Streamlit : infrastructure et architecture.

Affiche le schema d'architecture du projet, le statut des services
Docker, et les liens vers les outils CI/CD.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="12 - Infrastructure", layout="wide")
st.title(":building_construction: Infrastructure")

# =====================================================================
# Section 1 : Architecture
# =====================================================================
st.header("Architecture du pipeline MLOps")

st.markdown("""
```
XPS (training, GPU)                    NUC3 (serving/monitoring, CPU)
+-------------------+                 +----------------------------+
| dvc pull          |                 | Docker Desktop             |
| python train.py   |--MLflow logs--->|   api (FastAPI + ONNX)     |
| export ONNX       |                 |   demo (Streamlit)         |
| dvc push model    |--model.onnx---->|   prometheus               |
+-------------------+   (via DVC)     |   grafana                  |
                                      +----------------------------+
```

**Services Docker** :

| Service | Port | Role |
|---------|------|------|
| api | 8000 | FastAPI + ONNX Runtime (inference) |
| demo | 8501 | Streamlit (ce portfolio) |
| prometheus | 9090 | Scraping metriques /metrics |
| grafana | 3000 | Dashboards de monitoring |
""")

st.divider()

# =====================================================================
# Section 2 : Statut Docker
# =====================================================================
st.header("Statut des services Docker")

try:
    import subprocess

    result = subprocess.run(
        ["docker", "compose", "ps", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        import json

        # docker compose ps --format json retourne un JSON par ligne
        services = []
        import contextlib

        for line in result.stdout.strip().split("\n"):
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError):
                    services.append(json.loads(line))

        if services:
            import pandas as pd

            df = pd.DataFrame(services)
            display_cols = [
                c for c in ["Name", "Service", "State", "Status", "Ports"] if c in df.columns
            ]
            st.dataframe(
                df[display_cols] if display_cols else df, use_container_width=True, hide_index=True
            )
        else:
            st.info("Aucun service Docker en cours d'execution.")
    else:
        st.info("Docker Compose non disponible ou aucun service actif.")
except FileNotFoundError:
    st.info("Docker n'est pas installe ou pas dans le PATH.")
except Exception as e:
    st.warning(f"Erreur lors de la verification Docker : {e}")

st.divider()

# =====================================================================
# Section 3 : CI/CD
# =====================================================================
st.header("CI/CD")

st.markdown("""
**GitHub Actions** : pipeline automatise sur chaque push/PR.

| Job | Description |
|-----|-------------|
| lint | ruff check + ruff format --check |
| typecheck | mypy (strict) |
| docstrings | interrogate (100%) |
| test | pytest + coverage |
| build | docker compose build |

**Repository** : [DagsHub - LoicFocraud/Champy_Classifier](https://dagshub.com/LoicFocraud/Champy_Classifier)
""")

st.divider()

# =====================================================================
# Section 4 : Stack technique
# =====================================================================
st.header("Stack technique")

st.markdown("""
| Composant | Techno | Justification |
|-----------|--------|---------------|
| ML Framework | PyTorch + torchvision | Standard industrie, GPU support |
| Modele | ResNet50 (transfer learning) | Prouve sur le projet |
| Data versioning | DVC (DagsHub remote) | Deja en place |
| Experiment tracking | MLflow (DagsHub) | Integre DagsHub, gratuit |
| API serving | FastAPI | Async, Pydantic, OpenAPI auto |
| Demo UI | Streamlit (multi-page) | Python natif, rapide |
| Monitoring | Prometheus + Grafana | Standard industrie |
| Drift detection | Evidently AI | Rapports HTML |
| Containerisation | Docker Compose | Suffisant pour l'echelle |
| CI/CD | GitHub Actions | Integre DagsHub |
| Linting | Ruff | Rapide, tout-en-un |
| Type checking | Mypy (strict) | Rigueur |
| Tests | Pytest + pytest-cov | Standard |
| Config | Pydantic Settings + YAML | Validation auto |
""")
