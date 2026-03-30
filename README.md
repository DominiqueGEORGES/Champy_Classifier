# Champy Classifier - Pipeline MLOps

[![CI](https://github.com/LoicFocraud/Champy_Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/LoicFocraud/Champy_Classifier/actions/workflows/ci.yml)

Classification de champignons (30 especes de France metropolitaine) avec un pipeline MLOps complet.

**Equipe** : FOCRAUD Loic, GEORGES Dominique, PREGASSAME Saravana, SCHNEIDER Lionel
**Cadre** : TFE Master AI (DataScientest, RNCP niveau 7)

## Resultats

| Metrique | Valeur |
|----------|--------|
| Test accuracy | 83.9% |
| Test F1 macro | 77.8% |
| Classes | 30 especes |
| Images | 20 572 (curatees depuis 646K brutes) |
| Modele | ResNet50 (transfer learning PyTorch) |
| Inference | ONNX Runtime (CPU, < 50ms/image) |

## Architecture

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

## Installation

```powershell
# Cloner le repo
git clone https://dagshub.com/LoicFocraud/Champy_Classifier.git
cd Champy_Classifier

# Installer les dependances
pip install -e ".[dev]"
pre-commit install

# Recuperer les donnees (DVC)
dvc pull
```

## Utilisation

### Curation et split des donnees

```powershell
python data/curate.py         # Pipeline de curation depuis les CSV bruts
python data/data_split.py     # Split stratifie 70/15/15 (seed=42)
```

### Entrainement (sur XPS avec GPU)

```powershell
python -m src.training.train --config configs/training/default.yaml
```

### Export ONNX

```powershell
python -m src.models.export_onnx
```

### API FastAPI (inference)

```powershell
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Demo Streamlit

```powershell
streamlit run demo/app.py
```

### Docker Compose (tout-en-un)

```powershell
docker compose up -d    # api + demo + prometheus + grafana
docker compose ps       # statut des services
docker compose down     # arret
```

| Service | Port | URL |
|---------|------|-----|
| API FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

### Tests et qualite

```powershell
pytest tests/unit/ -v              # Tests unitaires
ruff check src/ tests/ demo/       # Linting
mypy src/                          # Type checking
interrogate src/ -c pyproject.toml # Docstrings
```

## Structure du projet

```
Champy_Classifier/
+-- src/
|   +-- config.py              # Pydantic Settings (MLflow, Training, Serving)
|   +-- data/
|   |   +-- dataset.py         # PyTorch Dataset + transforms
|   |   +-- dataloader.py      # DataLoader factory + WeightedRandomSampler
|   +-- models/
|   |   +-- resnet.py          # ResNet50 transfer learning
|   |   +-- export_onnx.py     # Export ONNX + validation
|   +-- training/
|   |   +-- train.py           # Boucle d'entrainement (AMP, MLflow)
|   |   +-- evaluate.py        # Metriques, confusion matrix
|   |   +-- callbacks.py       # Early stopping, checkpointing
|   +-- serving/
|   |   +-- app.py             # FastAPI (predict, health, metrics)
|   |   +-- schemas.py         # Pydantic request/response
|   |   +-- middleware.py      # Metriques Prometheus
|   +-- monitoring/
|       +-- drift.py           # Detection de drift (Evidently)
+-- demo/
|   +-- app.py                 # Streamlit - page d'accueil
|   +-- lib/                   # Helpers partages
|   +-- pages/                 # 12 pages du portfolio MLOps
+-- data/
|   +-- curate.py              # Pipeline de curation from scratch
|   +-- data_split.py          # Split stratifie reproductible
+-- configs/training/
|   +-- default.yaml           # Hyperparametres par defaut
+-- docker/
|   +-- Dockerfile.api         # Image API
|   +-- Dockerfile.demo        # Image Streamlit
+-- docker-compose.yml         # Orchestration 4 services
+-- tests/unit/                # 51 tests unitaires
```

## Stack technique

| Composant | Techno |
|-----------|--------|
| ML Framework | PyTorch + torchvision |
| Inference | ONNX Runtime |
| Data versioning | DVC (DagsHub) |
| Experiment tracking | MLflow (DagsHub) |
| API | FastAPI |
| Demo | Streamlit (12 pages) |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Qualite | Ruff, Mypy, Interrogate, pre-commit |
| Containerisation | Docker Compose |
