# CLAUDE.md -- Champy Classifier (MLOps TFE)

## Projet

Classification de champignons (30 espèces, ~700K images) dans le cadre d'un Master AI (DataScientest, RNCP niveau 7). L'objectif est de démontrer une maîtrise complète de la chaîne MLOps, pas juste du ML.

Repo : DagsHub `LoicFocraud/Champy_Classifier`
DVC remote : DagsHub (S3-compatible)
Tracking : MLflow (DagsHub intégré)

## Environnement de développement

**OS : Windows 11 Pro - PowerShell natif (pas de WSL)**
**Docker : Docker Desktop for Windows**

Conséquences :
- Toutes les commandes shell sont en **PowerShell**, pas bash
- Les chemins utilisent `pathlib.Path` dans le code Python (jamais de `/` ou `\` hardcodé)
- Le Makefile est remplacé par `tasks.py` (Python invoke, cross-platform)
- Docker Desktop gère les containers Linux via son backend Hyper-V
- Les volumes Docker utilisent la syntaxe Windows : `${PWD}/models:/app/models:ro`
- Les scripts `.sh` sont remplacés par des scripts `.ps1` ou intégrés dans `tasks.py`
- Les line endings doivent être LF dans le repo (`.gitattributes` avec `* text=auto eol=lf`)

## Architecture cible

```
Champy_Classifier/
├── CLAUDE.md                    # Ce fichier
├── LOGBOOK.md                   # Cahier de bord MLOps (pour le mémoire)
├── PLAYBOOK.md                  # Référentiel MLOps réutilisable
├── .claude/
│   └── skills/
│       └── champy-mlops/
│           └── SKILL.md         # Patterns MLOps pour Claude Code
├── .env.example                 # Variables d'environnement (template)
├── .env                         # Variables réelles (JAMAIS committé)
├── .gitignore
├── .gitattributes               # Line endings LF
├── .dvc/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Lint, tests, build images
│       └── cd.yml               # Deploy (optionnel)
├── data/
│   ├── raw/                     # Géré par DVC (700K images)
│   ├── processed/               # Splits train/val/test (DVC)
│   └── data_split.py            # Script de split reproductible
├── src/
│   ├── __init__.py
│   ├── config.py                # Pydantic Settings, chemins, hyperparams
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset + transforms
│   │   └── dataloader.py        # DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py            # ResNet50 fine-tuning
│   │   └── registry.py          # Model registry helpers (MLflow)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Boucle d'entraînement
│   │   ├── evaluate.py          # Métriques, confusion matrix
│   │   └── callbacks.py         # Early stopping, checkpointing
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI inference server
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── middleware.py        # Logging, metrics Prometheus
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py             # Data/concept drift detection
│       └── metrics.py           # Custom Prometheus metrics
├── demo/
│   ├── app.py                   # Streamlit - page d'accueil (vue pipeline)
│   ├── lib/
│   │   ├── data_utils.py        # Helpers : scan disque, stats images
│   │   ├── mlflow_utils.py      # Helpers : search_runs, artifacts, registry
│   │   ├── api_utils.py         # Helpers : appels FastAPI, Prometheus
│   │   └── viz.py               # Helpers : plots Plotly/Matplotlib réutilisables
│   ├── pages/
│   │   ├── 01_donnees_brutes.py      # EDA, distribution classes, qualité
│   │   ├── 02_nettoyage.py           # Avant/après, doublons, corruptions
│   │   ├── 03_augmentation.py        # Exemples visuels des transforms
│   │   ├── 04_split.py               # Stratification, stats par split
│   │   ├── 05_entrainement.py        # Courbes live depuis MLflow
│   │   ├── 06_evaluation.py          # Confusion matrix, F1, GradCAM
│   │   ├── 07_model_registry.py      # Versions, staging/prod, ONNX benchmark
│   │   ├── 08_prediction.py          # Upload image, top-5, confiance live
│   │   ├── 09_api.py                 # Swagger embed, latence, throughput
│   │   ├── 10_monitoring.py          # Métriques Prometheus, Grafana embed
│   │   ├── 11_drift.py               # Rapports Evidently on-demand
│   │   └── 12_infrastructure.py      # Docker, CI/CD, architecture
│   └── assets/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docker/
│   ├── Dockerfile.train         # Image entraînement (GPU)
│   ├── Dockerfile.api           # Image API inference
│   ├── Dockerfile.demo          # Image Streamlit
│   └── Dockerfile.monitoring    # Prometheus + Grafana (ou all-in-one)
├── docker-compose.yml           # Orchestration complète (NUC3)
├── docker-compose.dev.yml       # Override dev (volumes, ports debug)
├── docker-compose.train.yml     # Compose training (XPS GPU)
├── configs/
│   ├── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   │       └── champy.json      # Dashboard pré-configuré
│   └── training/
│       └── default.yaml         # Hyperparams par défaut
├── notebooks/
│   └── eda.ipynb                # Exploration data (optionnel)
├── tasks.py                     # Task runner (invoke, remplace Makefile)
├── pyproject.toml               # Dependencies + config Ruff/Mypy
├── requirements.txt             # Fallback pip
└── README.md
```

## INVARIANTS -- NE JAMAIS VIOLER

1. **Reproductibilité** : Tout run d'entraînement DOIT être traçable (seed fixé, hyperparams loggués dans MLflow, données versionnées DVC).
2. **Pas de données dans Git** : Les images passent UNIQUEMENT par DVC. `.dvc` files dans git, pas les images.
3. **Pas de secrets dans le code** : Tout via `.env` + `pydantic-settings`. Le `.env` est dans `.gitignore`.
4. **Tests avant merge** : Aucun push sur `main` sans que les tests passent (CI).
5. **Docker reproductible** : Chaque service doit tourner dans son container. Pas de "ça marche sur ma machine".
6. **Config séparée du code** : Hyperparams dans `configs/`, pas hardcodés.
7. **ONNX export** : Le modèle servi en production est au format ONNX (ou TorchScript). Pas de PyTorch brut en inference.
8. **Type hints partout** : Tout le code Python est typé. Mypy doit passer.
9. **Docstrings en français** : Fonctions publiques documentées (Google style, en français). `interrogate` doit passer à 100%.
10. **Pas d'em dash** : Utiliser des tirets normaux (-) dans tout le texte généré.
11. **Cross-platform** : Pas de commandes bash, pas de chemins Unix hardcodés. PowerShell + pathlib.
12. **Streamlit zéro hardcoded** : AUCUNE valeur (accuracy, nb images, noms de classes, etc.) n'est écrite en dur dans le Streamlit. Tout est lu dynamiquement aux sources (MLflow, disque, API, Prometheus). Si la source n'est pas dispo, afficher un message clair, pas une valeur par défaut silencieuse.
13. **Pre-commit obligatoire** : Aucun commit ne passe sans les hooks (ruff, mypy, interrogate). `pre-commit install` sur chaque machine.

## Stack technique

| Composant | Techno | Justification |
|-----------|--------|---------------|
| ML Framework | PyTorch + torchvision | Standard industrie, GPU support |
| Modèle | ResNet50 (transfer learning) | Prouvé sur le projet (95.4% accuracy) |
| Data versioning | DVC (DagsHub remote) | Déjà en place |
| Experiment tracking | MLflow (DagsHub) | Intégré DagsHub, gratuit |
| API serving | FastAPI | Async, Pydantic natif, OpenAPI auto |
| Demo UI | Streamlit (multi-page) | Rapide, Python natif |
| Monitoring | Prometheus + Grafana | Standard industrie |
| Drift detection | Evidently AI | Rapports data/model drift |
| Containerisation | Docker Desktop + Compose | Tout dockerisé, Hyper-V backend |
| CI/CD | GitHub Actions | Intégré DagsHub/GitHub |
| Linting | Ruff | Rapide, remplace flake8+isort+black |
| Type checking | Mypy | Rigueur |
| Tests | Pytest + pytest-cov | Standard |
| Config | Pydantic Settings + YAML | Validation automatique |
| Task runner | invoke (tasks.py) | Cross-platform, Python natif, remplace Make |
| Pre-commit | pre-commit + interrogate | Hooks automatiques : lint, types, docstrings |

## Infrastructure - Répartition des machines

### NUC3 - Hub MLOps (Windows 11 Pro, Ryzen AI 9, 96GB RAM, pas de GPU dédié)
Rôle : héberge TOUT sauf l'entraînement GPU.
- Développement principal (VS Code + Claude Code)
- API FastAPI (inference ONNX sur CPU - largement suffisant pour du serving)
- Streamlit demo
- Prometheus + Grafana (via Docker Desktop)
- MLflow (si instance locale, sinon DagsHub)
- Docker Compose principal (`docker-compose.yml`)

Le Ryzen AI 9 a un NPU (XDNA) mais celui-ci n'est PAS exploitable par PyTorch.
L'inference ONNX sur CPU avec 96GB RAM est très performante pour du ResNet50 (< 50ms/image).

### XPS 15 9520 (x1 ou x2) - Training GPU
Specs : i7-12700H, RTX 3050 Ti (4GB VRAM), Windows 11
Rôle : entraînement uniquement.
- Batch size max ~16-32 en mixed precision (fp16) pour ResNet50
- Utiliser `torch.cuda.amp` systématiquement
- Si 4GB VRAM insuffisant : gradient accumulation (accumulate_steps=2 ou 4)
- Le training pousse les métriques vers MLflow (DagsHub) automatiquement
- Le modèle exporté (ONNX) est ensuite déployé sur le NUC3
- CUDA + cuDNN installés nativement (pas besoin de Docker GPU sur Windows)

### Workflow inter-machines

```
XPS (training, Windows)                 NUC3 (serving/monitoring, Windows)
┌─────────────────┐                    ┌──────────────────────────┐
│ dvc pull         │                    │ Docker Desktop           │
│ python train.py  │--MLflow logs------>│   api (FastAPI + ONNX)   │
│ export ONNX      │                    │   demo (Streamlit)       │
│ dvc push model   │--model.onnx------>│   prometheus             │
└─────────────────┘   (via DVC/        │   grafana                │
                       scp/shared)      └──────────────────────────┘
```

Transfert du modèle XPS -> NUC3 :
- Option A : DVC push/pull (le modèle ONNX est un artefact DVC)
- Option B : Dossier partagé réseau (SMB)
- Option C : MLflow model registry (download depuis DagsHub)
- Option D : `scp` via OpenSSH (intégré Windows 11)

### Training sur le NUC3 (CPU fallback)

Le NUC3 PEUT entraîner sur CPU (96GB RAM = pas de contrainte mémoire), mais c'est 5-10x plus lent qu'avec la RTX 3050 Ti. Acceptable pour :
- Debug rapide (sous-ensemble de données)
- Fine-tuning léger (quelques epochs)
- Tests de pipeline complets

Pour un run complet (700K images, 15-20 epochs), préférer le XPS avec GPU.

### Estimations de temps (un XPS, RTX 3050 Ti, batch=16, fp16)

- ~45 min/epoch (estimation, dépend du nombre de workers DataLoader)
- ~15-20 epochs avant convergence (avec early stopping)
- Total : ~12-15h pour un run complet

Avec deux XPS, on peut paralléliser des expériences (ex: un teste lr=1e-3, l'autre lr=3e-4).

## Conventions de code

### Langue
- **Noms de variables, fonctions, classes, modules** : anglais (`train_loader`, `ModelConfig`, `split_dataset`)
- **Toute la documentation** : français (docstrings, commentaires, README, LOGBOOK, messages d'erreur utilisateur)
- **Messages de commit** : anglais (format conventionnel `feat:`, `fix:`, etc.)
- **Docstrings** : Google style, en français

Exemple de docstring :
```python
def split_dataset(
    data_dir: Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Divise le dataset en train/val/test de manière stratifiée.

    Charge la liste d'exclusion (excluded.json) et ne conserve que
    les images originales. Le split est reproductible grâce au seed.

    Args:
        data_dir: Répertoire contenant les images par classe.
        ratios: Proportions train/val/test (doit sommer à 1.0).
        seed: Graine pour la reproductibilité.

    Returns:
        Dictionnaire avec clés 'train', 'val', 'test' et listes de chemins.

    Raises:
        FileNotFoundError: Si data_dir n'existe pas.
        ValueError: Si les ratios ne somment pas à 1.0.
    """
```

### Style et qualité
- Python 3.11+
- Ruff pour le formatting ET le linting (config dans `pyproject.toml`)
- Imports : stdlib, third-party, local (Ruff gère automatiquement)
- Naming : snake_case partout sauf classes (PascalCase)
- Logging : `loguru` (pas print)
- Pas de notebooks en production (OK pour EDA uniquement)
- **Chemins : `pathlib.Path` exclusivement** (jamais de string concatenation)
- **Pas de commandes bash dans le code** : tout passe par Python ou PowerShell
- Type hints sur toutes les fonctions (Mypy strict)

### Pre-commit (cohérence automatique)

Chaque commit passe automatiquement par ces hooks (`.pre-commit-config.yaml`) :

| Hook | Rôle |
|------|------|
| ruff (check + format) | Linting et formatage |
| mypy | Vérification des types |
| interrogate | Vérifie que toutes les fonctions/classes ont une docstring |
| trailing-whitespace | Supprime les espaces en fin de ligne |
| end-of-file-fixer | Assure un newline en fin de fichier |
| check-yaml | Valide les fichiers YAML |
| check-toml | Valide pyproject.toml |
| check-added-large-files | Empêche de commiter des fichiers > 500KB (images !) |

Installation :
```powershell
pip install pre-commit
pre-commit install
```

Si un hook échoue, le commit est bloqué. Aucune exception.

## Task runner (invoke) - Commandes principales

```powershell
# Depuis PowerShell, à la racine du projet :
invoke setup          # Install deps, pull DVC data
invoke train          # Lance entraînement (local GPU ou Docker)
invoke serve          # Lance API + Demo + Monitoring (Docker Compose)
invoke stop           # Arrête les services Docker
invoke test           # Pytest + coverage
invoke lint           # Ruff check + mypy
invoke format         # Ruff format
invoke build          # Build toutes les images Docker
invoke export-onnx    # Export modèle ONNX
invoke clean          # Nettoyage __pycache__, .pytest_cache, etc.
```

## Workflow MLOps attendu

### 1. Data Pipeline
```
DVC pull -> Split (train/val/test, stratifié) -> Transforms (augmentation) -> DataLoader
```
- Split ratio : 70/15/15
- Augmentation : RandomHorizontalFlip, RandomRotation(15), ColorJitter, Normalize(ImageNet)
- Validation/test : Resize + CenterCrop + Normalize uniquement

### 2. Training Pipeline (sur XPS)
```
Config YAML -> Train loop (AMP) -> Log metrics MLflow -> Save best model -> Register model MLflow
```
- Mixed precision (AMP) obligatoire
- Early stopping (patience=5, monitor=val_loss)
- Checkpointing du meilleur modèle
- Log : loss, accuracy, F1 (macro), confusion matrix, learning curves
- Hyperparams loggués : lr, batch_size, epochs, optimizer, scheduler, seed

### 3. Model Registry
```
MLflow register -> Stage (Staging/Production) -> Export ONNX -> Version tag
```

### 4. Serving Pipeline (sur NUC3, Docker Desktop)
```
Load ONNX model -> FastAPI endpoints -> Prometheus metrics -> Health checks
```
Endpoints :
- `POST /predict` : image -> top-5 prédictions avec scores
- `GET /health` : status + model version
- `GET /metrics` : Prometheus format
- `GET /model/info` : metadata modèle

### 5. Monitoring (sur NUC3, Docker Desktop)
```
Prometheus scrape /metrics -> Grafana dashboards -> Alertes (optionnel)
```
Métriques :
- Latence inference (p50, p95, p99)
- Requêtes/seconde
- Distribution des classes prédites (drift proxy)
- Confiance moyenne des prédictions
- Erreurs HTTP

### 6. Demo Streamlit - Portfolio MLOps interactif (sur NUC3, Docker Desktop)

**Principe fondamental : ZERO HARDCODED.**
Le Streamlit ne stocke rien, ne calcule rien de permanent. C'est une vitrine dynamique qui tire toutes ses données aux sources en temps réel. Si on relance un training ou si on change de modèle, les pages se mettent à jour automatiquement.

**Rôle du Streamlit vs les autres outils** :
- **MLflow** = carnet de labo (source de vérité pour les expériences). Public cible : data scientists.
- **Grafana** = salle de contrôle (monitoring temps réel). Public cible : ops/SRE.
- **Streamlit** = vitrine narrative (raconte l'histoire du projet). Public cible : jury, clients, managers. Il consomme MLflow et Prometheus, il ne les remplace pas.

**Sources de données par page** :

| Pages | Source | Méthode |
|-------|--------|---------|
| 01-04 Data | Disque (DVC) + artefacts JSON | `pathlib` scan, `json.load()` |
| 05 Training | MLflow | `mlflow.search_runs()`, `mlflow.get_run()` |
| 06 Evaluation | MLflow artefacts | `mlflow.artifacts.download_artifacts()` |
| 07 Registry | MLflow Model Registry | `mlflow.search_model_versions()` |
| 08 Prediction | ONNX Runtime ou FastAPI | `ort.InferenceSession` ou `httpx.post()` |
| 09 API | FastAPI | `httpx.get("/docs")`, `/health`, `/metrics` |
| 10 Monitoring | Prometheus | `httpx.get()` sur API PromQL |
| 11 Drift | Evidently | Génération on-demand, rapport HTML |
| 12 Infra | Docker, GitHub API | `docker ps`, GitHub Actions API |

**Construction incrémentale** : Les pages Streamlit se construisent au fil des étapes. Quand l'étape 2 (data pipeline) est terminée, on crée les pages 01-04. Quand le training est fait, pages 05-06. Etc. Le Streamlit grandit avec le projet.

**Pattern standard pour chaque page** :
```python
# JAMAIS ça :
accuracy = 0.954

# TOUJOURS ça :
try:
    runs = mlflow.search_runs(order_by=["metrics.val_acc DESC"], max_results=1)
    accuracy = runs.iloc[0]["metrics.val_acc"]
    st.metric("Best accuracy", f"{accuracy:.1%}")
except Exception as e:
    st.warning(f"MLflow non disponible : {e}")
```

**Helpers partagés** : Le code d'accès aux sources est factorisé dans `demo/lib/` (mlflow_utils.py, data_utils.py, api_utils.py, viz.py) pour éviter la duplication entre pages.

## Docker Compose - Services (NUC3, Docker Desktop)

```yaml
services:
  api:           # FastAPI inference ONNX (port 8000)
  demo:          # Streamlit (port 8501)
  prometheus:    # Scraping metrics (port 9090)
  grafana:       # Dashboards (port 3000)
```

Sur XPS, le training peut se lancer :
- **Nativement** (recommandé, accès direct au GPU CUDA) :
  ```powershell
  python -m src.training.train --config configs/training/default.yaml
  ```
- Via Docker (nécessite NVIDIA Container Toolkit pour Windows) :
  ```powershell
  docker compose -f docker-compose.train.yml run --rm --gpus all train
  ```

Note : Docker GPU sur Windows nécessite Docker Desktop + WSL2 backend + NVIDIA drivers.
Si pas de WSL2, le training se fait nativement sur le XPS (plus simple, même résultat).

## Règles pour Claude Code

1. **Toujours lire ce fichier en premier** avant de toucher au code.
2. **Proposer les changements**, ne pas les appliquer silencieusement sur des fichiers existants sans confirmation.
3. **Un commit = une fonctionnalité**. Messages de commit en anglais, format conventionnel : `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `ci:`.
4. **Tester avant de déclarer terminé**. Si tu crées du code, crée aussi le test.
5. **Vérifier les contraintes VRAM (4GB)** avant de proposer des architectures ou batch sizes.
6. **Ne jamais hardcoder** de chemins absolus. Utiliser `pathlib.Path` et la config.
7. **Documenter en français** : docstrings (Google style), commentaires, messages d'erreur. Noms de variables/fonctions/classes en anglais.
8. **Si un INVARIANT risque d'être violé**, s'arrêter et demander confirmation.
9. **Distinguer les contextes machine** : training = XPS (GPU natif), serving/monitoring = NUC3 (Docker Desktop, CPU).
10. **ONNX inference sur CPU** : ne jamais supposer qu'un GPU est dispo côté serving.
11. **Environnement Windows** : PowerShell, pas bash. Pas de `rm -rf`, `grep`, `sed`, `awk`. Utiliser les équivalents Python ou PowerShell.
12. **Mettre à jour LOGBOOK.md et PLAYBOOK.md** à chaque fin d'étape (voir SKILL.md pour le détail).
13. **Pre-commit doit passer** : avant de déclarer un fichier terminé, vérifier que `ruff check`, `ruff format --check`, `mypy`, et `interrogate` passent sur ce fichier. Chaque fonction publique DOIT avoir une docstring en français.
14. **Cohérence entre fichiers** : même style de docstring, mêmes patterns d'import, même gestion d'erreurs, même utilisation de loguru dans tout le projet. Si un pattern existe déjà dans un fichier, le réutiliser à l'identique.
