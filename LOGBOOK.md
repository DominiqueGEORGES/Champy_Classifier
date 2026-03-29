# Champy Classifier - Cahier de bord MLOps

## Informations générales

- **Projet** : Classification de champignons (30 espèces, ~700K images)
- **Cadre** : TFE Master AI (DataScientest, RNCP niveau 7, promotion 2026)
- **Equipe** : [à compléter]
- **Repo** : DagsHub - LoicFocraud/Champy_Classifier
- **Date de début** : [à compléter]
- **Date de soutenance** : [à compléter]

---

## Etape 0 - Cadrage et architecture

**Date** : 2026-03-28
**Objectif** : Définir l'architecture MLOps cible, la répartition hardware, les conventions.

### Décisions prises

| Décision | Choix | Alternatives envisagées | Justification |
|----------|-------|------------------------|---------------|
| ML Framework | PyTorch | TensorFlow, JAX | Standard industrie, écosystème torchvision, compatibilité ONNX |
| Modèle de base | ResNet50 (transfer learning) | EfficientNet, ViT | Déjà validé sur le projet (95.4% accuracy au TFE), bon rapport complexité/performance |
| Data versioning | DVC (remote DagsHub) | Git LFS, Lakefs | Déjà en place, intégré DagsHub |
| Experiment tracking | MLflow (DagsHub) | W&B, Neptune | Gratuit, intégré DagsHub, self-hosted possible |
| API serving | FastAPI | Flask, TorchServe | Async natif, Pydantic, OpenAPI auto-générée |
| Format inference | ONNX Runtime | TorchScript, TensorRT | Portable, CPU-optimisé, pas de dépendance PyTorch en prod |
| Demo | Streamlit | Gradio, React | Python natif, multi-page, rapide à prototyper |
| Monitoring | Prometheus + Grafana | Datadog, ELK | Standard industrie, open source, léger |
| Drift detection | Evidently AI | Alibi Detect, NannyML | API simple, rapports HTML intégrés, gratuit |
| CI/CD | GitHub Actions | GitLab CI, Jenkins | Intégré DagsHub/GitHub |
| Linting | Ruff | Black + Flake8 + isort | Un seul outil, 10-100x plus rapide |
| Containerisation | Docker Compose | Kubernetes | Suffisant pour l'échelle du projet, pas d'overhead K8s |

### Répartition hardware

| Machine | Rôle | Specs clés |
|---------|------|-----------|
| NUC3 | Hub MLOps (dev, serving, monitoring) | Ryzen AI 9, 96GB RAM, pas de GPU dédié |
| XPS 9520 #1 | Training GPU | i7-12700H, RTX 3050 Ti (4GB VRAM) |
| XPS 9520 #2 (optionnel) | Training GPU parallèle | Idem |

### Artefacts produits
- `CLAUDE.md` - Fichier de gouvernance Claude Code
- `.claude/skills/champy-mlops/SKILL.md` - Patterns MLOps
- `LOGBOOK.md` - Cahier de bord MLOps
- `PLAYBOOK.md` - Referentiel MLOps reutilisable
- `tasks.py` - Task runner invoke (remplace Makefile)

---

## Etape 1 - Configuration et structure du projet

**Date** : 2026-03-28
**Objectif** : Mettre en place pyproject.toml, config Pydantic, .env, .gitignore, structure des repertoires.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Migration framework | PyTorch | Rester sur Keras/TF | Standard industrie, meilleur ecosysteme ONNX, plus flexible pour fine-tuning, AMP natif |
| Gestion deps | pyproject.toml (hatchling) + uv | requirements.txt seul | Standard PEP 621, sections [dev] separees, config Ruff/Mypy/Pytest integree |
| Config runtime | Pydantic Settings | python-decouple, dynaconf | Validation automatique des types, integration .env + YAML, coherent avec FastAPI |
| Python version | >=3.11 | 3.12 only | 3.11 minimum pour match_case et perf, mais compatible 3.12 |
| Task runner | invoke (tasks.py) | Makefile, Nox | Cross-platform Windows/Linux, Python natif |
| Line endings | LF force (.gitattributes) | Mixed | Evite les diffs parasites entre Windows et CI Linux |

### Problemes rencontres
- `requirements.txt` existant etait en encodage UTF-16 avec espaces entre chaque caractere - inutilisable, regenere proprement. Attention : meme apres regeneration, le fichier repassait en UTF-16 silencieusement (probleme Windows + outil d'ecriture). Solution : forcer l'ecriture en UTF-8 explicitement.
- `pyproject.toml` initial etait un dump brut de `pip freeze` (190 deps dont TensorFlow, yfinance, jupyter) - nettoye pour ne garder que les deps du projet
- Aucun `__init__.py` dans les sous-packages de `src/` - ajoutes (les dossiers existaient mais etaient vides)
- `.gitignore` ne couvrait que `/data` et `/models` - etendu a .env, caches, IDE, etc.
- `.gitignore` excluait par erreur `data.dvc` et `models.dvc` - corrige : ces fichiers .dvc DOIVENT etre dans git (c'est le principe meme de DVC)
- `httpx` et `plotly` manquants dans les deps principales alors qu'ils sont necessaires pour les helpers Streamlit (`demo/lib/`) et les appels API

### Artefacts produits
- `pyproject.toml` - deps PyTorch + plotly + httpx + config Ruff/Mypy/Pytest
- `requirements.txt` - fallback pip depuis pyproject.toml (UTF-8)
- `.env.example` - template variables d'environnement
- `.gitattributes` - LF force + binaires declares
- `.gitignore` - complete (env, caches, data, models, IDE) - data.dvc/models.dvc conserves dans git
- `src/config.py` - Pydantic Settings (MLflow, Training, Serving) avec chargement YAML
- `configs/training/default.yaml` - hyperparams par defaut
- `src/{data,training,inference,models,serving,monitoring}/__init__.py`
- `demo/{__init__.py,lib/__init__.py}` + structure `pages/`, `assets/`
- `tasks.py` - task runner invoke (18 taches)

### Metriques / Resultats
- pyproject.toml : 190 deps -> ~32 deps directes + 4 dev
- Sous-packages src/ : 6 avec __init__.py (data, training, inference, models, serving, monitoring)
- Structure demo/ creee : lib/, pages/, assets/

---

## Etape 2 - Data pipeline

**Date** : 2026-03-28
**Objectif** : DVC pull, split stratifie, Dataset PyTorch, DataLoader, augmentation.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Source de travail | data/processed/ (25 850 images) | data/raw/ (646K) | Deja nettoyees et classees par le notebook 0, equilibrees |
| Immutabilite des donnees | Exclusion list (excluded.json) | Suppression des doublons | data/processed/ est partage via DVC avec l'equipe, ne jamais modifier |
| **Option B : originaux uniquement** | Exclure les 13 690 augmentations TF, garder 12 131 originaux | Option A (tout garder), Option C (hybride) | Les augmentations TF ne sont pas reproductibles (non seedees), les originaux suffisent, PyTorch gerera ses propres augmentations + WeightedRandomSampler pour l'equilibrage |
| Detection doublons | Hash MD5 partiel (8KB debut + 4KB fin + taille) | Hash complet, perceptual hash | Suffisant pour des fichiers identiques, rapide sur 25K images |
| Gestion doublons | Garder l'ID le plus bas, exclure l'autre | Supprimer, deplacer dans quarantaine | Coherent avec l'immutabilite, tracable, reversible |
| Split ratio | 70/15/15 | 80/10/10 | Distribution naturelle desequilibree (52-900/classe), besoin de val/test suffisants |
| Augmentation | PyTorch transforms au training | Reutiliser les augmentations TF existantes | Reproductible, seedable, plus de controle |
| Docstrings | Francais (Google style), verifie par interrogate | Anglais | Convention du projet, mieux pour le memoire |
| Pre-commit | ruff + mypy + interrogate + hooks generaux | Verification manuelle | Garantit la qualite a chaque commit |
| Notebooks legacy | Deplaces dans notebooks/legacy/ | Supprimer | Conserves pour reference, plus dans le chemin principal |

### Sous-etape 2a - Etat des lieux

**Constat** : les donnees sont deja propres et equilibrees (850-900 images/classe, 0 corruption).
Le gros du nettoyage a ete fait par le notebook 0 (646K -> 25 850).
Dimensions variables (320x240 majoritaire ~55%), necessitent Resize+CenterCrop dans le pipeline.

### Sous-etape 2b - Nettoyage par exclusion (Option B - originaux uniquement)

**Politique** : donnees source immutables, filtrage par liste d'exclusion, pas de suppression.

Audit complet du pipeline notebook 0 (voir rapport d'audit dans la conversation) :
- Les 646K images raw viennent de mushroomobserver.org (thumbnails 320px)
- Le notebook 0 filtre (merge top30, confidence >= 92, filtre ResNet50 ImageNet) puis resamble (under a 900, over a 850 avec augmentations TF)
- **12 160 originaux** (copies bit-pour-bit de raw) + **13 690 augmentees** (transforms TF non seedees)

Decision Option B : exclure toutes les augmentations TF + les 29 doublons.
- 13 690 images `_N.jpg` exclues (raison : `tf_augmentation_legacy`)
- 29 doublons entre originaux exclus (raison : `duplicate`)
- **12 131 images originales retenues**

Le desequilibre naturel est accepte (52 a 900 par classe). PyTorch gerera l'equilibrage via `WeightedRandomSampler`.

### Sous-etape 2c - Split stratifie sur originaux (v2)

Script `data/data_split.py` (docstrings FR) : charge `excluded.json` (13 719 exclusions), split stratifie 70/15/15 avec seed=42.
Stratification verifiee sur toutes les classes, y compris les plus petites (Russula emetica, 52 images : 69.2/15.4/15.4%).

### Problemes rencontres
- Le repo contient 646K images brutes (data/raw/) mais seulement 25 850 sont exploitables (data/processed/)
- Les augmentations TF du notebook 0 ne sont pas reproductibles (transforms aleatoires non seedees) - exclues
- Le dataset naturel est tres desequilibre (ratio max/min = 17.3x) - necessitera WeightedRandomSampler
- `reset_index(drop=True)` dans la boucle for du notebook 0 (bug de perf, non corrige car on ne relance pas)
- Incoherence de nommage des modeles entre notebook 3 (.keras) et notebook 4 (.h5)

### Artefacts produits
- `data/raw_stats.json` - rapport complet des donnees brutes
- `data/excluded.json` - 13 719 exclusions (13 690 augmentees + 29 doublons)
- `data/cleaning_report.json` - rapport avant/apres (25 850 -> 12 131)
- `data/data_split.py` - script reproductible avec docstrings FR
- `data/split_manifest.csv` - 12 132 lignes (header + 12 131 entrees)
- `data/split_stats.json` - stats par classe par split
- `.pre-commit-config.yaml` - hooks ruff + mypy + interrogate
- `notebooks/legacy/` - 5 notebooks archives

### Sous-etape 2d - Dataset PyTorch + DataLoader

`src/data/dataset.py` : classe `MushroomDataset` qui lit le manifest CSV et charge les images
depuis `data/processed/`. Transforms configurables :
- Train : Resize(256), RandomCrop(224), RandomHorizontalFlip, RandomRotation(15), ColorJitter, Normalize(ImageNet)
- Val/Test : Resize(256), CenterCrop(224), Normalize(ImageNet)

`src/data/dataloader.py` : factory de DataLoaders avec :
- `WeightedRandomSampler` pour le train (poids inversement proportionnels a la frequence de classe)
- `num_workers=0` par defaut (Windows), configurable via TrainingConfig
- `pin_memory=True` pour le transfert GPU
- `drop_last=True` sur le train (eviter un dernier batch trop petit)
- Label map partage entre les 3 splits (construit depuis le train, reutilise pour val/test)

Tests unitaires : 23 tests, 23 passed (test_dataset.py + test_dataloader.py)

### Metriques / Resultats
- Images : 25 850 (processed) -> 12 131 (originaux retenus apres exclusion)
- Classes : 30, desequilibrees naturellement (52 a 900 par classe)
- Split : train=8 491, val=1 819, test=1 821
- Stratification : 70.0% +/- 0.8% sur toutes les classes
- Tests : 23 passed (15 dataset + 8 dataloader)
- Pre-commit : tous les hooks passent (ruff, mypy, interrogate)

---

## Etape 3 - Training pipeline

**Date** : [à compléter]
**Objectif** : Boucle d'entraînement PyTorch, AMP, MLflow tracking, early stopping, checkpointing.

### Décisions prises

| Décision | Choix | Alternatives envisagées | Justification |
|----------|-------|------------------------|---------------|
| Optimizer | AdamW | SGD+momentum, LARS | Convergence rapide, weight decay intégré |
| Scheduler | CosineAnnealingLR | StepLR, OneCycleLR | Smooth decay, bon pour fine-tuning |
| Batch size | 16 (fp16) | 32 | Contraint par 4GB VRAM |
| | | | |

### Problèmes rencontrés
-

### Artefacts produits
-

### Métriques / Résultats
- Best val accuracy :
- Best val F1 (macro) :
- Nombre d'epochs avant convergence :
- Temps total d'entraînement :

---

## Etape 4 - Model registry et export

**Date** : [à compléter]
**Objectif** : Enregistrer le modèle dans MLflow, promouvoir en Production, exporter ONNX.

### Décisions prises

| Décision | Choix | Alternatives envisagées | Justification |
|----------|-------|------------------------|---------------|
| Format export | ONNX (opset 17) | TorchScript | Portable, optimisé CPU, runtime léger |
| | | | |

### Problèmes rencontrés
-

### Artefacts produits
-

### Métriques / Résultats
- Taille modèle ONNX :
- Latence inference CPU (NUC3) :
- Ecart accuracy PyTorch vs ONNX :

---

## Etape 5 - API serving (FastAPI)

**Date** : [à compléter]
**Objectif** : API REST pour inference, métriques Prometheus, health checks.

### Décisions prises

| Décision | Choix | Alternatives envisagées | Justification |
|----------|-------|------------------------|---------------|
| | | | |

### Problèmes rencontrés
-

### Endpoints implémentés
- `POST /predict` :
- `GET /health` :
- `GET /metrics` :
- `GET /model/info` :

### Métriques / Résultats
- Latence p50 :
- Latence p95 :
- Throughput max :

---

## Etape 6 - Demo Streamlit

**Date** : [à compléter]
**Objectif** : Interface de démonstration multi-page.

### Pages implémentées
1. Prédiction (upload + top-5 + GradCAM) :
2. Exploration dataset :
3. Métriques modèle (MLflow) :
4. Monitoring live :

### Captures d'écran
[à ajouter]

---

## Etape 7 - Monitoring (Prometheus + Grafana)

**Date** : [à compléter]
**Objectif** : Dashboards de monitoring, détection de drift.

### Métriques monitorées
-

### Alertes configurées
-

### Drift detection (Evidently)
-

---

## Etape 8 - Dockerisation

**Date** : [à compléter]
**Objectif** : Containeriser tous les services, docker-compose fonctionnel.

### Images Docker

| Image | Base | Taille | Build time |
|-------|------|--------|-----------|
| Dockerfile.train | | | |
| Dockerfile.api | | | |
| Dockerfile.demo | | | |

### Docker Compose
- Services :
- Volumes :
- Networks :

---

## Etape 9 - CI/CD (GitHub Actions)

**Date** : [à compléter]
**Objectif** : Pipeline automatisé lint + test + build.

### Workflows
- `ci.yml` :
- `cd.yml` :

### Résultats
- Coverage :
- Temps de build CI :

---

## Etape 10 - Tests et couverture

**Date** : [à compléter]
**Objectif** : Tests unitaires + intégration, coverage > 80%.

### Résultats
- Tests unitaires :
- Tests intégration :
- Coverage globale :

---

## Bilan final

### Ce qui a bien fonctionné
-

### Difficultés majeures
-

### Améliorations possibles (hors scope TFE)
-

### Temps passé par étape

| Etape | Temps estimé | Temps réel |
|-------|-------------|-----------|
| 0 - Cadrage | | |
| 1 - Config | | |
| 2 - Data | | |
| 3 - Training | | |
| 4 - Registry | | |
| 5 - API | | |
| 6 - Demo | | |
| 7 - Monitoring | | |
| 8 - Docker | | |
| 9 - CI/CD | | |
| 10 - Tests | | |
