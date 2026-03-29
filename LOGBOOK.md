# Champy Classifier - Cahier de bord MLOps

## Informations générales

- **Projet** : Classification de champignons (30 espèces, ~700K images)
- **Cadre** : TFE Master AI (DataScientest, RNCP niveau 7, promotion 2026)
- **Equipe** : [à compléter]
- **Repo** : DagsHub - LoicFocraud/Champy_Classifier
- **Date de debut** : 2026-03-28
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

### Pages Streamlit 01-04 (construction incrementale)

Portfolio Streamlit demarre avec les 4 premieres pages (donnees).
Principe zero hardcoded : chaque page lit ses donnees depuis les rapports JSON generes.

Pages creees :
- `demo/app.py` - vue d'ensemble du pipeline, statut de chaque etape (detection dynamique des artefacts)
- `demo/pages/01_donnees_brutes.py` - distribution classes, formats, dimensions, galerie interactive
- `demo/pages/02_nettoyage.py` - avant/apres (25 850 -> 12 131), raisons d'exclusion, detail des fichiers
- `demo/pages/03_augmentation.py` - demonstration live des transforms PyTorch (original vs augmentees)
- `demo/pages/04_split.py` - distribution par classe par split, verification stratification, desequilibre

Helpers partages : `demo/lib/data_utils.py` (load_json, load_manifest, scan_classes, get_random_images)

---

## Etape 3 - Training pipeline

**Date** : 2026-03-29
**Objectif** : Boucle d'entrainement PyTorch, AMP, MLflow tracking, early stopping, checkpointing.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Optimizer | AdamW | SGD+momentum, LARS | Convergence rapide, weight decay integre |
| Scheduler | CosineAnnealingLR | StepLR, OneCycleLR | Smooth decay, bon pour fine-tuning |
| Batch size | 16 (fp16) | 32 | Contraint par 4GB VRAM RTX 3050 Ti |
| Architecture tete | Dropout(0.3) + Linear(2048, 30) | Dense(512)+Dense(30), GlobalAvgPool | Simple, evite l'overfitting, 2048 features ResNet |
| Mixed precision | AMP (autocast + GradScaler) | FP32 complet | Obligatoire pour 4GB VRAM, ~2x plus rapide |
| Gradient accumulation | Configurable (defaut 1) | Aucune | Permet batch effectif 32 ou 64 si VRAM insuffisant |
| Logging | loguru | print, logging stdlib | Formatage riche, rotation fichiers, pas de config |
| @torch.no_grad() | with torch.no_grad() dans le corps | decorateur | Le decorateur rend la fonction "untyped" pour mypy du pre-commit |

### Sous-etape 3a - Script d'entrainement

`src/training/train.py` : pipeline complet lancable en CLI.
- Chargement config YAML via `TrainingConfig.from_yaml()`
- Seed global (torch, numpy, random, cudnn)
- Detection automatique GPU/CPU
- Boucle train avec AMP + gradient accumulation
- Validation a chaque epoch (loss, accuracy, F1 macro)
- CosineAnnealingLR scheduler
- Early stopping + checkpointing du meilleur modele

`src/models/resnet.py` : creation ResNet50 avec tete personnalisee.
- Poids ImageNet V2 pre-entraines
- Tete : Dropout(0.3) + Linear(2048, 30)
- Freeze/unfreeze progressif du backbone (par layer)

### Sous-etape 3b - MLflow tracking

Integre dans la boucle de train :
- Log des hyperparams (config.model_dump())
- Metriques par epoch (train_loss, val_loss, val_acc, val_f1_macro, lr)
- Artefacts finaux : confusion_matrix.png, learning_curves.png, metrics.json
- URI et credentials depuis src/config.py (.env)

### Sous-etape 3c - Callbacks

`src/training/callbacks.py` :
- `EarlyStopping` : patience configurable, mode min/max, min_delta
- `ModelCheckpoint` : sauvegarde checkpoint complet (model + optimizer + epoch + best_score)

### Sous-etape 3d - Evaluation

`src/training/evaluate.py` :
- `evaluate_model()` : accuracy, F1 macro, classification report
- `save_confusion_matrix()` : PNG annotee, normalisable
- `save_learning_curves()` : PNG loss + metriques par epoch
- `save_metrics_json()` : JSON pour Streamlit

### Problemes rencontres
- Le decorateur `@torch.no_grad()` rend les fonctions "untyped" pour le mypy du pre-commit (mirrors-mypy sans torch). Solution : utiliser `with torch.no_grad()` dans le corps.
- Les `type: ignore` pour les generiques PyTorch (DataLoader, Dataset) ne sont pas necessaires avec le mypy du pre-commit qui ne connait pas torch. Il faut les retirer sinon `unused-ignore` bloque.
- loguru, matplotlib, scikit-learn n'etaient pas installes dans l'env systeme du NUC3.

### Artefacts produits
- `src/training/train.py` - script d'entrainement complet
- `src/training/callbacks.py` - EarlyStopping + ModelCheckpoint
- `src/training/evaluate.py` - metriques + artefacts visuels
- `src/models/resnet.py` - ResNet50 transfer learning
- `tests/unit/test_callbacks.py` - 9 tests (EarlyStopping + ModelCheckpoint)
- `tests/unit/test_evaluate.py` - 6 tests (confusion matrix, courbes, JSON)

### Metriques / Resultats
- Tests : 38 passed (23 existants + 15 nouveaux)
- Pre-commit : tous les hooks passent
- Best val accuracy : [a completer apres run sur XPS]
- Best val F1 (macro) : [a completer apres run sur XPS]
- Nombre d'epochs avant convergence : [a completer apres run sur XPS]
- Temps total d'entrainement : [a completer apres run sur XPS]

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
