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

## Etape 2bis - Pipeline de curation from scratch depuis raw/

**Date** : 2026-03-30
**Objectif** : Reconstruire le pipeline de donnees depuis les CSV bruts, sans dependre du notebook 0 ni du filtre ResNet50 ImageNet.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Source de donnees | data/raw/Mushrooms_images/ (20 572) | data/processed/ (12 131 originaux) | 70% de donnees en plus, pipeline reproductible, pas de dependance ResNet50 |
| Filtre ResNet50 | Pas de filtre | Filtre top-3 ImageNet classes champignon | Non reproductible (modele telechargeable varie), inegal (75% filtre sur Amanita muscaria vs 8% sur Russula olivacea), les images sont deja labelisees par des experts (GBIF confidence >= 94) |
| Augmentation statique | Aucune | Over-sampling TF (notebook 0) | PyTorch gere ses propres augmentations au training (reproductible, seedable) |
| Equilibrage | WeightedRandomSampler au training | Resampling statique, class weights | Pas de duplication de donnees, equilibrage dynamique a chaque epoch |
| Conflits d'especes | Retirer les 12 images avec 2 especes | Garder la premiere | Label ambigu, pas fiable pour l'entrainement |

### Investigation menee

Audit complet du pipeline du notebook 0 :
1. Le notebook 0 charge observations_mushroom.csv (647K obs, 11 999 especes)
2. Croise avec champignons_france_top30.csv (30 especes) -> 20 592 obs
3. Filtre GBIF confidence >= 92 -> 20 592 (aucune filtree, min=94)
4. Filtre ResNet50 ImageNet "est-ce un champignon ?" -> retire 40.9% (8 423 images)
5. Resample : under-sample a 900, over-sample a 850 avec augmentations TF non seedees
6. Resultat : 12 160 originaux + 13 690 augmentees = 25 850 images

Le filtre ResNet50 est tres inegal par espece (8% a 75% de rejet).
Les augmentations TF ne sont pas reproductibles (transforms aleatoires non seedees).

### Notre pipeline (data/curate.py)

```
observations_mushroom.csv (647 623 obs)
    |  inner join sur gbif_info/species
champignons_france_top30.csv (30 especes)
    |  20 592 observations
Filtre confiance GBIF >= 92 (aucune retiree)
    |  20 592
Dedup image_lien (8 doublons meme espece retires)
    |  20 584
Retrait conflits especes (12 images 2 especes retires)
    |  20 572
Verification fichiers raw/ (100% presents)
    |  20 572 images finales, 30 classes
```

### Problemes rencontres
- Le filtre confiance >= 92 ne retire rien (toutes les obs sont deja >= 94)
- 12 images ont un conflit d'especes (pas juste 2 comme identifie initialement - la recherche de conflits couvre tout le CSV, pas juste les top30)
- Le .venv du projet n'a pas pip, il faut utiliser le python systeme pour les scripts de curation
- Le manifest change de format : `path` contient `image_lien` (ex: `120022.jpg`) au lieu de `Espece/120022.jpg`

### Artefacts produits
- `data/curate.py` - pipeline de curation reproductible
- `data/curated_manifest.csv` - 20 572 lignes (image_lien, species)
- `data/curation_report.json` - rapport detaille de chaque etape
- `data/raw_stats.json` - regenere avec les stats des 20 572 images
- `data/cleaning_report.json` - regenere avec le nouveau pipeline
- `data/split_manifest.csv` - regenere (20 572 images)
- `data/split_stats.json` - regenere
- `data/data_split.py` - reecrit pour lire curated_manifest.csv
- `src/data/dataloader.py` - mis a jour pour pointer vers raw/Mushrooms_images/
- `data/excluded.json` - supprime (plus necessaire)

### Metriques / Resultats
- Images : 20 572 (vs 12 131 avant, +69%)
- Classes : 30
- Distribution : min=58 (Russula emetica), max=3 579 (Amanita muscaria), ratio=61.7x
- Split : train=14 400, val=3 085, test=3 087
- Tests : 38 passed

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

### Metriques / Resultats (premier run, 2026-03-30)

**Hardware** : XPS 9520, RTX 3050 Ti (4 GB VRAM), batch=16, fp16 (AMP)
**Run MLflow** : https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow/#/experiments/0/runs/1e7b1dda43ca467ead7c2c887ffdbece

| Metrique | Valeur |
|----------|--------|
| Best val accuracy | 83.2% (epoch 30) |
| Best val F1 macro | 78.7% (epoch 30) |
| Test accuracy | 83.9% |
| Test F1 macro | 77.8% |
| Meilleur checkpoint | epoch 27 (val_loss=0.7483) |
| Temps total | 48.1 min (30 epochs, ~93s/epoch) |
| Early stopping | Non declenche (30 epochs complets) |

**Points forts** (classes bien predites) :
- Coprinus comatus : 92% F1
- Craterellus cornucopioides : 92% F1
- Schizophyllum commune : 92% F1

**Points faibles** (classes rares et visuellement similaires) :
- Russula vesca : 20% F1 (9 images test seulement)
- Russula rosea : 58% F1
- Russula emetica : 67% F1

**Analyse** : les Russules sont un genre de champignons visuellement tres similaires (memes formes, couleurs proches). Les classes rares (< 100 images) souffrent du manque de donnees d'entrainement malgre le WeightedRandomSampler. Les classes a moins de 15 images test donnent des metriques instables (Russula vesca : 9 images).

- Tests code : 38 passed (23 existants + 15 nouveaux)
- Pre-commit : tous les hooks passent

---

## Etape 4 - Model registry et export

**Date** : 2026-03-30
**Objectif** : Exporter le modele PyTorch en ONNX, valider, comparer les sorties.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Format export | ONNX opset 17 (legacy exporter) | TorchScript, dynamo exporter | Portable, CPU-optimise, pas de dep PyTorch en prod |
| Exporteur | torch.onnx.export(dynamo=False) | Nouveau dynamo exporter | Le dynamo exporter (torch 2.11) produit un fichier de 240 KB au lieu de 90 MB - bug/incompatibilite avec la conversion opset 17. Legacy stable. |
| Axes dynamiques | batch_size dynamique | Taille fixe | Permet l'inference batch variable sans re-export |
| Validation | onnx.checker + comparaison numerique | Checker seul | La comparaison PyTorch vs ONNX (max_diff < 1e-4) garantit l'equivalence |

### Problemes rencontres
- Le nouveau dynamo exporter de torch 2.11 exporte un modele de 240 KB (au lieu de 90 MB) avec un warning "Failed to convert to target version 17". Le fichier passe onnx.checker mais est quasi vide. Solution : `dynamo=False` pour forcer l'exporteur legacy.
- `onnxscript` est une dependance requise par torch 2.11 pour l'export ONNX, meme en mode legacy (il est importe avant le fallback).

### Artefacts produits
- `src/models/export_onnx.py` - script d'export CLI complet (charge checkpoint, export, valide, compare, sauve class_names)
- `models/best_model.onnx` - modele ONNX exporte (89.8 MB)
- `models/class_names.json` - 30 noms d'especes (ordre alphabetique = label_map)
- `tests/unit/test_export_onnx.py` - 4 tests
- Pages Streamlit 05-07 (entrainement, evaluation, model registry)
- `demo/lib/mlflow_utils.py` - helpers MLflow + fallback local

### Metriques / Resultats
- Taille modele ONNX : 89.8 MB (vs 270 MB checkpoint PyTorch = 3x compression)
- Ecart PyTorch vs ONNX : max_diff=0.000006 (< 1e-4 = OK)
- Tests : 51 passed (47 existants + 4 nouveaux)
- Pre-commit : tous les hooks passent

---

## Etape 5 - API serving (FastAPI)

**Date** : 2026-03-30
**Objectif** : API REST pour inference, metriques Prometheus, health checks.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Framework | FastAPI | Flask, TorchServe | Async natif, Pydantic, OpenAPI auto-generee |
| Runtime inference | ONNX Runtime (CPU) | PyTorch, TensorRT | Portable, pas de dep PyTorch en prod, CPU-optimise |
| Preprocessing | Numpy pur (identique val/test) | torchvision transforms | Evite la dep torch en prod, meme resultat |
| Metriques | prometheus-client natif | starlette-prometheus | Controle fin des labels, pas de dep supplementaire |
| Schemas | Pydantic v2 models | dicts bruts | Validation automatique, documentation OpenAPI |
| Graceful degradation | 503 si modele absent | 500, refuser de demarrer | Permet de deployer l'API avant d'avoir le modele |

### Endpoints implementes
- `POST /predict` : upload image -> top-5 predictions avec scores de confiance
- `GET /health` : statut du service + etat du modele (healthy/no_model)
- `GET /metrics` : metriques Prometheus (latence, predictions, confiance, erreurs)
- `GET /model/info` : metadata du modele (classes, input_shape, version)

### Problemes rencontres
- Les decorateurs FastAPI (`@app.get`, `@app.post`) rendent les fonctions "untyped" pour mypy du pre-commit (mirrors-mypy sans starlette). Solution : `# type: ignore[misc]` sur chaque decorateur.
- B008 ruff : `File()` dans les arguments par defaut. Solution : extraire dans une constante module-level `_FILE_PARAM`.
- Le preprocessing numpy doit etre identique aux transforms val/test (Resize 256, CenterCrop 224, Normalize ImageNet). Verifie par test unitaire.

### Artefacts produits
- `src/serving/app.py` - serveur FastAPI complet (4 endpoints)
- `src/serving/schemas.py` - 5 modeles Pydantic
- `src/serving/middleware.py` - 5 metriques Prometheus
- `tests/unit/test_api.py` - 9 tests (health, predict mock, metrics, model_info, preprocess)

### Metriques / Resultats
- Tests : 47 passed (38 existants + 9 nouveaux)
- Latence p50 : [a mesurer apres deploiement]
- Latence p95 : [a mesurer apres deploiement]

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

**Date** : 2026-03-30
**Objectif** : Containeriser tous les services, docker-compose fonctionnel.

### Images Docker

| Image | Base | Contenu |
|-------|------|---------|
| docker/Dockerfile.api | python:3.11-slim | FastAPI + ONNX Runtime + Prometheus |
| docker/Dockerfile.demo | python:3.11-slim | Streamlit + Plotly + data artifacts |

### Docker Compose (docker-compose.yml)
- Services : api (port 8000), demo (port 8501), prometheus (port 9090), grafana (port 3000)
- Volumes : models/ monte en read-only sur l'API, grafana-data pour la persistance
- Healthcheck : api verifie /health toutes les 30s
- Demo depend de api (service_healthy)

### Artefacts produits
- `docker/Dockerfile.api` - image API inference
- `docker/Dockerfile.demo` - image Streamlit
- `docker-compose.yml` - orchestration 4 services
- `configs/prometheus.yml` - scrape l'API sur /metrics
- `.dockerignore` - exclut venv, data raw, tests, caches

### Taille / Build time
- [a mesurer apres premier build]

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
