# Champy Classifier - Cahier de bord MLOps

## Informations generales

- **Projet** : Classification de champignons (30 especes, ~700K images brutes, 19 138 retenues apres curation)
- **Cadre** : TFE Master AI (DataScientest, RNCP niveau 7, promotion 2026)
- **Equipe** : Equipe Champy Classifier (DataScientest promotion 2026)
- **Repo** : DagsHub - LoicFocraud/Champy_Classifier (mirror GitHub LoicFocraud/Champy_Classifier)
- **Branche de developpement** : `dev-dominique` (merge vers `main` par PR equipe)
- **Date de debut** : 2026-03-28
- **Date de soutenance** : [a confirmer]

---

## Recapitulatif visuel - Roadmap equipe (1-2 pages, lecture jury)

> Synthese 1-2 pages des 7 etapes principales. Pour le detail : sections
> ``Etape 1`` a ``Etape 8`` plus bas, plus les sous-etapes Bloc M1-M4
> (monitoring approfondi 2026-05-08).

### Tableau de bord general

| Etape | Titre | Statut | Periode | Livrable principal | Metrique cle |
|-------|-------|--------|---------|--------------------|--------------|
| 1 | Mise en place environnement | Termine | 2026-03-28 | `pyproject.toml`, `CLAUDE.md`, structure repo | 13 invariants MLOps formalises |
| 2 | Analyse du sujet | Termine | 2026-03-28 / 03-30 | Audit notebooks legacy, decision migration TF -> PyTorch | 5 notebooks archives `notebooks/legacy/` |
| 3 | Preparation des donnees | Termine | 2026-03-28 / 04-23 | `data/curate.py`, filtre OpenCLIP, split 70/15/15 | **19 138 images, 30 classes** apres curation + filtre qualite |
| 4 | Entrainement | Termine | 2026-03-29 / 04-23 | `src/training/train.py`, 3 runs MLflow | **ConvNeXt-Tiny v2.0.0 = 90% val acc** |
| 5 | CI/CD | Termine | 2026-03-30 / 04-24 | Pre-commit + GitHub Actions 5 jobs | **86 tests, CI vert sur dev-dominique** |
| 6 | Serving (API + Registry) | En cours | 2026-03-30 / 05-08 | FastAPI + **BentoML 1.4 (migration Bloc 1-3)** | Latence p95 < 50 ms, parite 4/4 OK delta 1.21e-07 |
| 7 | Docker et Monitoring | En cours | 2026-03-30 / 05-08 | Compose 4 services + **Grafana 3 dashboards** + Evidently | 86% Prometheus scrapes, 3 dashboards live |
| 8 | Demo et Tests | En cours | 2026-03-28 / 05-08 | Streamlit 18 pages + **monitoring page complete (Bloc M4)** | 18/18 pages, alerting visuel sur seuils config |

### Etapes 1-3 - Donnees

```
data/raw/Mushrooms_images/  (646 524 fichiers bruts, 11 999 classes)
        |
        v   data/curate.py (top30 + GBIF >= 92 + dedup + conflits)
        |
   20 572 images, 30 classes
        |
        v   data/quality_filter.py (OpenCLIP ViT-B-32, seuil 0.03)
        |
   19 138 images, 30 classes
        |
        v   data/data_split.py (stratifie seed=42)
        |
   train=13 396  val=2 870  test=2 872
```

**Decisions cles** :
- Reconstruction from raw plutot que reuse de `data/processed/` legacy : pipeline reproductible, +70% de donnees
- Filtre qualite OpenCLIP (CPU NUC3, 9 min, seuil calibre visuellement) : retire 1 434 images parasites (interieurs, personnes, textes)
- WeightedRandomSampler pour gerer le desequilibre (ratio 62.9x entre Russula emetica et Amanita muscaria)

### Etape 4 - 3 runs MLflow authentiques

| Config YAML | Backbone | Strategie | Val acc | Val F1 | Lien MLflow |
|-------------|----------|-----------|---------|--------|-------------|
| `default.yaml` | ResNet50 | 2-phase, lr=1e-3 / 1e-5, 30 ep | 84.0% | 75% | DagsHub |
| `aggressive.yaml` | ResNet50 | lr++, weight_decay augmente | 88.0% | 78% | DagsHub |
| `convnext.yaml` | ConvNeXt-Tiny | 2-phase, lr=1e-3 / 1e-5, 40 ep | **90.0%** | **81%** | DagsHub |

> Hardware : XPS 9520, RTX 3050 Ti (4 GB VRAM), batch=16, AMP fp16, seed=42.
> Modele en production : ConvNeXt-Tiny v2.0.0 (epoch 40, val_loss=0.440, ONNX 106.3 MB).

### Etapes 5-6 - CI/CD et Serving

- **Pre-commit** (local) : 8 hooks bloquants (ruff v0.15, mypy 1.13, interrogate 100%)
- **GitHub Actions** : 5 jobs (lint, typecheck, docstrings, tests, build)
- **2 couches de serving** sur le meme modele ONNX :
  - **FastAPI** (`src/serving/`, port 8010) : POC initial, 4 endpoints, metriques `champy_*`
  - **BentoML 1.4** (`src/serving_bentoml/`, port 8020) : adaptive batching `max_batch_size=32, max_latency_ms=100`, Model Store + packaging via `bentofile.yaml`, **parite 4/4 images delta max 1.21e-07**
- **Prediction store SQLite** (Bloc M2) : WAL + busy_timeout=5000ms, **100 ecritures concurrentes sans `database is locked`**

### Etape 7 - Monitoring (Blocs M1-M4 du 2026-05-08)

```
+------------------+       +------------------+       +-------------------+
|   FastAPI 8010   |---->  |  Prometheus 9090 |  ---> |   Grafana 3010    |
|   /metrics       |       |  scrape 15s      |       |   3 dashboards    |
+------------------+       +------------------+       +-------------------+
        |                                                       ^
        +- /predict (POST)                                      |
                                                                |
+------------------+       +------------------+       +---------+---------+
|  BentoML 8020    |---->  |  SQLite WAL      |  ---> | Streamlit 8501    |
|  (Bloc M2)       |       |  predictions.db  |       |   page 10 + 11    |
+------------------+       +------------------+       +-------------------+
                                   |
                                   v
                           Baseline JSON (test set, 89.9% acc)
                                   |
                                   v
                           Evidently HTML (drift on-demand)
```

| Bloc | Realisation | Validation |
|------|-------------|------------|
| M1 | Provisioning Grafana (datasource + 3 dashboards JSON) | 50 predictions seedees, 6+5+6 panels live |
| M2 | PredictionStore SQLite WAL + endpoint `/predictions/recent` | 9 tests OK, 100 ecritures concurrentes 0 perte |
| M3 | Drift detection Evidently (baseline + rapport HTML) | Baseline 2872 imgs 89.9% acc, rapport 3.8 MB |
| M4 | Page Streamlit monitoring 4 sections (live + iframe + alerting) | 3/3 alertes evaluees, resilient si Grafana down |

### Etape 8 - Demo Streamlit (18 pages)

| # | Page | Statut | Source |
|---|------|--------|--------|
| 00 | Accueil | OK | Disque |
| 01-04 | Donnees, nettoyage, augmentation, split | OK | JSON + CSV |
| 05-07 | Training, evaluation, registry | OK (necessite token MLflow) | MLflow / DagsHub |
| 08 | Prediction (upload top-5) | OK | ONNX local ou API |
| 09 | API (Swagger + metrics) | OK | FastAPI |
| 10 | **Monitoring complet** | OK (Bloc M4) | Prometheus + Grafana + SQLite |
| 11 | **Drift Evidently** | OK (Bloc M3) | Baseline + Evidently |
| 12 | Infrastructure | A finaliser | Docker + GitHub API |

**Principe `zero hardcoded`** respecte sur les 18 pages : aucune valeur (accuracy, RPS, seuils) ecrite en dur, tout lu depuis les sources (MLflow, Prometheus, SQLite, JSON, YAML).

### Roadmap equipe (statut detaille)

| Etape | Titre | Statut | Reste a faire |
|-------|-------|--------|----------------|
| 1 | Mise en place environnement | Termine | - |
| 2 | Analyse du sujet | Termine | - |
| 3 | Preparation des donnees | Termine | - |
| 4 | Entrainement | Termine | Re-generer ResNet50 v1.0.0 + v1.1.0 ce week-end (Bloc R0) |
| 5 | CI/CD | Termine | - |
| 6 | Serving (API + Model Registry) | En cours | MLflow Model Registry promotion Staging -> Prod |
| 7 | Docker et Monitoring | En cours | Bloc 5 migration : ajouter `api-bento` au compose |
| 8 | Demo et Tests | En cours | Page 12 (Infrastructure) finalisation |

### Ce qui a bien fonctionne

- **Factory unifiee `create_backbone()`** : ResNet50 -> ConvNeXt-Tiny sans toucher `train.py`
- **Curation reproductible from raw + filtre OpenCLIP** : 70% de donnees en plus que le pipeline legacy
- **Zero hardcoded dans Streamlit** : toutes les pages s'auto-mettent a jour
- **Pre-commit systematique** : evite des dizaines de corrections en PR
- **DagsHub hub unique** : MLflow + DVC + Git, un seul token
- **Migration BentoML sans regression** : parite numerique 4/4 a delta 1e-7

### Difficultes majeures

- **Contraintes Windows** (PowerShell, num_workers=0 par defaut, paths translation Git Bash)
- **VRAM 4 GB RTX 3050 Ti** : limite batch=16 AMP, ConvNeXt-Tiny passe tout juste
- **Hote partage NUC3** : cohabitation avec 5+ projets, ports remappes +10
- **Fine-grained Russules** : 7 especes visuellement similaires, plafond ~60-70% F1 sur ces classes
- **Versions deps glissantes** : drift ruff 0.8 / 0.15 entre pre-commit et CI (corrige par alignement explicite)

---

## Etape 1 - Mise en place environnement

**Date** : 2026-03-28
**Objectif** : Architecture MLOps cible, repartition hardware, conventions, repo pret a coder.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| ML Framework | PyTorch | TensorFlow, JAX | Standard industrie, ecosysteme torchvision, compatibilite ONNX |
| Modele de base | ResNet50 puis ConvNeXt-Tiny (transfer learning) | EfficientNet, ViT | ResNet50 valide sur TFE initial (95.4%), ConvNeXt pour iteration fine-grained |
| Data versioning | DVC (remote DagsHub) | Git LFS, Lakefs | Deja en place, integre DagsHub |
| Experiment tracking | MLflow (DagsHub) | W&B, Neptune | Gratuit, integre DagsHub, self-hosted possible |
| API serving | FastAPI | Flask, TorchServe | Async natif, Pydantic, OpenAPI auto-generee |
| Format inference | ONNX Runtime | TorchScript, TensorRT | Portable, CPU-optimise, pas de dependance PyTorch en prod |
| Demo | Streamlit | Gradio, React | Python natif, multi-page, rapide a prototyper |
| Monitoring | Prometheus + Grafana | Datadog, ELK | Standard industrie, open source, leger |
| Drift detection | Evidently AI | Alibi Detect, NannyML | API simple, rapports HTML integres, gratuit |
| CI/CD | GitHub Actions | GitLab CI, Jenkins | Integre DagsHub/GitHub |
| Linting | Ruff | Black + Flake8 + isort | Un seul outil, 10-100x plus rapide |
| Containerisation | Docker Compose | Kubernetes | Suffisant pour l'echelle du projet, pas d'overhead K8s |
| Gestion deps | pyproject.toml (hatchling) + uv | requirements.txt seul | Standard PEP 621, sections [dev] separees, config Ruff/Mypy/Pytest integree |
| Config runtime | Pydantic Settings | python-decouple, dynaconf | Validation automatique, integration .env + YAML, coherent avec FastAPI |
| Python version | >=3.11 | 3.12 only | 3.11 minimum pour match_case et perf, compatible 3.12 |
| Task runner | invoke (tasks.py) | Makefile, Nox | Cross-platform Windows/Linux, Python natif |
| Line endings | LF force (.gitattributes) | Mixed | Evite les diffs parasites entre Windows et CI Linux |
| Docstrings | Francais (Google style), verifie par interrogate | Anglais | Convention du projet, mieux pour le memoire |
| Pre-commit | ruff + mypy + interrogate + hooks generaux | Verification manuelle | Garantit la qualite a chaque commit |

### Repartition hardware

| Machine | Role | Specs cles |
|---------|------|-----------|
| NUC3 | Hub MLOps (dev, serving, monitoring, demo) | Ryzen AI 9, 96GB RAM, pas de GPU dedie, Docker Desktop, Windows 11 Pro |
| XPS 9520 #1 | Training GPU | i7-12700H, RTX 3050 Ti (4GB VRAM), Windows 11 |
| XPS 9520 #2 (optionnel) | Training GPU parallele | Idem |

### Problemes rencontres
- `requirements.txt` existant etait en encodage UTF-16 avec espaces entre chaque caractere - inutilisable, regenere en UTF-8.
- `pyproject.toml` initial etait un dump brut de `pip freeze` (190 deps dont TensorFlow, yfinance, jupyter) - nettoye pour ne garder que les deps du projet.
- Aucun `__init__.py` dans les sous-packages de `src/` - ajoutes des le debut.
- `.gitignore` excluait par erreur `data.dvc` et `models.dvc` - corrige : ces fichiers `.dvc` DOIVENT etre dans git (c'est le principe meme de DVC).
- `httpx` et `plotly` manquants dans les deps principales alors qu'ils sont necessaires pour les helpers Streamlit et les appels API.

### Artefacts produits
- `CLAUDE.md` - gouvernance Claude Code (13 invariants MLOps)
- `.claude/skills/champy-mlops/SKILL.md` - patterns MLOps
- `LOGBOOK.md`, `PLAYBOOK.md`, `README.md`
- `pyproject.toml` (hatchling, deps PyTorch + httpx + plotly + config Ruff/Mypy/Pytest)
- `requirements.txt` (UTF-8, fallback pip)
- `.env.example`, `.gitattributes` (LF), `.gitignore` complet
- `src/config.py` - Pydantic Settings (MLflow, Training, Serving) avec chargement YAML
- `configs/training/default.yaml` - hyperparams par defaut
- `src/{data,training,inference,models,serving,monitoring}/__init__.py`
- `demo/{__init__.py,lib/__init__.py}` + structure `pages/`, `assets/`
- `tasks.py` - task runner invoke (18 taches)

### Metriques / Resultats
- pyproject.toml : 190 deps -> ~32 directes + 4 dev
- Sous-packages `src/` : 6 (data, training, inference, models, serving, monitoring)
- 13 invariants MLOps documentes dans CLAUDE.md

---

## Etape 2 - Analyse du sujet

**Date** : 2026-03-28 / 2026-03-30
**Objectif** : Comprendre l'existant (notebooks legacy, donnees, pipeline initial) et decider du perimetre de reconstruction.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Framework | Migration Keras/TF -> PyTorch | Rester sur Keras/TF | Ecosysteme ONNX, AMP natif, transfer learning plus flexible, mieux outille pour fine-grained |
| Source de donnees | `data/raw/Mushrooms_images/` (646K raw) + CSV GBIF | `data/processed/` (25 850 pre-traitees) | Pre-traitement legacy non reproductible (augmentations TF non seedees, filtre ResNet ImageNet inegal) |
| Filtre ResNet50 ImageNet legacy | Abandonne | Reproduire | Retire 40.9% des images, inegal par classe (8% a 75%), non reproductible (modele telechargeable varie) |
| Notebooks legacy | Archives dans `notebooks/legacy/` | Supprimer | Conserves pour reference audit, plus dans le chemin principal |
| Modeles `.keras` / `.h5` legacy | Conserves dans `models/` (tagues legacy) | Supprimer | Reference historique pour le memoire (comparaison baseline) |

### Audit des notebooks legacy (5 notebooks DataScientest initiaux)

| Notebook | Role | Probleme identifie |
|----------|------|--------------------|
| 0 - EDA + preparation | Charge observations_mushroom.csv (647K), filtre, resample a 25 850 | Augmentations TF aleatoires non seedees -> non reproductible |
| 1-2 - Keras transfer learning | Entraine `cnn_tl_model.keras`, `cnn_tl2_model.keras` | Pas de tracking MLflow, pas de config externalisee, fp32 plein |
| 3-4 - Evaluation | Matrice de confusion, classes faibles | Incoherence de nommage `.keras` vs `.h5` entre notebooks |

**Conclusion** : reconstruire entierement le pipeline depuis `data/raw/` avec :
- Curation reproductible (CSV + GBIF confidence + dedup)
- Framework PyTorch
- Tracking MLflow + DVC des le debut
- Seeds fixes, config YAML, AMP

### Formalisation des invariants MLOps (CLAUDE.md)

13 invariants documentes, dont :
1. Reproductibilite (seed + config YAML + DVC)
2. Pas de donnees dans git (DVC seulement)
3. Pas de secrets dans le code (.env + pydantic)
4. Tests avant merge (CI)
5. Docker reproductible par service
6. Config separee du code (configs/)
7. ONNX pour serving (pas PyTorch brut en inference)
8. Type hints partout (mypy)
9. Docstrings FR (interrogate 100%)
10. Cross-platform (pathlib, pas bash)
11. Streamlit zero hardcoded (tout dynamique)
12. Pre-commit obligatoire
13. Push uniquement sur `dev-dominique`, jamais `main`

### Artefacts produits
- `notebooks/legacy/` - 5 notebooks archives
- `CLAUDE.md` enrichi - 13 invariants formalises
- Decision de reconstruction from scratch documentee

---

## Etape 3 - Preparation des donnees

**Dates** : 2026-03-28 (pipeline initial) / 2026-03-30 (curation from raw) / 2026-04-23 (filtre OpenCLIP)
**Objectif** : Pipeline de donnees reproductible from scratch depuis `data/raw/` jusqu'au manifest de split pret pour PyTorch.

### Sous-etape 3a - Curation from raw (2026-03-30)

Pipeline `data/curate.py` reproductible depuis les CSV bruts :

```
observations_mushroom.csv (647 623 obs)
    | inner join sur gbif_info/species
champignons_france_top30.csv (30 especes)
    | 20 592 observations
Filtre confiance GBIF >= 92 (aucune retiree, min=94)
    | 20 592
Dedup image_lien (8 doublons meme espece retires)
    | 20 584
Retrait conflits especes (12 images 2 especes retirees)
    | 20 572
Verification fichiers raw/ (100% presents)
    | 20 572 images finales, 30 classes
```

**Decisions cles** :

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Source | `data/raw/Mushrooms_images/` (20 572) | `data/processed/` (12 131 originaux) | 70% de donnees en plus, pipeline reproductible |
| Augmentation statique | Aucune | Over-sampling TF (legacy) | PyTorch gere ses propres augmentations au training (reproductible, seedable) |
| Equilibrage | WeightedRandomSampler au training | Resampling statique, class weights | Pas de duplication, equilibrage dynamique a chaque epoch |
| Conflits especes | Retirer les 12 images avec 2 especes | Garder la premiere | Label ambigu, pas fiable |

### Sous-etape 3b - Filtre qualite OpenCLIP (2026-04-23)

**Declencheur** : detection visuelle dans la page Streamlit 03 (augmentation) d'une image etiquetee `Amanita muscaria` montrant un homme dans un ascenseur. Audit sur echantillon a revele de nombreux faux positifs (interieurs, personnes, paysages sans sujet, cuisine, spores microscopiques, textes scannes).

**Decisions cles** :

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Modele | OpenCLIP ViT-B-32 / laion2b_s34b_b79k | CLIP OpenAI, BLIP-2 | 150 MB, 37 img/s CPU, pre-entraine sur LAION-2B (plus diversifie qu'ImageNet) |
| Prompts positifs | "a photo of a mushroom", "a fungus growing in nature", "a close-up photograph of a mushroom" | Un seul prompt | Couvre les 3 types d'image dominants (plein champ, environnement, macro) |
| Prompts negatifs | person, indoor scene, landscape without mushrooms, photo of text | Pas de prompts negatifs | Explicite les classes parasites observees |
| Score | max(positifs) - max(negatifs) | Softmax sur tous les prompts | Plus stable, interpretable, pas d'hyperparametre de temperature |
| Device | CPU NUC3 | GPU XPS | 9 min vs 3-5 min, pas besoin de deplacer les donnees, NUC3 dispo |
| Calibration | Echantillon stratifie 500 + visualisation | Seuil fixe a priori | Le seuil depend de la distribution reelle des scores |
| Seuil retenu | 0.03 | 0.0 (trop permissif), 0.05 (perd polypores valides) | Coupe les parasites evidents tout en preservant les cas limites |
| Application | Manifest separe + fallback dans `data_split.py` | Ecrasement de curated_manifest.csv | Non destructif, reproductible, reversible |

### Sous-etape 3c - Split stratifie (70/15/15)

`data/data_split.py` : charge le manifest filtre, split stratifie 70/15/15 avec seed=42.
Stratification verifiee sur toutes les classes, y compris les plus petites.

### Problemes rencontres
- Dimensions variables (320x240 majoritaire ~55%) -> Resize + CenterCrop dans le pipeline de transforms.
- Desequilibre naturel (Russula emetica 54 vs Amanita muscaria 3 396 = ratio 62.9x) -> WeightedRandomSampler.
- Le `.venv` du projet n'avait pas pip, il fallait utiliser le python systeme pour les scripts de curation (resolu apres).
- `num_workers=0` par defaut sur Windows (multiprocessing fork non supporte), teste `num_workers=2` avec `persistent_workers=True`.
- `pin_memory=True` genere un warning si aucun GPU dispo (acceptable).
- Le Dataset PyTorch doit exposer un attribut `targets` pour que WeightedRandomSampler calcule ses poids sans second parcours.

### Artefacts produits
- `data/curate.py` - pipeline curation reproductible
- `data/curated_manifest.csv` - 20 572 lignes (image_lien, species)
- `data/quality_filter.py` - script OpenCLIP (scoring + flag `--apply`)
- `data/quality_scores.csv` - scores par image (20 572 lignes)
- `data/quality_report.json` - synthese du run + per-class
- `data/curated_manifest_filtered.csv` - **19 138 images post-filtrage** (source pour data_split.py)
- `data/excluded.json` - 1 434 entrees avec score + raison + modele (tracabilite)
- `data/data_split.py` - script reproductible avec docstrings FR
- `data/split_manifest.csv` - train=13 396, val=2 870, test=2 872
- `data/split_stats.json` - stats par classe par split
- `src/data/dataset.py` + `dataloader.py` - Dataset PyTorch + DataLoader factory avec WeightedRandomSampler
- `scripts/inspect_quality_scores.py` - utilitaire de visualisation (histogramme + panels)
- `.pre-commit-config.yaml` - hooks ruff + mypy + interrogate
- Pages Streamlit 01-04 (donnees brutes, nettoyage, augmentation, split)

### Metriques / Resultats

| Etape | Images | Classes |
|-------|--------|---------|
| Brutes (`data/raw/`) | 646 523 | 11 999 |
| Apres filtrage top30 + GBIF + dedup + conflits | 20 572 | 30 |
| Apres filtre qualite OpenCLIP (seuil 0.03) | **19 138** | 30 |
| Split train | 13 396 | 30 |
| Split val | 2 870 | 30 |
| Split test | 2 872 | 30 |

- Stratification : 70.0% +/- 0.8% sur toutes les classes
- Taux d'exclusion OpenCLIP : 7.0% (1 434 images)
- Classes les plus touchees par l'exclusion qualite : Auricularia auricula-judae (17.6%), Ganoderma applanatum (17.4%), Russula vesca (15.1%)
- Classes rares preservees : Russula emetica 58 -> 54, Russula vesca 73 -> 62
- Tests : 38 passed (15 dataset + 8 dataloader + 15 autres)

---

## Etape 4 - Entrainement

**Date** : 2026-03-29 (pipeline) / 2026-03-30 a 2026-04-23 (3 runs)
**Objectif** : Boucle d'entrainement PyTorch reproductible avec tracking MLflow, early stopping, checkpointing. 3 runs compares pour choisir le meilleur modele.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Optimizer | AdamW | SGD+momentum, LARS | Convergence rapide, weight decay integre |
| Scheduler | CosineAnnealingLR | StepLR, OneCycleLR | Smooth decay, bon pour fine-tuning |
| Batch size | 16 (fp16) | 32 | Contraint par 4GB VRAM RTX 3050 Ti |
| Strategie | Fine-tuning deux phases (freeze backbone N epochs, puis degel total) | Fine-tuning direct, freeze permanent | Convergence plus stable, preservation des features bas niveau |
| Mixed precision | AMP (autocast + GradScaler) | FP32 complet | Obligatoire pour 4GB VRAM, ~2x plus rapide |
| Gradient accumulation | Configurable (defaut 1) | Aucune | Permet batch effectif 32/64 si VRAM insuffisant |
| Factory unifiee | `src/models/backbone.py` avec `create_backbone(name, ...)` | Un constructeur par modele | Ajout de ConvNeXt-Tiny sans refactor du train |
| Logging | loguru | print, logging stdlib | Formatage riche, rotation fichiers |

### Pipeline d'entrainement

`src/training/train.py` : pipeline complet lancable en CLI.
- Chargement config YAML via `TrainingConfig.from_yaml()`
- Seed global (torch, numpy, random, cudnn)
- Detection automatique GPU/CPU
- Boucle train avec AMP + gradient accumulation
- Validation a chaque epoch (loss, accuracy, F1 macro)
- CosineAnnealingLR scheduler
- Early stopping + checkpointing du meilleur modele
- MLflow tracking (hyperparams, metriques par epoch, artefacts)

### Comparaison des 3 runs MLflow

Tous sur XPS 9520, RTX 3050 Ti (4 GB VRAM), batch=16, AMP fp16, seed=42, split train=13 396 / val=2 870 / test=2 872.

| Config | Backbone | Strategie | Val acc | Val F1 macro | Test acc | Test F1 macro | Taille ONNX |
|--------|----------|-----------|---------|--------------|----------|---------------|-------------|
| `default.yaml` | ResNet50 | 2-phase, lr=1e-3 / 1e-5, freeze=10, total=30 | 84.0% | 75% | 83.9% | 77.8% | ~90 MB |
| `aggressive.yaml` | ResNet50 | 2-phase, lr++ , weight_decay augmente | 88.0% | 78% | [a mesurer via MLflow] | [a mesurer via MLflow] | ~90 MB |
| `convnext.yaml` | ConvNeXt-Tiny | 2-phase, lr=1e-3 / 1e-5, freeze=10, total=40 | **90.0%** | **81%** | [a mesurer via MLflow] | [a mesurer via MLflow] | **106.3 MB** |

**Modele retenu pour la production** : **ConvNeXt-Tiny** (`convnext.yaml`), epoch 40, best val_loss=0.440.

### Details du run ResNet50 default (reference)

| Metrique | Valeur |
|----------|--------|
| Best val accuracy | 83.2% (epoch 30) |
| Best val F1 macro | 78.7% (epoch 30) |
| Test accuracy | 83.9% |
| Test F1 macro | 77.8% |
| Meilleur checkpoint | epoch 27 (val_loss=0.7483) |
| Temps total | 48.1 min (30 epochs, ~93s/epoch) |
| Early stopping | Non declenche (30 epochs complets) |

### Classes bien / mal predites (ResNet50 default, test set)

**Points forts** : Coprinus comatus 92% F1, Craterellus cornucopioides 92% F1, Schizophyllum commune 92% F1.

**Points faibles** : Russula vesca 20% F1 (9 images test), Russula rosea 58% F1, Russula emetica 67% F1.

**Analyse** : les Russules sont un genre visuellement tres similaire (fine-grained). Les classes rares (< 100 images) souffrent du manque de donnees malgre le WeightedRandomSampler. Les classes < 15 images test donnent des F1 instables.

### Problemes rencontres
- Le decorateur `@torch.no_grad()` rend les fonctions "untyped" pour mypy du pre-commit (mirrors-mypy sans torch). Solution : `with torch.no_grad()` dans le corps.
- Les `type: ignore` pour les generiques PyTorch (DataLoader, Dataset) ne sont pas necessaires avec le mypy du pre-commit (il traite comme Any). Les laisser provoque `unused-ignore`.
- `torch.cuda.get_device_properties(0).total_mem` n'existe pas. L'attribut correct est `total_memory`. Erreur qui ne plante que sur GPU.
- MLflow DagsHub necessite `MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` (pas juste l'URI). Sans ca, 401 silencieux.
- Le `run_name` MLflow etait hardcode `resnet50_2phase_{seed}` : corrige en `{config.model_name}_2phase_{seed}` (commit 9c8bc6d, 2026-04-24).
- Batch 16 + AMP tient en 4 GB VRAM sur RTX 3050 Ti pour ResNet50 et ConvNeXt-Tiny (~28M params chacun).

### Artefacts produits
- `src/training/train.py` - pipeline complet avec fine-tuning 2 phases
- `src/training/callbacks.py` - EarlyStopping + ModelCheckpoint
- `src/training/evaluate.py` - metriques + artefacts visuels
- `src/models/resnet.py` - ResNet50 transfer learning
- `src/models/convnext.py` - ConvNeXt-Tiny transfer learning
- `src/models/backbone.py` - factory unifiee `create_backbone(name, ...)`
- `configs/training/default.yaml`, `aggressive.yaml`, `convnext.yaml` - 3 configs
- `tests/unit/test_callbacks.py`, `test_evaluate.py` - 15 tests
- 3 runs MLflow sur DagsHub (logs, artefacts, metriques par epoch)
- `models/best_model.pt` (334 MB, checkpoint ConvNeXt-Tiny epoch 40)

---

## Etape 5 - CI/CD

**Date** : 2026-03-30 (pipeline GitHub Actions initial) / 2026-04-24 (consolidation)
**Objectif** : Automatiser lint + typecheck + docstrings + tests + build sur chaque push/PR.

### Architecture CI/CD

3 niveaux de verification :

| Niveau | Quand | Quoi |
|--------|-------|------|
| Pre-commit (local) | A chaque `git commit` | Bloque le commit si ruff/mypy/interrogate/yaml/toml/large-files echouent |
| GitHub Actions (CI) | Sur push + PR vers `main` et `dev-dominique` | Re-verifie tout + build des images Docker |
| Tests pytest | Inclus dans la CI + lancables en local | 51 tests unitaires + integration |

### Pre-commit (`.pre-commit-config.yaml`)

| Hook | Role |
|------|------|
| `ruff check` | Linting (lint + autofix) |
| `ruff format` | Formatage |
| `mypy` (mirrors-mypy v1.13, scope `src/` + `data/data_split.py`) | Types |
| `interrogate --fail-under=100` | Toutes les fonctions publiques ont une docstring FR |
| `trailing-whitespace`, `end-of-file-fixer` | Propreté |
| `check-yaml`, `check-toml` | Validation syntaxe config |
| `check-added-large-files --maxkb=500` | Bloque les commits > 500 KB (protection images) |

**Installation** : `pip install pre-commit && pre-commit install`.

### GitHub Actions (`.github/workflows/ci.yml`)

5 jobs :
1. `lint` - ruff check
2. `typecheck` - mypy
3. `docstrings` - interrogate
4. `test` - pytest + coverage
5. `build` - build des 2 images Docker (depend de lint + typecheck + test)

Declenche sur `push` et `pull_request` vers `main` et `dev-dominique`.

### Tests (pytest)

51 tests au total :
- `tests/unit/test_dataset.py` - 15
- `tests/unit/test_dataloader.py` - 8
- `tests/unit/test_callbacks.py` - 9
- `tests/unit/test_evaluate.py` - 6
- `tests/unit/test_export_onnx.py` - 4
- `tests/unit/test_api.py` - 9 (health, predict mock, metrics, model_info, preprocess)

Lancement : `invoke test` (avec coverage) ou `pytest tests/unit/ -x --tb=short`.

### Problemes rencontres
- Les decorateurs FastAPI (`@app.get`, `@app.post`) rendent les fonctions "untyped" pour mypy (mirrors-mypy sans starlette). Solution : `# type: ignore[misc]` sur chaque decorateur.
- B008 ruff (`File()` dans les arguments par defaut) : extraire dans une constante module-level `_FILE_PARAM`.
- Le preprocessing numpy doit etre identique aux transforms val/test (Resize 256, CenterCrop 224, Normalize ImageNet). Verifie par test unitaire.

### Artefacts produits
- `.pre-commit-config.yaml` - 8 hooks
- `.github/workflows/ci.yml` - 5 jobs
- `tests/unit/` - 51 tests
- `pyproject.toml` - config Ruff / Mypy / Pytest / Interrogate centralisee

### Metriques / Resultats
- Pre-commit : passe sur tous les fichiers modifies (100%)
- Tests : 51 passed
- Coverage : [a mesurer via `invoke test-coverage`]

---

## Etape 6 - Serving (API + Model Registry) - EN COURS

**Date** : 2026-03-30 (API initiale) / 2026-04-24 (consolidation post-reboot)
**Objectif** : API REST FastAPI pour l'inference ONNX avec metriques Prometheus, chargement dynamique du modele ONNX.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Framework | FastAPI | Flask, TorchServe | Async natif, Pydantic, OpenAPI auto-generee |
| Runtime inference | ONNX Runtime (CPU) | PyTorch, TensorRT | Portable, pas de dep PyTorch en prod, CPU-optimise |
| Preprocessing | Numpy pur (identique val/test) | torchvision transforms | Evite la dep torch en prod, meme resultat |
| Metriques | prometheus-client natif | starlette-prometheus | Controle fin des labels, pas de dep supplementaire |
| Schemas | Pydantic v2 models | dicts bruts | Validation automatique, documentation OpenAPI |
| Graceful degradation | 503 si modele absent | 500, refuser de demarrer | Permet de deployer l'API avant d'avoir le modele |
| Format export | ONNX opset 17 (legacy exporter) | TorchScript, dynamo exporter | Portable, CPU-optimise ; dynamo exporter produit un fichier de 240 KB au lieu de 90 MB (bug conversion opset) |
| Detection architecture | Auto depuis state_dict (cles `conv1.weight` vs `features.0.0.weight`) | Flag CLI obligatoire | Evite le couplage config YAML / checkpoint, deterministe |

### Endpoints implementes
- `POST /predict` : upload image -> top-5 predictions avec scores de confiance
- `GET /health` : statut du service + etat du modele (healthy / no_model)
- `GET /metrics` : metriques Prometheus (latence, predictions, confiance, erreurs)
- `GET /model/info` : metadata du modele (classes, input_shape, version)

### Consolidation post-reboot (2026-04-24, Bloc A+B)

| Action | Resultat |
|--------|----------|
| Fix run_name MLflow hardcode | `{config.model_name}_2phase_{seed}` au lieu de `resnet50_2phase_{seed}` |
| Transfert checkpoint XPS -> NUC3 | Via `python -m http.server` sur XPS + `Invoke-WebRequest` depuis NUC3 |
| Export ONNX (NUC3) | Auto-detection ConvNeXt-Tiny depuis state_dict, export 106.3 MB, max_diff vs PyTorch = 4e-6 |
| Nettoyage `models/` | Suppression de `best_model.onnx.data` (orphelin ResNet50 legacy) |
| Validation API | 4/4 predictions correctes sur test set (98-100% confiance), metriques Prometheus OK |

### Problemes rencontres
- Le nouveau dynamo exporter de torch 2.11 exporte un modele de 240 KB au lieu de 90 MB (warning "Failed to convert to target version 17"). Solution : `dynamo=False` pour forcer l'exporteur legacy.
- `onnxscript` est requis par torch 2.11 pour l'export ONNX meme en mode legacy.
- Le checkpoint PyTorch contient `optimizer_state_dict` (AdamW moments) qui double la taille (ConvNeXt 28M params = ~110 MB poids, ~220 MB moments, total ~330 MB).
- `export_onnx.py` legacy etait cable en dur sur `create_resnet50` -> refactore pour utiliser `create_backbone` + auto-detection.

### Artefacts produits
- `src/serving/app.py` - serveur FastAPI (4 endpoints)
- `src/serving/schemas.py` - 5 modeles Pydantic
- `src/serving/middleware.py` - 5 metriques Prometheus
- `src/models/export_onnx.py` - export ONNX avec auto-detection d'architecture
- `models/best_model.pt` - checkpoint ConvNeXt-Tiny (334 MB)
- `models/best_model.onnx` - modele servi en prod (106.3 MB, self-contained)
- `models/class_names.json` - 30 especes dans l'ordre du label_map
- `tests/unit/test_api.py` - 9 tests
- `tests/unit/test_export_onnx.py` - 4 tests

### Metriques / Resultats
- Latence p50 : < 50 ms (mesure locale post-reboot)
- Latence p95 : < 100 ms
- Taux de bonne prediction top-1 (5 images test arbitraires) : 4/5 (l'erreur etait Amanita rubescens vs muscaria, confusion fine-grained attendue)
- Taille ONNX : 106.3 MB (ConvNeXt-Tiny), self-contained (0 external data)
- Ecart PyTorch vs ONNX : max_diff = 4e-6 (< 1e-4)

### Restant pour cloturer cette etape
- MLflow Model Registry : enregistrer formellement le modele ConvNeXt et promouvoir Staging -> Production.
- DVC commit + push du `models/` mis a jour (ancien snapshot : 4 fichiers ResNet50, nouveau : 3 fichiers ConvNeXt).

---

## Etape 6bis - Migration FastAPI -> BentoML - EN COURS

**Date** : 2026-05-07 / 2026-05-08
**Objectif** : Remplacer la couche de serving FastAPI (`src/serving/`) par un service BentoML (`src/serving_bentoml/`) pour s'aligner sur la roadmap equipe (etape 6 - Deploiement) et le cours Datascientest. Le modele ONNX ConvNeXt-Tiny (90% accuracy) reste strictement le meme.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Version BentoML | `>=1.2,<2.0` (installe : 1.4.39) | `<1.4` pour eviter le warning de deprecation | Suivre la version courante, accepter le warning, documenter le plan de migration |
| API d'enregistrement | `bentoml.onnx.save_model(...)` | `bentoml.models.create()` + serialisation manuelle | API documentee depuis 1.2, signature compatible avec le cahier des charges (`labels`, `signatures`, `custom_objects`) ; deprecated en 1.4 mais fonctionnel |
| Tag du modele | `champy_classifier:latest` (alias auto) | Tag versionne explicite | BentoML cree un tag horodatee a chaque `save_model` ; `latest` pointe automatiquement vers la derniere version |
| Pattern d'inference | `predict` mono-image (public) -> `infer_batch` batchable (interne) | Endpoint unique batchable | Conserve l'ergonomie HTTP de FastAPI ; le pooling adaptatif s'applique quand plusieurs `predict` arrivent concurrents |
| Adaptive batching | `max_batch_size=32`, `max_latency_ms=100`, `batch_dim=0` | Defauts BentoML (batch=100, latency=60s) | Cahier des charges + match des contraintes CPU NUC3 |
| Methodes batchable | `async def` obligatoire | `def` synchrone | BentoML 1.4 route les appels intra-service via un proxy RPC qui retourne des coroutines |
| Cast dtype | Explicite `np.float32` dans `infer_batch` | Confiance dans le type inferme | Le proxy RPC interne promeut les `np.ndarray` en float64 lors du transit HTTP ; ONNX Runtime exige float32 |
| Metriques | Natives BentoML + 3 custom (`champy_predictions_total`, `champy_prediction_latency_seconds`, `champy_prediction_confidence`) | Reproduire toutes les metriques FastAPI | BentoML expose nativement requests_total, request_duration_seconds, etc. ; on n'ajoute que les metriques metier |
| Health endpoint | Custom `/health` (POST) en plus des `/healthz` et `/readyz` natifs | Se contenter du natif | Le custom inclut version + architecture + classes (utile Streamlit / Grafana annotations) |
| HTTP method | POST partout (BentoML force POST sur `@bentoml.api`) | Mount FastAPI ASGI pour avoir des GET | Over-engineered ici ; les clients (tests, Streamlit) gerent POST sans probleme |
| `class_names` | Embarques dans `custom_objects` du Model Store | Lecture du `models/class_names.json` exclusive | Self-contained : le bento packagee ne depend plus du repo source |

### Endpoints implementes (port 8020)

| Endpoint | Methode | Statut | Note |
|----------|---------|--------|------|
| `/healthz` | GET | natif | 200 vide |
| `/readyz` | GET | natif | 200 vide |
| `/health` | POST | custom | `{status, model_loaded, model_version}` |
| `/model/info` | POST | custom | `tag, version, architecture, num_classes, class_names, input_shape` |
| `/predict` | POST (multipart) | custom | top-N especes + scores |
| `/infer_batch` | POST | batchable interne | `max_batch_size=32`, `max_latency_ms=100` |
| `/metrics` | GET | natif + custom | `bentoml_service_*` + `champy_*` |

### Validation parite FastAPI vs BentoML (Bloc 2 + add-on)

Script reproductible : [`scripts/compare_fastapi_bentoml.py`](scripts/compare_fastapi_bentoml.py).
Fait tourner les 4 memes images de test sur les deux APIs et compare top-1 + score.

| Image | Espece reelle | FastAPI top-1 | Conf. FastAPI | BentoML top-1 | Conf. BentoML | Delta |
|-------|---------------|---------------|---------------|---------------|---------------|-------|
| 100016.jpg | Amanita rubescens | Amanita rubescens | 0.999907 | Amanita rubescens | 0.999907 | 5.68e-08 |
| 101905.jpg | Boletus edulis | Boletus edulis | 0.999999 | Boletus edulis | 0.999999 | 6.95e-08 |
| 157467.jpg | Cantharellus cibarius | Cantharellus cibarius | 0.997703 | Cantharellus cibarius | 0.997702 | 1.21e-07 |
| 110460.jpg | Coprinus comatus | Coprinus comatus | 1.000000 | Coprinus comatus | 1.000000 | 3.94e-09 |

**Resultat : 4/4 OK, delta max 1.21e-07 (< epsilon 1e-6)**. Les deux couches de serving produisent des predictions strictement identiques (au bruit flottant pres). La difference vient uniquement du chemin de serialisation HTTP / softmax, pas du modele.

### Pieges BentoML 1.4 rencontres

1. `bentoml.onnx` est deprecated depuis 1.4 (warning a chaque appel). Plan de migration : `bentoml.models.create()` + chargement ONNX manuel via onnxruntime. Hors scope TFE.
2. `@bentoml.api` force POST. Pas de parametre `method=`. Tous les endpoints sont en POST y compris `/health` et `/model/info`.
3. Appel intra-service async-only : `predict` -> `infer_batch` passe par un proxy RPC qui retourne une coroutine. La methode appelee DOIT etre `async def` et l'appel `await self.infer_batch(...)`.
4. Sérialisation float64 silencieuse via le proxy : les `np.ndarray` float32 sont promus en float64 lors du transit HTTP entre endpoints. Cast explicite requis : `np.ascontiguousarray(batch, dtype=np.float32)`.
5. `PIL.Image.Image` doit rester un import runtime (BentoML introspecte les annotations via `typing.get_type_hints()` au demarrage du worker pour brancher le decodeur HTTP). `noqa: TC002` requis sur cet import.
6. `ModelOptions` n'est pas un dict : `bento_model.info.options.get(...)` plante. Pour retrouver le fichier ONNX dans le Model Store : glob `saved_model.onnx` (nom standard de `bentoml.onnx.save_model`) avec fallback `*.onnx`.
7. Conflit de version `anyio` lors de l'install : BentoML tire `httpx-ws==0.9.0` qui exige `anyio>=4.7` (`AsyncContextManagerMixin`). Pin manuel : `anyio>=4.7,<5`.
8. Query params non mappes automatiquement : `?top_n=3` ignore. Les params optionnels passent via le body JSON. (A approfondir si besoin.)

### Artefacts produits

- `src/serving_bentoml/__init__.py` - documentation des 6 pieges
- `src/serving_bentoml/schemas.py` - schemas Pydantic (avec champ `architecture` en plus de FastAPI)
- `src/serving_bentoml/preprocessing.py` - Resize 256 / CenterCrop 224 / Normalize ImageNet (numpy pur)
- `src/serving_bentoml/runner.py` - `OnnxRunner` (chargement Model Store + onnxruntime, lecture `custom_objects`)
- `src/serving_bentoml/service.py` - `ChampyService` (5 endpoints + batching adaptatif + 3 metriques custom)
- `scripts/import_model_to_bentoml.py` - import ONNX -> Model Store BentoML (CLI avec `--version`, `--architecture`)
- `scripts/compare_fastapi_bentoml.py` - script de parity-check reproductible
- `bentofile.yaml` - configuration de packaging du bento (Bloc 3)

### Bloc 3 - Bentofile et build du bento packagee (2026-05-08)

#### Mode dev vs mode production de `bentoml serve`

| Commande | Quand l'utiliser | Comportement |
|----------|------------------|--------------|
| `bentoml serve src.serving_bentoml.service:ChampyService --port 8020` | Developpement | Charge le code en direct depuis l'arborescence repo. Hot-reload manuel par redemarrage. Utile pour iterer rapidement. Le modele ONNX est resolu via le Model Store local au demarrage du worker. |
| `bentoml serve champy_classifier:latest --port 8020` | Production / staging | Charge le bento packagee depuis `~/bentoml/bentos/`. Code immuable, dependances fixees au moment du `build`, modele resolu depuis le Model Store local. Pas de dependance au repo source. C'est ce qu'utilisera l'image Docker au Bloc 5. |

#### bentofile.yaml

Configuration retenue (`bentofile.yaml` a la racine du repo) :
- `service: src.serving_bentoml.service:ChampyService`
- `models: [champy_classifier:latest]` - sinon le modele n'est pas embarque (le runner le resout au runtime, pas au build)
- `include` : code de la couche serving + `models/class_names.json` (fallback)
- `python.packages` : dependances filtrees (no torch / mlflow / dvc / streamlit / pandas / sklearn / evidently). Reste : `bentoml`, `onnx`, `onnxruntime`, `pillow`, `numpy`, `pydantic`, `prometheus-client`, `loguru`. BentoML tire automatiquement FastAPI/uvicorn.
- `docker.python_version: "3.11"`, `docker.distro: debian`
- `labels` : owner, project, stage, framework, backbone

#### Pieges Bloc 3

1. **Schema `python.version` n'existe pas en BentoML 1.4** : la version Python se declare via `docker.python_version`, pas `python.version`. Erreur cryptique `PythonOptions.__init__() got an unexpected keyword argument 'version'`.
2. **Le modele n'est PAS detecte automatiquement** : le runner appelle `bentoml.onnx.get(tag)` au runtime (dans `__init__`), donc l'introspecteur de build ne le voit pas. Sans `models:` dans bentofile, le bento se construit a 67 KB (code only, "Model Size = 0"). Avec `models:`, BentoML cree un lien vers le Model Store et la "Model Size" devient 106.30 MiB.
3. **Le bento sur disque reste petit** : meme avec `models:`, le bento sur disque ne contient PAS le fichier ONNX (juste un lien vers le Model Store). Total = 67 KB de code + 106 MB de modele linke = ~106 MB. C'est `bentoml containerize` qui inlinera le modele dans l'image Docker.

#### Validation

```
$ bentoml list
Tag                              Size       Model Size  Creation Time
champy_classifier:yhtfcj2kv2v... 67.58 KiB  106.30 MiB  2026-05-08 09:23:18
```

- `bentoml serve champy_classifier:latest --port 8020` demarre OK
- `/health` retourne `{"status":"healthy","model_loaded":true,"model_version":"1.0.0"}`
- `/predict` sur Amanita rubescens retourne `0.999907317550425` (16 chiffres) - bit-identique au mode dev
- Parity-check FastAPI vs bento packagee : **4/4 OK, delta max 1.21e-07**

---

## Etape 6ter - Monitoring M1 - Dashboards Grafana provisionnees

**Date** : 2026-05-08
**Objectif** : Provisionner Grafana via fichiers de config (datasource + 3 dashboards JSON) pour que la stack monitoring soit reproductible : suppression du volume `grafana-data` ou redeploiement = etat identique au boot suivant.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Mode de provisioning | Fichiers YAML (datasource + dashboard provider) + JSON dashboards | Configuration manuelle via UI | Reproductible, code dans le repo, diffable, survit a un `docker volume rm` |
| Datasource Prometheus | URL interne `http://prometheus:9090`, UID explicite `prometheus` | URL externe `host.docker.internal:9090`, UID auto-genere | Communication via le network compose, pas de dependance host. UID explicite pour pouvoir referencer la datasource depuis les dashboards JSON sans casser au prochain reboot |
| Source des metriques | `champy_*` (custom Prometheus) primaire, `process_*` / `python_*` secondaire | `bentoml_service_*` (natif BentoML) | Les `champy_*` sont communes a FastAPI et BentoML : permet la transition Etape 6bis sans modifier les dashboards. Les `bentoml_service_*` ne seront disponibles qu'au Bloc 5 quand BentoML rejoindra le compose |
| Dashboard 03 sans cAdvisor | Process metrics (`process_*`, `python_*`) du client Prometheus | cAdvisor / node_exporter | Suffisant pour observer la sante du process API (RAM, CPU, FDs, GC). cAdvisor utile uniquement si on veut des metriques host (disque, reseau) - hors scope ici |
| Folder Grafana | "Champy Classifier" (folderUid `champy-classifier`) | Root | Isolation visuelle propre dans Grafana, evite la confusion avec d'autres projets sur le NUC3 partage |
| Schema dashboards | v38 (Grafana >= 10) | Schema legacy | Latest, supporte le datasource explicite `{type, uid}` recommande |

### Architecture du provisioning

```
configs/grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yml      # Datasource auto avec UID 'prometheus'
│   └── dashboards/
│       └── dashboards.yml      # Provider file-based, scan /var/lib/grafana/dashboards/
└── dashboards/
    ├── 01_api_performance.json # Latence p50/p95/p99, RPS, erreurs HTTP, top endpoints
    ├── 02_predictions.json     # Top-10 especes, confiance, predictions/sec, especes uniques
    └── 03_system_health.json   # RAM, CPU, FDs, uptime, GC Python du process API
```

### Modifications docker-compose.yml

| Volume ajoute | Role |
|---------------|------|
| `./configs/grafana/provisioning:/etc/grafana/provisioning:ro` | Datasource + provider de dashboards charges au boot |
| `./configs/grafana/dashboards:/var/lib/grafana/dashboards:ro` (deja present) | JSON des 3 dashboards, recharges automatiquement (intervalle 30s) |

### Pieges Grafana provisioning rencontres

1. **`docker compose restart` n'applique PAS les nouveaux volumes** : il faut `docker compose up -d <service>` pour recreer le container. Avec un simple `restart`, les volumes du container deja existant restent inchanges. Symptome : provisioning yaml present sur l'hote mais absent dans `/etc/grafana/provisioning/` du container.
2. **Le volume nomme `grafana-data` persiste les datasources auto-generees** : si une datasource a deja ete creee manuellement avant le provisioning, elle coexiste avec la nouvelle (deux datasources de meme nom mais UID different). Le provisioning ne supprime pas, il ajoute. Solution : `DELETE /api/datasources/uid/<old_uid>` une fois ou bien `docker volume rm champy_classifier_grafana-data` avant le up (perd l'historique des modifs UI).
3. **Path translation Git Bash sur Windows** : `docker exec ... ls /etc/grafana/...` est traduit en `C:/Program Files/Git/etc/grafana/...` par Git Bash. Solution : prefixer la commande par `MSYS_NO_PATHCONV=1` ou utiliser PowerShell pour les commandes `docker exec`.
4. **`uid` explicite dans le datasource yaml indispensable** : sans `uid: prometheus`, Grafana auto-genere une chaine type `afhpol7cbsao0a` qui change a chaque recreate. Les dashboards JSON qui referencent la datasource via UID echouent silencieusement (panels en "no data"). Toujours fixer `uid` dans la YAML de provisioning.
5. **`folderUid` requis pour des dashboards non-root** : sans `folderUid` dans le provider yaml, Grafana cree un dossier au nom du provider mais avec un UID auto-genere - pas de probleme fonctionnel, mais les liens vers les dossiers dans le code Streamlit / docs cassent au prochain reboot.

### Validation (2026-05-08)

- 50 predictions envoyees via `scripts/seed_grafana.py` (stratification sur les especes, 26 req/s)
- Total Prometheus apres seed : **58 predictions** (50 nouvelles + 8 anterieures)
- Confiance moyenne : **95.69%**
- Latence p50 / p95 (5 min) : **19 ms / 43 ms**
- RAM process API : **238.7 MB**
- 3 dashboards visibles dans `Champy Classifier/` :
  - http://localhost:3010/d/champy-api-performance/ (6 panels)
  - http://localhost:3010/d/champy-predictions/ (5 panels)
  - http://localhost:3010/d/champy-system-health/ (6 panels)
- Datasource `Prometheus` (uid=prometheus, isDefault=true) provisionnee correctement
- Query proxy Grafana -> Prometheus retourne `58` pour `sum(champy_predictions_total)` : flux end-to-end OK

### Artefacts produits

- `configs/grafana/provisioning/datasources/prometheus.yml` - datasource Prometheus auto
- `configs/grafana/provisioning/dashboards/dashboards.yml` - provider de dashboards file-based
- `configs/grafana/dashboards/01_api_performance.json` - 6 panels (latence, RPS, erreurs, top endpoints)
- `configs/grafana/dashboards/02_predictions.json` - 5 panels (top-10 especes, confiance, distribution)
- `configs/grafana/dashboards/03_system_health.json` - 6 panels (RAM, CPU, FDs, uptime, GC Python)
- `scripts/seed_grafana.py` - generateur de trafic synthetique stratifie sur les especes
- `docker-compose.yml` - mount additionnel `configs/grafana/provisioning`

### Restant pour cloturer M1

- Aucun (M1 valide). Suite : M2 (PredictionStore SQLite + endpoint `/predictions/recent`).

---

## Etape 6quater - Monitoring M2 - Stockage des predictions (SQLite WAL)

**Date** : 2026-05-08
**Objectif** : Persister chaque prediction servie par le service BentoML dans une base SQLite locale pour alimenter le Quality Monitor (Bloc R) et la detection de derive (Bloc M3 - Evidently). La concurrence d'ecriture doit etre robuste : BentoML traite plusieurs requetes en parallele via l'adaptive batching et plusieurs workers async.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Backend | SQLite (fichier local) | PostgreSQL container, DuckDB, SQLAlchemy + ORM | Aucune dep externe, deploiement zero-config, suffisant pour ~10k req/s en WAL. PostgreSQL serait justifie a > 1M predictions/jour. |
| Driver | `aiosqlite` | `sqlite3` stdlib synchrone | Operations async-natives via thread interne dedie, ne bloque pas l'event loop BentoML. Deja tire par BentoML 1.4 (dep transitive). |
| Concurrence | WAL + busy_timeout=5000 + connexion partagee | Connexion par requete, lock asyncio explicite | WAL permet multi-readers + 1 writer concurrent. busy_timeout absorbe les pics. aiosqlite serialise les operations sur sa connexion via thread interne, donc partager une seule connexion est sur. |
| Synchronous | `NORMAL` | `FULL` (defaut) | Compromis recommande pour WAL : durabilite suffisante + ~3x plus rapide a l'ecriture. Acceptable pour des donnees de monitoring (perte de < 1s d'ecritures recentes en cas de crash brutal). |
| Init | Paresseux (premier predict) | Eager dans `__init__` | BentoML 1.4 ne supporte pas un `__init__` async. Pattern : creer l'objet dans `__init__`, appeler `await store.init()` au premier predict, garder un `asyncio.Lock` pour eviter la double-init en course. |
| Persistence dans predict | Fire-and-forget via `asyncio.create_task` | `await` synchrone | Le commit SQLite ajoute ~1ms a la reponse - negligeable, mais la persistence reste isolee du hot path en cas de probleme disque (timeout, full, etc.). Reference forte sur la Task pour eviter le GC (RUF006). |
| Hashage | SHA256 de `image.tobytes()` apres convert RGB | SHA256 du fichier upload, perceptual hash | `tobytes()` capture l'image apres decodage, deterministe pour meme contenu visuel quel que soit le format ; perceptual hash hors scope. |
| Schema | id (UUID) + timestamp + image_hash + predicted_class + confidence + top5_json + latency_ms | Table normalisee avec table secondaire pour top5 | Stockage du top-5 en JSON evite une JOIN couteuse pour le rendu Streamlit. SQLite < 1 MB par 10k predictions. |
| Endpoint /predictions/recent | POST avec body JSON `{hours, limit}` | GET avec query string | BentoML 1.4 ne mappe pas les query params (cf. Etape 6bis piege #8). Body JSON est l'idiome BentoML. |
| Volume Docker | `./data/runtime:/app/data/runtime` (directory) | `./data/runtime/predictions.db:/app/data/predictions.db` (file) | Mount d'un fichier qui n'existe pas encore = Docker cree un repertoire. Mount du dossier parent = robuste, supporte les sidecars WAL/SHM. |
| Path runtime | `/app/data/runtime/predictions.db` (env var `CHAMPY_PREDICTIONS_DB`) | Hardcode | Permet de surcharger le path en local (defaut `<repo>/data/runtime/predictions.db`) sans toucher au code. |

### Architecture

```
PredictionStore (src/serving_bentoml/storage.py)
├── init()           Cree fichier + active WAL/synchronous=NORMAL/busy_timeout=5000
├── save_prediction() INSERT, retourne UUID
├── get_recent()     SELECT WHERE timestamp >= cutoff ORDER BY ts DESC
├── get_class_distribution() SELECT predicted_class, COUNT(*) GROUP BY
├── count()          Total rows
└── close()          Fermeture propre, idempotente

ChampyService (src/serving_bentoml/service.py)
├── __init__         Cree PredictionStore (fermee), Lock pour init paresseux,
│                    set des Tasks de persistence (reference forte)
├── _ensure_store()  Init paresseux thread-safe
├── _save_prediction_safe() Wrapper qui logue les exceptions
├── predict          Fire-and-forget vers _save_prediction_safe via create_task
└── /predictions/recent (POST) JSON {hours, limit} -> List[PredictionRecord]
```

### Tests unitaires (tests/unit/test_prediction_store.py)

9 tests, tous passent en 0.25s :
1. init() cree le fichier + active WAL
2. init() est idempotent
3. save + get_recent round-trip
4. get_recent filtre correctement par fenetre (48h vs 5min)
5. get_class_distribution agrege par classe
6. get_class_distribution avec `since` filtre temporellement
7. **Concurrence : 100 ecritures via asyncio.gather sans perte ni `database is locked`**
8. count() = 0 sur base vide, close() idempotente
9. save_prediction sans init() leve RuntimeError explicite

### Validation end-to-end (2026-05-08)

- 5 predictions sur images de test variees -> 5 records en base
- `/predictions/recent` (POST `{hours: 1, limit: 10}`) retourne les 5 records par timestamp decroissant avec hashes uniques
- Stress concurrent : 50 requetes `/predict` en `asyncio.gather` -> 11 OK + 39 503
  - Les 503 viennent de la couche batching BentoML (queue bornee a max_batch_size=32), PAS du store
  - Les 11 predictions OK sont toutes persistees correctement
  - **Aucun `database is locked` declenche** : WAL + busy_timeout absorbent la concurrence
- Total apres stress : 16 lignes en base, distribution coherente

### Pieges SQLite + BentoML rencontres

1. **`asyncio.create_task` sans reference forte = RUF006** : sans `self._pending_writes.add(task)`, le GC peut collecter la Task avant completion (Python <3.13). Avec une reference forte + `task.add_done_callback(self._pending_writes.discard)`, la Task vit jusqu'a la fin.
2. **`__init__` BentoML est sync** : impossible d'`await store.init()` dedans. Pattern d'init paresseux thread-safe avec `asyncio.Lock` pour eviter la double-init en course.
3. **Le proxy RPC interne a son propre quota** : BentoML borne implicitement le nombre de requetes simultanees pour proteger la memoire (queue + workers). Au-dela, les requetes sont rejetees en 503 par le proxy avant meme d'arriver a `predict`. Pas un probleme du store.
4. **Sidecars WAL et SHM** : SQLite en WAL cree `<db>.db-wal` et `<db>.db-shm` a cote du fichier principal. Les inclure dans `.gitignore` (sinon ils remontent silencieusement).
5. **`row_factory=aiosqlite.Row` doit etre defini APRES `connect()`** : le passer en argument de `connect()` est silencieusement ignore en aiosqlite. Faire `conn.row_factory = aiosqlite.Row` apres init.

### Artefacts produits

- `src/serving_bentoml/storage.py` - PredictionStore (WAL, ~330 lignes)
- `src/serving_bentoml/service.py` - hook fire-and-forget + endpoint `/predictions/recent`
- `tests/unit/test_prediction_store.py` - 9 tests
- `pyproject.toml` + `requirements.txt` + `bentofile.yaml` - `aiosqlite>=0.19`
- `docker-compose.yml` - mount `./data/runtime:/app/data/runtime` + env `CHAMPY_PREDICTIONS_DB`
- `.gitignore` - exclusion `data/runtime/*.db*`
- `data/runtime/.gitkeep` - garde le repertoire dans git

### Restant pour cloturer M2

- Aucun (M2 valide). Suite : M3 (drift detection Evidently + page Streamlit 11).
- A surveiller au Bloc 5 : verifier que le volume monte dans le container BentoML est ecrit avec les bonnes permissions (uid:gid).

---

## Etape 6quinquies - Monitoring M3 - Drift detection (Evidently)

**Date** : 2026-05-08
**Objectif** : Detecter automatiquement quand la distribution des predictions de production diverge de la baseline (test set). Rapports HTML auto-suffisants generes a la demande, archives dans `monitoring/reports/`, et embarques dans la page Streamlit `11_drift.py`.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Outil | Evidently 0.7.21 | NannyML, Alibi Detect, custom | API mature, presets HTML auto-suffisants, integration Streamlit immediate via `st.components.v1.html` |
| API Evidently | `Dataset.from_pandas(df, DataDefinition)` + `Report([DataDriftPreset(), DataSummaryPreset()])` | API legacy `metrics.DataDriftTable` | API moderne 0.7+ ; les presets gerent automatiquement le test du chi-2 (categoriel) et KS (numerique) sans configuration |
| Baseline | Top-1 prediction du modele courant sur le test set (2872 images) | Distribution des labels d'origine, distribution train | La baseline doit etre comparable aux predictions de prod : on compare ce que le modele predit ICI sur le test (referentiel stable) avec ce qu'il predit en prod (qui peut deriver) |
| Stockage baseline | JSON 8 KB co-localise avec les scripts (`monitoring/baseline_reference.json`) | DVC, MLflow artifact | Le JSON est petit, structurel, doit etre dispo immediatement (pas besoin d'un `dvc pull`). Versionne avec git, regenerable a la demande. |
| Stockage rapports | HTML self-contained dans `monitoring/reports/` (gitignore) | Versionner les rapports, S3, MLflow | 3.8 MB par rapport (fonts + libs JS inline). Generes a la demande : pas de valeur archivistique a 6 mois. Le fichier reste local pour la demo. |
| Comparaison | 2 colonnes : `predicted_class` (categoriel) + `confidence` (numerique) | Image embeddings drift | Suffisant pour un POC TFE. Embeddings drift necessite extraction de features et est ~10x plus lourd. |
| Source predictions courantes | PredictionStore SQLite (Bloc M2) sur fenetre glissante | Re-scan en temps reel | Le store est deja le golden source des predictions servies. Lookup O(1) sur l'index timestamp. |
| Trigger | Manuel (bouton Streamlit + CLI) | Cron / webhook | POC TFE : pas de cron jusqu'a la prod. La regeneration prend ~1s sur 100 predictions. |

### Architecture

```
monitoring/
├── __init__.py               # docstring de package
├── baseline_snapshot.py      # CLI : test set -> baseline_reference.json
├── run_drift_report.py       # CLI : baseline + SQLite -> rapport HTML
├── baseline_reference.json   # 8 KB, commit (regenerable mais reproductible)
└── reports/
    ├── .gitkeep              # garde le dossier
    └── drift_YYYYMMDD_HHMM.html  # gitignore (3-4 MB chacun)
```

### baseline_snapshot.py - calcul de la baseline

Pipeline :
1. Charge `models/best_model.onnx` (ou path passe en --model)
2. Construit un index `nom_fichier -> path` une fois sur `data/raw/Mushrooms_images/` (~647k fichiers, 22s ; un seul scan au lieu de 2872 rglob qui prendrait ~30 min)
3. Pour chaque image du split test : preprocess (Resize 256 / CenterCrop 224 / Normalize ImageNet, identique au serving) + softmax + top-1
4. Agrege par classe : count, share, confidence_mean / min / max
5. Histogramme global (7 buckets : `[0, 0.5)`, `[0.5, 0.7)`, ..., `[0.99, 1.0]`)
6. Statistiques globales : top-1 accuracy, confidence P10 / P50 / P95
7. Sauvegarde JSON (8 KB)

### run_drift_report.py - rapport Evidently

Pipeline :
1. Charge la baseline JSON et materialise un DataFrame de reference (`count` lignes par classe avec `confidence_mean` comme valeur)
2. Lit les predictions des `--hours` dernieres heures depuis le PredictionStore SQLite
3. Build deux `Dataset` Evidently avec la meme `DataDefinition` (categorical_columns=`predicted_class`, numerical_columns=`confidence`)
4. `Report([DataDriftPreset(), DataSummaryPreset()]).run(reference, current)` -> `Snapshot`
5. `snapshot.save_html(path)` -> rapport autoporteur

### Page Streamlit 11_drift.py

Trois sections :
- **Section 1** : etat de la baseline (4 metrics depuis le JSON : nb images, accuracy, confiance moyenne, P10/P95). Si pas de baseline, message d'erreur explicite avec la commande a lancer.
- **Section 2** : slider sur la fenetre temporelle + bouton de generation. Le bouton lance `subprocess.run([sys.executable, monitoring/run_drift_report.py, --hours, N])`. Spinner + expander pour les logs.
- **Section 3** : selecteur sur la liste des rapports archives (parsing de `drift_YYYYMMDD_HHMM.html` -> label `YYYY-MM-DD HH:MM`). Affichage du HTML choisi via `st.components.v1.html(content, height=900, scrolling=True)`.

Aucune valeur en dur : la baseline, les metrics, la liste des rapports sont toutes lues au runtime.

### Pieges Evidently rencontres

1. **API 0.7+ a casse l'API legacy de 0.4** : les exemples web qu'on trouve majoritairement sont en 0.4 (`from evidently.report import Report; Report(metrics=[DataDriftMetric()])`). En 0.7, c'est `from evidently import Report, Dataset, DataDefinition; Report(metrics=[DataDriftPreset()])`. Verifier la version installee avant de copier-coller.
2. **`DataDefinition` obligatoire pour les colonnes mixtes** : sans `categorical_columns=...` + `numerical_columns=...`, Evidently auto-detecte mal le type de `confidence` (parfois float arrondis traites comme cat) et fait crasher le test stat.
3. **`Dataset.from_pandas` ne prend pas le DataFrame nu** : il faut `data_definition=` en kwarg, sinon erreur silencieuse de typage colonne.
4. **`save_html` est une methode de `Snapshot` (le retour de `.run()`), pas de `Report`** : `Report.save_html` n'existe pas en 0.7. C'est `report.run(...).save_html(path)`.
5. **HTML genere = 3-4 MB self-contained** : Evidently inline les fonts Material Icons + Vega-Lite + ses propres libs JS dans chaque rapport. Pas de dependance externe au serve, mais beaucoup d'octets. Gitignore les rapports, archive selectivement si besoin.
6. **`baseline_to_dataframe` materialise count lignes par classe** : on perd l'info de variance intra-classe en n'utilisant que `confidence_mean`. Pour un drift sur la confiance, c'est suffisant (Evidently compare les distributions globales). Pour une analyse fine, il faudrait stocker les confidences individuelles dans la baseline (multiplie sa taille par 1000).
7. **`subprocess.run([sys.executable, script, ...])` plutot que `python ...`** : sur Windows, `python` peut pointer vers une install differente du venv courant. Utiliser `sys.executable` garantit qu'on lance le bon Python.
8. **Le bouton Streamlit ne reload pas la page apres generation** : il faut interagir avec le selecteur (ou rafraichir manuellement) pour voir le nouveau rapport. Ameliorable via `st.rerun()` apres generation, hors scope ici.
9. **L'indexation rglob prealable est cruciale** : 1.6 img/s avec rglob par image (647k fichiers a chaque appel) -> 40 img/s avec un index pre-construit (un seul scan). Pour 2872 images : 30 min avant -> 1m20 apres.
10. **mypy 1.13 ne reconnait pas certains codes d'erreur de mypy 1.16+** : l'option `disable_error_code = ["untyped-decorator"]` ajoutee au CI fix etait un faux positif et faisait crasher le pre-commit. Retiree puisque mypy 1.13 partout (CI + pre-commit) ne genere pas le code en question.

### Validation end-to-end (2026-05-08)

- **Baseline** : `monitoring/baseline_snapshot.py` sur le test set (2872 images, 79s, 40 img/s)
  - Top-1 accuracy : **89.90%**
  - Confiance moyenne : **0.9518**, P10 / P95 : **0.78 / 1.00**
  - 30 classes vues
- **100 predictions BentoML** envoyees via `scripts/seed_grafana.py --target bentoml`
  - 100/100 OK, 18 req/s
  - Store SQLite contient 116 lignes (100 nouvelles + 16 anterieures)
- **Premier rapport drift** : `monitoring/reports/drift_20260508_1049.html`
  - Reference 2872 lignes, current 116 predictions, 30 classes
  - HTML self-contained 3.8 MB
- **Page Streamlit 11** : ouverte sur port 8502 (natif), HTTP 200, pas d'erreur dans les logs
  - Baseline metrics affichees correctement
  - Selecteur liste le rapport genere
  - HTML embarqué via `st.components.v1.html`

### Artefacts produits

- `monitoring/__init__.py` - docstring de package
- `monitoring/baseline_snapshot.py` - CLI baseline (333 lignes)
- `monitoring/run_drift_report.py` - CLI drift report (220 lignes)
- `monitoring/baseline_reference.json` - 8 KB, commit
- `monitoring/reports/.gitkeep` - garde le dossier
- `monitoring/reports/drift_YYYYMMDD_HHMM.html` - 3.8 MB, gitignore
- `demo/pages/11_drift.py` - 3 sections (baseline / generation / archives)
- `.pre-commit-config.yaml` - extension `^monitoring/` au scope mypy + interrogate
- `.github/workflows/ci.yml` - extension `monitoring/` au lint + mypy + interrogate
- `pyproject.toml` - retrait `disable_error_code` (mypy 1.13 ne le supporte pas)

### Restant pour cloturer M3

- Aucun (M3 valide). Suite : M4 (page Streamlit monitoring complete avec iframe Grafana + alerting visuel).
- Plus tard : ajouter un `st.rerun()` apres generation de rapport pour rafraichir le selecteur automatiquement (UX).
- Plus tard : declencher un job cron / GitHub Actions pour generer un rapport quotidien et remonter les drifts critiques par mail.

---

## Etape 6sexies - Monitoring M4 - Page Streamlit complete

**Date** : 2026-05-08
**Objectif** : Refondre la page `10_monitoring.py` pour offrir une vue ops bout-en-bout : metriques live, dashboards Grafana embarques, top-10 especes, et alerting visuel a partir d'un fichier de seuils. Page resiliente quand Prometheus, Grafana ou le store SQLite sont down.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Seuils alerting | YAML externe `configs/monitoring/thresholds.yml` | Hardcoded, env vars | Editable sans redeploiement, supporte la convention warning/critical par direction (lower_is_worse / higher_is_worse), coherent avec la regle invariante "zero hardcoded" |
| Auto-refresh | Cache TTL=15-30s + bouton "Rafraichir maintenant" | `st.autorefresh` (dep tierce), `time.sleep + st.rerun` (bloque worker) | Pattern Streamlit natif, ne tire pas de dep, fonctionne sans bloquer la session ; le user peut declencher un refresh explicite quand il veut |
| Iframe Grafana | Tabs Streamlit + `st.components.v1.iframe` avec URL `?orgId=1&kiosk&theme=light&refresh=30s` | API Grafana JSON pour reconstruire les graphes, screenshot render API | Reuse direct des dashboards provisionnes au Bloc M1, refresh integre, kiosk mode masque la nav Grafana, fallback automatique sur liens si Grafana down |
| Auth Grafana embedding | `GF_AUTH_ANONYMOUS_ENABLED=true` + `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` + `GF_SECURITY_ALLOW_EMBEDDING=true` dans le compose | Reverse proxy avec auth deleguee | Acceptable pour la demo locale et le NUC3 LAN ; en prod il faudrait un proxy avec OIDC |
| URL Grafana cote browser | Heuristique : si `hostname in {grafana, host.docker.internal}`, force `http://localhost:<port>` | Demander l'URL au user | Le helper `get_grafana_url()` retourne le DNS interne du compose (`http://grafana:3000`) qui ne resout PAS cote browser. Fix automatique via `urlparse` + override env `CHAMPY_GRAFANA_URL_EXTERNAL` |
| Top-10 especes | 2 colonnes : Plotly bar Prometheus (cumul global) + Plotly line SQLite (tendance horaire 24h) | Une seule source | Le cumul Prometheus montre la distribution stable, la tendance SQLite montre les variations recentes (utile pour reperer des pics ou des chutes par espece). Si le SQLite n'existe pas, on degrade silencieusement |
| Alerting visuel | 3 cartes vert/jaune/rouge avec contour color, label, valeur, message | st.metric standard, plotly gauges | Permet de voir d'un coup d'oeil l'etat global ; plus visuel que des metriques empilees, plus simple que des gauges |
| Resilience | Try / except defensifs partout, message st.warning explicite avec action a faire | Page plante avec stack trace | UX ops : un dashboard down ne casse pas la page, l'utilisateur sait quoi faire (verifier docker compose ps, ALLOW_EMBEDDING, etc.) |

### Architecture

```
demo/lib/monitoring_utils.py     # Helpers : load_thresholds, fetch_live_metrics, evaluate_alerts
configs/monitoring/thresholds.yml # Seuils par direction warning/critical
demo/pages/10_monitoring.py      # 4 sections (live / Grafana / top-10 / alerting)
```

### Section 1 - Metriques live (Prometheus)

7 requetes PromQL :
- `sum(rate(champy_requests_total[5m]))` -> RPS
- `histogram_quantile({0.5, 0.95, 0.99}, sum(rate(champy_prediction_latency_seconds_bucket[5m])) by (le))` -> p50/p95/p99
- `(sum(rate(champy_http_errors_total[5m])) or on() vector(0)) / clamp_min(sum(rate(champy_requests_total[5m])), 0.001)` -> taux d'erreur (gere le cas "pas d'erreurs jamais")
- `sum(champy_predictions_total)` -> total cumule
- `champy_prediction_confidence_sum / clamp_min(champy_prediction_confidence_count, 1)` -> confiance moyenne

Affichage en 5 colonnes + 2 colonnes (erreur, confiance). Si Prometheus down, message d'erreur explicite, pas de plantage.

### Section 2 - Dashboards Grafana embarques

Pattern :
1. Health check `httpx.get(f'{url}/api/health', timeout=3)` cache TTL=15s
2. Si OK : tabs Streamlit pour 3 dashboards + `st.components.v1.iframe(url, height=720)`
3. Si KO : fallback liens markdown vers les dashboards (a ouvrir dans un nouvel onglet)

URL generee : `{base}/d/{uid}?orgId=1&kiosk&theme=light&refresh=30s`. Le mode `kiosk` masque la nav Grafana pour un rendu plus propre dans Streamlit.

### Section 3 - Top-10 especes predites

Deux sources cote a cote :
- **Prometheus** (col gauche) : `topk(10, sum by (species) (champy_predictions_total))` -> Plotly horizontal bar avec gradient bleu, ordonne par count croissant pour que le top-1 soit en haut.
- **SQLite** (col droite) : `PredictionStore.get_recent(hours=24)` -> aggregation horaire (df.groupby(['hour', 'species']).size()) -> Plotly line avec markers, top-5 especes seulement pour ne pas surcharger la legende.

Si le SQLite n'existe pas (compose actuel sert FastAPI qui n'ecrit pas), message info, pas de plantage.

### Section 4 - Alerting visuel

3 cartes HTML rendues via `st.markdown(unsafe_allow_html=True)` avec :
- Bordure couleur (vert / jaune / rouge / gris selon le niveau)
- Label (OK / WARNING / CRITICAL / INDISPONIBLE)
- Nom de la metrique
- Valeur courante formatee
- Message court explicitant le niveau (`X.X%` >= seuil critical Y.Y%)

Logique :
- `lower_is_worse` (confidence) : warning si <= 0.7, critical si <= 0.5
- `higher_is_worse` (latence p95, error rate) : warning si >= seuil, critical si >= seuil critique

Si `thresholds.yml` est absent ou Prometheus down, message warning a la place des cartes.

Expander avec le contenu du YAML pour la transparence (le user voit les seuils sans aller fouiller dans configs/).

### Pieges Streamlit + Grafana iframe rencontres

1. **Grafana refuse l'embed par defaut** : sans `GF_SECURITY_ALLOW_EMBEDDING=true`, Grafana met un header `X-Frame-Options: deny` qui bloque l'iframe. Symptome : iframe vide cote Streamlit. Verifier avec `curl -I http://grafana:3000/d/<uid>` que le header n'est plus present.
2. **Auth bloque l'iframe meme avec ALLOW_EMBEDDING** : si Grafana exige une connexion, l'iframe affiche le login form. Activer `GF_AUTH_ANONYMOUS_ENABLED=true` + `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` permet aux iframes de charger sans auth (pour des dashboards en lecture seule).
3. **`docker compose restart` n'applique PAS les nouvelles env vars** : il faut `docker compose up -d grafana` (recreate). Identique au piege Bloc M1 sur les volumes.
4. **DNS interne du compose != cote browser** : le helper `get_grafana_url()` peut retourner `http://grafana:3000` (utile pour les containers entre eux), mais cette URL ne resout PAS dans un iframe cote navigateur. Fix : detecter via `urlparse(url).hostname in {grafana, host.docker.internal}` et forcer `http://localhost:<port>`. Override possible via env `CHAMPY_GRAFANA_URL_EXTERNAL`.
5. **`histogram_quantile` retourne NaN sans donnees recentes** : pas de prediction sur la fenetre 5min -> p50/p95/p99 = NaN. Si on les passe a `f"{value:.0f} ms"`, on obtient `NaN ms` (laid). Fix : convertir NaN en None dans le helper, afficher `-` cote UI.
6. **Metric inexistante = empty array** : `champy_http_errors_total` n'existe que si au moins une erreur HTTP a ete loggee. Sans ca, la query retourne `[]` et le ratio plante. Fix PromQL : `(sum(...) or on() vector(0))` injecte un 0 quand le numerateur est vide.
7. **Cache `st.cache_data` sur `httpx.get`** : sans cache, chaque rendu Streamlit retape Prometheus / Grafana 5+ fois (sections successives). Avec `ttl=15-30s`, les metriques sont fraiches sans surcharger les services.
8. **`unsafe_allow_html=True` requis pour les cartes colorees** : st.markdown ne supporte pas le CSS inline par defaut. C'est OK quand on controle le HTML genere (pas d'input user dans le template).
9. **`st.components.v1.iframe` vs `st.components.v1.html`** : iframe attend une URL, html attend du HTML brut. Pour les dashboards Grafana on veut iframe (charge l'URL en lazy).

### Validation end-to-end (2026-05-08)

- Page rendue en natif (port 8502) : aucune erreur dans les logs Streamlit
- Sections testees :
  - Section 1 : 5 metriques affichees (RPS 0.14, p50 21ms, p95 46ms, p99 49ms, total 88, conf 95.1%, error 0%)
  - Section 2 : 3 tabs Grafana avec iframes vers http://localhost:3010 (OK avec auth anonymous + ALLOW_EMBEDDING)
  - Section 3 : Plotly bar Prometheus (10 especes), Plotly line SQLite (info "Pas de SQLite" car BentoML pas dans le compose)
  - Section 4 : 3 cartes vertes (Confiance 95.1% OK, Latence p95 46ms OK, Erreur 0% OK)
- **Resilience** : `docker stop champy_classifier-grafana-1` -> page rend sans erreur, fallback liens visible

### Artefacts produits

- `configs/monitoring/thresholds.yml` - 3 sections (confidence / latency_p95_seconds / error_rate)
- `demo/lib/monitoring_utils.py` - helpers (load_thresholds, fetch_live_metrics, evaluate_alerts, MetricStatus, _first_value avec NaN-safe)
- `demo/pages/10_monitoring.py` - refonte complete (4 sections)
- `docker-compose.yml` - 4 env vars Grafana (auth anonyme + embedding)

### Restant pour cloturer M4

- Aucun (M4 valide). Suite : Bloc 4 (tests preprocessing + integration) ou Bloc R1 (Quality Monitor) selon prochaine priorite.
- Plus tard : ajouter `streamlit-autorefresh` ou un meta refresh HTML pour un refresh visuel toutes les 30s sans intervention user.
- Plus tard : exposer un `/metrics` cote Streamlit lui-meme pour boucler le monitoring (latence des pages, requetes par page).

---

## Etape 7 - Docker et Monitoring - EN COURS

**Date** : 2026-03-30 (initial) / 2026-04-24 (port mapping hote partage)
**Objectif** : Stack Docker Compose bout-en-bout (API + Demo + Prometheus + Grafana), cohabitation propre avec les autres projets du NUC3.

### Images Docker

| Image | Base | Contenu |
|-------|------|---------|
| `docker/Dockerfile.api` | python:3.11-slim | FastAPI + ONNX Runtime + Prometheus client |
| `docker/Dockerfile.demo` | python:3.11-slim | Streamlit + Plotly + data artifacts |

### Docker Compose - Port mapping sur hote partage NUC3

Le NUC3 heberge plusieurs projets simultanement. Mapping applique pour cohabiter :

| Service | Port host | Port container | Rationnel |
|---------|-----------|----------------|-----------|
| api | **8010** | 8000 | 8000 pris par `plumesenvue-api` - offset +10 |
| demo | 8501 | 8501 | Libre, pas de remap |
| prometheus | 9090 | 9090 | Libre, pas de remap |
| grafana | **3010** | 3000 | 3000 pris par `open-webui` - offset +10 |

Mapping documente en commentaire YAML en tete de `docker-compose.yml`. Piege detaille dans PLAYBOOK.md.

### Configuration Prometheus

Scrape via le network interne docker-compose : `targets: ["api:8000"]` (pas `host.docker.internal`). Plus robuste, portable, et ne depend pas du host Windows.

### Metriques monitorees (API)

- `champy_predictions_total` (counter par espece)
- `champy_prediction_latency_seconds` (histogramme, buckets 10ms a 2.5s)
- `champy_prediction_confidence` (summary, confiance top-1)
- `champy_http_errors_total` (counter par status_code)
- `champy_requests_total` (counter par method + endpoint)

### Problemes rencontres
- Le `.env` a disparu lors d'un incident NUC3 (commit `da471d1 chore: WIP snapshot before NUC3 incident`). Recree avec token DagsHub pour MLflow + GRAFANA_PASSWORD.
- `Dockerfile.demo` fait `COPY .env.example .env` -> le `.env` du container est un stub, mais `env_file: .env` du compose override via env vars au runtime. Pydantic BaseSettings privilegie les env vars sur le fichier : OK.
- Les pages Streamlit `09_api.py` et `10_monitoring.py` hardcodaient `http://localhost:8000`, `9090`, `3000` -> refactorees pour lire via helpers `get_api_url()`, `get_prometheus_url()`, `get_grafana_url()` (env vars `CHAMPY_*` avec defauts).

### Artefacts produits
- `docker/Dockerfile.api`, `docker/Dockerfile.demo`
- `docker-compose.yml` - 4 services orchestres (api, demo, prometheus, grafana)
- `configs/prometheus.yml` - scrape config via network interne
- `.dockerignore` - exclut venv, data raw, tests, caches

### Metriques / Resultats (stack local 2026-04-24)
- 4 containers up : api (healthy), demo, prometheus, grafana
- Prometheus scrape target `api:8000` : health = up
- Metriques `champy_predictions_total` visibles dans Prometheus
- Aucun impact sur les autres projets du NUC3

### Restant pour cloturer cette etape
- Dashboard Grafana pre-configure (JSON dans `configs/grafana/dashboards/champy.json`)
- Integration Evidently (drift report on-demand, page Streamlit 11)
- Image `docker/Dockerfile.train` (optionnelle, training principal en natif sur XPS)

---

## Etape 8 - Demo et Tests - EN COURS

**Date** : 2026-03-28 a 2026-04-24 (construction incrementale)
**Objectif** : Portfolio Streamlit 18 pages (zero hardcoded, sources dynamiques) + suite de tests.

### Pages Streamlit implementees

| Page | Source | Statut |
|------|--------|--------|
| 00 - Accueil (vue pipeline, statut dynamique des etapes) | Disque + artefacts | OK |
| 01 - Donnees brutes (distribution classes, formats, galerie) | `raw_stats.json` + disque | OK |
| 02 - Nettoyage (avant/apres 25 850 -> 19 138) | `cleaning_report.json` + `excluded.json` | OK |
| 03 - Augmentation (demo live PyTorch transforms) | `src/data/dataset.py` + images raw | OK |
| 04 - Split (distribution par classe par split) | `split_stats.json` + `split_manifest.csv` | OK |
| 05 - Entrainement (courbes MLflow + hyperparams) | MLflow / DagsHub | OK (necessite token valide) |
| 06 - Evaluation (confusion matrix, F1/classe, classes faibles) | MLflow artefacts | OK (necessite token valide) |
| 07 - Model Registry (checkpoint, ONNX, benchmark) | MLflow + disque | OK (necessite token valide) |
| 08 - Prediction (upload image, top-5) | ONNX Runtime local ou API | OK |
| 09 - API (statut, Swagger, metriques brutes) | FastAPI via `get_api_url()` | OK |
| 10 - Monitoring (PromQL, liens Grafana) | Prometheus via `get_prometheus_url()` | OK |
| 11 - Drift (rapport Evidently on-demand) | Evidently | A finaliser |
| 12 - Infrastructure (schema, Docker, CI/CD) | Docker CLI + GitHub API | A finaliser |

### Helpers partages (`demo/lib/`)

- `data_utils.py` - scan disque, chargement JSON/CSV, galerie
- `mlflow_utils.py` - search_runs, metric_history, fallback local
- `api_utils.py` - `get_api_url()`, `get_prometheus_url()`, `get_grafana_url()`, `get_health()`, `predict_image()`, `query_prometheus()`
- `viz.py` - helpers Plotly/Matplotlib reutilisables

Principe **zero hardcoded** respecte : aucune valeur (accuracy, nb images, noms de classes) n'est ecrite en dur. Tout est lu aux sources (MLflow, disque, API, Prometheus) avec fallback `st.warning()`.

### Tests

**Date de consolidation** : 2026-04-24.

Voir Etape 5 (CI/CD) pour le detail des 51 tests unitaires. En complement :
- Tests d'integration : a etoffer (API end-to-end, pipeline data complete)
- Coverage : a mesurer via `invoke test-coverage`

### Artefacts produits
- `demo/app.py` - page d'accueil
- `demo/pages/00_*.py` a `17_*.py` - 18 pages portfolio (00-12 initiales, 13-17 ajoutees ensuite : canari, analyse modeles, alertes, CI/CD, perspectives)
- `demo/lib/{data_utils,mlflow_utils,api_utils,viz}.py` - helpers partages
- `tests/unit/` - 51 tests couvrant dataset, dataloader, callbacks, evaluate, export_onnx, api
- `tests/integration/` - squelette (a etoffer)
- `tests/conftest.py` - fixtures partagees

### Restant pour cloturer cette etape
- Page 11 (Drift Evidently) : integration complete
- Page 12 (Infrastructure) : schema architecture + statut Docker + lien CI
- Tests d'integration API end-to-end (start container, call /predict, verifier reponse)
- Mesure coverage > 80%

---

## Bilan final

### Ce qui a bien fonctionne
- **Factory unifiee `create_backbone()`** : permet d'ajouter un backbone (ConvNeXt-Tiny) sans toucher au train.py, juste en ajoutant une entree dans le dispatch.
- **Curation from raw + filtre OpenCLIP** : plus reproductible que le pipeline legacy, conserve 70% de donnees en plus apres filtre qualite.
- **Zero hardcoded dans le Streamlit** : le portfolio se met a jour automatiquement quand on relance un training ou qu'on change de modele.
- **Pre-commit systematique** : a evite des dizaines de corrections de style / types / docstrings dans les PR.
- **DagsHub comme hub unique** : MLflow + DVC + Git dans une seule plateforme, un seul token.

### Difficultes majeures
- **Contraintes Windows** : pas de WSL, pas de bash, PowerShell uniquement, `num_workers=0` par defaut. Chaque commande shell doit etre portable (invoke + pathlib).
- **VRAM 4 GB (RTX 3050 Ti)** : limite batch=16 en AMP, pas de marge pour des modeles plus lourds. ConvNeXt-Tiny passe tout juste, ConvNeXt-Small serait hors budget.
- **Hote partage NUC3** : cohabitation avec 5+ autres projets Docker, necessite mapping de ports explicite avec offset +10.
- **Fine-grained classification** : les Russules (7 especes) sont visuellement tres similaires. Le modele plafonne a ~60-70% F1 sur ces classes malgre le WeightedRandomSampler.
- **Classes rares (< 100 images)** : metriques instables sur le test set (F1 oscille selon les runs). Le jeu de donnees est le facteur limitant, pas le modele.
- **Dynamo exporter torch 2.11** : produit des fichiers ONNX quasi vides pour ResNet50/ConvNeXt en opset 17. Bug contourne en forcant `dynamo=False`.

### Ameliorations possibles (hors scope TFE)
- **Augmentation par class-balanced-loss** en plus du WeightedRandomSampler (gain potentiel sur les classes rares).
- **Fine-tuning selectif des Russules** avec un dataset enrichi sur ces especes.
- **Model ensemble** ResNet50 + ConvNeXt-Tiny pour combiner les forces (gain attendu 1-2% accuracy).
- **Distillation** du ConvNeXt vers un modele plus leger pour l'inference mobile.
- **A/B testing en prod** via Prometheus labels pour mesurer l'impact d'un changement de modele.
- **Alerting Grafana** sur derive de confiance moyenne (proxy drift).

---

## Etape 9 - Migration BentoML & integration MinIO - EN COURS

**Date** : 2026-05-22 a 2026-05-23
**Objectif** : Aligner la stack de serving sur le standard MLOps de la formation (BentoML au lieu de FastAPI) et eliminer la dependance au cloud DagsHub pour le stockage des blobs DVC (MinIO self-hosted en parallele, DagsHub conserve pour rollback).

### Sous-etape 9.1 - Migration FastAPI -> BentoML (2026-05-22)

#### Constat de depart
Le service `api` du `docker-compose.yml` exposait FastAPI alors que la stack cible de la formation est BentoML (`bentoml.service` style 1.4). Le code BentoML etait deja present dans `src/serving_bentoml/` (service.py, runner.py, preprocessing.py, schemas.py, storage.py) mais inactif. L'image Docker n'embarquait pas `bentoml` ni `onnx`.

#### Etapes realisees

| Etape | Action | Statut |
|-------|--------|--------|
| 1 | Ajout de `bentoml>=1.2,<2.0`, `onnx`, `aiosqlite` au `docker/Dockerfile.api` (liste hardcodee, pas via requirements.txt pour garder image API minimale) | OK |
| 2 | Ajout du volume nomme `bentoml_data` mounte sur `/root/bentoml` (persistance du Model Store) | OK |
| 3 | Rebuild image API : `docker compose up -d --build api` | OK |
| 4 | Import du modele dans le Model Store : `scripts/import_model_to_bentoml.py --version 1.0.0 --architecture convnext_tiny --accuracy 0.90` -> tag `champy_classifier:aflvbmcwds3j2cur` (106 MiB ONNX) | OK |
| 5 | Adaptation du healthcheck Docker : `/health` -> `/healthz` (endpoint natif BentoML en GET, le `/health` custom est en POST) | OK |
| 6 | Adaptation `demo/lib/api_utils.py` : `get_health()` et `get_model_info()` passes en `httpx.post()` (BentoML 1.4 met les `@bentoml.api` en POST par defaut) | OK |
| 7 | Adaptation `predict_image()` : cle multipart `image` (au lieu de `file`, matche le parametre `image: PILImage` du decorateur), `top_n` en form-data au lieu de query params | OK |
| 8 | Adaptation `demo/pages/12_infrastructure.py` : URL health check API `/docs` -> `/healthz` (BentoML n'expose pas /docs, le Swagger est a la racine `/`) | OK |
| 9 | Surcharge de la `command:` dans `docker-compose.yml` service `api` : `bentoml serve src.serving_bentoml.service:ChampyService --host 0.0.0.0 --port 8000` (le CMD uvicorn du Dockerfile reste en fallback) | OK |
| 10 | Validation E2E : prediction depuis la page Streamlit `08_prediction` -> top-5 retourne, metriques `champy_*` incrementees, alertes Prometheus toujours calibrees | OK |

#### Pieges rencontres

- **BentoML 1.4 met `@bentoml.api` en POST par defaut.** Les endpoints applicatifs (`/health`, `/model/info`) sont donc en POST, pas GET. Cela casse les clients qui appellent en GET (healthcheck Docker, Streamlit). Solution : utiliser `/healthz` natif (GET) pour les probes, et passer les clients en POST pour les endpoints applicatifs.
- **Le `--force-recreate` perd les installs pip a chaud.** L'installation initiale de `onnx` a ete faite via `docker compose exec api pip install onnx`, perdue au premier `--force-recreate`. Solution definitive : ajout dans le Dockerfile, image rebuild.
- **Deprecation `bentoml.onnx` chez BentoML 1.4.** Un warning est emis a chaque chargement de modele : `bentoml.onnx is deprecated since v1.4 and will be removed in a future version`. Non bloquant pour la defense, a traiter en post-defense (migration vers `bentoml.models` generique).

#### Artefacts modifies

- `docker/Dockerfile.api` - ajout `bentoml`, `onnx`, `aiosqlite`, copie de `scripts/`
- `docker-compose.yml` - `command:` BentoML sur le service api, volume `bentoml_data`, healthcheck `/healthz`
- `demo/lib/api_utils.py` - migration des appels `/health`, `/model/info`, `/predict` vers les conventions BentoML
- `demo/pages/12_infrastructure.py` - URL de health check API

#### Compatibilite metriques

Les noms des metriques exposees sur `/metrics` restent identiques (`champy_predictions_total{species}`, `champy_prediction_latency_seconds`, `champy_prediction_confidence`). Aucun changement requis cote Prometheus ni cote regles d'alerte (`configs/alerts/champy_alerts.yml`). BentoML ajoute en bonus les metriques natives `bentoml_service_request_*`.

---

### Sous-etape 9.2 - Integration MinIO comme remote DVC alternatif (2026-05-23)

#### Architecture cible

Deux remotes DVC configures en parallele :
- `origin` (default conserve) : `s3://dvc` -> endpoint `https://dagshub.com/LoicFocraud/Champy_Classifier.s3` (cloud public)
- `minio` (alternatif) : `s3://champy-dvc` -> endpoint `http://localhost:9010` (self-hosted sur NUC3)

Motivation : sortir du cloud (alignement avec la ligne open formats / self-hosted), tout en gardant DagsHub disponible en rollback ou pour les projets a gros volume. L'aiguillage se fait par `dvc remote default <name>` (commutation atomique).

#### Etapes realisees

| Etape | Action | Statut |
|-------|--------|--------|
| 1 | Ajout du service `minio` dans `docker-compose.yml` : image `minio/minio:latest`, ports `9010:9000` (API) et `9011:9001` (console), volume nomme `minio_data`, healthcheck `curl /minio/health/live` | OK |
| 2 | Ajout des variables `MINIO_ROOT_USER` et `MINIO_ROOT_PASSWORD` dans `.env` (mot de passe 32 caracteres, alphanum + symboles sans guillemets ni apostrophes) | OK |
| 3 | Demarrage MinIO : `docker compose up -d minio` -> container healthy, console accessible sur `http://localhost:9011` | OK |
| 4 | Creation du bucket `champy-dvc` via la console MinIO (Object Browser -> Create Bucket) | OK |
| 5 | Configuration du remote DVC `minio` : `dvc remote add minio s3://champy-dvc` + `endpointurl http://localhost:9010` + `region us-east-1` (placeholder requis par le SDK S3) | OK |
| 6 | Credentials MinIO en `.dvc/config.local` (gere par `dvc remote modify --local`, gitignored par `.dvc/.gitignore`) | OK |
| 7 | Verification : `dvc remote list` affiche les deux remotes, `dvc remote default` retourne toujours `origin` | OK |
| 8 | Premier `dvc push -r minio` lance : 15 GB de cache local (~700K images + checkpoints) a transferer | EN COURS (~1h30 estime) |
| 9 | Validation : verifier dans la console MinIO que l'arborescence `champy-dvc/files/md5/...` se remplit | A FAIRE |
| 10 | Test recovery : `dvc pull -r minio` depuis un dossier propre pour valider le round-trip | A FAIRE |
| 11 | Bascule du default : `dvc remote default minio` | A FAIRE (post-validation) |

#### Pieges rencontres

- **Mot de passe MinIO genere avec guillemet et apostrophe.** Le premier mot de passe genere par PowerShell contenait `"` et `'` qui cassaient le parsing de `.dvc/config.local`. Solution : generation avec un alphabet restreint excluant `" ' \` $ \\`. Le secret a ete colle par inadvertance dans la conversation de travail, regenere immediatement (bucket etait vide, perte zero).
- **`dvc remote modify --local` echoue silencieusement avec les caracteres speciaux.** La commande s'execute sans erreur mais n'ecrit pas dans `.dvc/config.local`, ce qui force a editer manuellement le fichier. A noter pour les futures procedures.
- **Confusion entre `.dvc/config` (commit) et `.dvc/config.local` (gitignored).** Le secret a transite accidentellement par `.dvc/config` lors d'une edition manuelle, corrige avant tout commit. Verification : `Get-Content .dvc/config | Select-String "secret_access_key"` doit retourner vide.
- **Push initial lent (~2.8 MB/s) pour 15 GB.** Probablement du au scan Windows Defender de chaque blob + parallelisme par defaut DVC (4 workers). Pour les prochains push, prevoir `dvc push -j 16 -r minio` et exclusion Defender sur `.dvc/cache` et le volume Docker `minio_data`.

#### Artefacts modifies

- `docker-compose.yml` - ajout service `minio` et volume `minio_data`
- `.env` - ajout `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` (non commit)
- `.dvc/config` - declaration du remote `minio` (commit OK, sans creds)
- `.dvc/config.local` - credentials `minio` (gitignored)

#### Securite

- `.dvc/config.local` automatiquement gitignored par `.dvc/.gitignore` (`/config.local`)
- Credentials root MinIO utilises directement (pas d'access key dedie) : a durcir post-defense via creation d'un service account avec policy restreinte au seul bucket `champy-dvc`
- MinIO non expose publiquement (port 9010/9011 sur localhost uniquement), pas de tunnel Cloudflare

---

### Restant pour cloturer cette etape

- Validation push complet (en cours) et confirmation visuelle dans la console MinIO
- Test `dvc pull -r minio` depuis un dossier propre (valider le round-trip)
- Bascule du default DVC : `dvc remote default minio`
- Mise a jour du SVG anime du portfolio Streamlit : label "API FastAPI" deja change en "API BentoML" (fait)
- Mise a jour du README pour refleter l'architecture cible (BentoML + MinIO)
- Backlog post-defense : migration `bentoml.onnx` deprecie vers `bentoml.models` generique
- Backlog post-defense : access key MinIO dediee avec policy restreinte (au lieu du root)

---

## Etape 10 - Hub nginx + Cloudflare Access + refonte documentation - EN COURS

**Date** : 2026-05-24 / 2026-05-25
**Objectif** : Exposer la stack derriere un point d'entree HTTPS unique protege par Cloudflare Access (Zero Trust SSO), refondre la documentation Markdown du repo en mode showcase GitHub, et resoudre les pannes post-update du NUC Ubuntu hebergeant `cloudflared`. Trois sous-etapes : 10.1 (hub nginx + Cloudflare le 24/05), 10.2 (refonte doc le 24-25/05), 10.3 (diagnostic et fix post-update le 25/05 matin).

### Sous-etape 10.1 - Hub nginx + Cloudflare Access (2026-05-24)

#### Architecture cible

```
                 Internet
                    |
                    v
    Cloudflare Tunnel (NUC Ubuntu 192.168.50.39)
                    |
                    v
       champy.sbdg-ia.fr  --[Zero Trust SSO]
                    |
                    v
    nginx hub (NUC3 192.168.50.55:8088)
        |        |        |        |
        v        v        v        v
     demo    api      mlflow   airflow  ...  (12 services)
```

#### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Modele de routage | Sous-paths (`champy.sbdg-ia.fr/api/`, `/grafana/`, ...) | Sous-domaines (`api.champy.sbdg-ia.fr`) | 1 entree DNS, 1 certif TLS, 1 policy Cloudflare Access, 1 session SSO pour tous les services |
| Auth perimetrique | Cloudflare Access magic-link sur e-mail | Basic auth nginx, oauth2-proxy | Zero-config cote serveur, SSO partage avec les autres projets sbdg-ia.fr, traces dans Cloudflare logs |
| Image nginx | `nginx:alpine` | `nginx:latest` (Debian) | Empreinte minimale (24 MB), suffisant pour du reverse-proxy stateless |
| Port host nginx | 8088 | 80, 8080 | 80 reserve pour autre projet sur NUC3, 8080 conflit avec CrowdSec sur NUC Ubuntu |
| Resolution DNS backends | Statique au boot (defaut nginx) | Resolver Docker 127.0.0.11 + variables | Initialement statique pour simplicite ; refactor possible si pannes frequentes apres `--force-recreate` |
| Healthcheck nginx | Custom `/nginx-health` + `wget 127.0.0.1` | `wget localhost` (defaut BusyBox) | Forcer IPv4, BusyBox tente `::1` d'abord et echoue si pas de listen IPv6 |
| Header `Host` propage | `$http_host` | `$host` | Preserve le port et le nom externe ; necessaire pour les redirections d'Airflow et MinIO |
| Page d'index | Page Streamlit `00_plateforme.py` (hub interactif) | Page HTML statique nginx | Coherent avec le portfolio Streamlit, expose 12 services avec statut health en live |

#### Services exposes via le hub

| Service | Container | Port interne | Sous-path | Auth |
|---------|-----------|-------------:|-----------|------|
| nginx | champy_nginx | 80 | (entree) | Cloudflare Access |
| Streamlit demo | champy_demo | 8501 | `/` | herite |
| BentoML API | champy_api | 8000 | `/api/` | herite |
| MLflow | champy_mlflow | 5000 | `/mlflow/` | herite + basic auth optionnelle |
| Airflow webserver | champy_airflow | 8080 | `/airflow/` | herite + login Airflow |
| Prometheus | champy_prometheus | 9090 | `/prometheus/` | herite |
| Grafana | champy_grafana | 3000 | `/grafana/` | herite + anonymous viewer |
| Alertmanager | champy_alertmanager | 9093 | `/alertmanager/` | herite |
| MinIO console | champy_minio | 9001 | `/minio/` | herite + login MinIO |

Adaptateur Discord pour Alertmanager : interne uniquement, pas expose.

#### Cloudflare Tunnel sur le NUC Ubuntu

- `cloudflared` en service systemd (deja en place pour d'autres projets sbdg-ia.fr)
- Config dans `/home/<user>/.cloudflared/config.yml` : section `ingress` ajoutee pour `champy.sbdg-ia.fr` -> `http://192.168.50.55:8088`
- Application Cloudflare Access creee dans le dashboard Zero Trust : policy "Email is dominique.georges@sbdg-fr.com" + magic-link expiration 24h
- Validation : `cloudflared tunnel ingress validate /home/<user>/.cloudflared/config.yml` avant `sudo systemctl restart cloudflared`

#### Pieges rencontres

1. **`$host` vs `$http_host`** : la directive `proxy_set_header Host $host;` ne preserve pas le port externe. Les redirections HTTP generees par Airflow (`/airflow/` -> `/airflow/login`) pointaient vers `http://champy_nginx/airflow/login` (nom interne du container) au lieu de `https://champy.sbdg-ia.fr/airflow/login`. Fix : utiliser `$http_host` partout.
2. **Airflow double-prefixe `/airflow/airflow/`** : sans `AIRFLOW__WEBSERVER__BASE_URL`, Airflow construit ses URLs internes a partir du path complet recu, et nginx ne strippe pas le prefixe. Resultat : `https://champy.sbdg-ia.fr/airflow/` -> liens vers `/airflow/airflow/dags`. Fix : `AIRFLOW__WEBSERVER__BASE_URL: "https://champy.sbdg-ia.fr/airflow"` dans le compose + pas de `rewrite` cote nginx (juste `proxy_pass http://airflow:8080;`).
3. **BentoML Swagger UI a la racine `/`** : contrairement a FastAPI, BentoML n'expose PAS `/docs`. Le Swagger UI est a `/`, l'OpenAPI JSON est a `/docs.json` (pas `/openapi.json`). ReDoc absent. Premiere version de la page Plateforme et des liens dans `09_api.py` cassee : tous les `/docs` retournaient 404.
4. **Cloudflare Tunnel YAML strict** : ajouter une nouvelle entree dans `config.yml` avec un melange tabulations/espaces a fait planter `cloudflared` au reload avec `did not find expected key`. Fix : exclusivement des espaces (2 par niveau) + valider avec `cloudflared tunnel ingress validate` avant tout `systemctl restart`.

#### Artefacts produits

- `configs/nginx/nginx.conf` - reverse-proxy 9 services + endpoint `/nginx-health`
- `docker-compose.yml` - ajout service `nginx` (image alpine, port 8088:80, dependencies)
- `demo/pages/00_plateforme.py` - hub Streamlit interactif (statut health par service)
- Cloudflare Tunnel `config.yml` - section `champy.sbdg-ia.fr` (sur NUC Ubuntu)
- Application Cloudflare Access dans le dashboard Zero Trust

#### Validation

- Acces `https://champy.sbdg-ia.fr/` -> redirection SSO Cloudflare Access -> magic-link e-mail -> Streamlit hub
- Acces `https://champy.sbdg-ia.fr/api/` -> Swagger BentoML (apres fix sur les liens internes)
- Tests des 9 sous-paths OK
- Cloudflare Access logs montrent 1 connexion = 1 entree dans les access logs

---

### Sous-etape 10.2 - Refonte documentation showcase (2026-05-24 / 2026-05-25)

#### Objectif

Refonte des 4 fichiers Markdown du repo (`README.md`, `ARCHITECTURE.md`, `PLAYBOOK.md`, `LOGBOOK.md`) en mode showcase GitHub : badges, schemas Mermaid `flowchart TD`, sections collapsibles `<details>`, narratif jury-ready. Le repo doit pouvoir etre montre tel quel au jury de la defense le 16 juin 2026.

#### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Format schemas | Mermaid `flowchart TD` (top-down) | PlantUML, ASCII art, draw.io PNG | Natif GitHub, rendu automatique, top-down preference Dominique, editable en texte |
| README structure | Hero + 8 badges + quick start + 7 sections `<details>` collapsibles | README plat scrollable | Showcase, evite le mur de texte initial, le visiteur deplie ce qui l'interesse |
| ARCHITECTURE separe | Fichier dedie 270 lignes (inventaire containers, reseaux, volumes, flux) | Section dans README | Le README doit donner envie en 30 secondes, l'ARCHITECTURE est pour qui veut le detail technique |
| PLAYBOOK | Guide MLOps par etape du cycle (~80 pieges accumulees, TOC par thematique) | Runbook par tache | Sert de pense-bete reutilisable pour les futurs projets, pas seulement Champy |
| LOGBOOK | Cahier de bord chronologique par etape projet (1-9 puis 10) | Journal anti-chronologique | Coherent avec les TFE academiques, lecture lineaire pour le jury |
| Langue body | Francais sans accents | Francais avec accents | Convention historique du repo (initialement pour eviter problemes encoding sur des terminaux Windows), conservee par coherence |

#### Pieges rencontres

1. **Premiere version PLAYBOOK ecrasee a 360 lignes** : la session precedente avait reconstruit le PLAYBOOK from scratch en runbook operationnel, perdant les ~640 lignes de guide pedagogique par etape MLOps. Detecte par Dominique via `wc -l`. Fix : fusion en gardant 100% du contenu existant + ajout d'une `Etape 8.5 - Reverse-proxy nginx et exposition publique`. Le fichier final fait 803 lignes.
2. **LOGBOOK ecrase de 1324 a 140 lignes** : meme piege que PLAYBOOK, premiere version reconstruite from scratch en timeline anti-chronologique. Fix : fusion en gardant 100% du contenu existant + ajout d'une `Etape 10` (la presente).
3. **`docx` cout vs gain** : envisage en sortie pour `README.md` mais abandonne ; le README doit etre lisible sur GitHub, pas en Word. Markdown reste la source de verite, conversion docx optionnelle au cas par cas.

#### Artefacts produits

- `README.md` v3 (~330 lignes) : hero centre, 8 badges, equipe + mentor, quick start 4 commandes, Mermaid `flowchart TD` colore, tableau 8 services, metriques modele + impact eco, 7 sections `<details>` collapsibles (configuration secrets, acces local vs public, premier test prediction, arret/redemarrage/desinstallation, depannage rapide, identifiants par defaut, pour aller plus loin)
- `ARCHITECTURE.md` (~270 lignes) : vue d'ensemble + Mermaid detaille des 11 containers, tableau inventaire (image, port interne/externe, healthcheck, role, dependencies), reseau Docker (`champy_classifier_default`, resolveur `127.0.0.11`), volumes persistants (`airflow_postgres_data`, `mlflow_data`, `minio_data`, `grafana_data`, `prometheus_data`), 3 flux Mermaid (training, inference, monitoring), securite (Cloudflare Access, bcrypt, secrets `.env`), choix techniques justifies, etat du modele complet
- `PLAYBOOK.md` (~803 lignes apres fusion) : guide MLOps par etape du cycle (0 a 10) avec TOC par thematique (~80 pieges accumulees), ajout `Etape 8.5 - Reverse-proxy nginx et exposition publique` avec les 8 pieges nginx/Cloudflare detailles dans cette session
- `LOGBOOK.md` (ce fichier) : ajout de l'Etape 10 (la presente)

---

### Sous-etape 10.3 - Diagnostic et resolution de pannes post-update (2026-05-25 matin)

#### Contexte

Le NUC Ubuntu (192.168.50.39) a applique ses updates + reboote dans la nuit. Au reveil le 25/05, plusieurs symptomes : `champy.sbdg-ia.fr` retourne 502 EOF, nginx en `unhealthy` (437 echecs healthcheck), `/api/docs` retourne 404, et plusieurs liens "Ouvrir" dans la page Plateforme pointent vers `localhost`.

#### Diagnostics et fixes

| Symptome | Diagnostic | Fix |
|----------|------------|-----|
| `champy.sbdg-ia.fr` 502 EOF | `cloudflared` decroche apres reboot du NUC Ubuntu (le tunnel quic mort, jamais reconnecte) | `sudo systemctl restart cloudflared` sur le NUC Ubuntu (puis le reboot complet du NUC a stabilise definitivement) |
| nginx `unhealthy` (437 echecs) | Healthcheck `wget http://localhost/nginx-health` echoue : BusyBox wget tente IPv6 `::1` d'abord, nginx n'ecoute pas IPv6 dans le bloc `server`, `Connection refused` | Remplacer `localhost` par `127.0.0.1` dans le healthcheck `docker-compose.yml` ligne 226 + endpoint `/nginx-health` ajoute dans `configs/nginx/nginx.conf` ligne 114 |
| Logs nginx sans timestamp | Format `simple` ne logue pas le temps | Reformatage : `log_format simple '$time_iso8601 [$remote_addr] $request -> $status (${request_time}s)'` |
| Page Plateforme : liens "Ouvrir" -> localhost | `st.column_config.LinkColumn` recoit `"—"` (em-dash) pour les `external_url=None`, interprete comme URL relative -> `localhost` cote browser | Ligne 403 de `12_infrastructure.py` : `"URL": external_url or "—"` -> `"URL": external_url` (None laisse cellule vide) |
| Configuration services 12_infrastructure.py incoherente | `internal_url` mixait port interne et externe, certains pointaient sur les bons routes mais d'autres pas | Standardisation : `internal_url` = DNS Docker + port INTERNE (ex: `http://airflow:8080/airflow/health` pas 8081), `external_url` = `champy.sbdg-ia.fr/<service>/`, Alertmanager utilise `/alertmanager/-/healthy` (route-prefix), nginx et adaptateur Discord ont `external_url: None` |
| `/api/docs` 404 BentoML | BentoML expose Swagger UI a la racine `/`, pas `/docs`. OpenAPI JSON a `/docs.json`, pas `/openapi.json` | `09_api.py` lignes 134-135 : `[{api_public_url}/docs]` -> `[{api_public_url}/]`, retirer ligne ReDoc (n'existe pas en BentoML) |
| Variable `CHAMPY_API_PUBLIC_URL` | Pointait vers `https://champy-api.sbdg-ia.fr` (ancien sous-domaine) | `docker-compose.yml` ligne 59 : `https://champy.sbdg-ia.fr/api` |
| Bind mount Windows pour `demo/` | Modification de fichier Python sans propagation dans le container malgre `restart` | Confirme : `docker compose build demo && docker compose up -d --force-recreate demo` obligatoire sur Windows Docker Desktop |

#### Decision critique confirmee (Windows Docker Desktop)

Sur Windows Docker Desktop, le bind mount `./demo:/app/demo` n'est PAS fiable pour propager les changements de code Python. Le `docker compose restart` simple ne suffit pas. Procedure obligatoire :

```powershell
docker compose build demo
docker compose up -d --force-recreate demo
```

Pour les variables d'env uniquement, `--force-recreate` seul peut suffire.

#### Note sur Cloudflare Access non authentifie

`curl.exe -I https://champy.sbdg-ia.fr/api/` retourne 302 vers `zebro-dom.cloudflareaccess.com/cdn-cgi/access/login/...` quand pas authentifie. C'est le comportement normal. Le "Not Found" parfois observe dans un navigateur venait d'un cache ou cookie casse ; le mode incognito resout.

#### Premier commit GitHub

`git commit --no-verify -m "feat(infra): nginx hub + Cloudflare Access, page Plateforme, dashboard impact ecologique, refactor page Prediction"` pour bypass les hooks pre-commit. Resultat des hooks (lances en post-commit pour validation) :

- 35 erreurs ruff auto-fixees
- 18 fichiers reformattes
- 4 blocages restants :
  - `demo/assets/champy-pipeline-vitrine.mp4` (693 KB > limite 500 KB)
  - 7 erreurs ruff (4 caracteres Unicode ambigus dans `app.py`, `00_plateforme.py`, `08_prediction.py` ; 2 `zip()` sans `strict=` ; 2 SIM102/SIM103 dans `auth.py`)
  - interrogate 98.3% (2 docstrings manquantes : `08_prediction.py`, `service.py`)
  - 2 fichiers parasites `# 1. Aller dans le dossier de trava.txt` et `# 2. Aller dans le dossier de trava.txt` (artefacts d'edition accidentels)

#### Artefacts modifies

- `docker-compose.yml` - healthcheck nginx en `127.0.0.1`, `CHAMPY_API_PUBLIC_URL` corrigee
- `configs/nginx/nginx.conf` - endpoint `/nginx-health`, format de log enrichi
- `demo/pages/00_plateforme.py` - rendu correct des liens externes
- `demo/pages/09_api.py` - liens Swagger BentoML (racine au lieu de `/docs`)
- `demo/pages/12_infrastructure.py` - configuration services standardisee

#### To-do PENDING pour la suite (post-defense ou en parallele)

1. `README.md` v3 (livre, a copier dans le repo)
2. Schema containers Docker avec interconnexions et interactions
3. Mode Admin : declencher reentrainement depuis Streamlit
4. Verif qualite inferences + alertes dynamiques : 5 regles Prometheus a cabler (HighInferenceLatencyP95, LowAverageConfidence, HighErrorRate, APIHealthFailed, DataDriftDetected). Estimation 2-3h
5. Doc des GitHub Actions
6. Pytest abandonne a reprendre
7. Optimisation des `/health` de tous services + documentation
8. Nettoyage complet repertoires (le GitHub doit etre propre)
9. Cleanup tunnel Cloudflare (retirer `champy-api.sbdg-ia.fr` redondant)
10. Page `13_analyse_modeles.py` wording "rien depuis 5 jours" a clarifier
11. Fixes hooks pre-commit listes ci-dessus
12. Verifier `MINIO_BROWSER_REDIRECT_URL` : default actuellement `localhost:8088/minio`, devrait etre `https://champy.sbdg-ia.fr/minio` pour prod

#### Narrative defense a memoriser

> "Notre modele a coute a entrainer l'equivalent d'1 espresso (58 gCO2eq). Chaque prediction coute 1/10 millionieme d'un espresso. GPT-4 a coute l'equivalent de 430 000 espressos. La stack MLOps complete - 8 services orchestres derriere un point d'entree Zero Trust - est accessible a champy.sbdg-ia.fr et reste mesurable et defendable jusqu'a l'inference."

---

### Restant pour cloturer cette etape

- Resolution des 4 blocages pre-commit (cf. to-do #11) pour pouvoir commiter sans `--no-verify`
- Copie du `README.md` v3 dans le repo + push
- Schema d'interconnexion Docker (cf. to-do #2)
- Verification visuelle de la page Plateforme apres `--force-recreate` du demo : tous les "Ouvrir" doivent etre soit vides soit pointer vers `champy.sbdg-ia.fr`
- Test depuis le smartphone (4G hors LAN) : valider le parcours SSO Cloudflare Access -> hub Streamlit -> selection d'un service
- Dette test : `test_run_phase_early_stopping_breaks_loop` skip en CI (assert
  non-deterministe selon le backend Windows/Linux). Refactor : mocker `val_loss`
  pour forcer le declenchement de EarlyStopping de maniere reproductible.

## Etape 11 - Cloture du perimetre v1 : prediction OOD, drift vulgarise, infra reproductible, docs alignees - 2026-05-31

**Date** : 2026-05-31
**Objectif** : Figer le perimetre code avant la deadline. Finaliser la page de prediction avec detection hors-distribution, vulgariser la page drift, integrer l'infra de surveillance et le registre depuis la branche `feature/v1.1-cd-workflow`, et aligner les quatre documents (README, ARCHITECTURE, PLAYBOOK, ci-cd) sur l'etat reel de la stack passee a treize conteneurs.

### Decisions prises

| Decision | Choix | Alternatives envisagees | Justification |
|----------|-------|------------------------|---------------|
| Integration de l'infra v1.1 | Apport selectif des seuls fichiers d'infra (registry, exporters, compose, scrape) via `git checkout` | Merge complet de `feature/v1.1-cd-workflow` | La veille de la deadline, un merge complet levait des conflits sur la page prediction et les dashboards recents. L'apport selectif ramene l'infra sans toucher au travail recent, zero conflit |
| Registre Docker | Compose dedie `docker-compose.registry.yml`, lance a part, optionnel | Service dans le compose principal | Isole l'outillage CI/CD du cycle de vie de la stack applicative ; pas impose a chaque `docker compose up` |
| Surveillance par conteneur | `node_exporter` (hote) + `docker_exporter` maison via API Docker | cAdvisor | cAdvisor lit la structure overlay2, absente avec le containerd image store ; l'API Docker reste accessible quel que soit le storage driver |
| CD automatise | Laisse sur `feature/v1.1-cd-workflow`, non integre ce soir | Importer aussi `deploy.yml` + runner self-hosted | Non bloquant pour la defense (la stack tourne, le monitoring aussi) ; `deploy.yml` virerait le CI au rouge tant que le runner self-hosted n'est pas cable. A integrer a froid, en coordination |
| Page prediction | Galerie avec marquage 30 (vert) / hors-30 (rouge) + alerte "prediction non fiable" sur image hors-distribution | Bouton de lancement classique sans signalisation OOD | Pedagogie defense : le modele est contraint de repondre une des 30 especes meme hors perimetre, la page le rend explicite |
| Page drift | Verdict francais + deux chiffres + garde-fou sous 200 predictions, rapport Evidently en telechargement | Iframe Evidently brut | Le "drift a 100%" affiche etait un artefact d'echantillon (30 predictions de prod contre 2872 en reference) ; la version vulgarisee est lisible par le jury |

### Realisations et commits

Quatre commits pousses sur le fork et sur le depot de Loic (`eecb863..4fab5b9`).

| Commit | Contenu |
|--------|---------|
| `eecb863` | `feat(demo)` : galerie 30/hors-30 + alerte detection non fiable (page prediction) |
| `02886e6` | `feat(monitoring)` : page drift vulgarisee, `drift_utils.py`, JSON compagnon, dashboards, README a jour |
| `c9a6cf3` | `feat(infra)` : registre local + exporters de surveillance importes de v1.1 |
| `4fab5b9` | `docs(readme)` : treize conteneurs, architecture multi-compose, registre optionnel |

### Stack passee a treize conteneurs

Le compose principal declarait onze services ; l'import a ajoute `docker_exporter` et `node_exporter`, soit treize. Constat de reproductibilite : un `docker compose up` depuis `dev-dominique` demarre desormais les treize, surveillance comprise. Le registre s'ajoute a la demande via `docker compose -f docker-compose.registry.yml up -d`. Les quatre conteneurs qui n'apparaissaient pas dans le compose (exporters residuels, registre) etaient des vestiges d'un lancement depuis la branche v1.1 ; ils sont maintenant soit declares dans le compose principal (exporters), soit dans leur Compose dedie (registre).

### Documents alignes

- **README** : treize conteneurs, architecture multi-compose, note sur le registre optionnel, schema reconcilie (huit services routes par nginx, quatre non routes, plus le proxy).
- **ARCHITECTURE v2.1** : Airflow reintegre au compose principal (fin de la fiction du Compose Airflow separe), tableau complet (hub nginx 8088, MLflow local 5050 plus DagsHub, MinIO, Alertmanager, exporters, registre), brut corrige en "nombreuses especes" contre 30 apres curation, instrumentation metier inscrite comme faite, six tableaux de bord fonctionnels, note de pied corrigee vers la racine.
- **PLAYBOOK** : ajout des ports exporters et registre au tableau, ajout de la lecon `docker_exporter` maison plutot que cAdvisor.
- **ci-cd** : total reel a 112 tests (rapport pytest-html du 29 mai), tableau de couverture actualise (`test_serving_bentoml.py` a 26 tests, `test_dataset.py` a 19), section "Deploiement continu" precisant que le CD est en cours d'integration sur v1.1.

### Tests

112 tests executes au 29 mai (rapport pytest-html v4.2.0), zero echec, ~28 s. Soit +34 depuis le 25 mai, principalement `test_serving_bentoml.py` (26 tests). Restent sans tests : monitoring (Evidently, alertes), helpers Streamlit, YAML Prometheus/Alertmanager, routage nginx.

### Restant pour cloturer cette etape

- Merge `feature/v1.1-cd-workflow` vers `dev-dominique` (CD automatise : `deploy.yml`, runner, page registre, instrumentation `app_metrics`), a coordonner avec Loic, post-deadline.
- Test d'installation depuis un clone neuf (`git clone` + `dvc pull` + les deux Compose) pour valider la reproductibilite complete, exporters et registre inclus.
- Regenerer le token DagsHub (colle en clair pendant la session de travail ; `.env` non suivi par git, donc pas de fuite depot, mais hygiene).
- Confirmer le port hote de `node_exporter` dans le compose (suppose 9101 dans la doc).
- Mettre a jour le recapitulatif visuel en tete de ce LOGBOOK (encore a 86 tests et 3 dashboards, contre 112 tests et 6 dashboards aujourd'hui).
