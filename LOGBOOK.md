# Champy Classifier - Cahier de bord MLOps

## Informations generales

- **Projet** : Classification de champignons (30 especes, ~700K images brutes, 19 138 retenues apres curation)
- **Cadre** : TFE Master AI (DataScientest, RNCP niveau 7, promotion 2026)
- **Equipe** : Equipe Champy Classifier (DataScientest promotion 2026)
- **Repo** : DagsHub - LoicFocraud/Champy_Classifier (mirror GitHub LoicFocraud/Champy_Classifier)
- **Branche de developpement** : `dev-dominique` (merge vers `main` par PR equipe)
- **Date de debut** : 2026-03-28
- **Date de soutenance** : [a confirmer]

### Roadmap equipe (reference)

| Etape | Titre | Statut |
|-------|-------|--------|
| 1 | Mise en place environnement | Termine |
| 2 | Analyse du sujet | Termine |
| 3 | Preparation des donnees | Termine |
| 4 | Entrainement | Termine |
| 5 | CI/CD | Termine |
| 6 | Serving (API + Model Registry) | En cours |
| 7 | Docker et Monitoring | En cours |
| 8 | Demo et Tests | En cours |

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
**Objectif** : Portfolio Streamlit 12 pages (zero hardcoded, sources dynamiques) + suite de tests.

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
- `demo/pages/01_*.py` a `12_*.py` - 12 pages portfolio
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
