# Playbook MLOps - Guide de référence

> Référentiel construit au fil du projet Champy Classifier.
> Objectif : servir de pense-bête et de base pour tout futur projet MLOps.
> Chaque étape est enrichie au fur et à mesure, avec les commandes, les pièges, et les "pourquoi".

---

## Etape 0 - Cadrage

**But** : Definir le perimetre, la stack, et la repartition des responsabilites avant d'ecrire une ligne de code.

**A produire** :
- [x] Tableau des choix techniques (techno / alternatives / justification)
- [x] Schema d'architecture (machines, services, flux de donnees)
- [x] CLAUDE.md ou equivalent (gouvernance, invariants, conventions)
- [x] Arborescence cible du repo

**Pieges connus** :
- Prevoir le contexte OS des le depart (Windows vs Linux) : impacte le task runner, les chemins, les line endings, Docker GPU
- Ne pas sous-estimer le temps de cadrage : un CLAUDE.md bien fait evite des dizaines de decisions ad hoc plus tard

**Commandes cles** :
- Aucune (etape de reflexion)

**Duree typique** : 1-2 jours

---

## Etape 1 - Structure du projet et config

**But** : Repo pret a coder - dependances, config, linting, .gitignore, task runner.

**A produire** :
- [x] `pyproject.toml` (dependances, config Ruff, Mypy)
- [x] `src/config.py` (Pydantic Settings, chargement .env + YAML)
- [x] `.env.example` + `.env` (dans .gitignore)
- [x] `.gitignore` complete
- [x] `.gitattributes` (line endings LF)
- [x] `tasks.py` (invoke, cross-platform)
- [x] Structure des repertoires src/, tests/, docker/, configs/

**Pieges connus** :
- `pip freeze` dans pyproject.toml : ne jamais dumper toutes les deps transitives, garder uniquement les deps directes avec version minimum (ex: `torch>=2.2` pas `torch==2.2.1`)
- `requirements.txt` genere par certains outils peut avoir un encodage UTF-16 sous Windows : verifier l'encodage avant de commiter
- Les `__init__.py` manquants dans les sous-packages causent des erreurs d'import silencieuses : les creer des le debut, meme vides
- `.gitignore` minimaliste = risque de commiter des `.env`, `__pycache__/`, ou des modeles de plusieurs centaines de MB : partir d'un template genereux
- `.gitattributes` avec `* text=auto eol=lf` est indispensable en equipe mixte Windows/Linux pour eviter les diffs de line endings
- Pydantic Settings : `env_prefix` doit correspondre exactement au prefixe des variables dans `.env` (sensible a la casse)
- YAML + Pydantic : utiliser `yaml.safe_load()` puis passer le dict au constructeur Pydantic, pas de parsing custom
- `.gitignore` et fichiers `.dvc` : ne JAMAIS exclure les fichiers `data.dvc` / `models.dvc` du git. C'est le principe meme de DVC : les fichiers `.dvc` sont les pointeurs legers qui permettent de retrouver les donnees. Les exclure = perdre le lien vers les donnees versionnees.
- `requirements.txt` sous Windows : certains outils d'ecriture generent du UTF-16 silencieusement. Verifier avec `file requirements.txt` ou ouvrir en hex editor. pip ne parse pas l'UTF-16.
- Penser a `httpx` et `plotly` des le depart si le projet inclut un Streamlit qui appelle une API et affiche des graphiques : ce sont des deps de premiere classe, pas des extras
- Installer `pre-commit` + `interrogate` des l'etape 1, pas en rattrapage. Chaque fichier cree sans docstring devra etre repris plus tard. Le cout de la retro-documentation est 3x plus eleve que de documenter au fil de l'eau.
- Archiver les notebooks legacy dans un sous-dossier (`notebooks/legacy/`) plutot que les supprimer : ils contiennent les decisions implicites du projet initial et sont utiles pour l'audit.

**Commandes cles** :
```powershell
pip install -e ".[dev]"    # install deps + dev tools
invoke setup               # ou pip install -r requirements.txt && dvc pull
```

**Duree typique** : 0.5-1 jour

---

## Etape 2 - Data pipeline

**But** : Données versionnées, split reproductible, DataLoader prêt.

**A produire** :
- [ ] DVC configuré et fonctionnel (`dvc pull` ramène les données)
- [ ] Script de split stratifié (train/val/test)
- [ ] Dataset PyTorch (avec transforms)
- [ ] DataLoader factory (configurable via YAML)
- [ ] Vérification : distribution des classes, nombre d'images par split

**Pieges connus** :
- Les donnees brutes (raw) et les donnees nettoyees (processed) peuvent avoir des ordres de grandeur tres differents (646K vs 25K). Toujours verifier ce qu'on a reellement avant de planifier batch sizes et temps d'entrainement.
- Ne jamais supprimer ou modifier des fichiers partages via DVC. Utiliser une liste d'exclusion (excluded.json) que les scripts consomment. Cela rend le nettoyage tracable, reversible, et evite les conflits DVC entre coequipiers.
- Les dimensions d'images dans un dataset scrape sont rarement uniformes. Prevoir Resize + CenterCrop dans le pipeline de transforms, pas seulement Normalize.
- La detection de doublons par hash MD5 partiel (debut + fin + taille) est un bon compromis vitesse/fiabilite pour des images identiques. Pour des near-duplicates (recadrages, recompressions), il faudrait du perceptual hashing (hors scope ici).
- Generer des rapports JSON (raw_stats.json, cleaning_report.json) des cette etape : le Streamlit et les tests les consommeront plus tard sans recalcul.
- Le split doit produire un fichier de mapping (CSV ou JSON avec chemins relatifs), PAS des copies d'images. Copier 25K+ images dans train/val/test triple l'espace disque et complique DVC. Le Dataset PyTorch peut lire les images depuis processed/ via le manifest.
- Toujours verifier la stratification apres le split : le ratio par classe doit etre stable (+/- 0.5%). Un desequilibre cache peut biaiser l'evaluation.
- Si le dataset source contient des augmentations pre-calculees (ex: transforms TF non seedees), les exclure et laisser le framework de training gerer ses propres augmentations. Les augmentations non reproductibles violent le principe de reproductibilite.
- Un dataset naturellement desequilibre (ratio 61x entre la classe la plus grande et la plus petite) necessite un mecanisme d'equilibrage au training (WeightedRandomSampler, class weights, ou over-sampling en ligne), pas un resampling statique des donnees.
- Privilegier un pipeline de curation reproductible (CSV + filtre + dedup) plutot que de dependre d'un filtre par modele tiers (ex: ResNet50 ImageNet). Un tel filtre est non deterministe (le modele telechargeable peut changer) et inegal selon les classes.
- Quand un dataset communautaire a un champ de confiance (ex: GBIF confidence), l'utiliser comme filtre de qualite au lieu d'inventer un filtre ad hoc.
- Attention aux conflits de labels : une meme image peut etre assignee a 2 especes differentes dans les donnees communautaires. Toujours verifier et retirer les conflits.
- Un filtre de qualite visuel est necessaire pour les datasets scrapes, meme quand les labels sont corrects. Les observateurs community uploadent parfois des images hors-sujet (interieur, personnes, cuisine, textes, graphes) sur les fiches d'especes. OpenCLIP (ViT-B-32, ~150 MB, 37 img/s sur CPU) fait tres bien le tri : score = max(prompts_positifs) - max(prompts_negatifs), seuil calibre visuellement sur un echantillon stratifie (top/bottom/borderline panels). Pour 20K images, compter ~10 min sur CPU moderne. Le seuil doit etre calibre empiriquement (pas a priori) : la distribution reelle des scores depend du dataset et des prompts.
- Appliquer le filtre de qualite APRES la curation par CSV (GBIF confidence + dedup + conflits), pas avant. L'ordre : sources brutes -> labels propres -> images propres. Produire un manifest filtre separe (curated_manifest_filtered.csv) et faire en sorte que data_split.py prefere ce manifest s'il existe, sinon retombe sur curated_manifest.csv. Non destructif, reversible.
- Garder excluded.json avec les scores pour chaque image exclue (score + raison + modele), pour pouvoir auditer les decisions plus tard, affiner le seuil, ou detecter des derives de distribution entre runs.
- Le Dataset PyTorch doit exposer un attribut `targets` (tensor d'indices de classes) pour que le WeightedRandomSampler puisse calculer ses poids. Sans ca, il faut un second parcours des donnees.
- Toujours construire le label_map depuis le split train et le partager avec val/test. Si chaque split construit le sien, les indices peuvent ne pas correspondre.
- Sur Windows, `num_workers=0` par defaut dans le DataLoader (multiprocessing fork non supporte). Tester `num_workers=2` avec `persistent_workers=True` si le chargement est un goulot.
- `pin_memory=True` ameliore le transfert CPU->GPU, mais genere un warning si aucun GPU n'est disponible. Acceptable (pas bloquant).

**Commandes cles** :
```powershell
dvc pull
python -c "from data.scan import scan_processed; scan_processed()"  # ou script ad hoc
invoke split-data
```

**Duree typique** : 1-2 jours

---

## Etape 3 - Training pipeline

**But** : Entraînement reproductible, traçé dans MLflow, avec early stopping.

**A produire** :
- [x] Boucle d'entrainement (AMP si GPU)
- [x] Early stopping + checkpointing
- [x] MLflow logging (params, metriques par epoch, artefacts)
- [x] Config YAML externalisee (lr, batch_size, epochs, etc.)
- [x] Seed fixe pour reproductibilite

**Pieges connus** :
- Le decorateur `@torch.no_grad()` rend les fonctions "untyped" pour mypy quand torch n'est pas installe dans l'env mypy (cas mirrors-mypy de pre-commit). Utiliser `with torch.no_grad():` dans le corps a la place.
- Les `type: ignore` pour les generiques PyTorch (DataLoader[X], Dataset[X]) ne sont pas necessaires quand mypy ne connait pas torch (il les traite comme Any). Les laisser provoque des erreurs `unused-ignore`.
- `torch.backends.cudnn.benchmark = True` accelere les convolutions a taille fixe, mais `deterministic = True` est prioritaire pour la reproductibilite. Ne pas activer benchmark en mode reproductible.
- Sur Windows, AMP (mixed precision) fonctionne nativement avec CUDA sans configuration supplementaire. Par contre, `GradScaler` doit recevoir `device=device.type` et non `device="cuda"` sinon ca plante sur CPU.
- Gradient accumulation : diviser la loss par `accumulation_steps` a chaque forward, et ne faire `optimizer.step()` que toutes les N iterations. Oublier la division = loss trop grande = divergence.
- `loguru` est plus simple que le module `logging` stdlib mais n'est pas installe par defaut. L'ajouter dans requirements.txt et pyproject.toml.
- `torch.cuda.get_device_properties(0).total_mem` n'existe PAS. L'attribut correct est `total_memory`. Erreur subtile qui ne plante que sur GPU.
- MLflow sur DagsHub necessite `MLFLOW_TRACKING_USERNAME` et `MLFLOW_TRACKING_PASSWORD` dans le .env, pas juste l'URI. Sans ca, erreur 401 silencieuse.
- Batch 16 + AMP tient en 4 GB VRAM sur RTX 3050 Ti pour ResNet50 (~93s/epoch sur 20K images). Pas besoin de gradient accumulation pour ce modele.
- Les classes a moins de ~15 images dans le split test donnent des metriques F1 instables. Interpreter avec prudence (Russula vesca : 9 images test -> F1 oscille entre 0% et 40% selon le run).
- Les especes visuellement similaires du meme genre (ex: 7 Russules) sont les plus dures a separer. C'est un probleme de fine-grained classification, pas de pipeline.

**Commandes cles** :
```powershell
python -m src.training.train --config configs/training/default.yaml
invoke train --config configs/training/default.yaml
```

**Duree typique** : 2-3 jours (code) + 12-15h d'entrainement (XPS, RTX 3050 Ti)

---

## Etape 4 - Model registry et export

**But** : Modèle versionné, promu, exporté en format optimisé pour la prod.

**A produire** :
- [x] Export ONNX avec validation
- [x] Comparaison numerique PyTorch vs ONNX
- [x] Script d'export automatise (CLI)
- [ ] Enregistrement dans MLflow Model Registry
- [ ] Promotion Staging -> Production

**Pieges connus** :
- Le nouveau dynamo exporter de torch >= 2.9 est le defaut mais peut produire des fichiers quasi vides lors de la conversion d'opset. Toujours verifier la taille du fichier ONNX (ResNet50 = ~90 MB, pas 240 KB). Utiliser `dynamo=False` si probleme.
- `onnxscript` est requis par torch >= 2.9 pour l'export ONNX. L'ajouter dans les deps meme si on utilise le legacy exporter.
- Toujours comparer les sorties numeriques (pas juste onnx.checker) : generer N inputs aleatoires, comparer les logits PyTorch vs ONNX, max_diff < 1e-4.
- Sauvegarder class_names.json a cote du modele ONNX : l'API en a besoin pour mapper les indices de sortie vers les noms d'especes.
- Les axes dynamiques (batch_size) evitent de devoir re-exporter si on change la taille du batch d'inference.

**Commandes cles** :
```powershell
python -m src.models.export_onnx
python -m src.models.export_onnx --checkpoint models/best_model.pt --output models/best_model.onnx
```

**Duree typique** : 0.5 jour (script + validation + debug dynamo)

---

## Etape 5 - API serving

**But** : API REST prête pour l'inference, avec métriques et health checks.

**A produire** :
- [ ] FastAPI avec endpoint /predict (top-N + scores)
- [ ] /health, /metrics (Prometheus), /model/info
- [ ] Pydantic schemas (request/response)
- [ ] Chargement ONNX Runtime (CPU)
- [ ] Préprocessing identique au training (transforms)
- [ ] Tests unitaires de l'API

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke serve
# Test rapide :
# Invoke-RestMethod -Uri http://localhost:8000/health
```

**Durée typique** : 1-2 jours

---

## Etape 6 - Demo (Streamlit)

**But** : Interface de démonstration visuelle du pipeline complet.

**A produire** :
- [ ] Page prédiction (upload + top-5 + GradCAM)
- [ ] Page exploration dataset (galerie, stats)
- [ ] Page métriques modèle (depuis MLflow)
- [ ] Page monitoring (métriques live)

**Pieges connus** :
- Le Streamlit est un portfolio narratif, pas un outil de prod. Il consomme MLflow et Prometheus, il ne les remplace pas.
- Principe zero hardcoded : ne jamais ecrire de valeur en dur (accuracy, nb images, noms de classes). Tout lire dynamiquement depuis les fichiers JSON, MLflow, ou l'API. Si la source n'est pas dispo, afficher un `st.warning()`.
- Les pages Streamlit se construisent incrementalement au fil des etapes. Ne pas attendre la fin du projet pour tout creer d'un coup.
- Factoriser le code d'acces aux sources dans `demo/lib/` (data_utils, mlflow_utils, api_utils). Evite la duplication entre pages.
- Les imports lourds (torch, plotly, pandas) doivent etre faits dans le try/except pour ne pas planter la page si le module manque.
- `use_container_width=True` sur les images et graphiques pour un rendu responsive.

**Commandes cles** :
```powershell
streamlit run demo/app.py    # lancement local
invoke serve                 # inclut le container Streamlit
```

**Duree typique** : 2-3 jours (incrementalement)

---

## Etape 7 - Monitoring

**But** : Visualiser la santé du modèle en production.

**A produire** :
- [ ] Prometheus : scrape /metrics de l'API
- [ ] Grafana : dashboard pré-configuré (latence, throughput, distribution classes)
- [ ] Drift detection (Evidently) : rapport périodique ou on-demand
- [ ] Alertes (optionnel) : seuils sur latence, confiance, erreurs

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke serve   # Prometheus + Grafana inclus
# Grafana : http://localhost:3000
# Prometheus : http://localhost:9090
```

**Durée typique** : 1-2 jours

---

## Etape 8 - Dockerisation

**But** : Tout tourne en containers, reproductible sur n'importe quelle machine.

**A produire** :
- [ ] Dockerfile.api (Python slim + ONNX Runtime)
- [ ] Dockerfile.demo (Streamlit)
- [ ] Dockerfile.train (PyTorch + CUDA, optionnel)
- [ ] docker-compose.yml (orchestration complète)
- [ ] docker-compose.dev.yml (override dev)
- [ ] .dockerignore

**Pièges connus** :
- **Cohabitation sur hôte partagé** : sur une machine qui héberge plusieurs projets (plusieurs docker-compose en parallèle), les ports standards (8000, 3000, 9090, etc.) entrent rapidement en collision. Avant tout `docker compose up`, auditer les ports occupés :
  ```powershell
  Get-NetTCPConnection -State Listen | Where-Object {$_.LocalPort -lt 10000} `
    | Select-Object -ExpandProperty LocalPort | Sort-Object -Unique
  ```
  Appliquer un **offset projet explicite** (ex: +10) sur chaque port déjà occupé par un autre service. Exemple Champy : API `8010:8000` (8000 pris), Grafana `3010:3000` (3000 pris), Streamlit `8501:8501` et Prometheus `9090:9090` (libres, pas de remap). Documenter le mapping en commentaire YAML en tête de `docker-compose.yml` pour que le rationnel survive au temps.
- **Port host vs port interne du container** : un remap `8010:8000` ne change que le port host. Les containers d'un même compose se parlent par nom de service sur le port interne (ex: Prometheus scrape `api:8000`, pas `host.docker.internal:8010`). Utiliser `host.docker.internal` uniquement si un service docker doit atteindre un process tournant nativement sur le host (rare, et casse la portabilité Linux/Mac).
- **Zéro hardcoded d'URL dans la demo** : Streamlit ne doit jamais écrire `http://localhost:8000` en dur. Passer par une env var (ex: `CHAMPY_API_URL`, `CHAMPY_PROMETHEUS_URL`, `CHAMPY_GRAFANA_URL`) avec un défaut raisonnable, exposée par un helper `get_api_url()` / `get_prometheus_url()` / `get_grafana_url()` dans `demo/lib/`. Sinon, dès qu'on change le mapping des ports, la demo affiche des liens morts.
- **Nommer les env vars avec un préfixe projet** (`CHAMPY_*`) plutôt que générique (`API_URL`, `PROMETHEUS_URL`) : sur un hôte partagé, les env vars sans préfixe se télescopent entre projets et on passe 30 min à comprendre pourquoi Streamlit tape sur la mauvaise API.

**Commandes clés** :
```powershell
invoke build
invoke serve
docker compose ps
```

**Durée typique** : 1-2 jours

---

## Etape 9 - CI/CD

**But** : Automatiser lint, tests, build sur chaque push.

**A produire** :
- [ ] `.github/workflows/ci.yml` (lint + test + build images)
- [ ] `.github/workflows/cd.yml` (deploy, optionnel)
- [ ] Badge CI dans README.md
- [ ] Branch protection sur `main`

**Pièges connus** :
-

**Commandes clés** :
- Push sur GitHub/DagsHub -> pipeline automatique

**Durée typique** : 0.5-1 jour

---

## Etape 10 - Tests et couverture

**But** : Coverage > 80%, tests unitaires + intégration.

**A produire** :
- [ ] Tests unitaires (src/data, src/models, src/serving)
- [ ] Tests intégration (API end-to-end, pipeline data)
- [ ] conftest.py (fixtures partagées)
- [ ] Rapport de couverture HTML

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke test
invoke test-unit
invoke test-integration
```

**Durée typique** : 2-3 jours (en parallèle avec le code)

---

## Aide-mémoire transversal

### Commandes invoke les plus utilisées
```powershell
invoke --list          # Toutes les commandes disponibles
invoke setup           # Setup initial
invoke train           # Entraînement
invoke serve           # Lancer tous les services
invoke stop            # Arrêter les services
invoke test            # Tests + coverage
invoke lint            # Vérification qualité
invoke status          # Etat Docker + DVC + Git
invoke clean           # Nettoyage
```

### Ports par défaut
| Service | Port |
|---------|------|
| FastAPI | 8000 |
| Streamlit | 8501 |
| Prometheus | 9090 |
| Grafana | 3000 |
| MLflow (si local) | 5000 |

### Checklist "avant de commit"
- [ ] `invoke lint` passe
- [ ] `invoke test` passe
- [ ] Pas de secrets dans le code
- [ ] LOGBOOK.md à jour
- [ ] PLAYBOOK.md enrichi si nouvelle leçon apprise
