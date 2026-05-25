# Playbook MLOps - Guide de référence

> Référentiel construit au fil du projet Champy Classifier.
> Objectif : servir de pense-bête et de base pour tout futur projet MLOps.
> Chaque étape est enrichie au fur et à mesure, avec les commandes, les pièges, et les "pourquoi".

---

## Table des matières par thématique

> Le playbook est organisé chronologiquement par étape MLOps (cadrage,
> data, training, ...). Cette TOC permet de naviguer **par sujet** entre
> les ~80 pièges accumulés. Chaque entrée pointe vers la section qui
> contient le piège (Ctrl+F sur le mot-clé pour le retrouver précisément).

### Git, DVC et données partagées
- [Fichiers `.dvc` doivent rester dans git](#etape-1---structure-du-projet-et-config) (les exclure casse le lien vers les données versionnées)
- [`requirements.txt` UTF-16 silencieux sur Windows](#etape-1---structure-du-projet-et-config) (pip ne parse pas l'UTF-16)
- [Conflits `git stash` après `git pull` sur artefacts générés](#etape-2---data-pipeline) (manifests CSV, reports JSON regénérés par d'autres)
- [`git pull && dvc pull` comme séquence atomique](#etape-2---data-pipeline) (les `.dvc` suivent git, les données pas)
- [Token DagsHub unique pour MLflow + DVC + Git + AWS_SECRET_ACCESS_KEY](#etape-1---structure-du-projet-et-config)

### Windows / PowerShell
- [`.env` fragile sur reboot Windows](#etape-1---structure-du-projet-et-config) (considérer comme reconstructible)
- [Path translation Git Bash dans `docker exec`](#etape-7---monitoring) (`MSYS_NO_PATHCONV=1` ou PowerShell direct)
- [`num_workers=0` par défaut sur Windows](#etape-2---data-pipeline) (multiprocessing fork non supporté)
- [`pin_memory=True` warning sans GPU](#etape-2---data-pipeline) (acceptable, pas bloquant)
- [Cohabitation hôte partagé : audit ports + offset +10](#etape-8---dockerisation)
- [Préfixer env vars avec `CHAMPY_*`](#etape-8---dockerisation) (sinon télescope avec autres projets)
- [Bind mount Windows Docker Desktop non fiable pour le code Python live](#etape-85---reverse-proxy-nginx-et-exposition-publique) (rebuild + force-recreate obligatoire)

### MLflow
- [MLflow 401/403 silencieux si env vars pas exportées](#etape-3---training-pipeline)
- [Token DagsHub périssable](#etape-3---training-pipeline) (révoquer/régénérer côté UI)
- [`load_dotenv` en début de script avant `import mlflow`](#etape-3---training-pipeline)

### PyTorch / Training
- [`@torch.no_grad()` casse mypy pre-commit](#etape-3---training-pipeline) (utiliser `with torch.no_grad():`)
- [`type: ignore` sur generics PyTorch inutiles avec mirrors-mypy](#etape-3---training-pipeline)
- [`torch.cuda.get_device_properties(0).total_memory`, pas `total_mem`](#etape-3---training-pipeline)
- [Gradient accumulation : diviser la loss par `accumulation_steps`](#etape-3---training-pipeline)
- [Classes < 15 images test = F1 instable](#etape-3---training-pipeline)
- [Espèces visuellement similaires (7 Russules) = plafond F1](#etape-3---training-pipeline)

### ONNX export
- [Dynamo exporter torch >=2.9 produit fichier vide](#etape-4---model-registry-et-export) (forcer `dynamo=False`)
- [`onnxscript` requis même en mode legacy](#etape-4---model-registry-et-export)
- [Comparer sorties numériques (max_diff < 1e-4)](#etape-4---model-registry-et-export) (pas juste `onnx.checker`)
- [Export agnostique de l'architecture : auto-detection state_dict](#etape-4---model-registry-et-export)
- [`.onnx.data` orphelin entre exports successifs](#etape-4---model-registry-et-export)
- [Transfert checkpoint XPS -> NUC3 via `python -m http.server`](#etape-4---model-registry-et-export)
- [Taille `.pt` (334 MB) != taille déployée (110 MB ONNX)](#etape-4---model-registry-et-export)

### BentoML 1.4
- [`bentoml.onnx` deprecated en 1.4](#etape-5---api-serving) (mais fonctionnel, plan migration `bentoml.models.create()`)
- [`@bentoml.api` force POST](#etape-5---api-serving) (pas de paramètre `method=`)
- [Appel intra-service async-only](#etape-5---api-serving) (`predict` -> `infer_batch` via proxy RPC)
- [Sérialisation float64 silencieuse via le proxy](#etape-5---api-serving) (cast `dtype=np.float32` requis)
- [`PIL.Image.Image` doit rester import runtime](#etape-5---api-serving) (`noqa: TC002`)
- [`ModelOptions` n'est pas un dict](#etape-5---api-serving) (glob `saved_model.onnx` à la place)
- [Schema `python.version` n'existe pas en 1.4](#etape-5---api-serving) (utiliser `docker.python_version`)
- [Modèle non auto-détecté au build sans `models: [tag]`](#etape-5---api-serving) (Model Size = 0)
- [Conflit `anyio` / `httpx-ws`](#etape-5---api-serving) (pin `anyio>=4.7`)
- [Query params HTTP non mappés automatiquement](#etape-5---api-serving) (passer en body JSON)
- [`bentoml serve <module>` (dev) vs `bentoml serve <bento_tag>` (prod)](#etape-5---api-serving)
- [Swagger UI à la racine `/`, OpenAPI JSON à `/docs.json`](#etape-85---reverse-proxy-nginx-et-exposition-publique) (pas `/docs` ni `/openapi.json`)

### SQLite + async (PredictionStore)
- [`PRAGMA journal_mode=WAL` obligatoire pour la concurrence](#etape-7---monitoring)
- [`PRAGMA busy_timeout=5000` anti-`database is locked`](#etape-7---monitoring)
- [`PRAGMA synchronous=NORMAL` recommandé pour WAL](#etape-7---monitoring) (~3x plus rapide que FULL)
- [`row_factory` après `connect()` en aiosqlite](#etape-7---monitoring)
- [Sidecars `.db-wal` et `.db-shm` à gitignore](#etape-7---monitoring)
- [Partager une connexion aiosqlite entre coroutines est sûr](#etape-7---monitoring) (thread interne sérialise)
- [Init async + `__init__` sync](#etape-7---monitoring) (lazy init + `asyncio.Lock`)
- [Fire-and-forget `asyncio.create_task` exige référence forte](#etape-7---monitoring) (sinon GC → RUF006)
- [Mount Docker dossier > fichier](#etape-7---monitoring) (sidecars + fichier inexistant)
- [Hash image via `image.tobytes()` après `convert("RGB")`](#etape-7---monitoring)
- [SQLite WAL ~10k req/s, PostgreSQL au-delà de 1M lignes](#etape-7---monitoring)

### Evidently 0.7+ (drift detection)
- [API moderne `Dataset.from_pandas` + `Report([Preset()])`](#etape-7---monitoring) (≠ legacy 0.4 des tutos)
- [`DataDefinition` obligatoire pour cat/num](#etape-7---monitoring)
- [`save_html` sur `Snapshot` (retour de `.run()`)](#etape-7---monitoring) (pas sur `Report`)
- [HTML self-contained 3-4 MB](#etape-7---monitoring) (gitignore les rapports)
- [`DataDriftPreset` chi-2 + KS par défaut](#etape-7---monitoring) (PSI / Wasserstein possibles)
- [Materialiser baseline depuis aggregats perd la variance](#etape-7---monitoring)
- [Index rglob pré-construit sur dataset scrape](#etape-7---monitoring) (1.6 → 40 img/s)
- [`subprocess.run` avec `sys.executable`](#etape-7---monitoring) (pas `python` sur Windows)
- [`st.components.v1.html(html, height=H, scrolling=True)`](#etape-7---monitoring) (height obligatoire)

### Docker / Compose / Grafana
- [Cohabitation hôte partagé + offset +10 sur ports occupés](#etape-8---dockerisation)
- [Port host vs port interne du container](#etape-8---dockerisation) (containers se parlent sur port interne)
- [`docker compose restart` n'applique PAS les nouveaux volumes / env vars](#etape-7---monitoring) (`up -d` requis)
- [Volume nommé `grafana-data` persiste les datasources manuelles](#etape-7---monitoring)
- [`uid` explicite obligatoire dans datasource yaml](#etape-7---monitoring)
- [`folderUid` recommandé pour stabilité des liens](#etape-7---monitoring)
- [`schemaVersion >= 36` requis pour Grafana 10+](#etape-7---monitoring)
- [Datasource ref doit être `{type, uid}` objet, pas string](#etape-7---monitoring)
- [`prometheus_client` expose `process_*` / `python_*` automatiquement](#etape-7---monitoring) (pas besoin cAdvisor)
- [Privilégier métriques applicatives custom (`champy_*`)](#etape-7---monitoring) sur natives (`bentoml_service_*`)

### Reverse-proxy nginx et exposition publique (Cloudflare Tunnel + Access)
- [nginx cache les IPs des backends et casse après `--force-recreate`](#etape-85---reverse-proxy-nginx-et-exposition-publique) (resolver 127.0.0.11 ou restart nginx)
- [`$host` vs `$http_host` dans `proxy_set_header`](#etape-85---reverse-proxy-nginx-et-exposition-publique) (utiliser `$http_host` pour préserver les redirections externes)
- [BentoML : Swagger UI à `/`, pas `/docs`](#etape-85---reverse-proxy-nginx-et-exposition-publique) (et OpenAPI à `/docs.json`, pas `/openapi.json`)
- [Airflow double-préfixe `/airflow/airflow/`](#etape-85---reverse-proxy-nginx-et-exposition-publique) (`AIRFLOW__WEBSERVER__BASE_URL` + pas de réécriture nginx)
- [Bind mount Windows non fiable pour code Python live](#etape-85---reverse-proxy-nginx-et-exposition-publique) (build + force-recreate obligatoire)
- [Streamlit `runOnSave` ne fonctionne pas sur Windows Docker Desktop](#etape-85---reverse-proxy-nginx-et-exposition-publique)
- [Cloudflare Tunnel YAML strict sur l'indentation](#etape-85---reverse-proxy-nginx-et-exposition-publique) (valider avec `cloudflared tunnel ingress validate`)
- [Healthcheck nginx `localhost` foire sur IPv6 BusyBox](#etape-85---reverse-proxy-nginx-et-exposition-publique) (forcer `127.0.0.1`)

### Streamlit + iframes
- [Grafana refuse l'embed par défaut](#etape-7---monitoring) (`GF_SECURITY_ALLOW_EMBEDDING=true`)
- [Auth bloque l'iframe même avec ALLOW_EMBEDDING](#etape-7---monitoring) (auth anonyme + Viewer requis)
- [DNS interne compose != côté browser](#etape-7---monitoring) (heuristique `urlparse`)
- [`histogram_quantile` retourne NaN sans données récentes](#etape-7---monitoring) (convertir en `None`)
- [Métric Prometheus inexistante = empty array](#etape-7---monitoring) (`or on() vector(0)` en PromQL)
- [Cache `st.cache_data(ttl=15-30)` indispensable sur requêtes externes](#etape-7---monitoring)
- [`st.components.v1.iframe` (URL) vs `html` (HTML brut)](#etape-7---monitoring)
- [Cartes alerting via `st.markdown(unsafe_allow_html=True)`](#etape-7---monitoring) (st.metric trop limité)
- [Charger seuils depuis YAML, pas hardcoded](#etape-7---monitoring) (regle "zero hardcoded")
- [Résilience défensive partout : try/except + `st.warning`](#etape-7---monitoring)

### Streamlit zero hardcoded (demo)
- [Streamlit = portfolio narratif, pas outil de prod](#etape-6---demo-streamlit) (consomme MLflow/Prom, ne les remplace pas)
- [Aucune valeur écrite en dur](#etape-6---demo-streamlit) (tout depuis JSON/MLflow/API/Prometheus)
- [Pages incrementalement au fil des etapes](#etape-6---demo-streamlit)
- [Imports lourds dans try/except](#etape-6---demo-streamlit) (pour ne pas planter la page)
- [`use_container_width=True` pour les graphiques](#etape-6---demo-streamlit)

### CI/CD GitHub Actions
- [Drift de version ruff entre pre-commit et CI](#etape-9---cicd) (aligner explicitement les versions)
- [`pip install -r requirements.txt` source de vérité](#etape-9---cicd) (vs liste figée dans le workflow)
- [`pip install torch --index-url cpu` pour les tests CI](#etape-9---cicd) (éviter le pull CUDA inutile)
- [Cache `pip` via `actions/setup-python@v5 cache: pip`](#etape-9---cicd) (~30s/job après le 1er run)
- [`concurrency` group + `cancel-in-progress`](#etape-9---cicd) (évite saturation runners)
- [`dorny/paths-filter@v3` pour skip Docker build](#etape-9---cicd) (PR documentaires)
- [mypy 1.13 ne supporte pas `disable_error_code = ["untyped-decorator"]`](#etape-9---cicd) (introduit en 1.16)

### Tests
- [Coverage > 80% objectif](#etape-10---tests-et-couverture)

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
- **Sur DagsHub, un seul token sert pour tout** (MLflow + DVC + Git auth + API). Ne pas chercher a separer les credentials par service : copier le meme token dans `MLFLOW_TRACKING_PASSWORD`, `DAGSHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY` (pour DVC S3-compatible). Eviter les noms d'env vars tentants comme `AWS_ACCESS_KEY_ID` qui font croire qu'il faut un compte AWS reel : c'est juste le token DagsHub reutilise. Documenter ce fait en tete du `.env.example` pour eviter qu'un coequipier pense a tort qu'il manque un acces AWS.
- **Un `.env` est fragile sur Windows** : en cas de reboot force, crash Docker Desktop, ou copie de repo, le `.env` peut disparaitre silencieusement alors qu'il est exclu de git. Considerer le `.env` comme **reconstructible a partir d'une source sure** (gestionnaire de mots de passe personnel ou notes internes de l'equipe), pas comme un artefact stable. Garder `.env.example` a jour pour que la reconstruction soit immediate.

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
- **Conflits `git stash` apres `git pull`** : les fichiers generes localement (manifests CSV, reports JSON, dumps de stats) qui ne sont pas dans `.gitignore` et qui ont ete regeneres entre-temps par un coequipier creent des conflits a chaque pull. Deux regles pour eviter ca :
  1. **Ne jamais commiter les artefacts generes** (`data/*.csv`, `data/*.json`, `models/*.pt`, `models/*.onnx`). Soit ils sont dans `.gitignore`, soit ils sont versionnes par DVC (`.dvc` pointeurs dans git).
  2. **Si tu dois commiter un CSV generé ponctuellement** (ex: un rapport d'audit), le placer dans `reports/` avec un timestamp (`audit_YYYY-MM-DD.csv`) pour qu'il ne soit plus jamais regenere et que les pulls ne creent pas de conflit.
  Quand le conflit arrive quand meme : `git stash` les modifs locales, `git pull --rebase`, regenerer les artefacts, et `git stash drop` (ne jamais `git stash pop` si les artefacts seront regeneres de toute facon).
- **Les fichiers `.dvc` suivent git mais les donnees qu'ils pointent sont ailleurs** : si tu changes un fichier dans `data/` ou `models/`, le `.dvc` associe n'est pas mis a jour automatiquement. Il faut faire `dvc commit data.dvc` (ou equivalent) pour enregistrer le nouveau hash, puis `git commit` du `.dvc` pour que le pointeur soit versionnne. Piege inverse : un `git pull` ramene le nouveau `.dvc`, mais les donnees locales restent celles d'avant tant que `dvc pull` n'est pas lance. Adopter la regle `git pull && dvc pull` comme sequence atomique.

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
- **MLflow 403 / 401 si les env vars ne sont pas exportees dans le shell courant** : meme si le `.env` contient les bonnes valeurs, lancer un script Python qui ne charge pas explicitement `.env` (via `python-dotenv` ou `pydantic-settings`) partira avec des credentials vides. Sous PowerShell, `$env:MLFLOW_TRACKING_PASSWORD` ne se propage QUE dans la session courante ; un nouveau terminal repart blanc. Deux patterns qui marchent :
  1. **Dans le code** (recommande) : `from dotenv import load_dotenv; load_dotenv(".env")` en tout debut de script, AVANT `import mlflow`. Pydantic Settings fait ca automatiquement si `env_file=".env"` est declare.
  2. **Dans le shell** (debug rapide) : `Get-Content .env | ForEach-Object { if ($_ -match '^([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }` avant de lancer le script.
  Si `mlflow.search_runs()` retourne silencieusement une erreur "To use authentication, you must first: Get your default access token..." c'est le symptome typique : le token est pas lu. Verifier avec `echo $env:MLFLOW_TRACKING_PASSWORD` dans la meme session que le lancement.
- **Token DagsHub perissable** : le token peut etre revoque, regenere, ou lie a un compte qui n'a plus les droits sur le repo. En cas de 401 persistent meme avec env vars correctement lues, aller sur `https://dagshub.com/user/settings/tokens` et regenerer. Le .env doit etre mis a jour partout (XPS + NUC3 + CI secrets).

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
- **Le script d'export ONNX doit etre agnostique de l'architecture** : ne pas cabler `create_resnet50` en dur, sinon changer de backbone casse l'export. Deux approches qui marchent :
  1. **Flag CLI `--model <name>`** avec defaut sur `config.model_name` (necessite de maintenir la coherence entre le YAML et le checkpoint).
  2. **Auto-detection depuis les cles du state_dict** (recommande pour de l'automation) : ResNet50 expose `conv1.weight`, ConvNeXt-Tiny expose `features.0.0.weight`, etc. Lever `ValueError` si aucune cle caracteristique n'est trouvee.
  Utiliser une factory unifiee `create_backbone(model_name, ...)` pour que l'export, le train et les tests utilisent le meme code.
- **Fichier `.onnx.data` orphelin quand on change de modele** : pour les modeles > 2 GB, ONNX genere un fichier externe de poids `model.onnx.data` a cote du `.onnx` (protobuf limit). Pour des modeles < 2 GB (cas de ResNet50 et ConvNeXt-Tiny, ~90-110 MB), le `.onnx` est self-contained et le `.data` ne devrait pas exister. Si un `.onnx.data` trainait d'un export precedent, le supprimer explicitement avant de re-exporter, ou mieux, nettoyer le dossier `models/` avant chaque export. Verification : `onnx.load(path, load_external_data=False)` puis compter les `initializer.external_data` refs (doit etre 0 pour un self-contained).
- **Transferer un checkpoint entre machines locales** (ex: XPS training -> NUC3 serving) : la methode la plus simple est `python -m http.server 8888` dans `models/` sur la source, puis `Invoke-WebRequest "http://<hostname>:8888/best_model.pt" -OutFile "best_model.pt"` sur la destination. Pas de setup SSH, pas de cloud, pas de quota DVC. Une fois le transfert fait, Ctrl+C sur le serveur. Alternative plus propre pour un transfert recurrent : DVC push sur la source + DVC pull sur la destination (utilise le remote DagsHub comme proxy), mais beaucoup plus lent pour un fichier de 300 MB.
- **Taille du checkpoint .pt != taille du modele deploye** : un checkpoint PyTorch contient typiquement `model_state_dict` + `optimizer_state_dict` + `scheduler_state_dict` + `scaler_state_dict` + metadonnees. L'optimiseur AdamW stocke 2 moments par parametre, donc le checkpoint pese ~3x la taille des poids seuls. Exemple ConvNeXt-Tiny : 28M params = ~110 MB de poids, mais 334 MB de checkpoint. L'ONNX exporte ne garde que les poids (~110 MB). Ne pas s'alarmer si le `.pt` semble gros : c'est normal.

**Commandes cles** :
```powershell
python -m src.models.export_onnx
python -m src.models.export_onnx --checkpoint models/best_model.pt --output models/best_model.onnx
python -m src.models.export_onnx --model convnext_tiny   # force l'architecture si besoin

# Transfert checkpoint XPS -> NUC3
# Sur le XPS (source) :
cd D:\<repo>\models; python -m http.server 8888
# Sur le NUC3 (destination) :
cd D:\<repo>\models; Invoke-WebRequest "http://<xps-hostname>:8888/best_model.pt" -OutFile "best_model.pt"
```

**Duree typique** : 0.5 jour (script + validation + debug dynamo)

---

## Etape 5 - API serving

**But** : API REST prête pour l'inference, avec métriques et health checks.

**A produire** :
- [ ] Service de serving (FastAPI ou BentoML) avec endpoint /predict (top-N + scores)
- [ ] /health, /metrics (Prometheus), /model/info
- [ ] Pydantic schemas (request/response)
- [ ] Chargement ONNX Runtime (CPU)
- [ ] Préprocessing identique au training (transforms)
- [ ] Tests unitaires de l'API
- [ ] (BentoML) bentofile.yaml + bento packagee testee via `bentoml serve <tag>`

**Pieges connus (BentoML 1.4)** :
- `bentoml.onnx` est deprecated depuis 1.4 (warning a chaque appel) mais fonctionnel. `bentoml.onnx.save_model(name, onnx_model, signatures, labels, metadata, custom_objects)` reste l'API la plus simple. Plan de migration cible : `bentoml.models.create()` + chargement ONNX manuel via onnxruntime. Hors scope si on cible 1.2-1.3.
- **`@bentoml.api` force POST** : il n'y a pas de parametre `method=` dans 1.4. Tous les endpoints (`/predict`, `/health`, `/model/info`, ...) sont en POST, style RPC. Pour des GET, monter une ASGI app via `@bentoml.asgi_app`. Ne pas attendre du REST classique : adapter le client (Streamlit, tests).
- **Appel intra-service async-only** : un endpoint qui en appelle un autre (par exemple `predict` -> `infer_batch` pour beneficier du batching adaptatif) passe par un proxy RPC interne qui retourne une coroutine. La methode appelee DOIT etre `async def` et l'appel DOIT etre `await self.infer_batch(...)`. Un appel synchrone fait planter avec `anyio.NoEventLoopError: Not running inside an AnyIO worker thread`.
- **Sérialisation float64 silencieuse via le proxy interne** : les `np.ndarray` float32 sont promus en float64 lors du transit HTTP entre endpoints intra-service. ONNX Runtime exige float32 ; sinon `InvalidArgument: Unexpected input data type. Actual: (tensor(double)), expected: (tensor(float))`. Cast explicite : `np.ascontiguousarray(batch, dtype=np.float32)` dans la methode batchable avant l'inference.
- **`PIL.Image.Image` doit rester un import runtime** : BentoML introspecte les annotations via `typing.get_type_hints()` au demarrage du worker pour brancher le decodeur d'image HTTP. Donc PAS de `TYPE_CHECKING` block sur cet import. Utiliser `from PIL.Image import Image as PILImage  # noqa: TC002` pour faire taire ruff.
- **`ModelOptions` n'est pas un dict** : `bento_model.info.options.get(...)` plante (`AttributeError: 'ModelOptions' object has no attribute 'get'`). Pour retrouver le fichier ONNX dans le Model Store : glob `saved_model.onnx` (nom standard de `bentoml.onnx.save_model`) avec fallback sur `*.onnx`. Pour les `class_names` packagees via `custom_objects`, lire `bento_model.custom_objects['class_names']`.
- **Schema bentofile.yaml en 1.4** : il n'existe PAS de cle `python.version`. La version Python se declare via `docker.python_version`. Sinon `TypeError: PythonOptions.__init__() got an unexpected keyword argument 'version'`.
- **Le modele n'est PAS detecte automatiquement au build** : si le runner appelle `bentoml.onnx.get(tag)` au runtime (dans `__init__`), l'introspecteur de build ne le voit pas. Sans cle `models: [tag]` dans bentofile, le bento se construit sans modele (Model Size = 0). Avec, le bento reste petit sur disque (lien vers le Model Store) mais expose 100+ MB de "Model Size" dans `bentoml list`. Au containerize, le modele est inline dans l'image Docker.
- **Conflit `anyio` / `httpx-ws` au moment de l'install** : BentoML 1.4 tire `httpx-ws==0.9.0` qui exige `anyio>=4.7` (`AsyncContextManagerMixin`). Si `anyio` est deja installe en version <4.7 dans le venv (ex: par `httpx`), `import bentoml` echoue avec `AttributeError: module 'anyio' has no attribute 'AsyncContextManagerMixin'`. Pin manuel : `anyio>=4.7,<5`.
- **Query params HTTP non mappes automatiquement** : `?top_n=3` est ignore par `@bentoml.api` en 1.4. Les params optionnels d'une methode passent via le body JSON, pas la query string. Si on veut absolument query string, il faut wrapper dans une ASGI app.
- **`bentoml serve <module>` (dev) vs `bentoml serve <bento_tag>` (prod)** : le premier charge le code en direct depuis l'arborescence repo (utile pour iterer rapidement, redemarrer manuellement pour appliquer un changement). Le second charge le bento packagee depuis `~/bentoml/bentos/`, code immuable, dependances fixees au `build`. C'est la version `<tag>` qui sera utilisee dans l'image Docker (Etape 8).

**Commandes clés (BentoML)** :
```powershell
# Importer le modele ONNX dans le Model Store
python scripts/import_model_to_bentoml.py
bentoml models list

# Mode dev (hot reload manuel)
bentoml serve src.serving_bentoml.service:ChampyService --port 8020

# Build + serve du bento packagee
bentoml build
bentoml list
bentoml serve champy_classifier:latest --port 8020

# Containerize (Etape 8)
bentoml containerize champy_classifier:latest
```

**Durée typique** : 1-2 jours (FastAPI seul) / 2-3 jours (FastAPI + migration BentoML)

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
- [x] Prometheus : scrape /metrics de l'API
- [x] Grafana : dashboard pré-configuré via fichiers de provisioning (latence, throughput, distribution classes, sante process)
- [ ] Drift detection (Evidently) : rapport périodique ou on-demand
- [ ] Alertes (optionnel) : seuils sur latence, confiance, erreurs

**Pieges connus (Streamlit + Grafana iframe + alerting visuel)** :
- **Grafana refuse l'embed par defaut** : sans `GF_SECURITY_ALLOW_EMBEDDING=true`, Grafana met un header `X-Frame-Options: deny` qui bloque l'iframe. Symptome : iframe vide cote Streamlit. Verifier avec `curl -I http://grafana:3000/d/<uid>` que le header n'est plus la apres l'env var.
- **Auth bloque l'iframe meme avec ALLOW_EMBEDDING** : si Grafana exige login, l'iframe affiche le formulaire d'auth (peu pratique). Activer `GF_AUTH_ANONYMOUS_ENABLED=true` + `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` autorise les iframes en lecture seule sans auth. Acceptable en demo / LAN, en prod il faut un reverse proxy avec auth deleguee (OIDC / OAuth2).
- **DNS interne compose != cote browser** : un helper qui retourne `http://grafana:3000` est valide pour des appels container -> container (Prometheus scrape, Streamlit qui interroge l'API Grafana), mais l'iframe est resolue par le NAVIGATEUR du client qui ne connait pas le DNS interne. Detecter le hostname `grafana` / `host.docker.internal` et forcer `http://localhost:<port>` cote browser. Permettre une override via env var (`CHAMPY_GRAFANA_URL_EXTERNAL`) pour les setups non-localhost.
- **`docker compose restart` n'applique PAS les nouvelles env vars** : il faut `docker compose up -d <service>` (recreate). Identique au piege provisioning des volumes.
- **`histogram_quantile` retourne NaN sans donnees recentes** : pas de prediction sur la fenetre 5min -> p50/p95/p99 = NaN. Si on formatte avec `f"{value:.0f} ms"` on obtient `NaN ms`. Convertir NaN en None dans le helper Python (`math.isnan`) et afficher `-` cote UI.
- **Metric Prometheus inexistante = tableau vide** : un Counter PromQL n'apparait pas tant qu'il n'a pas ete incremente au moins une fois (cas typique : `champy_http_errors_total` au boot, aucune erreur encore). Sans protection, le ratio errors/requests plante. Fix PromQL : `(sum(rate(metric[5m])) or on() vector(0))` injecte un 0 quand le numerateur est vide.
- **Cache `st.cache_data` indispensable sur les requetes externes** : sans cache, chaque rerun Streamlit retape Prometheus / Grafana 5-10 fois. Avec `ttl=15-30s`, les metriques sont fraiches sans surcharger. Pour invalider explicitement : `st.cache_data.clear()`.
- **`st.components.v1.iframe` (URL) vs `st.components.v1.html` (HTML brut)** : iframe charge une URL en lazy (idéal Grafana / rapports HTML stockes), html prend du contenu et l'inline (idéal pour rendre du HTML local). Hauteur obligatoire dans les deux cas (sinon iframe a 0px).
- **Cartes alerting colorees via `st.markdown(html, unsafe_allow_html=True)`** : st.metric est limite a 1 valeur + 1 delta. Pour des cartes vert/jaune/rouge avec contour, label, valeur, message, on est oblige de passer par un `<div>` avec CSS inline. C'est OK tant que le contenu HTML est genere par le code (pas d'input user injecte).
- **Charger les seuils depuis YAML, pas hardcoded** : la regle "zero hardcoded" du Streamlit s'applique aux seuils d'alerting comme aux autres parametres. `configs/monitoring/thresholds.yml` rend l'edition des seuils possible sans toucher au code et sans redeploiement. Convention warning/critical par direction (`lower_is_worse` / `higher_is_worse`) couvre les deux familles de metriques (confidence vs latence/erreur).
- **Resilience defensive partout** : try/except autour de chaque appel reseau (Prometheus, Grafana health, SQLite) avec un message st.warning explicite + action a faire (verifier docker compose ps, ouvrir un terminal, etc.). Une page de monitoring qui crash quand son backend est down est inutile : c'est exactement le moment ou on en a besoin.

**Pieges connus (Evidently 0.7+ pour drift detection)** :
- **L'API a casse entre 0.4 et 0.7** : la majorite des tutoriels web utilisent l'API legacy (`from evidently.report import Report; Report(metrics=[DataDriftMetric()])`). En 0.7+, c'est `from evidently import Report, Dataset, DataDefinition` et `Report(metrics=[DataDriftPreset()])`. Toujours verifier la version (`evidently.__version__`) avant de copier un exemple.
- **`Dataset.from_pandas(df, data_definition=...)` necessite la `DataDefinition` en kwarg** : sans, Evidently auto-detecte les types et confond souvent les float arrondis avec des categories. Symptome : test stat qui plante ou drift faux positif.
- **`save_html` est une methode du `Snapshot` (retour de `.run()`)**, pas du `Report` : `report.run(reference, current).save_html(path)`. `Report.save_html` n'existe pas.
- **HTML self-contained = 3-4 MB par rapport** : Evidently inline les fonts Material Icons + Vega-Lite JS + ses libs. Bonne nouvelle : aucune dependance externe au serve. Mauvaise : gitignore les rapports si on en genere des dizaines, ou archiver selectivement (S3, MLflow). Stocker les rapports dans un dossier `reports/` avec timestamp dans le nom + un `.gitkeep` pour preserver l'arborescence.
- **DataDriftPreset par defaut applique chi-2 (categoriel) + KS (numerique)** : c'est rarement le bon choix sur des distributions tres desequilibrees ou bimodales. Si on a des colonnes specifiques avec une distribution connue (ex: confidence en U inverse), specifier `num_method="psi"` (Population Stability Index) ou `num_method="wasserstein"` via le parametre `per_column_method`.
- **Materialiser une baseline a partir d'aggregats perd la variance** : si la baseline JSON ne contient que `confidence_mean` par classe, le DataFrame reconstruit aura toutes les valeurs egales par groupe. Drift est detecte sur la moyenne globale + la distribution des classes, pas sur la dispersion intra-classe. Pour une analyse fine, il faut stocker les confidences individuelles (multiplie la taille par ~1000 sur 2872 images).
- **Indexer les images avant le calcul de baseline** : `Path.rglob(name)` est O(M) ou M = nb fichiers totaux. Sur un dataset scrape (`data/raw/Mushrooms_images/` contient 647k fichiers), faire un rglob par image cherchee = O(N x M) = 30 minutes. Faire un seul scan O(M) qui construit `dict[name -> path]` puis lookup O(1) = 80s pour 2872 images.
- **Trigger Streamlit -> subprocess.run avec `sys.executable`** : `subprocess.run(["python", ...])` peut pointer vers un Python different du venv (Windows). `sys.executable` garantit le bon interpreteur. `capture_output=True` + `text=True` pour recuperer logs.
- **`st.components.v1.html(html, height=H, scrolling=True)` pour embed** : le `height` est obligatoire (sinon iframe a 0 px). Compter generously (900-1200) pour les rapports Evidently qui sont longs. `scrolling=True` essentiel.
- **Rapports drift = donnees structurees, pas des modeles** : ne pas les versionner avec DVC. Les regenerer a la demande depuis la baseline + le store de predictions est plus propre, plus rapide, et evite l'inflation du stockage DVC.

**Commandes cles (drift detection)** :
```powershell
# Calcul de la baseline (une fois, ~1m20 sur le test set complet 2872 images)
python monitoring/baseline_snapshot.py

# Generation d'un rapport drift sur les 24 dernieres heures
python monitoring/run_drift_report.py --hours 24

# Sur une fenetre custom + chemin de baseline alternatif
python monitoring/run_drift_report.py --hours 6 --baseline monitoring/baseline_v1.json

# Ouvrir le dernier rapport
explorer.exe monitoring\reports\drift_*.html  # Windows
```

**Pieges connus (Grafana provisioning)** :
- **`docker compose restart` n'applique PAS les nouveaux volumes** : il faut `docker compose up -d <service>` pour recreer le container quand on ajoute un mount au compose. Sinon le nouveau dossier `provisioning/` reste vide dans le container alors qu'il est present sur l'hote. `restart` redemarre seulement le process dans le container existant.
- **Le volume nomme `grafana-data` persiste l'etat UI entre redemarrages** : si une datasource a ete creee a la main avant le provisioning, elle coexiste avec celle provisionnee (deux entrees de meme nom, UID different). Le provisioning ne supprime jamais. Soit `DELETE /api/datasources/uid/<old_uid>` une fois, soit `docker volume rm <project>_grafana-data` (perd l'historique UI : custom dashboards crees a la main, panels modifies, etc.).
- **`uid` explicite obligatoire dans le datasource yaml** : sans `uid: prometheus`, Grafana auto-genere une chaine type `afhpol7cbsao0a` qui change a chaque recreate du container. Les dashboards JSON qui referencent `{"datasource": {"uid": "prometheus"}}` echouent silencieusement (panels affichent "no data" sans message d'erreur). Toujours fixer `uid` explicitement dans la YAML.
- **`folderUid` explicite recommande dans le provider de dashboards** : sans `folderUid: champy-classifier` dans le provider yaml, Grafana cree un dossier au nom du provider mais avec un UID auto-genere. Les liens vers les dossiers dans les pages Streamlit ou la doc cassent au prochain reboot. Avec un UID explicite, les URLs `https://grafana/dashboards/f/<uid>/` restent stables.
- **Path translation Git Bash sur Windows** : `docker exec ... ls /etc/grafana/...` est traduit en `C:/Program Files/Git/etc/grafana/...` par MSYS. Symptome : `ls: cannot access...`. Solution : prefixer la commande par `MSYS_NO_PATHCONV=1` (POSIX style) ou utiliser PowerShell directement.
- **Schema version compte** : Grafana >=10 attend `schemaVersion >= 36` dans les dashboards JSON. Avec un schema plus ancien, les panels affichent "no data" meme si la datasource est OK. Utiliser `schemaVersion: 38` (au moment de la redaction) pour les nouveaux dashboards.
- **Le datasource reference dans les dashboards JSON doit etre `{"type": "prometheus", "uid": "prometheus"}`** (objet), PAS la string `"Prometheus"` (qui marchait en Grafana 8 mais est silencieusement ignoree en 10+). Si on copie un dashboard exporte d'une vieille install, faire un find/replace.
- **Cohabitation sur hote partage : audit prealable des ports** : avant `docker compose up`, lister les ports occupes :
  ```powershell
  Get-NetTCPConnection -State Listen | Where-Object {$_.LocalPort -lt 10000} `
    | Select-Object -ExpandProperty LocalPort | Sort-Object -Unique
  ```
- **`prometheus_client` expose `process_*` et `python_*` automatiquement** : pas besoin d'ajouter cAdvisor ou node_exporter pour observer la sante du process API. RAM (`process_resident_memory_bytes`), CPU (`rate(process_cpu_seconds_total[1m])`), file descriptors, uptime (`time() - process_start_time_seconds`), GC Python (`python_gc_collections_total`). cAdvisor reste utile uniquement pour les metriques host (disque, reseau, memoire totale).
- **Choix des metriques pour les dashboards** : si on a une couche de serving qui peut changer (FastAPI -> BentoML), referencer les metriques **applicatives custom** (`champy_*`) plutot que les metriques natives du framework (`bentoml_service_*`). Les premieres survivent au changement de framework, les secondes pas. Ajouter les metriques natives en complement, pas en source primaire.

**Commandes clés** :
```powershell
invoke serve   # Prometheus + Grafana inclus
# Grafana : http://localhost:3010 (admin / GRAFANA_PASSWORD du .env)
# Prometheus : http://localhost:9090

# Generer du trafic pour alimenter les dashboards
python scripts/seed_grafana.py --n 50 --target fastapi

# Verifier le provisioning depuis l'API Grafana
curl -u admin:$env:GRAFANA_PASSWORD http://localhost:3010/api/datasources
curl -u admin:$env:GRAFANA_PASSWORD http://localhost:3010/api/search?type=dash-db
```

**Durée typique** : 1-2 jours

### Stockage des predictions pour le monitoring (SQLite WAL)

**A produire** :
- [x] Schema minimal : id (UUID), timestamp, image_hash (SHA256), predicted_class, confidence, top5_json, latency_ms
- [x] Driver async pour ne pas bloquer l'event loop du serving
- [x] Fire-and-forget depuis le hot path predict
- [x] Endpoint /predictions/recent + /predictions/distribution
- [x] Tests de concurrence (gather de 100+ ecritures)

**Pieges connus (SQLite + BentoML)** :
- **`PRAGMA journal_mode=WAL` est obligatoire pour la concurrence** : SQLite en mode `DELETE` (defaut) verrouille le fichier entier pour chaque ecriture. WAL permet plusieurs lecteurs simultanes pendant qu'un ecriveur ecrit. Sans WAL, on voit `database is locked` au moindre appel concurrent.
- **`PRAGMA busy_timeout=5000` est l'arme anti-`database is locked`** : sans, SQLite remonte l'erreur immediatement si un autre ecriveur tient le verrou. Avec, il attend jusqu'a 5s avant d'echouer. La latence n'augmente jamais (sauf cas pathologique) car le verrou WAL est tres court.
- **`PRAGMA synchronous=NORMAL` est le bon compromis pour WAL** : `FULL` (defaut) fait un fsync apres chaque transaction (durable mais ~3x plus lent). `OFF` ne fsync jamais (rapide mais perd les ecritures en cas de crash). `NORMAL` fsync seulement au checkpoint WAL : durabilite suffisante pour des donnees de monitoring (perte de < 1s en cas de crash brutal).
- **`row_factory` doit etre assigne APRES connect()** : le passer en argument du `connect()` est silencieusement ignore en aiosqlite. Faire `conn.row_factory = aiosqlite.Row` apres l'ouverture, sinon les `fetchall()` retournent des tuples au lieu de dicts.
- **WAL cree des sidecars `.db-wal` et `.db-shm`** : a inclure dans `.gitignore` et `.dockerignore`, sinon ils remontent silencieusement dans les commits ou les images Docker. Ils sont reconstruits automatiquement au prochain open.
- **Partager une connexion aiosqlite entre coroutines est sur** : aiosqlite execute chaque operation dans un thread interne dedie qui serialise les appels. Pas besoin d'`asyncio.Lock` autour des read/write. Inversement, partager un `sqlite3.Connection` (sync stdlib) entre threads sans `check_same_thread=False` plante.
- **Init async + `__init__` sync** : si le framework de serving impose un constructeur synchrone (cas BentoML 1.4), on ne peut pas faire `await store.init()` dans `__init__`. Pattern : creer l'objet dans `__init__`, declencher `init()` au premier appel, proteger avec un `asyncio.Lock` pour eviter la double-init en course.
- **Fire-and-forget avec `asyncio.create_task` exige une reference forte** : sans `self._pending.add(task) ; task.add_done_callback(self._pending.discard)`, le GC peut collecter la Task avant la fin de l'ecriture (Python <3.13). Symptome : ecritures perdues sous charge, sans erreur dans les logs. Ruff RUF006 attrape ce cas.
- **Mount Docker : preferer un dossier a un fichier** : `./data/runtime/predictions.db:/app/data/predictions.db` echoue si le fichier n'existe pas sur l'hote (Docker cree un dossier vide a sa place). `./data/runtime:/app/data/runtime` mount le repertoire entier, supporte les sidecars WAL/SHM, et le fichier .db est cree par l'application.
- **Hash d'image : `image.tobytes()` apres convert("RGB")** plutot que le contenu du fichier upload : permet la deduplication par contenu visuel meme si le client re-encode/recompresse. SHA256 hexdigest = 64 chars, indexable.
- **Volumetrie SQLite acceptable jusqu'a ~1M lignes** : au-dela, envisager PostgreSQL container (pas plus complexe avec Docker Compose). SQLite WAL tient confortablement ~10k req/s en ecriture sur SSD moderne pour le schema decrit.

**Commandes cles** :
```powershell
# Inspecter la base
python -c "import asyncio; from src.serving_bentoml.storage import PredictionStore; from pathlib import Path; \
  asyncio.run((lambda s: (s.init(), print('rows:', s.count()), s.close()))(PredictionStore(Path('data/runtime/predictions.db'))))"

# Reset de la base (perte de l'historique)
rm data/runtime/predictions.db data/runtime/predictions.db-wal data/runtime/predictions.db-shm
```

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

## Etape 8.5 - Reverse-proxy nginx et exposition publique

**But** : Exposer la stack derrière un point d'entrée unique HTTPS, protégé par Cloudflare Access (Zero Trust), avec routage par sous-path vers chaque service.

**A produire** :
- [x] Reverse-proxy nginx interne (container `champy_nginx`, port 8088)
- [x] Configuration `configs/nginx/nginx.conf` avec `location` par service
- [x] Cloudflare Tunnel (`cloudflared` en service systemd sur le NUC Ubuntu)
- [x] Cloudflare Access application (Zero Trust, SSO e-mail magic-link)
- [x] Page **Plateforme** dans Streamlit comme hub interactif

**Pourquoi sous-paths plutôt que sous-domaines** :

| Critère | Sous-domaines (`api.champy.sbdg-ia.fr`) | Sous-paths (`champy.sbdg-ia.fr/api/`) |
|---|---|---|
| DNS | 1 entrée par service | 1 seule entrée |
| Certificat TLS | Wildcard requis | Simple |
| Cloudflare Access | 1 policy par sous-domaine | 1 seule policy |
| CORS | Configuration par origine | Origin unique |
| Session SSO | Une session par service | Une session pour tout |

Coût : configuration nginx plus délicate (gestion fine des préfixes par service).

**Pieges connus** :

- **nginx cache les IPs des backends et casse après `--force-recreate`** : nginx résout les noms DNS (par exemple `grafana`, `mlflow`) au démarrage et garde les IPs en cache. Quand un container backend est recréé (`docker compose up -d --force-recreate <service>`), son IP Docker change, mais nginx pointe encore vers l'ancienne. Symptôme : 502 Bad Gateway sur les routes concernées.
  - **Fix rapide** : `docker compose restart nginx` après tout `--force-recreate` d'un backend.
  - **Fix permanent** : ajouter dans le bloc `http` de `nginx.conf` :
    ```nginx
    resolver 127.0.0.11 valid=30s ipv6=off;
    ```
    Et utiliser des variables dans les `proxy_pass` :
    ```nginx
    set $upstream http://api:8000;
    proxy_pass $upstream;
    ```

- **`$host` vs `$http_host` dans `proxy_set_header`** : la directive `proxy_set_header Host $host;` ne préserve pas le port et utilise parfois le nom interne du container. Symptôme : les redirections HTTP générées par certains backends (notamment Airflow et MinIO) cassent et pointent vers `champy_nginx` au lieu de `champy.sbdg-ia.fr`.
  - **Fix** : utiliser `proxy_set_header Host $http_host;` dans tous les blocs `location`. `$http_host` préserve l'en-tête `Host:` original envoyé par le client.

- **BentoML expose Swagger UI à la racine `/`, pas à `/docs`** : contrairement à FastAPI, BentoML utilise sa propre convention. Symptôme : `https://champy.sbdg-ia.fr/api/docs` retourne 404.
  - **Fix** : utiliser `https://champy.sbdg-ia.fr/api/` (avec slash final) pour Swagger. Le schéma OpenAPI JSON est à `/api/docs.json` (pas `/openapi.json`). **ReDoc n'est pas disponible**. Mettre à jour tous les liens dans le code Streamlit en conséquence (vérifier avec `Get-ChildItem -Recurse | Select-String -Pattern "/docs|/openapi.json|/redoc"`).

- **Airflow double-préfixe le path : `/airflow/airflow/`** : Airflow construit ses URLs internes en concaténant `AIRFLOW__WEBSERVER__BASE_URL` avec le path de chaque route. Si on configure mal, les liens du menu donnent `/airflow/airflow/dags`.
  - **Fix** : dans `docker-compose.yml`, définir :
    ```yaml
    environment:
      AIRFLOW__WEBSERVER__BASE_URL: "https://champy.sbdg-ia.fr/airflow"
    ```
    Et dans `nginx.conf`, **ne pas stripper le préfixe** (utiliser `proxy_pass http://airflow:8080;` sans modification du path, surtout pas de `rewrite`).

- **Bind mount Windows Docker Desktop non fiable pour le code Python live** : sur Windows, le bind mount `./demo:/app/demo` ne propage pas les changements de manière fiable. Symptôme : modification d'un fichier Python, restart du container, mais l'ancien code continue de s'exécuter.
  - **Fix** : après toute modification de code, **rebuild + force-recreate obligatoire** :
    ```powershell
    docker compose build demo
    docker compose up -d --force-recreate demo
    ```
    `docker compose restart` seul n'est pas suffisant. Le `restart` redémarre seulement le process dans le container existant ; il faut recréer le container pour que le nouveau code soit pris en compte.

- **Streamlit `runOnSave` ne fonctionne pas sur Windows Docker Desktop** : le mécanisme de file watching de Streamlit utilise `inotify` (Linux). À travers un bind mount Windows → Linux, les événements ne sont pas remontés correctement.
  - **Fix** : désactiver `runOnSave` dans `.streamlit/config.toml` (ou le laisser activé sans s'attendre à ce qu'il fonctionne) et adopter la procédure `build + force-recreate` ci-dessus.

- **Cloudflare Tunnel YAML strict sur l'indentation** : `cloudflared` refuse de démarrer après modification du fichier `config.yml` avec une erreur `did not find expected key`. YAML est strict sur l'indentation. Un mélange de tabulations et d'espaces, ou un commentaire mal placé, peut casser le parsing.
  - **Fix** : valider avec `cloudflared tunnel ingress validate /home/<user>/.cloudflared/config.yml` avant `systemctl restart cloudflared`. Utiliser exclusivement des espaces pour l'indentation (2 espaces de niveau).

- **Healthcheck nginx `localhost` foire sur IPv6 BusyBox** : le `wget` de l'image `nginx:alpine` (BusyBox) tente IPv6 (`::1`) d'abord. Si nginx n'écoute pas sur IPv6 dans le bloc `server`, le healthcheck échoue en boucle avec `Connection refused`. Symptôme : container `(unhealthy)` alors que nginx sert les requêtes normalement.
  - **Fix** : forcer IPv4 dans le healthcheck. Dans `docker-compose.yml` :
    ```yaml
    healthcheck:
      test: ["CMD-SHELL", "wget -q -O /dev/null http://127.0.0.1/nginx-health || exit 1"]
    ```
    Et ajouter un endpoint dédié `/nginx-health` dans `nginx.conf` :
    ```nginx
    location = /nginx-health {
        access_log off;
        add_header Content-Type text/plain;
        return 200 'OK\n';
    }
    ```

**Cloudflare Tunnel + Access (recettes)** :

```bash
# Sur le NUC Ubuntu (où tourne cloudflared)
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -n 30 --no-pager
sudo systemctl restart cloudflared

# Validation du fichier de config avant restart
cloudflared tunnel ingress validate /home/<user>/.cloudflared/config.yml

# Tester depuis le NUC Ubuntu que la liaison vers le NUC3 fonctionne
curl -I http://192.168.50.55:8088/nginx-health
```

**Configuration nginx type pour un nouveau service** :

```nginx
# Bloc à ajouter dans http { server { ... } } de configs/nginx/nginx.conf
location /<service>/ {
    proxy_pass http://<service>:<port>/;
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix /<service>;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

**Commandes clés** :

```powershell
# Reload nginx sans downtime après modification de la config
docker exec champy_nginx nginx -t                # Test syntaxe
docker exec champy_nginx nginx -s reload         # Reload sans coupure

# Diagnostic healthcheck nginx
docker inspect champy_nginx --format '{{json .State.Health}}'

# Test direct des routes en local
curl.exe -I http://localhost:8088/grafana/
curl.exe -I http://localhost:8088/api/
curl.exe -I http://localhost:8088/airflow/

# Test depuis Internet (renvoie 302 vers Cloudflare Access si pas authentifié)
curl.exe -I "https://champy.sbdg-ia.fr/api/" -L
```

**Durée typique** : 1 jour (setup initial) + 2-3 itérations sur les pièges (~ 0.5 jour chacune)

---

## Etape 9 - CI/CD

**But** : Automatiser lint, tests, build sur chaque push.

**A produire** :
- [x] `.github/workflows/ci.yml` (5 jobs : lint, typecheck, docstrings, tests, build)
- [ ] `.github/workflows/cd.yml` (deploy, optionnel)
- [x] Badge CI dans README.md
- [ ] Branch protection sur `main`

**Pieges connus** :
- **Drift de version ruff entre pre-commit et CI** : si `mirrors-ruff` du pre-commit est pinne (ex: v0.8.0) mais le CI installe `pip install ruff>=0.4` (= latest), les regles de format different. Symptome : pre-commit local accepte un fichier que le CI rejette (ou inversement). Fix : aligner explicitement les versions (pin `mirrors-ruff` au meme tag que la version utilisee localement, ex: v0.15.0). Verifier regulierement avec `pre-commit autoupdate`.
- **`pip install -r requirements.txt` est la source de verite** : maintenir une liste de deps figee dans le workflow CI casse a chaque ajout (cas reel : `aiosqlite`, `mlflow` ajoutes au Bloc M2 mais oublies dans le CI -> `ModuleNotFoundError` sur les tests). Le `requirements.txt` du repo est la seule liste a maintenir, le CI la consomme. Penser a regenerer `requirements.txt` a chaque ajout de dep dans `pyproject.toml`.
- **`pip install torch --index-url https://download.pytorch.org/whl/cpu` pour les tests CI** : sinon torch tire la variante CUDA (~2 GB), inutile sur les runners GitHub. La variante CPU pese ~150 MB et suffit pour les tests d'inference (ONNX Runtime CPU deja).
- **Cache `pip` via `actions/setup-python@v5 cache: pip`** : sans cache, chaque job reinstalle les deps en ~3 minutes. Avec cache, ~30 secondes apres le 1er run. Le cache est invalide automatiquement au changement de `requirements.txt`.
- **`concurrency` group + `cancel-in-progress: true`** : sur des push successifs rapides (`git push --force` ou `git rebase`), GitHub Actions empile les runs et sature la queue gratuite. Cancel-in-progress annule les runs precedents sur la meme branche. A ajouter au top du workflow.
- **`dorny/paths-filter@v3` pour skip le job docker build** : sur une PR purement documentaire (LOGBOOK + dashboards), rebuilder les images Docker (~2-3 min) est une perte. Le filter declenche le build seulement si `docker/`, `compose`, `requirements`, `src/`, `configs/` ou `demo/` ont change.
- **mypy 1.13 ne supporte pas `disable_error_code = ["untyped-decorator"]`** : ce code n'a ete ajoute qu'en mypy 1.16. Pin a 1.13 partout (CI + pre-commit) ou retirer la directive. Sinon le pre-commit local plante avec "Invalid error code(s)" alors que le CI le traite en warning.
- **Decorateurs FastAPI / BentoML "untyped" pour mirrors-mypy** : `mirrors-mypy` (pre-commit) n'installe pas FastAPI/BentoML stubs, donc tous les `@app.post`, `@bentoml.api` sont "untyped". Solution : `# type: ignore[misc]` sur chaque decorateur. En mypy >= 1.16, le code separe `untyped-decorator` rend les `[misc]` insuffisants - mais on garde 1.13 pour eviter ca.
- **Verifier orphelins avant suppression** : avant de supprimer un fichier qui fait planter `interrogate` (ex: `# Script python a faire a martir du notebook` sans docstring), `grep -r` pour s'assurer qu'aucun import / config ne le reference. Cas reel : 4 fichiers vestiges du fork supprimes au CI fix, libere 96.5% -> 100% docstrings coverage.

**Commandes cles** :
```powershell
# Pousser et suivre le CI
git push origin dev-dominique
gh run list --workflow=ci.yml --limit=3
gh run view <run_id> --log-failed   # extraire les erreurs

# Reproduire le CI en local
ruff check src/ data/data_split.py data/curate.py demo/lib/ monitoring/ scripts/ tests/
ruff format --check src/ data/data_split.py data/curate.py demo/lib/ monitoring/ scripts/ tests/
mypy src/ monitoring/ data/data_split.py --ignore-missing-imports
interrogate src/ monitoring/ scripts/ -c pyproject.toml -v
pytest tests/unit/ -v --tb=short
```

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

### Ports par défaut (mapping actuel du projet Champy)

| Service | Port interne | Port externe (host) | Sous-path via nginx hub |
|---|---:|---:|---|
| nginx (hub) | 80 | **8088** | — |
| Streamlit demo | 8501 | 8501 | `/` |
| BentoML API | 8000 | 8010 | `/api/` |
| MLflow | 5000 | 5050 | `/mlflow/` |
| Airflow webserver | 8080 | 8081 | `/airflow/` |
| PostgreSQL Airflow | 5432 | 5433 | (interne uniquement) |
| Prometheus | 9090 | 9090 | `/prometheus/` |
| Grafana | 3000 | 3010 | `/grafana/` |
| Alertmanager | 9093 | 9193 | `/alertmanager/` |
| Alertmanager Discord adapter | 9094 | — | (interne uniquement) |
| MinIO S3 API | 9000 | 9010 | (via console) |
| MinIO console web | 9001 | 9011 | `/minio/` |

**Point d'entrée public** : `https://champy.sbdg-ia.fr/<sous-path>/` (protégé par Cloudflare Access SSO).

### Checklist "avant de commit"
- [ ] `invoke lint` passe
- [ ] `invoke test` passe
- [ ] Pas de secrets dans le code (vérifier que `.env` est dans `.gitignore`)
- [ ] LOGBOOK.md à jour
- [ ] PLAYBOOK.md enrichi si nouvelle leçon apprise
- [ ] Si modif code Streamlit sur Windows : `docker compose build demo && docker compose up -d --force-recreate demo` testé
- [ ] Si modif nginx.conf : `docker exec champy_nginx nginx -t` passe + `docker compose restart nginx`
- [ ] Si modif `cloudflared` : `cloudflared tunnel ingress validate config.yml` passe + `sudo systemctl restart cloudflared`
