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
- Un dataset naturellement desequilibre (ratio 17x entre la classe la plus grande et la plus petite) necessite un mecanisme d'equilibrage au training (WeightedRandomSampler, class weights, ou over-sampling en ligne), pas un resampling statique des donnees.

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
- [ ] Boucle d'entraînement (AMP si GPU)
- [ ] Early stopping + checkpointing
- [ ] MLflow logging (params, métriques par epoch, artefacts)
- [ ] Config YAML externalisée (lr, batch_size, epochs, etc.)
- [ ] Seed fixé pour reproductibilité

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke train --config configs/training/default.yaml
```

**Durée typique** : 2-3 jours (code) + temps d'entraînement

---

## Etape 4 - Model registry et export

**But** : Modèle versionné, promu, exporté en format optimisé pour la prod.

**A produire** :
- [ ] Enregistrement dans MLflow Model Registry
- [ ] Promotion Staging -> Production
- [ ] Export ONNX (ou TorchScript)
- [ ] Validation : comparaison accuracy PyTorch vs ONNX
- [ ] Script d'export automatisé

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke export-onnx
```

**Durée typique** : 0.5-1 jour

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

**Pièges connus** :
-

**Commandes clés** :
```powershell
invoke serve   # inclut le container Streamlit
```

**Durée typique** : 2-3 jours

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
-

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
