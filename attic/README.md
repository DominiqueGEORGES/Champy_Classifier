# attic/ - fichiers archivés

Ce dossier regroupe des fichiers retirés du chemin principal du projet lors de la
préparation à la publication. Ils sont **conservés ici plutôt que supprimés** (geste
réversible) : aucun n'est référencé par le code, la CI, `tasks.py`, les DAGs Airflow ou
`docker-compose`. La suppression définitive reste une décision ultérieure.

| Fichier | Origine | Raison |
|---|---|---|
| `Test_domi.txt` | racine | Fichier de test personnel (contenu : `test`). |
| `tests/Test.jpg` | `tests/` | Image orpheline, aucun test ne la lit. |
| `gradcam_validation.png` | racine | Sortie générée par `scripts/validate_pytorch_model.py`. |
| `main.py` | racine | Stub `uv init` (`print(...)`), jamais un point d'entrée réel. |
| `scripts/fix_streamlit_accents.py` | `scripts/` | Correctif d'encodage one-off (auto-documenté « à exécuter une fois puis supprimer »). |
| `demo/home.py` | `demo/` | Landing Streamlit obsolète, doublon de `demo/app.py` (valeurs en dur). |
| `demo/lib/datetime_utils.py` | `demo/lib/` | Helper jamais importé (code mort). |
| `demo/lib/_path_setup.py` | `demo/lib/` | Helper jamais importé (boilerplate `sys.path` dupliqué à la main dans les pages). |
| `src/inference/__init__.py` | `src/inference/` | Package vide, jamais importé. |
| `src/monitoring/__init__.py` | `src/monitoring/` | Package vide ; le monitoring réel est dans `monitoring/` à la racine. |

Voir `docs/INVENTAIRE_PRE_PUBLIC.md` pour le détail des constats.
