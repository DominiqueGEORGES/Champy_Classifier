"""Page d'accueil du portfolio Streamlit Champy Classifier.

Affiche une vue d'ensemble du pipeline MLOps avec le statut
de chaque etape, determine dynamiquement en verifiant la
presence des artefacts correspondants.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Racine du projet (demo/ est un sous-dossier de la racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="Champy Classifier - MLOps Portfolio",
    page_icon=":mushroom:",
    layout="wide",
)

st.title(":mushroom: Champy Classifier")
st.subheader("Portfolio MLOps - Classification de champignons (30 especes)")

st.markdown("""
Ce portfolio interactif retrace l'ensemble du pipeline MLOps
du projet Champy Classifier, de l'exploration des donnees brutes
au monitoring en production.

**Principe fondamental** : aucune valeur n'est ecrite en dur.
Toutes les statistiques, metriques et visualisations sont lues
dynamiquement aux sources (fichiers JSON, MLflow, API, Prometheus).
""")

st.divider()

# --- Statut des etapes (detection dynamique des artefacts) ---
st.header("Statut du pipeline")


def check_artifact(path: Path) -> str:
    """Retourne une icone selon l'existence d'un artefact.

    Args:
        path: Chemin vers le fichier ou repertoire a verifier.

    Returns:
        Chaine avec emoji de statut.
    """
    if path.exists():
        return ":white_check_mark:"
    return ":x:"


# Definition des etapes et de leurs artefacts de validation
pipeline_steps = [
    ("1. Donnees brutes", DATA_DIR / "raw_stats.json", "Exploration et statistiques"),
    ("2. Nettoyage", DATA_DIR / "cleaning_report.json", "Exclusion doublons et augmentations TF"),
    (
        "3. Augmentation",
        PROJECT_ROOT / "src" / "data" / "dataset.py",
        "Transforms PyTorch configurables",
    ),
    ("4. Split", DATA_DIR / "split_stats.json", "Split stratifie 70/15/15"),
    ("5. Entrainement", PROJECT_ROOT / "models" / "model.onnx", "Training PyTorch + MLflow"),
    (
        "6. Evaluation",
        PROJECT_ROOT / "models" / "model.onnx",
        "Metriques, confusion matrix, GradCAM",
    ),
    ("7. Model Registry", PROJECT_ROOT / "models" / "model.onnx", "MLflow Model Registry"),
    ("8. Prediction", PROJECT_ROOT / "src" / "serving" / "app.py", "API FastAPI + inference ONNX"),
    ("9. API", PROJECT_ROOT / "src" / "serving" / "app.py", "Endpoints REST"),
    ("10. Monitoring", PROJECT_ROOT / "configs" / "prometheus.yml", "Prometheus + Grafana"),
    ("11. Drift", PROJECT_ROOT / "src" / "monitoring" / "drift.py", "Evidently AI"),
    ("12. Infrastructure", PROJECT_ROOT / "docker-compose.yml", "Docker Compose + CI/CD"),
]

cols = st.columns(3)
for i, (name, artifact, description) in enumerate(pipeline_steps):
    col = cols[i % 3]
    status = check_artifact(artifact)
    col.markdown(f"{status} **{name}**")
    col.caption(description)

st.divider()

# --- Metriques rapides si disponibles ---
st.header("Metriques cles")

try:
    import json

    with open(DATA_DIR / "split_stats.json", encoding="utf-8") as f:
        split_stats = json.load(f)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images retenues", f"{split_stats['total']:,}")
    col2.metric("Classes", len(split_stats["per_class"]))
    col3.metric("Split train", f"{split_stats['splits']['train']:,}")
    col4.metric("Seed", split_stats["seed"])
except FileNotFoundError:
    st.warning("Statistiques de split non disponibles (data/split_stats.json manquant).")
except Exception as e:
    st.warning(f"Erreur lors du chargement des metriques : {e}")

st.divider()
st.caption("Utilisez le menu lateral pour naviguer entre les etapes du pipeline.")
