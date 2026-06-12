"""Page Streamlit dédiée à l'analyse et l'interprétation des modèles.

Lit les snapshots versionnés produits par ``scripts/generate_analysis.py``
dans ``docs/analysis/`` et présente l'analyse de façon structurée :
- lecture pédagogique des courbes MLflow ;
- comparaison chiffrée des trois runs (default, aggressive, ConvNeXt) ;
- analyse de la préparation des données ;
- pipeline d'augmentation actuel ;
- axes d'amélioration identifiés ;
- narratifs prêts à l'emploi pour le jury.

La page ne fait **aucun appel MLflow direct** : tout est lu depuis les
fichiers JSON statiques. Si une analyse plus récente est nécessaire,
relancer le script générateur depuis la racine du projet ::

    python -m scripts.generate_analysis

Un sélecteur d'historique en haut de page permet de naviguer dans les
analyses passées.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from demo import auth

auth.setup_page(min_role="user")  # ou "guest" pour pages publiques

# ---------------------------------------------------------------------------
# Configuration et helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "docs" / "analysis"
CURRENT_FILE = ANALYSIS_DIR / "current.json"

RUN_KEY_TO_LABEL = {
    "resnet50_default": "ResNet50 default",
    "resnet50_aggressive": "ResNet50 aggressive",
    "convnext_tiny": "ConvNeXt-Tiny",
}


@st.cache_data
def load_snapshot(path_str: str) -> dict:
    """Lit un fichier JSON d'analyse, avec cache Streamlit."""
    with Path(path_str).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_versioned_snapshots() -> list[Path]:
    """Liste les snapshots versionnés (sauf ``current.json``), du plus récent au plus ancien."""
    if not ANALYSIS_DIR.exists():
        return []
    return sorted(
        (p for p in ANALYSIS_DIR.glob("*.json") if p.name != "current.json"),
        reverse=True,
    )


def format_snapshot_label(path: Path) -> str:
    """Convertit un nom de fichier de snapshot en label lisible localisé."""
    name = path.stem
    try:
        dt = datetime.strptime(name, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)
        local = dt.astimezone()
        return local.strftime("%d/%m/%Y à %Hh%M")
    except ValueError:
        return name


def format_pct(value, decimals: int = 1) -> str:
    """Formate un nombre entre 0 et 1 en pourcentage français, ou ``—`` si None."""
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f} %"


def format_delta_pct(delta) -> str:
    """Formate un delta de proportion en points de pourcentage signés."""
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f} pts"


def get_metric(run_data, *candidates):
    """Retourne la première métrique trouvée parmi celles fournies, ou None."""
    if run_data is None:
        return None
    for c in candidates:
        v = run_data.get("metrics", {}).get(c)
        if v is not None:
            return v
    return None


def format_duration(seconds) -> str:
    """Formate une durée en secondes en ``HhMM``, ou ``—`` si None."""
    if seconds is None:
        return "—"
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h{minutes:02d}"


def format_relative_age(iso_timestamp: str) -> str:
    """Retourne ``il y a X minutes/heures/jours`` à partir d'un ISO 8601 UTC."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"il y a {int(delta.total_seconds() / 60)} minutes"
        if hours < 24:
            return f"il y a {int(hours)} heures"
        return f"il y a {int(hours / 24)} jours"
    except (ValueError, TypeError):
        return "—"


def build_param_row(runs: dict, label: str, key: str) -> dict:
    """Construit une ligne de tableau comparatif pour un paramètre des trois runs."""
    values = {}
    for run_key in ["resnet50_default", "resnet50_aggressive", "convnext_tiny"]:
        run = runs.get(run_key)
        if run is None:
            values[RUN_KEY_TO_LABEL[run_key]] = "—"
        else:
            raw = run.get("params", {}).get(key)
            if raw is None:
                raw = run.get("metrics", {}).get(key)
            values[RUN_KEY_TO_LABEL[run_key]] = str(raw) if raw is not None else "—"
    return {"Métrique": label, **values}


def build_metric_row(runs: dict, label: str, *candidates: str) -> dict:
    """Construit une ligne de tableau comparatif pour une métrique des trois runs."""
    values = {}
    for run_key in ["resnet50_default", "resnet50_aggressive", "convnext_tiny"]:
        v = get_metric(runs.get(run_key), *candidates)
        values[RUN_KEY_TO_LABEL[run_key]] = format_pct(v)
    return {"Métrique": label, **values}


def build_gap_row(runs: dict, label: str = "Écart accuracy / F1") -> dict:
    """Construit la ligne d'écart accuracy moins F1 macro pour les trois runs."""
    values = {}
    for run_key in ["resnet50_default", "resnet50_aggressive", "convnext_tiny"]:
        acc = get_metric(runs.get(run_key), "test_accuracy", "val_acc")
        f1 = get_metric(runs.get(run_key), "test_f1_macro", "val_f1_macro")
        gap = (acc - f1) if (acc is not None and f1 is not None) else None
        values[RUN_KEY_TO_LABEL[run_key]] = format_delta_pct(gap)
    return {"Métrique": label, **values}


# ---------------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Analyse des modèles - Champy",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Analyse et interprétation des modèles")
st.markdown(
    "Synthèse des trois runs d'entraînement, des courbes MLflow, des choix "
    "de préparation des données et des axes d'amélioration identifiés."
)


# ---------------------------------------------------------------------------
# Sélecteur d'historique et chargement du snapshot
# ---------------------------------------------------------------------------

if not CURRENT_FILE.exists():
    st.error(
        "Aucun snapshot d'analyse disponible. Lance d'abord le générateur "
        "depuis la racine du projet :\n\n"
        "```\npython -m scripts.generate_analysis\n```"
    )
    st.stop()

versioned = list_versioned_snapshots()

col_select, col_age = st.columns([2, 1])

with col_select:
    options = ["Dernière analyse (current)"] + [format_snapshot_label(p) for p in versioned]
    paths_by_option = {"Dernière analyse (current)": CURRENT_FILE}
    paths_by_option.update({format_snapshot_label(p): p for p in versioned})

    selected_label = st.selectbox(
        "Version de l'analyse",
        options=options,
        index=0,
        help="Sélectionne une version historique pour comparer l'évolution des analyses.",
    )
    selected_path = paths_by_option[selected_label]

snapshot = load_snapshot(str(selected_path))

with col_age:
    age = format_relative_age(snapshot.get("generated_at", ""))
    st.metric(
        label="Généré",
        value=age,
        help=f"Snapshot généré le {snapshot.get('generated_at', '?')}",
    )

git_commit = snapshot.get("git_commit", "?")
git_branch = snapshot.get("git_branch", "?")
generator_version = snapshot.get("generator_version", "?")
st.caption(
    f"Snapshot v{generator_version} — branche `{git_branch}` — commit `{git_commit}` — "
    f"généré le {snapshot.get('generated_at', '?')}"
)

st.divider()


# ---------------------------------------------------------------------------
# Extraction des données fréquemment utilisées
# ---------------------------------------------------------------------------

runs = snapshot.get("runs", {})
comparison = snapshot.get("comparison", {})
narratives = snapshot.get("narratives", {})

run_default = runs.get("resnet50_default")
run_aggressive = runs.get("resnet50_aggressive")
run_convnext = runs.get("convnext_tiny")

acc_default = get_metric(run_default, "test_accuracy", "val_acc")
acc_aggressive = get_metric(run_aggressive, "test_accuracy", "val_acc")
acc_convnext = get_metric(run_convnext, "test_accuracy", "val_acc")

f1_default = get_metric(run_default, "test_f1_macro", "val_f1_macro")
f1_aggressive = get_metric(run_aggressive, "test_f1_macro", "val_f1_macro")
f1_convnext = get_metric(run_convnext, "test_f1_macro", "val_f1_macro")


# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------

tab_overview, tab_curves, tab_compare, tab_data, tab_improve, tab_pitch = st.tabs(
    [
        "Vue d'ensemble",
        "Lecture des courbes",
        "Comparaison des runs",
        "Qualité des données",
        "Axes d'amélioration",
        "Narratif soutenance",
    ]
)


# ===========================================================================
# Onglet 1 - Vue d'ensemble
# ===========================================================================

with tab_overview:
    st.header("Modèle final retenu pour la production")

    delta_acc = (
        acc_convnext - acc_aggressive
        if acc_convnext is not None and acc_aggressive is not None
        else None
    )
    delta_f1 = (
        f1_convnext - f1_aggressive
        if f1_convnext is not None and f1_aggressive is not None
        else None
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "ConvNeXt-Tiny")
    col2.metric(
        "Test accuracy",
        format_pct(acc_convnext),
        format_delta_pct(delta_acc) + " vs ResNet50 agg." if delta_acc is not None else None,
    )
    col3.metric(
        "Test F1 macro",
        format_pct(f1_convnext),
        format_delta_pct(delta_f1) + " vs ResNet50 agg." if delta_f1 is not None else None,
    )

    if run_convnext:
        st.markdown(f"**Run MLflow** : [{run_convnext['run_name']}]({run_convnext['mlflow_url']})")

    st.markdown(
        """
**Pourquoi ConvNeXt-Tiny est le modèle retenu :**

- Meilleure accuracy globale par rapport au ResNet50 aggressive
- Meilleur F1 macro, traduisant une meilleure équité entre classes
- Écart accuracy / F1 réduit, donc moins de biais sur les classes minoritaires
- Architecture plus moderne (2022) tout en restant légère

C'est ce modèle qui sera importé dans le BentoML Model Store sous la version `v2.0.0`.
        """
    )

    st.subheader("Stratégie d'entraînement appliquée")

    st.markdown(
        """
Les trois runs partagent la même stratégie **two-phase fine-tuning** :

1. **Phase 1 (epochs 1 à 10)** : backbone gelé, seule la tête de classification apprend.
   Learning rate élevé (`lr=1e-3`) pour permettre une calibration rapide.
2. **Phase 2 (epochs 11 à fin)** : backbone dégelé, fine-tuning complet du réseau.
   Learning rate divisé par 30 (`lr=3e-5`) pour éviter de détériorer les features
   pré-entraînées sur ImageNet.

Le scheduler **CosineAnnealingLR** réduit progressivement le learning rate sur l'ensemble
du training pour permettre une convergence fine en fin de course.
        """
    )

    st.subheader("Reproductibilité")

    repro_rows = []
    for key, run in [
        ("resnet50_default", run_default),
        ("resnet50_aggressive", run_aggressive),
        ("convnext_tiny", run_convnext),
    ]:
        if run is None:
            continue
        params = run.get("params", {})
        repro_rows.append(
            {
                "Run": RUN_KEY_TO_LABEL[key],
                "Seed": params.get("seed", "—"),
                "Run name MLflow": run.get("run_name", "—"),
                "Durée": format_duration(run.get("duration_seconds")),
                "Statut": run.get("status", "—"),
            }
        )

    if repro_rows:
        st.dataframe(pd.DataFrame(repro_rows), hide_index=True, use_container_width=True)

    st.markdown(
        "Tous les runs sont consultables sur DagsHub MLflow et reproductibles via "
        "leurs fichiers de configuration YAML versionnés dans le dépôt git."
    )


# ===========================================================================
# Onglet 2 - Lecture des courbes MLflow
# ===========================================================================

with tab_curves:
    st.header("Lecture pédagogique des courbes MLflow")

    st.markdown(
        "Chaque run MLflow expose six graphiques de métriques au fil des epochs. "
        "Les comprendre permet de raconter clairement le déroulement du training au jury."
    )

    st.subheader("Les quatre zones temporelles")

    st.markdown(
        """
Quel que soit le graphique observé, il faut le découper mentalement en quatre zones :

| Zone | Epochs | Ce qui s'y passe |
|---|---|---|
| **Phase 1** | 1 → 10 | Backbone gelé, seule la tête fc apprend. Calibration rapide. |
| **Transition** | epoch 10 | Dégel du backbone, chute brutale du learning rate (divisé par 30). |
| **Phase 2** | 11 → 30-40 | Fine-tuning complet, ajustements fins du backbone. |
| **Plateau / Stop** | fin | Convergence atteinte. Early stopping si le val_loss ne baisse plus. |

Cette structure en deux phases est la **signature visuelle** du training. Le jury va la chercher.
        """
    )

    st.subheader("Décodage des six graphiques")

    metrics_data = [
        {
            "Métrique": "lr",
            "Représente": "Learning rate effectif à chaque epoch",
            "Forme attendue": "Décroissance cosinus phase 1, chute à 3e-5 à l'epoch 10, puis cosinus jusqu'à zéro",
            "Lecture": "Confirme que le scheduler et la stratégie two-phase sont appliqués",
        },
        {
            "Métrique": "phase",
            "Représente": "Marqueur de phase (1 ou 2)",
            "Forme attendue": "Plateau à 1 jusqu'à epoch 10, saut à 2 ensuite",
            "Lecture": "Repère temporel pour relire les autres graphiques",
        },
        {
            "Métrique": "train_loss",
            "Représente": "Erreur du modèle sur les images d'entraînement",
            "Forme attendue": "Décroissance régulière, légère accélération à l'epoch 10",
            "Lecture": "Mesure à quel point le modèle se trompe sur ce qu'il connaît",
        },
        {
            "Métrique": "val_acc",
            "Représente": "Pourcentage de bonnes prédictions sur images jamais vues",
            "Forme attendue": "Montée régulière, bond à l'epoch 10-12, plateau en fin",
            "Lecture": "LA métrique principale pour juger la qualité de généralisation",
        },
        {
            "Métrique": "val_f1_macro",
            "Représente": "Score F1 moyen par classe (équité entre classes)",
            "Forme attendue": "Similaire à val_acc mais plus bas, oscillations possibles en phase 2",
            "Lecture": "L'écart avec val_acc révèle les classes mal traitées",
        },
        {
            "Métrique": "val_loss",
            "Représente": "Erreur du modèle sur les images jamais vues",
            "Forme attendue": "Décroissance, plateau en fin sans remontée",
            "Lecture": "Si elle remonte alors que train_loss baisse, c'est de l'overfitting",
        },
    ]

    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    st.subheader("Signaux d'alerte à savoir reconnaître")

    st.markdown(
        """
- **Overfitting** : train_loss continue à baisser, mais val_loss remonte. Le modèle apprend
  par cœur le train set au lieu de généraliser. **Non observé sur les trois runs Champy.**
- **Underfitting** : train_loss et val_loss plateauent haut, sans descendre. Le modèle est
  trop simple ou pas assez entraîné. **Non observé.**
- **Instabilité** : oscillations violentes en zigzag sur train_loss ou val_loss. Le learning
  rate est trop grand. **Non observé**, le CosineAnnealingLR évite ce piège.
- **Catastrophic forgetting** : chute brutale de val_acc à l'epoch de dégel du backbone.
  C'est ce qu'évite la division du learning rate par 30 en phase 2. **Non observé.**
        """
    )

    st.info(
        "Sur les trois runs Champy, aucun de ces signaux pathologiques n'est présent. "
        "Les courbes sont propres et la convergence est régulière."
    )


# ===========================================================================
# Onglet 3 - Comparaison des runs
# ===========================================================================

with tab_compare:
    st.header("Comparaison chiffrée des trois runs")

    st.markdown(
        "Les trois runs ont été lancés sous le même compte DagsHub entre le 12 et le 13 mai "
        "2026 sur le poste XPS2 (RTX 3050 Ti, 4 GB de VRAM). Ils sont strictement comparables."
    )

    if not any([run_default, run_aggressive, run_convnext]):
        st.warning("Aucun run disponible dans ce snapshot.")
    else:
        comparison_rows = [
            build_param_row(runs, "Architecture", "model_name"),
            build_param_row(runs, "Epochs configurés", "total_epochs"),
            build_param_row(runs, "Batch size", "batch_size"),
            build_param_row(runs, "lr phase 1", "lr_phase1"),
            build_param_row(runs, "lr phase 2", "lr_phase2"),
            build_param_row(runs, "Seed", "seed"),
            build_param_row(runs, "Early stopping patience", "early_stopping_patience"),
            build_metric_row(runs, "Test accuracy", "test_accuracy", "val_acc"),
            build_metric_row(runs, "Test F1 macro", "test_f1_macro", "val_f1_macro"),
            build_gap_row(runs),
        ]

        st.dataframe(
            pd.DataFrame(comparison_rows),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Analyse différentielle")

    gap_acc_def_agg = (
        acc_aggressive - acc_default
        if acc_aggressive is not None and acc_default is not None
        else None
    )
    gap_f1_def_agg = (
        f1_aggressive - f1_default
        if f1_aggressive is not None and f1_default is not None
        else None
    )
    gap_acc_agg_cnx = (
        acc_convnext - acc_aggressive
        if acc_convnext is not None and acc_aggressive is not None
        else None
    )
    gap_f1_agg_cnx = (
        f1_convnext - f1_aggressive
        if f1_convnext is not None and f1_aggressive is not None
        else None
    )

    st.markdown(
        f"""
**Default → aggressive (même architecture, hyperparamètres ajustés)** :

- Gain de {format_delta_pct(gap_acc_def_agg)} sur l'accuracy et {format_delta_pct(gap_f1_def_agg)} sur le F1 macro
- Cette amélioration provient principalement de l'augmentation du nombre d'epochs (30 → 40),
  du batch size plus élevé (16 → 24) et du learning rate phase 2 trois fois plus fort
  (`1e-5` → `3e-5`)
- Démontre que pour ResNet50, le default sous-utilisait le potentiel du modèle

**Aggressive ResNet50 → ConvNeXt-Tiny (changement d'architecture)** :

- Gain supplémentaire de {format_delta_pct(gap_acc_agg_cnx)} sur l'accuracy et {format_delta_pct(gap_f1_agg_cnx)} sur le F1 macro
- L'architecture ConvNeXt apporte plus que les hyperparamètres
- L'écart accuracy / F1 se réduit, signe d'une meilleure équité entre classes

**Justification scientifique** : ConvNeXt (Liu et al., 2022) est une architecture conçue
pour égaler les performances des Vision Transformers tout en gardant la structure
convolutionnelle classique. Sur des datasets de taille modérée comme le nôtre, elle est
réputée mieux généraliser que ResNet50 grâce à ses normalisations LayerNorm et ses kernels
plus larges.
        """
    )


# ===========================================================================
# Onglet 4 - Qualité des données
# ===========================================================================

with tab_data:
    st.header("Préparation des données : analyse honnête")

    st.markdown(
        "La qualité des données conditionne le plafond de performance du modèle. Voici une "
        "analyse sans complaisance de notre préparation."
    )

    st.subheader("Volume et répartition")

    volume_data = [
        {"Split": "Train", "Nombre d'images": "13 396", "Proportion": "70 %"},
        {"Split": "Validation", "Nombre d'images": "2 870", "Proportion": "15 %"},
        {"Split": "Test", "Nombre d'images": "2 872", "Proportion": "15 %"},
        {"Split": "**Total**", "Nombre d'images": "**19 138**", "Proportion": "**100 %**"},
    ]

    st.dataframe(pd.DataFrame(volume_data), hide_index=True, use_container_width=True)

    st.markdown(
        "**Nombre moyen d'images par classe** : environ 640 images en train. Pour du "
        "transfer learning à partir d'ImageNet, ce volume est honorable mais limite pour "
        "les classes minoritaires."
    )

    st.subheader("Le problème du déséquilibre")

    st.warning(
        "**Ratio de déséquilibre 1 à 62** entre la classe majoritaire et la classe "
        "minoritaire. C'est un déséquilibre sévère qui explique l'écart entre accuracy "
        "et F1 macro."
    )

    st.markdown(
        """
**Comment on a tenté de compenser** :

- **WeightedRandomSampler** activé pendant l'entraînement : les classes minoritaires sont
  tirées plus souvent que les classes majoritaires pour équilibrer artificiellement l'exposition.
- Cette compensation permet de gagner environ 8 points de F1 macro par rapport à un
  entraînement naïf, mais elle a ses limites : les classes pauvres restent les mêmes images
  vues encore et encore.

**Ce qui manque vraiment** :

- Plus de données réelles sur les classes minoritaires (iNaturalist, MushroomObserver)
- Augmentation différentielle plus agressive sur ces classes
- Audit qualité des annotations par un mycologue
        """
    )

    st.subheader("Pipeline d'augmentation actuel")

    st.markdown("Sept transformations appliquées pendant le training, dans cet ordre :")

    aug_data = [
        {
            "Étape": "1",
            "Transformation": "RandomResizedCrop",
            "Paramètres": "scale=(0.7, 1.0)",
            "Effet": "Zoom et recadrage aléatoires (70 à 100 % de l'image)",
        },
        {
            "Étape": "2",
            "Transformation": "RandomHorizontalFlip",
            "Paramètres": "p=0.5",
            "Effet": "Retournement horizontal une fois sur deux",
        },
        {
            "Étape": "3",
            "Transformation": "RandomAffine",
            "Paramètres": "degrees=15, translate=(0.1, 0.1)",
            "Effet": "Rotation +/- 15 degrés et translation +/- 10 %",
        },
        {
            "Étape": "4",
            "Transformation": "ColorJitter",
            "Paramètres": "0.3, 0.3, 0.3, 0.1",
            "Effet": "Variations de luminosité, contraste, saturation (+/- 30 %) et teinte (+/- 10 %)",
        },
        {
            "Étape": "5",
            "Transformation": "ToTensor",
            "Paramètres": "—",
            "Effet": "Conversion en tenseur PyTorch",
        },
        {
            "Étape": "6",
            "Transformation": "Normalize",
            "Paramètres": "ImageNet mean / std",
            "Effet": "Normalisation aux statistiques ImageNet",
        },
        {
            "Étape": "7",
            "Transformation": "RandomErasing",
            "Paramètres": "p=0.25",
            "Effet": "Efface un rectangle aléatoire 25 % du temps (Cutout)",
        },
    ]

    st.dataframe(pd.DataFrame(aug_data), hide_index=True, use_container_width=True)

    st.info(
        "Ce pipeline est conforme aux standards modernes du deep learning pour la "
        "classification d'images. Il couvre les variations naturelles principales : "
        "angle, cadrage, orientation, luminosité, occlusion partielle."
    )


# ===========================================================================
# Onglet 5 - Axes d'amélioration
# ===========================================================================

with tab_improve:
    st.header("Axes d'amélioration identifiés")

    st.markdown(
        "Cette section liste les pistes d'amélioration prioritaires en cas de version 3 "
        "du modèle, classées par rapport bénéfice / effort estimé."
    )

    st.subheader("1. Approche data-centric AI (impact estimé : +2 à +4 points)")

    st.markdown(
        """
Concept popularisé par Andrew Ng : dans la majorité des projets ML déployés en production,
l'amélioration la plus rentable vient de la **qualité des données**, pas de la sophistication
du modèle.

**Actions concrètes pour Champy** :

- **Compléter les classes minoritaires** via iNaturalist (https://www.inaturalist.org),
  MushroomObserver (https://mushroomobserver.org) et GBIF. Objectif : ramener le ratio
  de déséquilibre de 1:62 à 1:5.
- **Audit qualité des annotations** : vérifier les confusions classiques entre espèces
  visuellement proches (genres Cortinarius, Russula).
- **Suppression des doublons** : vérifier qu'aucune image n'apparaît dans plusieurs splits.
        """
    )

    st.subheader("2. MixUp et CutMix (impact estimé : +1 à +3 points sur F1 macro)")

    st.markdown(
        """
Techniques d'augmentation qui mélangent deux images de classes différentes :

- **MixUp** : superposition pondérée de deux images (`lambda * A + (1 - lambda) * B`), avec étiquette mixte
- **CutMix** : on découpe un rectangle d'une image A et on le colle dans une image B,
  l'étiquette est pondérée par la surface relative

Particulièrement efficaces sur les datasets déséquilibrés. Demandent une légère modification
du DataLoader.
        """
    )

    st.subheader("3. Augmentation différentielle par classe (impact estimé : +1 à +2 points)")

    st.markdown(
        """
Le pipeline actuel applique les **mêmes** transformations à toutes les classes. Or les
classes minoritaires bénéficieraient d'augmentations **plus agressives** (rotations plus
larges, color jitter plus intense) sans risquer de dégrader les classes majoritaires.

Implémentation : Dataset custom qui inspecte la classe avant d'appliquer le transform.
        """
    )

    st.subheader("4. Test-Time Augmentation (impact estimé : +0,5 à +1,5 point)")

    st.markdown(
        """
À l'inférence, on présente plusieurs versions augmentées de la même image au modèle,
on récupère N prédictions, et on **moyenne** les probabilités. Cette technique réduit
la variance des prédictions et améliore légèrement la précision.

Coût : multiplication par N du temps d'inférence (N typiquement entre 4 et 8).
        """
    )

    st.subheader("5. Optimisation des epochs (impact estimé : +0,5 à +1,5 point)")

    st.markdown(
        """
La courbe val_acc de ConvNeXt-Tiny n'a pas vraiment plateauté à 40 epochs : la métrique
progressait encore légèrement à la fin. Pousser à 60 epochs avec une patience plus élevée
sur l'early stopping pourrait grappiller quelques fractions de points supplémentaires.
        """
    )

    st.subheader("Tableau récapitulatif des priorités")

    priority_data = [
        {
            "Piste": "Compléter les données minoritaires",
            "Impact estimé": "+2 à +4 pts",
            "Effort": "Élevé (scraping, nettoyage)",
            "Priorité": "1",
        },
        {
            "Piste": "MixUp et CutMix",
            "Impact estimé": "+1 à +3 pts",
            "Effort": "Moyen (modification DataLoader)",
            "Priorité": "2",
        },
        {
            "Piste": "Augmentation différentielle",
            "Impact estimé": "+1 à +2 pts",
            "Effort": "Moyen (Dataset custom)",
            "Priorité": "3",
        },
        {
            "Piste": "Test-Time Augmentation",
            "Impact estimé": "+0,5 à +1,5 pt",
            "Effort": "Faible (modif inférence)",
            "Priorité": "4",
        },
        {
            "Piste": "Plus d'epochs",
            "Impact estimé": "+0,5 à +1,5 pt",
            "Effort": "Très faible (config)",
            "Priorité": "5",
        },
    ]

    st.dataframe(pd.DataFrame(priority_data), hide_index=True, use_container_width=True)


# ===========================================================================
# Onglet 6 - Narratif soutenance
# ===========================================================================

with tab_pitch:
    st.header("Narratifs prêts à l'emploi pour la soutenance")

    st.markdown(
        "Ces paragraphes sont **regénérés automatiquement** par le script "
        "`scripts/generate_analysis.py` à partir des valeurs MLflow live. "
        "Pas de désynchronisation possible entre les chiffres et les textes."
    )

    questions = [
        ("Question : Pourquoi avoir choisi ConvNeXt-Tiny plutôt que ResNet50 ?", "why_convnext"),
        ("Question : Expliquez votre stratégie de fine-tuning two-phase", "two_phase_strategy"),
        ("Question : Pourquoi cet écart entre accuracy et F1 macro ?", "accuracy_vs_f1"),
        ("Question : Quelles sont les limites de votre approche ?", "limitations"),
        ("Question : Que feriez-vous différemment avec plus de temps ?", "perspectives"),
        ("Question : Vos métriques sont-elles fiables et reproductibles ?", "reproducibility"),
    ]

    for title, key in questions:
        st.subheader(title)
        text = narratives.get(key, "(narratif non disponible dans ce snapshot)")
        st.markdown(f"> {text}")

    st.divider()
    st.caption(
        "Ces narratifs sont des points d'ancrage. Ils ne remplacent pas la pratique orale, "
        "mais fournissent la structure et le vocabulaire technique attendus par le jury."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Document généré dans le cadre de la préparation TFE Champy Classifier — "
    "Master AI DataScientest / Mines Paris PSL — Promotion 2026. "
    "Pour régénérer cette analyse à partir de MLflow : `python -m scripts.generate_analysis`"
)
