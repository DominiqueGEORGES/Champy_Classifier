"""Page Streamlit : détection de drift avec Evidently AI.

Permet de générer un rapport Evidently à la demande en comparant
la distribution des prédictions récentes (PredictionStore SQLite,
Bloc M2) avec la baseline calculée sur le test set
(``monitoring/baseline_reference.json``, Bloc M3 baseline_snapshot.py).

Source de vérité : aucune valeur n'est écrite en dur. La baseline est
lue dynamiquement, les rapports passés sont scannés au runtime, et la
génération d'un nouveau rapport invoque ``monitoring/run_drift_report.py``
via subprocess.

La section 3 affiche d'abord une lecture en clair du rapport (verdict, chiffres,
explication des termes), puis le rapport technique Evidently replié en dessous.
"""

from __future__ import annotations

# =====================================================================
# Imports standards
# =====================================================================
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =====================================================================
# Setup chemin projet
# =====================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =====================================================================
# Imports tiers
# =====================================================================

import streamlit as st

# =====================================================================
# Imports projet
# =====================================================================
from demo import auth
from demo.lib.drift_utils import DriftVerdict, build_drift_verdict

# =====================================================================
# Constantes
# =====================================================================

# Chemins (le repo layout est figé, pas d'override env nécessaire)
BASELINE_PATH = _PROJECT_ROOT / "monitoring" / "baseline_reference.json"
REPORTS_DIR = _PROJECT_ROOT / "monitoring" / "reports"
RUN_DRIFT_SCRIPT = _PROJECT_ROOT / "monitoring" / "run_drift_report.py"

# Mapping mois français pour formatage lisible (évite la dépendance
# à la locale système qui peut ne pas être installée dans le container).
MOIS_FR = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}

# Explication des termes techniques, en langage humain.
_TERM_EXPLANATION = """
Le modèle a appris à reconnaître les champignons sur un jeu d'images précis :
un certain mélange d'espèces, de cadrages, de lumières. C'est sa zone de confort.
Tant que les photos reçues en production ressemblent à celles-là, ses réponses
restent fiables.

La **dérive** mesure l'écart entre ce que le modèle a connu (la **référence**,
calculée sur le test set) et ce qu'il reçoit en ce moment (le **flux actuel**).
Quand l'écart devient trop grand, on lève une alerte, avant que la qualité ne se dégrade.

Deux mesures techniques apparaissent dans le rapport détaillé :

- **Distance de Wasserstein** : imaginez deux tas de sable, un par distribution.
  La distance, c'est l'effort minimal pour transformer le premier tas en le second.
  Plus elle est grande, plus les deux situations sont éloignées.
- **Divergence de Jensen-Shannon** : un score entre 0 et 1. Zéro signifie des
  distributions identiques, un signifie totalement différentes.
"""


# =====================================================================
# Fonctions utilitaires
# =====================================================================


def _parse_report_timestamp(path: Path) -> datetime | None:
    """Parse le timestamp depuis le nom de fichier ``drift_YYYYMMDD_HHMM.html``.

    Args:
        path: Chemin du fichier HTML.

    Returns:
        Timestamp parsé, ou None si le format ne correspond pas.
    """
    try:
        return datetime.strptime(path.stem.replace("drift_", ""), "%Y%m%d_%H%M")
    except ValueError:
        return None


def _format_date_fr(ts: datetime) -> str:
    """Formate un timestamp en français lisible.

    Args:
        ts: Timestamp à formater.

    Returns:
        Chaîne du type ``21 mai 2026 à 22h35``.
    """
    return f"{ts.day} {MOIS_FR[ts.month]} {ts.year} à {ts:%Hh%M}"


def _format_report_label(path: Path) -> str:
    """Formate un nom de rapport en libellé lisible pour le selectbox.

    Args:
        path: Chemin du fichier HTML.

    Returns:
        Libellé ``YYYY-MM-DD HH:MM (drift_*.html)``.
    """
    ts = _parse_report_timestamp(path)
    if ts is None:
        return path.name
    return f"{ts:%Y-%m-%d %H:%M}  ({path.name})"


def _safe_load_baseline() -> tuple[dict, dict]:
    """Charge les blocs ``metadata`` et ``global`` de la baseline.

    Returns:
        Le couple ``(metadata, global)``, ou ``({}, {})`` si indisponible.
    """
    if not BASELINE_PATH.exists():
        return {}, {}
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metadata", {}), data.get("global", {})
    except (OSError, json.JSONDecodeError):
        return {}, {}


def _load_report_summary(html_path: Path) -> dict | None:
    """Charge le résumé chiffré associé à un rapport, s'il existe.

    Le résumé est un JSON compagnon, même nom que le HTML mais en ``.json``,
    déposé par ``run_drift_report.py``. Il porte au minimum ``n_current`` et
    ``n_reference``.

    Args:
        html_path: Chemin du rapport HTML sélectionné.

    Returns:
        Le dictionnaire de résumé, ou None si absent ou illisible.
    """
    summary_path = html_path.with_suffix(".json")
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _render_verdict_banner(verdict: DriftVerdict) -> None:
    """Affiche le bandeau de verdict avec la couleur adaptée au statut.

    Args:
        verdict: Le verdict de dérive à afficher.
    """
    message = f"**{verdict.headline}**\n\n{verdict.detail}"
    if verdict.status == "drift":
        st.error(message)
    elif verdict.status == "stable":
        st.success(message)
    elif verdict.status == "no_data":
        st.info(message)
    else:
        st.warning(message)


def _render_key_figures(verdict: DriftVerdict) -> None:
    """Affiche les chiffres clés du verdict sous forme de métriques.

    Args:
        verdict: Le verdict de dérive à afficher.
    """
    col_current, col_reference, col_share = st.columns(3)
    col_current.metric("Prédictions analysées", verdict.n_current)
    col_reference.metric("Jeu de référence", verdict.n_reference)
    if verdict.drift_share is not None:
        col_share.metric("Colonnes en dérive", f"{verdict.drift_share:.0%}")
    else:
        col_share.metric("Colonnes en dérive", "n/a")


def _render_clear_reading(summary: dict | None) -> None:
    """Affiche la lecture en clair du rapport sélectionné.

    Si un résumé chiffré est disponible, affiche le verdict, les chiffres clés
    et la tendance de confiance. Sinon, indique comment l'activer. Dans tous les
    cas, propose l'explication des termes techniques.

    Args:
        summary: Résumé chiffré du rapport (compagnon JSON), ou None.
    """
    st.subheader("Lecture en clair")

    if summary is not None:
        meta, glob = _safe_load_baseline()
        verdict = build_drift_verdict(
            n_current=int(summary.get("n_current", 0)),
            n_reference=int(summary.get("n_reference", meta.get("n_images", 0))),
            dataset_drift=summary.get("dataset_drift"),
            drift_share=summary.get("share_of_drifted_columns"),
            confidence_current_mean=summary.get("confidence_current_mean"),
            confidence_reference_mean=summary.get(
                "confidence_reference_mean", glob.get("confidence_mean")
            ),
            confidence_current_std=summary.get("confidence_current_std"),
            confidence_reference_std=summary.get("confidence_reference_std"),
        )
        _render_verdict_banner(verdict)
        _render_key_figures(verdict)
        if verdict.confidence_trend:
            st.info(verdict.confidence_trend)
    else:
        st.caption(
            "Résumé chiffré indisponible pour ce rapport. Pour activer le verdict "
            "automatique, `run_drift_report.py` doit déposer un JSON compagnon "
            "(même nom, extension `.json`) à côté du HTML."
        )

    with st.expander("Que veut dire « dérive » ? (explication simple)"):
        st.markdown(_TERM_EXPLANATION)


# =====================================================================
# Authentification (lit access_policy.yaml)
# =====================================================================

auth.setup_page()

# =====================================================================
# Configuration de la page
# =====================================================================

st.set_page_config(page_title="11 - Drift", layout="wide")
st.title(":warning: Détection de drift")

st.markdown(
    """
La détection de drift surveille si la distribution des prédictions
en production s'écarte de la baseline (calculée sur le test set).
Le rapport HTML est généré par
[Evidently AI](https://github.com/evidentlyai/evidently) et combine :

- **Drift de classes** : la distribution des espèces prédites diverge-t-elle de la baseline ?
- **Drift de confiance** : les scores de confiance moyenne / P10 / P95 dérivent-ils ?
"""
)
st.divider()


# =====================================================================
# Section 1 : État de la baseline
# =====================================================================

st.header("1. Baseline de référence")

if not BASELINE_PATH.exists():
    st.error(
        f"Baseline manquante : `{BASELINE_PATH.relative_to(_PROJECT_ROOT)}`.\n\n"
        "La baseline est calculée une fois en faisant tourner l'inférence "
        "sur le test set. Lancer :\n\n"
        "```powershell\npython monitoring/baseline_snapshot.py\n```"
    )
else:
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    meta = baseline.get("metadata", {})
    glob = baseline.get("global", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images de référence", meta.get("n_images", 0))
    col2.metric("Top-1 accuracy", f"{glob.get('top1_accuracy', 0.0):.1%}")
    col3.metric("Confiance moyenne", f"{glob.get('confidence_mean', 0.0):.3f}")
    col4.metric(
        "Confiance P10 / P95",
        f"{glob.get('confidence_p10', 0.0):.2f} / {glob.get('confidence_p95', 0.0):.2f}",
    )
    st.caption(
        f"Baseline générée le {meta.get('generated_at', '?')[:19]} "
        f"sur le split `{meta.get('split', '?')}` du modèle "
        f"`{Path(meta.get('model_path', '?')).name}`."
    )

st.divider()


# =====================================================================
# Section 2 : Génération d'un nouveau rapport
# =====================================================================

st.header("2. Générer un nouveau rapport")

col_h, col_btn = st.columns([2, 1])
hours = col_h.slider(
    "Fenêtre temporelle (heures)",
    min_value=1,
    max_value=168,
    value=24,
    help="Prédictions stockées dans le SQLite sur cette fenêtre glissante.",
)
generate = col_btn.button(
    "Générer un rapport",
    disabled=not BASELINE_PATH.exists(),
    type="primary",
)

if generate:
    with st.spinner(f"Génération du rapport sur les {hours} dernières heures..."):
        result = subprocess.run(
            [
                sys.executable,
                str(RUN_DRIFT_SCRIPT),
                "--hours",
                str(hours),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode == 0:
        st.success("Rapport généré avec succès.")
        with st.expander("Logs de génération"):
            st.code(result.stderr or result.stdout, language="text")
    else:
        st.error("Échec de la génération. Voir les logs ci-dessous.")
        st.code(result.stderr or result.stdout, language="text")

st.divider()


# =====================================================================
# Section 3 : Rapports archivés
# =====================================================================

st.header("3. Rapports archivés")

reports = sorted(REPORTS_DIR.glob("drift_*.html"), reverse=True) if REPORTS_DIR.exists() else []

if not reports:
    st.info(
        "Aucun rapport pour le moment. Lance la génération ci-dessus après "
        "avoir envoyé quelques prédictions au service BentoML (le store "
        "SQLite alimente le rapport)."
    )
else:
    options = {p: _format_report_label(p) for p in reports}
    chosen = st.selectbox(
        "Sélectionner un rapport",
        options=list(options.keys()),
        index=0,
        format_func=lambda p: options[p],
    )
    st.caption(f"{len(reports)} rapport(s) disponibles dans `monitoring/reports/`")

    if chosen is not None:
        # =============================================================
        # Fiche d'identité du rapport sélectionné
        # =============================================================
        ts = _parse_report_timestamp(chosen)
        date_fr = _format_date_fr(ts) if ts else "Date inconnue"
        file_stat = chosen.stat()
        size_ko = file_stat.st_size / 1024

        st.subheader("Rapport sélectionné")

        col_a, col_b, col_c = st.columns([2, 2, 1])
        col_a.markdown(f"**Date** : {date_fr}")
        col_b.markdown(f"**Fichier** : `{chosen.name}`")
        col_c.markdown(f"**Taille** : {size_ko:.1f} Ko")

        # Lire une fois pour l'iframe et pour le bouton de téléchargement
        with open(chosen, encoding="utf-8") as f:
            html_content = f.read()

        st.download_button(
            label=":arrow_down: Télécharger ce rapport (HTML)",
            data=html_content,
            file_name=chosen.name,
            mime="text/html",
        )

        st.markdown("---")

        # =============================================================
        # Lecture en clair (verdict + explication), avant le rapport brut
        # =============================================================
        _render_clear_reading(_load_report_summary(chosen))

        # =============================================================
        # Rapport technique brut, replié pour ne pas noyer la lecture
        # =============================================================
        with st.expander("Rapport technique complet (Evidently)"):
            st.caption(
                "Note : la pagination en bas du tableau « Data Drift Summary » "
                "est native à Evidently et reste affichée même quand il n'y a "
                "qu'une seule page. C'est cosmétique."
            )
            st.components.v1.html(html_content, height=900, scrolling=True)
