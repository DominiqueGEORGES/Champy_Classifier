"""Page Streamlit : detection de drift avec Evidently.

Permet de generer un rapport Evidently on-demand en comparant
la distribution des predictions recentes (PredictionStore SQLite,
Bloc M2) avec la baseline calculee sur le test set
(``monitoring/baseline_reference.json``, Bloc M3 baseline_snapshot.py).

Source de verite : aucune valeur n'est ecrite en dur. La baseline est
lue dynamiquement, les rapports passes sont scannes au runtime, et la
generation d'un nouveau rapport invoque ``monitoring/run_drift_report.py``
via subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="11 - Drift", layout="wide")
st.title(":warning: Detection de drift")

# --- Chemins (configurables via env si on change le layout du repo)
BASELINE_PATH = _PROJECT_ROOT / "monitoring" / "baseline_reference.json"
REPORTS_DIR = _PROJECT_ROOT / "monitoring" / "reports"
RUN_DRIFT_SCRIPT = _PROJECT_ROOT / "monitoring" / "run_drift_report.py"

st.markdown(
    """
La detection de drift surveille si la distribution des predictions
en production s'ecarte de la baseline (calculee sur le test set).
Le rapport HTML est genere par
[Evidently AI](https://github.com/evidentlyai/evidently) et combine :

- **Drift de classes** : la distribution des especes predites diverge-t-elle de la baseline ?
- **Drift de confiance** : les scores de confiance moyenne / P10 / P95 derivent-ils ?
"""
)
st.divider()

# =====================================================================
# Section 1 : Etat de la baseline
# =====================================================================
st.header("1. Baseline de reference")

if not BASELINE_PATH.exists():
    st.error(
        f"Baseline manquante : `{BASELINE_PATH.relative_to(_PROJECT_ROOT)}`.\n\n"
        "La baseline est calculee une fois en faisant tourner l'inference sur le "
        "test set. Lancer :\n\n"
        "```powershell\npython monitoring/baseline_snapshot.py\n```"
    )
else:
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    meta = baseline.get("metadata", {})
    glob = baseline.get("global", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images de reference", meta.get("n_images", 0))
    col2.metric("Top-1 accuracy", f"{glob.get('top1_accuracy', 0.0):.1%}")
    col3.metric("Confiance moyenne", f"{glob.get('confidence_mean', 0.0):.3f}")
    col4.metric(
        "Confiance P10 / P95",
        f"{glob.get('confidence_p10', 0.0):.2f} / {glob.get('confidence_p95', 0.0):.2f}",
    )
    st.caption(
        f"Baseline generee le {meta.get('generated_at', '?')[:19]} "
        f"sur le split `{meta.get('split', '?')}` du modele "
        f"`{Path(meta.get('model_path', '?')).name}`."
    )

st.divider()

# =====================================================================
# Section 2 : Generation d'un nouveau rapport
# =====================================================================
st.header("2. Generer un nouveau rapport")

col_h, col_btn = st.columns([2, 1])
hours = col_h.slider(
    "Fenetre temporelle (heures)",
    min_value=1,
    max_value=168,
    value=24,
    help="Predictions stockees dans le SQLite sur cette fenetre glissante.",
)
generate = col_btn.button(
    "Generer un rapport",
    disabled=not BASELINE_PATH.exists(),
    type="primary",
)
if generate:
    with st.spinner(f"Generation du rapport sur les {hours} dernieres heures..."):
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
        st.success("Rapport genere avec succes.")
        with st.expander("Logs de generation"):
            st.code(result.stderr or result.stdout, language="text")
    else:
        st.error("Echec de la generation. Voir les logs ci-dessous.")
        st.code(result.stderr or result.stdout, language="text")

st.divider()

# =====================================================================
# Section 3 : Liste des rapports passes + selecteur
# =====================================================================
st.header("3. Rapports archives")

reports = sorted(REPORTS_DIR.glob("drift_*.html"), reverse=True) if REPORTS_DIR.exists() else []
if not reports:
    st.info(
        "Aucun rapport pour le moment. Lance la generation ci-dessus apres avoir "
        "envoye quelques predictions au service BentoML (le store SQLite alimente "
        "le rapport)."
    )
else:

    def _format_report(p: Path) -> str:
        """Formate un nom de rapport en libelle lisible.

        Args:
            p: Chemin du fichier HTML.

        Returns:
            Libelle ``YYYY-MM-DD HH:MM (drift_*.html)`` pour le selecteur.
        """
        try:
            ts = datetime.strptime(p.stem.replace("drift_", ""), "%Y%m%d_%H%M")
            return f"{ts:%Y-%m-%d %H:%M}  ({p.name})"
        except ValueError:
            return p.name

    options = {p: _format_report(p) for p in reports}
    chosen = st.selectbox(
        "Selectionner un rapport",
        options=list(options.keys()),
        index=0,
        format_func=lambda p: options[p],
    )
    st.caption(f"{len(reports)} rapport(s) disponibles dans `monitoring/reports/`")

    if chosen is not None:
        with open(chosen, encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=900, scrolling=True)
