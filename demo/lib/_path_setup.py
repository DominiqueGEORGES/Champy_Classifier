"""Configuration du sys.path pour les imports Streamlit.

Streamlit execute chaque page comme un script independant.
Le repertoire racine du projet doit etre dans sys.path pour
que les imports 'from src...' et 'from demo...' fonctionnent.

Usage (en tete de chaque page et de app.py) :
    import demo.lib._path_setup  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

# Racine du projet = 3 niveaux au-dessus de ce fichier
# demo/lib/_path_setup.py -> demo/lib -> demo -> racine
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
