"""Page CI/CD : statuts GitHub Actions + rapports de tests et couverture.

Source des statuts : API GitHub publique (rate limit 60/h sans token, suffisant avec cache TTL 5min).
Source des rapports : fichiers locaux `reports/pytest.html` et `reports/coverage.xml`
  (generes par `invoke test` ou recuperes depuis les artefacts GitHub Actions).

Aucune valeur en dur : owner/repo/branche lus via env vars `CHAMPY_GITHUB_*`.

Sections :
1. Statut du dernier run (badge + metriques principales)
2. Historique des derniers runs (statistiques + graphique + tableau)
3. Rapport pytest HTML embarque (self-contained)
4. Couverture parsee depuis coverage.xml (tableau par module)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# Configuration (env-driven, defauts raisonnables)
# =====================================================================

GITHUB_OWNER = os.environ.get("CHAMPY_GITHUB_OWNER", "LoicFocraud")
GITHUB_REPO = os.environ.get("CHAMPY_GITHUB_REPO", "Champy_Classifier")
GITHUB_BRANCH = os.environ.get("CHAMPY_GITHUB_BRANCH", "dev-dominique")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # optionnel, augmente le rate limit a 5000/h

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
PYTEST_HTML = REPORTS_DIR / "pytest.html"
COVERAGE_XML = REPORTS_DIR / "coverage.xml"
COVERAGE_HTML = REPORTS_DIR / "coverage" / "index.html"

CACHE_TTL = 300  # 5 minutes

GREEN = "#1F4E3D"
AMBER = "#B85C00"
CREAM = "#FAFAF5"


# =====================================================================
# Helpers
# =====================================================================


@st.cache_data(ttl=CACHE_TTL)
def fetch_workflow_runs(per_page: int = 30) -> list[dict]:
    """Recupere les derniers runs CI via l'API GitHub.

    Retourne une liste vide en cas d'erreur reseau (affiche un warning).
    """
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs"
    params: dict[str, str | int] = {"per_page": per_page, "branch": GITHUB_BRANCH}
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json().get("workflow_runs", [])
    except httpx.HTTPError as exc:
        st.warning(f"Impossible de joindre l'API GitHub : {exc}")
        return []


def status_to_label(status: str, conclusion: str | None) -> tuple[str, str]:
    """Convertit le couple status/conclusion en (symbole, libelle court)."""
    if status == "in_progress":
        return ("EN COURS", AMBER)
    if status == "queued":
        return ("EN ATTENTE", AMBER)
    if status == "completed":
        if conclusion == "success":
            return ("SUCCES", GREEN)
        if conclusion == "failure":
            return ("ECHEC", "#B00020")
        if conclusion == "cancelled":
            return ("ANNULE", "#666666")
        return (conclusion.upper() if conclusion else "INCONNU", "#666666")
    return ("INCONNU", "#666666")


def compute_duration(start: str, end: str) -> str:
    """Calcule la duree d'un run a partir des timestamps ISO 8601."""
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        delta = end_dt - start_dt
        total = int(delta.total_seconds())
        return f"{total // 60}m{total % 60:02d}s"
    except (ValueError, AttributeError):
        return "-"


def runs_to_dataframe(runs: list[dict]) -> pd.DataFrame:
    """Transforme la liste de runs API en DataFrame pret a afficher."""
    rows = []
    for run in runs:
        status = run.get("status", "")
        conclusion = run.get("conclusion")
        label, _ = status_to_label(status, conclusion)
        rows.append(
            {
                "Statut": label,
                "Commit": (run.get("display_title") or "")[:70],
                "Auteur": (run.get("triggering_actor") or {}).get("login", "-"),
                "Branche": run.get("head_branch", ""),
                "Duree": compute_duration(
                    run.get("run_started_at", ""),
                    run.get("updated_at", ""),
                ),
                "Date": (run.get("run_started_at") or "")[:16].replace("T", " "),
                "Lien": run.get("html_url", ""),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def parse_coverage_xml(path: Path) -> dict | None:
    """Parse coverage.xml (Cobertura format) pour produire un resume.

    Retourne None si le fichier n'existe pas.
    """
    if not path.exists():
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        line_rate = float(root.get("line-rate", "0")) * 100
        branch_rate = float(root.get("branch-rate", "0")) * 100
        lines_covered = int(root.get("lines-covered", "0"))
        lines_valid = int(root.get("lines-valid", "0"))

        # Par package (= dossier)
        packages = []
        for pkg in root.findall(".//package"):
            packages.append(
                {
                    "Module": pkg.get("name", ""),
                    "Couverture (%)": round(float(pkg.get("line-rate", "0")) * 100, 1),
                    "Lignes couvertes": _count_lines_in_pkg(pkg, covered=True),
                    "Lignes totales": _count_lines_in_pkg(pkg, covered=False),
                }
            )

        return {
            "line_rate": line_rate,
            "branch_rate": branch_rate,
            "lines_covered": lines_covered,
            "lines_valid": lines_valid,
            "packages": pd.DataFrame(packages).sort_values("Couverture (%)", ascending=False),
        }
    except (ET.ParseError, ValueError) as exc:
        st.warning(f"Erreur de parsing coverage.xml : {exc}")
        return None


def _count_lines_in_pkg(pkg: ET.Element, covered: bool) -> int:
    """Compte les lignes (couvertes ou totales) d'un package coverage."""
    total = 0
    for cls in pkg.findall(".//class"):
        for line in cls.findall(".//line"):
            if covered:
                if int(line.get("hits", "0")) > 0:
                    total += 1
            else:
                total += 1
    return total


def render_status_badge(label: str, color: str, size: str = "large") -> str:
    """Genere une carte HTML coloree avec le label centre."""
    font_size = "2.5rem" if size == "large" else "1.2rem"
    padding = "1.5rem" if size == "large" else "0.5rem"
    return f"""
    <div style="
        background-color: {color};
        color: white;
        padding: {padding};
        border-radius: 8px;
        text-align: center;
        font-size: {font_size};
        font-weight: 600;
        font-family: 'Manrope', sans-serif;
    ">
        {label}
    </div>
    """


# =====================================================================
# Mise en page
# =====================================================================

st.set_page_config(page_title="CI/CD", page_icon=":rocket:", layout="wide")
st.title("CI/CD - Statuts GitHub Actions + rapports")

st.caption(
    f"Repo : `github.com/{GITHUB_OWNER}/{GITHUB_REPO}` (branche `{GITHUB_BRANCH}`). "
    f"Cache TTL : {CACHE_TTL}s. "
    f"{'Token detecte (rate limit 5000/h)' if GITHUB_TOKEN else 'Pas de token (rate limit 60/h)'}."
)

if st.button("Rafraichir maintenant", help="Vide le cache et re-interroge l'API GitHub"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Section 1 : Dernier run
# ---------------------------------------------------------------------
st.header("Dernier run")

runs = fetch_workflow_runs(per_page=30)

if not runs:
    st.error("Aucun run recupere. Verifier la connexion ou le rate limit GitHub.")
    st.stop()

last_run = runs[0]
last_status = last_run.get("status", "")
last_conclusion = last_run.get("conclusion")
last_label, last_color = status_to_label(last_status, last_conclusion)

col_badge, col_metrics = st.columns([1, 2])

with col_badge:
    st.markdown(render_status_badge(last_label, last_color), unsafe_allow_html=True)

with col_metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Duree",
        compute_duration(
            last_run.get("run_started_at", ""),
            last_run.get("updated_at", ""),
        ),
    )
    c2.metric("Branche", last_run.get("head_branch", "-"))
    c3.metric(
        "Auteur",
        (last_run.get("triggering_actor") or {}).get("login", "-"),
    )

st.caption(f"Commit : `{(last_run.get('display_title') or '-')[:90]}`")
st.markdown(f"[Voir les logs sur GitHub]({last_run.get('html_url', '#')})")

st.divider()

# ---------------------------------------------------------------------
# Section 2 : Historique
# ---------------------------------------------------------------------
st.header(f"Historique ({len(runs)} derniers runs)")

df = runs_to_dataframe(runs)

if not df.empty:
    total = len(df)
    successes = (df["Statut"] == "SUCCES").sum()
    failures = (df["Statut"] == "ECHEC").sum()
    rate = successes / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", total)
    c2.metric("Reussites", successes)
    c3.metric("Echecs", failures)
    c4.metric("Taux de succes", f"{rate:.1f}%")

    # Graphique : taux de succes par jour (derniers 14)
    df_chart = df.copy()
    df_chart["DateJour"] = pd.to_datetime(df_chart["Date"], errors="coerce").dt.date
    df_chart["IsSuccess"] = (df_chart["Statut"] == "SUCCES").astype(int)
    df_chart = df_chart.dropna(subset=["DateJour"])
    df_daily = (
        df_chart.groupby("DateJour")
        .agg(total=("IsSuccess", "size"), success=("IsSuccess", "sum"))
        .reset_index()
        .tail(14)
    )
    df_daily["TauxSucces"] = (df_daily["success"] / df_daily["total"] * 100).round(1)

    if not df_daily.empty:
        fig = px.bar(
            df_daily,
            x="DateJour",
            y="TauxSucces",
            title="Taux de succes par jour (14 derniers jours)",
            labels={"DateJour": "Jour", "TauxSucces": "Succes (%)"},
            color="TauxSucces",
            color_continuous_scale=[AMBER, CREAM, GREEN],
            range_color=[0, 100],
        )
        fig.update_layout(height=320, margin={"t": 50, "b": 30, "l": 30, "r": 10})
        st.plotly_chart(fig, use_container_width=True)

    # Tableau detaille
    st.subheader("Detail des runs")
    st.dataframe(
        df.drop(columns=["Lien"]),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------
# Section 3 : Rapport pytest HTML
# ---------------------------------------------------------------------
st.header("Rapport de tests local")

if PYTEST_HTML.exists():
    rel_path = PYTEST_HTML.relative_to(REPO_ROOT)
    mtime = datetime.fromtimestamp(PYTEST_HTML.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Source : `{rel_path}` (genere le {mtime}). Regenerer avec `invoke test`.")
    with PYTEST_HTML.open(encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=700, scrolling=True)
else:
    st.warning(
        f"Aucun rapport de tests local trouve a `{PYTEST_HTML.relative_to(REPO_ROOT)}`. "
        "Lancer `invoke test` pour le generer."
    )

st.divider()

# ---------------------------------------------------------------------
# Section 4 : Couverture
# ---------------------------------------------------------------------
st.header("Couverture des tests")

cov_data = parse_coverage_xml(COVERAGE_XML)

if cov_data is None:
    st.warning(
        f"Aucun rapport de couverture local trouve a `{COVERAGE_XML.relative_to(REPO_ROOT)}`. "
        "Lancer `invoke test` pour le generer."
    )
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Couverture globale (lignes)", f"{cov_data['line_rate']:.1f}%")
    c2.metric("Lignes couvertes", f"{cov_data['lines_covered']:,}".replace(",", " "))
    c3.metric("Lignes totales", f"{cov_data['lines_valid']:,}".replace(",", " "))

    st.subheader("Couverture par module")
    st.dataframe(
        cov_data["packages"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Couverture (%)": st.column_config.ProgressColumn(
                "Couverture (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    if COVERAGE_HTML.exists():
        st.caption(
            f"Rapport HTML detaille : `{COVERAGE_HTML.relative_to(REPO_ROOT)}` "
            "(ouvrir directement dans le navigateur pour la navigation interactive)."
        )
