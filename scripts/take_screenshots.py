"""Capture les 12 pages Streamlit dans ``docs/screenshots/``.

Utilise Playwright en mode headless pour parcourir la demo Streamlit
servie sur le port configure (defaut http://localhost:8501) et
sauvegarder une capture PNG full-page de chaque page.

Pre-requis (a installer une seule fois) :

    pip install playwright
    python -m playwright install chromium

Usage :

    python scripts/take_screenshots.py
    python scripts/take_screenshots.py --base-url http://localhost:8502
    python scripts/take_screenshots.py --output docs/screenshots --wait 3.0

Le script attend ``--wait`` secondes apres chargement avant la capture
pour laisser Streamlit terminer ses ``st.spinner`` et requetes
externes (MLflow, Prometheus, ONNX Runtime). 3 secondes suffisent en
general ; augmenter si une page consomme MLflow / DagsHub avec
latence elevee.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "screenshots"
DEFAULT_BASE_URL = "http://localhost:8501"
DEFAULT_WAIT = 3.0
DEFAULT_VIEWPORT = (1440, 900)

# Mapping ``slug -> nom de fichier``. Les slugs sont les noms de fichiers
# des pages Streamlit dans demo/pages/ sans le prefixe numerique. Streamlit
# cree une URL ``/01_donnees_brutes`` (l'extension .py et les accents sont
# normalises). On utilise les noms ASCII pour le slug URL et un nom de
# fichier court et stable pour la sortie.
PAGES: list[tuple[str, str]] = [
    ("", "home.png"),  # Accueil = /
    ("01_données_brutes", "01_donnees_brutes.png"),
    ("02_nettoyage", "02_nettoyage.png"),
    ("03_augmentation", "03_augmentation.png"),
    ("04_split", "04_split.png"),
    ("05_entraînement", "05_entrainement.png"),
    ("06_évaluation", "06_evaluation.png"),
    ("07_model_registry", "07_model_registry.png"),
    ("08_prédiction", "08_prediction.png"),
    ("09_api", "09_api.png"),
    ("10_monitoring", "10_monitoring.png"),
    ("11_drift", "11_drift.png"),
    ("12_infrastructure", "12_infrastructure.png"),
]


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec ``base_url``, ``output``, ``wait``, ``viewport``.
    """
    parser = argparse.ArgumentParser(
        description="Capture les 12 pages Streamlit (Playwright headless).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL de base du Streamlit (sans slash final).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Repertoire de sortie pour les PNG.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT,
        help="Secondes a attendre apres chargement avant la capture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_VIEWPORT[0],
        help="Largeur du viewport en pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_VIEWPORT[1],
        help="Hauteur du viewport en pixels (la capture est full-page).",
    )
    return parser.parse_args()


def _check_playwright_available() -> None:
    """Verifie que Playwright est installe et que chromium est present.

    Affiche un message d'erreur explicite avec les commandes a lancer
    si l'un des deux manque, puis termine avec code 1.
    """
    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            "Playwright n'est pas installe. Installer avec :\n"
            "    pip install playwright\n"
            "    python -m playwright install chromium",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


async def _capture_pages(
    base_url: str,
    output_dir: Path,
    wait_seconds: float,
    viewport: tuple[int, int],
) -> int:
    """Lance Chromium headless et capture chaque page.

    Args:
        base_url: URL de base de Streamlit (ex: ``http://localhost:8501``).
        output_dir: Repertoire de sortie (cree si absent).
        wait_seconds: Secondes a attendre apres chargement de chaque page.
        viewport: Taille du viewport ``(width, height)``.

    Returns:
        Nombre de captures reussies.
    """
    from playwright.async_api import async_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
        )
        page = await context.new_page()
        for slug, filename in PAGES:
            url = f"{base_url}/{slug}".rstrip("/")
            output_path = output_dir / filename
            print(f"[{filename:<35}] {url}", flush=True)
            try:
                await page.goto(url, wait_until="networkidle", timeout=30_000)
                await page.wait_for_timeout(int(wait_seconds * 1000))
                await page.screenshot(path=str(output_path), full_page=True)
                n_ok += 1
            except Exception as exc:
                print(f"  ECHEC : {exc}", file=sys.stderr)
        await browser.close()
    return n_ok


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si toutes les captures ont reussi, 1 sinon.
    """
    args = parse_args()
    _check_playwright_available()

    print(
        f"Capture {len(PAGES)} pages depuis {args.base_url} vers {args.output.resolve()}",
        flush=True,
    )
    n_ok = asyncio.run(
        _capture_pages(
            base_url=args.base_url.rstrip("/"),
            output_dir=args.output,
            wait_seconds=args.wait,
            viewport=(args.width, args.height),
        )
    )
    print(f"Termine : {n_ok}/{len(PAGES)} captures.")
    return 0 if n_ok == len(PAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
