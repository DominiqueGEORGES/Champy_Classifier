"""Genere du trafic synthetique sur l'API d'inference pour alimenter Grafana.

Pioche N images de test stratifiees sur les especes pour garantir une
distribution variee dans les metriques Prometheus, puis envoie chaque
image en POST sur l'API FastAPI (port 8010) ou BentoML (port 8020 si
disponible). Utile pour valider visuellement les dashboards Grafana
apres provisioning.

Usage :
    python scripts/seed_grafana.py
    python scripts/seed_grafana.py --n 100 --target bentoml
    python scripts/seed_grafana.py --n 50 --target fastapi
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "Mushrooms_images"
SPLIT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"

TARGETS = {
    "fastapi": ("http://127.0.0.1:8010/predict", "file"),
    "bentoml": ("http://127.0.0.1:8020/predict", "image"),
}


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec ``n``, ``target``, et ``seed``.
    """
    parser = argparse.ArgumentParser(description="Seed Prometheus / Grafana avec du trafic.")
    parser.add_argument("--n", type=int, default=50, help="Nombre de predictions a envoyer.")
    parser.add_argument(
        "--target",
        choices=list(TARGETS),
        default="fastapi",
        help="API cible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed pour la selection des images.")
    return parser.parse_args()


def select_images(n: int, seed: int) -> list[Path]:
    """Selectionne ``n`` images stratifiees sur les especes du test set.

    Pioche au plus une image par espece tant que possible, puis recommence
    par espece si ``n`` excede le nombre de classes (30).

    Args:
        n: Nombre d'images a tirer.
        seed: Graine de reproductibilite.

    Returns:
        Liste de chemins d'images.
    """
    random.seed(seed)
    df = pd.read_csv(SPLIT_MANIFEST)
    test = df[df["split"] == "test"].copy()
    images: list[Path] = []
    species_pool = list(test["label"].unique())
    random.shuffle(species_pool)
    while len(images) < n:
        for species in species_pool:
            candidates = test[test["label"] == species]["path"].tolist()
            if not candidates:
                continue
            chosen = random.choice(candidates)
            matches = list(RAW_DIR.rglob(chosen))
            if matches:
                images.append(matches[0])
                if len(images) >= n:
                    break
    return images


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si toutes les requetes ont retourne 200.
    """
    args = parse_args()
    url, file_field = TARGETS[args.target]

    images = select_images(args.n, args.seed)
    logger.info(f"{len(images)} images selectionnees, cible : {url}")

    n_ok, n_fail = 0, 0
    start = time.perf_counter()
    with httpx.Client(timeout=30) as client:
        for i, image in enumerate(images, start=1):
            with open(image, "rb") as f:
                resp = client.post(
                    url,
                    files={file_field: (image.name, f, "image/jpeg")},
                )
            if resp.status_code == 200:
                top1 = resp.json()["predictions"][0]
                logger.info(
                    f"[{i:3d}/{len(images)}] {image.name:<20} -> "
                    f"{top1['species']} ({top1['confidence']:.2%})"
                )
                n_ok += 1
            else:
                logger.error(f"[{i:3d}] {image.name} -> HTTP {resp.status_code}")
                n_fail += 1

    duration = time.perf_counter() - start
    logger.info(
        f"Termine en {duration:.1f}s : {n_ok} OK / {n_fail} FAIL "
        f"({len(images) / duration:.1f} req/s)"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
