"""Compare les predictions de l'API FastAPI et du service BentoML.

Verifie que les deux couches de serving (FastAPI sur port 8010 et BentoML
sur port 8020) produisent des predictions strictement identiques sur le
meme jeu d'images de test. Le modele ONNX servi est le meme dans les deux
cas, donc l'ecart attendu est nul (ou inferieur a un epsilon de 1e-6 pour
absorber les fluctuations de softmax flottantes).

Usage :
    python scripts/compare_fastapi_bentoml.py
    python scripts/compare_fastapi_bentoml.py --images data/raw/Mushrooms_images/100016.jpg ...
    python scripts/compare_fastapi_bentoml.py --epsilon 1e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from loguru import logger

DEFAULT_FASTAPI_URL = "http://127.0.0.1:8010"
DEFAULT_BENTOML_URL = "http://127.0.0.1:8020"
DEFAULT_EPSILON = 1e-6

DEFAULT_IMAGES = [
    Path("data/raw/Mushrooms_images/100016.jpg"),
    Path("data/raw/Mushrooms_images/101905.jpg"),
    Path("data/raw/Mushrooms_images/157467.jpg"),
    Path("data/raw/Mushrooms_images/110460.jpg"),
]


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec URLs, liste d'images, et tolerance.
    """
    parser = argparse.ArgumentParser(
        description="Compare FastAPI vs BentoML sur les memes images de test."
    )
    parser.add_argument(
        "--fastapi-url",
        default=DEFAULT_FASTAPI_URL,
        help=f"URL de base de l'API FastAPI (defaut: {DEFAULT_FASTAPI_URL}).",
    )
    parser.add_argument(
        "--bentoml-url",
        default=DEFAULT_BENTOML_URL,
        help=f"URL de base du service BentoML (defaut: {DEFAULT_BENTOML_URL}).",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Chemins des images a tester (defaut: 4 images de test).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Tolerance sur les scores de confiance (defaut: {DEFAULT_EPSILON}).",
    )
    return parser.parse_args()


def call_fastapi(client: httpx.Client, url: str, image: Path) -> dict[str, float | str]:
    """Interroge l'endpoint /predict de l'API FastAPI.

    Args:
        client: Client HTTP partage.
        url: URL de base (sans /predict).
        image: Chemin de l'image a envoyer.

    Returns:
        Dictionnaire avec ``species`` et ``confidence`` du top-1.
    """
    with open(image, "rb") as f:
        resp = client.post(
            f"{url}/predict",
            files={"file": (image.name, f, "image/jpeg")},
            params={"top_n": 1},
            timeout=30,
        )
    resp.raise_for_status()
    top1 = resp.json()["predictions"][0]
    return {"species": top1["species"], "confidence": float(top1["confidence"])}


def call_bentoml(client: httpx.Client, url: str, image: Path) -> dict[str, float | str]:
    """Interroge l'endpoint /predict du service BentoML.

    Args:
        client: Client HTTP partage.
        url: URL de base (sans /predict).
        image: Chemin de l'image a envoyer.

    Returns:
        Dictionnaire avec ``species`` et ``confidence`` du top-1.
    """
    with open(image, "rb") as f:
        resp = client.post(
            f"{url}/predict",
            files={"image": (image.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    top1 = resp.json()["predictions"][0]
    return {"species": top1["species"], "confidence": float(top1["confidence"])}


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si parite atteinte sur toutes les images, 1 sinon.
    """
    args = parse_args()
    missing = [p for p in args.images if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"Image introuvable : {p}")
        return 1

    logger.info(f"Tolerance configuree : {args.epsilon}")
    logger.info(f"FastAPI : {args.fastapi_url}")
    logger.info(f"BentoML : {args.bentoml_url}")

    n_ok = 0
    n_fail = 0
    with httpx.Client() as client:
        for image in args.images:
            try:
                fastapi_pred = call_fastapi(client, args.fastapi_url, image)
                bentoml_pred = call_bentoml(client, args.bentoml_url, image)
            except Exception as exc:
                logger.error(f"[{image.name}] erreur reseau : {exc}")
                n_fail += 1
                continue

            same_species = fastapi_pred["species"] == bentoml_pred["species"]
            delta = abs(float(fastapi_pred["confidence"]) - float(bentoml_pred["confidence"]))
            same_score = delta <= args.epsilon

            status = "OK" if (same_species and same_score) else "FAIL"
            logger.info(
                f"[{status}] {image.name:<20} "
                f"FastAPI={fastapi_pred['species']} ({float(fastapi_pred['confidence']):.6f}) "
                f"| BentoML={bentoml_pred['species']} ({float(bentoml_pred['confidence']):.6f}) "
                f"| delta={delta:.2e}"
            )
            if same_species and same_score:
                n_ok += 1
            else:
                n_fail += 1

    logger.info("---")
    logger.info(f"Total : {n_ok} OK / {n_fail} FAIL sur {len(args.images)} images")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
