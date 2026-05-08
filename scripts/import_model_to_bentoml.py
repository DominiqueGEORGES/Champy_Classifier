"""Importe le modele ONNX dans le Model Store BentoML.

Ce script enregistre ``models/best_model.onnx`` dans le Model Store local
de BentoML sous le tag ``champy_classifier:latest``. Il est conçu pour etre
relance a chaque nouveau training : un nouveau tag horodatee est cree
automatiquement par BentoML, et l'alias ``latest`` pointe vers la derniere
version.

Le fichier ``models/class_names.json`` est packagee a cote du modele (via
le ``custom_objects`` de BentoML) pour que le service puisse retrouver
le mapping ``index -> espece`` sans dependre du repo source.

Usage :
    python scripts/import_model_to_bentoml.py
    python scripts/import_model_to_bentoml.py --model-path models/best_model.onnx
    python scripts/import_model_to_bentoml.py --version 1.1.0 --architecture convnext_tiny

Note : depuis BentoML 1.4 le sous-module ``bentoml.onnx`` emet un
``BentoMLDeprecationWarning``. L'API ``save_model`` reste fonctionnelle.
La migration vers ``bentoml.models.create()`` (API generique) est documentee
dans PLAYBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
from loguru import logger

# Racine du repo : on suppose que le script est lance depuis la racine ou
# depuis ``scripts/`` (pathlib resout les deux cas).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_model.onnx"
DEFAULT_CLASS_NAMES_PATH = REPO_ROOT / "models" / "class_names.json"
DEFAULT_MODEL_NAME = "champy_classifier"
DEFAULT_VERSION = "1.0.0"
DEFAULT_ARCHITECTURE = "convnext_tiny"


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Returns:
        Namespace avec les chemins et metadonnees du modele a importer.
    """
    parser = argparse.ArgumentParser(
        description="Importe un modele ONNX dans le Model Store BentoML."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Chemin du fichier ONNX a importer.",
    )
    parser.add_argument(
        "--class-names-path",
        type=Path,
        default=DEFAULT_CLASS_NAMES_PATH,
        help="Chemin du JSON contenant la liste des classes.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Nom du modele dans le Model Store BentoML.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help="Version logique a attacher en label.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        help="Backbone du modele (label informationnel).",
    )
    return parser.parse_args()


def import_model(
    model_path: Path,
    class_names_path: Path,
    name: str,
    version: str,
    architecture: str,
) -> str:
    """Importe le modele ONNX dans le Model Store BentoML.

    Args:
        model_path: Chemin vers ``best_model.onnx``.
        class_names_path: Chemin vers ``class_names.json``.
        name: Nom du modele dans le Model Store.
        version: Valeur du label ``version``.
        architecture: Valeur du label ``architecture``.

    Returns:
        Tag complet du modele cree (ex: ``champy_classifier:abcd1234``).

    Raises:
        FileNotFoundError: Si le fichier ONNX ou le JSON est absent.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modele ONNX introuvable : {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Fichier class_names.json introuvable : {class_names_path}")

    import bentoml  # import differe pour eviter le cout au moment de --help

    logger.info(f"Chargement du modele ONNX : {model_path}")
    onnx_model = onnx.load(str(model_path))

    with open(class_names_path, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    logger.info(f"Classes lues : {len(class_names)} especes")

    logger.info(f"Enregistrement dans le Model Store BentoML sous '{name}'...")
    bento_model = bentoml.onnx.save_model(
        name,
        onnx_model,
        signatures={"run": {"batchable": True, "batch_dim": 0}},
        labels={"version": version, "architecture": architecture},
        metadata={
            "num_classes": len(class_names),
            "input_shape": [1, 3, 224, 224],
            "source_file": str(model_path.name),
        },
        custom_objects={"class_names": class_names},
    )

    logger.success(f"Modele importe : {bento_model.tag}")
    logger.info(f"Architecture : {architecture}, version : {version}")
    logger.info(f"Path local : {bento_model.path}")
    return str(bento_model.tag)


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si succes, 1 sinon.
    """
    args = parse_args()
    try:
        tag = import_model(
            model_path=args.model_path,
            class_names_path=args.class_names_path,
            name=args.name,
            version=args.version,
            architecture=args.architecture,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception(f"Echec de l'import du modele : {exc}")
        return 1

    logger.info(f"Tag complet : {tag}")
    logger.info("Pour servir le modele : bentoml serve src.serving_bentoml.service:ChampyService")
    return 0


if __name__ == "__main__":
    sys.exit(main())
