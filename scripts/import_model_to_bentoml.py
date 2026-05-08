"""Importe un modele ONNX dans le Model Store BentoML.

Le script enregistre un fichier ONNX local dans le Model Store BentoML
local sous le tag ``<name>:<auto_id>`` avec ``latest`` qui pointe sur le
dernier import. Il est concu pour etre relance a chaque nouveau training
(une seule version par release de modele) et supporte plusieurs
architectures et versions cohabitant dans le meme Model Store.

Le fichier ``class_names.json`` correspondant est packagee dans les
``custom_objects`` du modele : le service BentoML retrouve le mapping
``index -> espece`` sans dependre du repo source. C'est important quand
on change de version de label_map entre deux trainings (ex: re-curation
qui retire ou renomme une espece).

Usage typique
-------------
::

    # Modele courant (defauts : ``models/best_model.onnx`` + ``class_names.json``)
    python scripts/import_model_to_bentoml.py

    # ConvNeXt v2.0.0 explicite (Bloc R0)
    python scripts/import_model_to_bentoml.py \
        --onnx-path models/convnext_tiny_v2.0.0.onnx \
        --version v2.0.0 \
        --architecture convnext_tiny \
        --accuracy 0.90 \
        --class-names-path models/class_names_v2.0.0.json

    # ResNet50 v1.0.0 (regenere ce week-end)
    python scripts/import_model_to_bentoml.py \
        --onnx-path models/resnet50_v1.0.0.onnx \
        --version v1.0.0 \
        --architecture resnet50 \
        --accuracy 0.84 \
        --mlflow-run-id 1e7b1dda43ca467ead7c2c887ffdbece

Note : depuis BentoML 1.4 le sous-module ``bentoml.onnx`` emet un
``BentoMLDeprecationWarning``. L'API ``save_model`` reste fonctionnelle.
La migration vers ``bentoml.models.create()`` (API generique) est
documentee dans PLAYBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx
from loguru import logger

# Racine du repo : on suppose que le script est lance depuis la racine ou
# depuis ``scripts/`` (pathlib resout les deux cas).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "best_model.onnx"
DEFAULT_CLASS_NAMES_PATH = REPO_ROOT / "models" / "class_names.json"
DEFAULT_MODEL_NAME = "champy_classifier"
DEFAULT_VERSION = "v2.0.0"
DEFAULT_ARCHITECTURE = "convnext_tiny"
DEFAULT_ACCURACY = 0.90


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Accepte les nouveaux flags Bloc R0 (``--onnx-path``, ``--accuracy``,
    ``--mlflow-run-id``) ainsi que l'ancien ``--model-path`` pour
    retrocompatibilite avec les scripts existants.

    Returns:
        Namespace avec les chemins et metadonnees du modele a importer.
    """
    parser = argparse.ArgumentParser(
        description="Importe un modele ONNX dans le Model Store BentoML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Sources : on accepte --onnx-path (nouveau) et --model-path (legacy)
    # pour ne pas casser les scripts ou Makefile qui referencent l'ancien
    # nom. Si les deux sont passes, --onnx-path l'emporte.
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Chemin du fichier ONNX a importer.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Alias retro-compatible de --onnx-path.",
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
    # Identite logique du modele : ces 3 champs partent en labels du
    # Model Store et sont visibles dans `bentoml models list`.
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help="Version logique du modele (ex: v1.0.0, v1.1.0, v2.0.0).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        help="Backbone du modele (resnet50, convnext_tiny, ...).",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=DEFAULT_ACCURACY,
        help="Accuracy test set (label informationnel, ex: 0.84, 0.88, 0.90).",
    )
    parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="ID du run MLflow (DagsHub) pour la tracabilite (optionnel).",
    )
    args = parser.parse_args()
    # Resolution de la source : --onnx-path prioritaire, sinon --model-path,
    # sinon defaut.
    args.onnx_path = args.onnx_path or args.model_path or DEFAULT_ONNX_PATH
    return args


def import_model(
    onnx_path: Path,
    class_names_path: Path,
    name: str,
    version: str,
    architecture: str,
    accuracy: float,
    mlflow_run_id: str | None = None,
) -> str:
    """Importe le modele ONNX dans le Model Store BentoML.

    Le script enregistre les metadonnees suivantes pour la tracabilite :

    - **labels** (visibles dans ``bentoml models list``) : ``version``,
      ``architecture``, ``accuracy``, eventuellement ``mlflow_run_id``.
    - **metadata** (lus par le service au runtime) : ``num_classes``,
      ``input_shape``, ``source_file``, ``accuracy``.
    - **custom_objects** : la liste des especes (``class_names``).

    Args:
        onnx_path: Chemin du fichier ONNX a importer.
        class_names_path: Chemin du JSON listant les especes.
        name: Nom du modele dans le Model Store.
        version: Valeur du label ``version``.
        architecture: Valeur du label ``architecture``.
        accuracy: Test accuracy du modele (0.0 a 1.0).
        mlflow_run_id: ID du run MLflow pour la tracabilite (optionnel).

    Returns:
        Tag complet du modele cree (ex: ``champy_classifier:abcd1234``).

    Raises:
        FileNotFoundError: Si le fichier ONNX ou le JSON est absent.
        ValueError: Si l'accuracy n'est pas dans [0, 1].
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"Modele ONNX introuvable : {onnx_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Fichier class_names.json introuvable : {class_names_path}")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy doit etre dans [0, 1], recu : {accuracy}")

    import bentoml  # import differe pour eviter le cout au moment de --help

    logger.info(f"Chargement du modele ONNX : {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))

    with open(class_names_path, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    logger.info(f"Classes lues : {len(class_names)} especes")

    # Labels = identite logique queryable, metadata = champs typiques du
    # service au runtime. On duplique 'accuracy' dans les deux pour qu'il
    # soit a la fois filtrable cote CLI et accessible cote service.
    labels: dict[str, str] = {
        "version": version,
        "architecture": architecture,
        "accuracy": f"{accuracy:.4f}",
    }
    if mlflow_run_id:
        labels["mlflow_run_id"] = mlflow_run_id

    metadata: dict[str, Any] = {
        "num_classes": len(class_names),
        "input_shape": [1, 3, 224, 224],
        "source_file": onnx_path.name,
        "accuracy": float(accuracy),
    }
    if mlflow_run_id:
        metadata["mlflow_run_id"] = mlflow_run_id

    logger.info(f"Enregistrement dans le Model Store BentoML sous '{name}'...")
    bento_model = bentoml.onnx.save_model(
        name,
        onnx_model,
        signatures={"run": {"batchable": True, "batch_dim": 0}},
        labels=labels,
        metadata=metadata,
        custom_objects={"class_names": class_names},
    )

    logger.success(f"Modele importe : {bento_model.tag}")
    logger.info(
        f"Architecture : {architecture}, version : {version}, "
        f"accuracy : {accuracy:.2%}"
        + (f", mlflow_run_id : {mlflow_run_id}" if mlflow_run_id else "")
    )
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
            onnx_path=args.onnx_path,
            class_names_path=args.class_names_path,
            name=args.name,
            version=args.version,
            architecture=args.architecture,
            accuracy=args.accuracy,
            mlflow_run_id=args.mlflow_run_id,
        )
    except (FileNotFoundError, ValueError) as exc:
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
