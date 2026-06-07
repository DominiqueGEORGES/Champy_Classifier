"""Deploiement depuis le registre MLflow vers le Model Store BentoML.

Recupere la version en Staging de 'champy-classifier' dans le registre MLflow,
l'exporte en ONNX, puis l'importe dans le Model Store BentoML. Les metadonnees
(version, architecture, accuracy, run_id) sont lues automatiquement depuis le
registre et le run associe : plus aucun flag a saisir a la main, plus aucune
copie de fichier entre machines.

A lancer sur la machine qui heberge le Model Store BentoML servi (le NUC3),
apres un entrainement qui a publie une version au registre.

Usage:
    python -m scripts.deploy_from_registry
    python -m scripts.deploy_from_registry --stage Staging
"""

from __future__ import annotations

import argparse

import mlflow.pytorch
from loguru import logger
from mlflow import MlflowClient
from scripts.import_model_to_bentoml import import_model

from src.config import MODELS_DIR, get_training_config
from src.models.export_onnx import compare_outputs, export_to_onnx, save_class_names, validate_onnx

REGISTERED_MODEL = "champy-classifier"


def main() -> int:
    """Point d'entree CLI : registre -> ONNX -> Model Store BentoML.

    Returns:
        Code de sortie : 0 si succes, 1 sinon.
    """
    parser = argparse.ArgumentParser(
        description="Deploie la version du registre MLflow vers le Model Store BentoML."
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        help="Stage du registre a deployer (Staging, Production).",
    )
    parser.add_argument(
        "--name",
        default="champy_classifier",
        help="Nom du modele dans le Model Store BentoML.",
    )
    args = parser.parse_args()

    config = get_training_config()
    client = MlflowClient()

    # 1. Resoudre la version du registre pour le stage demande, et lire ses
    #    metadonnees depuis le run associe (architecture, accuracy, run_id).
    versions = client.get_latest_versions(REGISTERED_MODEL, stages=[args.stage])
    if not versions:
        logger.error(f"Aucune version en {args.stage} pour le modele {REGISTERED_MODEL}.")
        return 1

    model_version = versions[0]
    run = client.get_run(model_version.run_id)
    architecture = run.data.params.get("model_name", "unknown")
    accuracy = float(run.data.metrics.get("test_accuracy", 0.0))
    logger.info(
        f"Version {model_version.version} ({args.stage}) - run {model_version.run_id} - "
        f"architecture {architecture} - accuracy {accuracy:.4f}"
    )

    # 2. Charger le modele PyTorch directement depuis le registre (pas de fichier
    #    rapatrie : MLflow telecharge l'artefact depuis MinIO).
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL}/{args.stage}")
    model.to("cpu").eval()

    # 3. Exporter en ONNX, valider, comparer PyTorch vs ONNX (fonctions existantes).
    onnx_path = MODELS_DIR / f"champy_{architecture}_v{model_version.version}.onnx"
    export_to_onnx(model, onnx_path, image_size=config.image_size)
    validate_onnx(onnx_path)
    compare_outputs(model, onnx_path, image_size=config.image_size)

    # 4. Generer la liste des especes (lue depuis le manifest de curation).
    class_names_path = MODELS_DIR / "class_names.json"
    save_class_names(class_names_path, num_classes=config.num_classes)

    # 5. Importer dans le Model Store BentoML, metadonnees renseignees
    #    automatiquement (plus de flags manuels, plus d'ecrasement distrait).
    tag = import_model(
        onnx_path=onnx_path,
        class_names_path=class_names_path,
        name=args.name,
        version=f"v{model_version.version}",
        architecture=architecture,
        accuracy=accuracy,
        mlflow_run_id=model_version.run_id,
    )
    logger.success(f"Deploiement termine : {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
