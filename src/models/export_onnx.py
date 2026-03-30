"""Export du modele PyTorch au format ONNX pour l'inference en production.

Charge le meilleur checkpoint (.pt), cree le modele ResNet50 avec la meme
architecture que le training, exporte en ONNX (opset 17), valide avec
onnx.checker, et optionnellement compare les predictions PyTorch vs ONNX.

Usage:
    python -m src.models.export_onnx
    python -m src.models.export_onnx --checkpoint models/best_model.pt --output models/best_model.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger

from src.config import MODELS_DIR, get_training_config
from src.models.resnet import create_resnet50


def load_checkpoint(
    checkpoint_path: Path,
    num_classes: int = 30,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Charge un checkpoint PyTorch et reconstruit le modele.

    Args:
        checkpoint_path: Chemin vers le fichier .pt du checkpoint.
        num_classes: Nombre de classes (doit correspondre au training).
        device: Device cible. Si None, utilise CPU.

    Returns:
        Modele PyTorch en mode evaluation, poids charges.

    Raises:
        FileNotFoundError: Si le checkpoint n'existe pas.
    """
    if not checkpoint_path.exists():
        msg = f"Checkpoint introuvable : {checkpoint_path}"
        raise FileNotFoundError(msg)

    device = device or torch.device("cpu")

    # Recreer le modele avec la meme architecture que le training
    model = create_resnet50(num_classes=num_classes, pretrained=False)

    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch", "?")
    best_score = checkpoint.get("best_score", "?")
    logger.info(
        f"Checkpoint charge : {checkpoint_path} " f"(epoch={epoch}, best_score={best_score})"
    )
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """Exporte le modele PyTorch au format ONNX.

    Utilise des axes dynamiques pour le batch size, permettant
    l'inference sur des batches de taille variable.

    Args:
        model: Modele PyTorch en mode eval.
        output_path: Chemin de sortie du fichier .onnx.
        image_size: Taille des images d'entree (carre).
        opset_version: Version de l'opset ONNX.

    Returns:
        Chemin du fichier ONNX sauvegarde.
    """
    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,  # Utiliser l'exporteur legacy (stable, pas de dep onnxscript)
    )

    logger.info(f"Modele ONNX exporte : {output_path}")
    logger.info(f"Taille : {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def validate_onnx(onnx_path: Path) -> bool:
    """Valide le fichier ONNX avec onnx.checker.

    Args:
        onnx_path: Chemin vers le fichier .onnx.

    Returns:
        True si le modele est valide.
    """
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info("Validation ONNX : OK")
    return True


def compare_outputs(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    image_size: int = 224,
    n_samples: int = 10,
) -> dict[str, float]:
    """Compare les sorties PyTorch et ONNX sur des entrees aleatoires.

    Genere n_samples images aleatoires et compare les logits
    des deux modeles pour verifier la coherence de l'export.

    Args:
        pytorch_model: Modele PyTorch en mode eval.
        onnx_path: Chemin vers le fichier ONNX.
        image_size: Taille des images.
        n_samples: Nombre d'echantillons de test.

    Returns:
        Dictionnaire avec 'max_abs_diff', 'mean_abs_diff', 'all_match'.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_diffs: list[float] = []

    for _ in range(n_samples):
        dummy = torch.randn(1, 3, image_size, image_size)

        # PyTorch
        with torch.no_grad():
            pt_output = pytorch_model(dummy).numpy()

        # ONNX Runtime
        ort_output = session.run(None, {input_name: dummy.numpy()})[0]

        diff = np.abs(pt_output - ort_output).max()
        max_diffs.append(float(diff))

    result = {
        "max_abs_diff": float(max(max_diffs)),
        "mean_abs_diff": float(np.mean(max_diffs)),
        "all_match": all(d < 1e-4 for d in max_diffs),
    }

    logger.info(
        f"Comparaison PyTorch vs ONNX ({n_samples} echantillons) : "
        f"max_diff={result['max_abs_diff']:.6f}, "
        f"match={'OK' if result['all_match'] else 'ECART DETECTE'}"
    )
    return result


def save_class_names(output_path: Path, num_classes: int = 30) -> Path:
    """Genere et sauvegarde la liste des noms de classes pour l'API.

    Lit le manifest de curation pour extraire les noms d'especes
    tries alphabetiquement (meme ordre que le label_map du Dataset).

    Args:
        output_path: Chemin de sortie du fichier JSON.
        num_classes: Nombre de classes attendu (verification).

    Returns:
        Chemin du fichier JSON sauvegarde.
    """
    from src.config import DATA_DIR

    curated_path = DATA_DIR / "curated_manifest.csv"
    if curated_path.exists():
        import pandas as pd

        df = pd.read_csv(curated_path)
        names = sorted(df["species"].unique().tolist())
    else:
        # Fallback : noms generiques
        names = [f"class_{i}" for i in range(num_classes)]
        logger.warning("curated_manifest.csv introuvable, noms generiques utilises")

    if len(names) != num_classes:
        logger.warning(f"Nombre de classes inattendu : {len(names)} vs {num_classes} attendu")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)
    logger.info(f"Noms de classes sauvegardes : {output_path} ({len(names)} classes)")
    return output_path


def main() -> None:
    """Point d'entree CLI pour l'export ONNX."""
    parser = argparse.ArgumentParser(description="Export du modele PyTorch en ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "best_model.pt"),
        help="Chemin vers le checkpoint PyTorch",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "best_model.onnx"),
        help="Chemin de sortie du fichier ONNX",
    )
    parser.add_argument("--opset", type=int, default=17, help="Version opset ONNX")
    args = parser.parse_args()

    config = get_training_config()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # 1. Charger le checkpoint
    model = load_checkpoint(checkpoint_path, num_classes=config.num_classes)

    # 2. Exporter en ONNX
    export_to_onnx(model, output_path, image_size=config.image_size, opset_version=args.opset)

    # 3. Valider
    validate_onnx(output_path)

    # 4. Comparer les sorties
    compare_outputs(model, output_path, image_size=config.image_size)

    # 5. Sauvegarder les noms de classes
    save_class_names(MODELS_DIR / "class_names.json", num_classes=config.num_classes)

    logger.info("Export ONNX termine avec succes")


if __name__ == "__main__":
    main()
