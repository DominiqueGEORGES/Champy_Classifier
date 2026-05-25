"""Validation isolee du modele PyTorch pour Grad-CAM.

Ce script charge le fichier .pt, verifie la structure du modele,
lance une prediction de test, calcule un Grad-CAM sur une image,
et sauvegarde une image overlay. Il ne touche pas a l'API en production.

Usage:
    python scripts/validate_pytorch_model.py path/to/image.jpg
    python scripts/validate_pytorch_model.py path/to/image.jpg --save-overlay overlay.png

Pre-requis:
    pip install grad-cam

Sortie attendue:
    - Affichage du top-5 PyTorch (devrait coincider avec top-5 ONNX)
    - Sauvegarde d'une image overlay montrant la heatmap superposee
    - Si la heatmap se concentre sur le champignon, c'est un succes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration (identique a src/serving/app.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "convnext_tiny_v2.0.0.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224
NUM_CLASSES = 30


# ---------------------------------------------------------------------------
# Chargement du modele
# ---------------------------------------------------------------------------
def load_pytorch_model() -> torch.nn.Module:
    """Charge le modele ConvNeXt-Tiny adapte a Champy (30 classes).

    Returns:
        Modele PyTorch en mode eval, pret pour l'inference et Grad-CAM.

    Raises:
        FileNotFoundError: si le fichier .pt n'existe pas.
        RuntimeError: si le state_dict ne charge pas correctement.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele PyTorch introuvable : {MODEL_PATH}")

    print(f"Chargement de {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)...")

    # Import differe pour eviter de charger torchvision si modele absent
    from torchvision.models import convnext_tiny

    # Reconstruction de la structure (sans poids ImageNet par defaut)
    model = convnext_tiny(weights=None)

    # Adaptation de la derniere couche aux 30 classes Champy
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, NUM_CLASSES)

    # Chargement du state_dict
    # On essaie weights_only=True (PyTorch 2.4+), sinon fallback
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # weights_only pas supporte ou checkpoint contient autre chose que des tensors
        state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Si le checkpoint est un dict avec une cle 'state_dict' ou 'model_state_dict'
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # Chargement
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Cles manquantes : {len(missing)} (echantillon : {missing[:3]})")
    if unexpected:
        print(f"  Cles inattendues : {len(unexpected)} (echantillon : {unexpected[:3]})")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Modele charge : {total_params:,} parametres")
    return model


# ---------------------------------------------------------------------------
# Preprocessing (identique a src/serving/app.py)
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocesse une image pour PyTorch ConvNeXt-Tiny.

    Applique le meme pipeline que src/serving/app.py.preprocess_image :
    Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet).

    Args:
        image_bytes: Contenu brut de l'image.

    Returns:
        Tuple (tensor pour le modele, image numpy [0,1] pour overlay).
    """
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Resize : cote le plus petit a 256px
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(h * 256 / w)
    else:
        new_h, new_w = 256, int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop 224x224
    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # Conversion en numpy [0, 1] - utile pour overlay
    arr = np.array(img, dtype=np.float32) / 255.0
    image_for_overlay = arr.copy()

    # Normalisation ImageNet
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW

    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor, image_for_overlay


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_top_k(
    model: torch.nn.Module, tensor: torch.Tensor, k: int = 5
) -> list[tuple[int, float]]:
    """Predit les top-K classes.

    Args:
        model: Modele PyTorch en mode eval.
        tensor: Tensor d'entree (1, 3, 224, 224).
        k: Nombre de predictions a retourner.

    Returns:
        Liste de tuples (class_id, probability) tries par probabilite decroissante.
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = torch.topk(probs, k=k)
    return [(int(idx), float(p)) for idx, p in zip(top_k.indices, top_k.values, strict=True)]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_class_id: int,
) -> np.ndarray:
    """Calcule la carte d'activation Grad-CAM pour une classe cible.

    Args:
        model: Modele PyTorch.
        tensor: Tensor d'entree (1, 3, 224, 224).
        target_class_id: Id de la classe pour laquelle expliquer la decision.

    Returns:
        Carte d'activation en grayscale, shape (224, 224), valeurs [0, 1].
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Pour ConvNeXt-Tiny (torchvision), la derniere couche conv est dans
    # model.features[-1]. Le dernier element de ce stage est un CNBlock.
    # On cible directement features[-1] qui represente la sortie du dernier stage.
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_id)]

    # cam() retourne un array shape (batch_size, H, W)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    return grayscale_cam


# ---------------------------------------------------------------------------
# Sauvegarde de l'overlay
# ---------------------------------------------------------------------------
def save_overlay(
    image_array: np.ndarray,
    grayscale_cam: np.ndarray,
    output_path: Path,
) -> None:
    """Sauvegarde une image avec heatmap superposee.

    Args:
        image_array: Image originale, shape (224, 224, 3), valeurs [0, 1].
        grayscale_cam: Carte Grad-CAM, shape (224, 224), valeurs [0, 1].
        output_path: Chemin du fichier PNG a creer.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    overlay = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
    Image.fromarray(overlay).save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Point d'entrée CLI : valide une prédiction PyTorch + génère un overlay Grad-CAM."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image_path", help="Chemin vers une image de test (JPEG ou PNG)")
    parser.add_argument(
        "--save-overlay",
        default="gradcam_validation.png",
        help="Fichier de sortie pour l'overlay (defaut : gradcam_validation.png)",
    )
    parser.add_argument(
        "--save-heatmap",
        default=None,
        help="Optionnel : fichier de sortie pour la heatmap brute en niveaux de gris",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Erreur : image introuvable : {image_path}", file=sys.stderr)
        return 1

    # Etape 1 : chargement du modele
    print("=" * 60)
    print("ETAPE 1 - Chargement du modele PyTorch")
    print("=" * 60)
    try:
        model = load_pytorch_model()
    except Exception as e:
        print(f"Erreur au chargement du modele : {e}", file=sys.stderr)
        return 2

    # Etape 2 : preprocessing
    print()
    print("=" * 60)
    print("ETAPE 2 - Preprocessing de l'image")
    print("=" * 60)
    print(f"Image : {image_path}")
    image_bytes = image_path.read_bytes()
    tensor, image_for_overlay = preprocess_image(image_bytes)
    print(f"  Tensor shape : {tuple(tensor.shape)}, dtype : {tensor.dtype}")
    print(f"  Image overlay shape : {image_for_overlay.shape}")

    # Etape 3 : prediction
    print()
    print("=" * 60)
    print("ETAPE 3 - Prediction PyTorch (top-5)")
    print("=" * 60)
    top5 = predict_top_k(model, tensor, k=5)

    # Chargement des noms de classes
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
        print(f"  Note : {CLASS_NAMES_PATH.name} introuvable, classes nommees class_N")

    for rank, (class_id, prob) in enumerate(top5, start=1):
        name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {rank}. {name:35s} {prob:.4f}")

    # Etape 4 : Grad-CAM
    target_class_id = top5[0][0]
    target_class_name = (
        class_names[target_class_id]
        if target_class_id < len(class_names)
        else f"class_{target_class_id}"
    )
    print()
    print("=" * 60)
    print("ETAPE 4 - Calcul Grad-CAM")
    print("=" * 60)
    print(f"Classe cible : {target_class_name} (id={target_class_id})")
    try:
        grayscale_cam = compute_gradcam(model, tensor, target_class_id)
    except Exception as e:
        print(f"Erreur Grad-CAM : {e}", file=sys.stderr)
        print(
            "Si l'erreur mentionne 'features[-1]' ou la structure du modele,",
            "il faut introspecter le modele : ajouter `print(model)` dans le script.",
            file=sys.stderr,
        )
        return 3
    print(f"  Heatmap shape : {grayscale_cam.shape}")
    print(f"  Heatmap range : [{grayscale_cam.min():.3f}, {grayscale_cam.max():.3f}]")

    # Etape 5 : sauvegarde
    print()
    print("=" * 60)
    print("ETAPE 5 - Sauvegarde des resultats")
    print("=" * 60)
    overlay_path = Path(args.save_overlay)
    save_overlay(image_for_overlay, grayscale_cam, overlay_path)
    print(f"  Overlay : {overlay_path.resolve()}")

    if args.save_heatmap:
        heatmap_path = Path(args.save_heatmap)
        heatmap_image = (grayscale_cam * 255).astype(np.uint8)
        Image.fromarray(heatmap_image).save(heatmap_path)
        print(f"  Heatmap brute : {heatmap_path.resolve()}")

    print()
    print("=" * 60)
    print("VALIDATION REUSSIE")
    print("=" * 60)
    print("Verification visuelle :")
    print(f"  Ouvre {overlay_path} et observe la heatmap.")
    print("  Si la zone chaude se concentre sur le champignon, le modele")
    print("  prend ses decisions sur les bons pixels. Si elle se concentre")
    print("  sur l'arriere-plan, il y a un biais a investiguer.")
    print()
    print("Coherence avec l'API ONNX :")
    print("  Compare le top-1 ci-dessus avec POST /predict sur la meme image.")
    print(f"  Les deux doivent retourner '{target_class_name}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
