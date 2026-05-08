"""Preprocessing d'image pour l'inference ONNX (service BentoML).

Reproduit STRICTEMENT le pipeline des transforms val/test du Dataset PyTorch :
``Resize(256)`` -> ``CenterCrop(224)`` -> ``ToTensor`` -> ``Normalize(ImageNet)``.

Utilise uniquement Pillow + numpy : aucune dependance a torch n'est introduite
dans la couche de serving (l'API ONNX en production doit etre la plus legere
possible). La coherence avec le pipeline PyTorch est verifiee par un test
unitaire dedie (``tests/unit/test_bentoml_preprocessing.py``).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

# Statistiques ImageNet pour la normalisation (identique au pipeline val/test)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224
RESIZE_SIZE = 256


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesse une image pour l'inference ONNX.

    Applique le meme pipeline que les transforms val/test du Dataset PyTorch :
    Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet).

    Args:
        image_bytes: Contenu brut de l'image (JPEG/PNG).

    Returns:
        Array numpy de forme (1, 3, 224, 224), float32, normalise ImageNet.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return preprocess_pil(img)


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Preprocesse une image PIL deja chargee.

    Variante de ``preprocess_image`` utile lorsque l'image est deja
    decodee (par exemple par BentoML qui passe directement un objet PIL).

    Args:
        img: Image PIL en RGB.

    Returns:
        Array numpy de forme (1, 3, 224, 224), float32, normalise ImageNet.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize : cote le plus petit a 256px (preserve le ratio)
    w, h = img.size
    if w < h:
        new_w = RESIZE_SIZE
        new_h = int(h * RESIZE_SIZE / w)
    else:
        new_h = RESIZE_SIZE
        new_w = int(w * RESIZE_SIZE / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop 224x224
    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # ToTensor + Normalize (HWC float [0, 1] -> CHW normalise)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]
