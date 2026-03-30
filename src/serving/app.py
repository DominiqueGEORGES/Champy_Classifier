"""Serveur FastAPI pour l'inference du classificateur de champignons.

Charge un modele ONNX et expose des endpoints REST pour la prediction,
le monitoring (Prometheus) et les health checks. Si le modele n'est pas
disponible, /health retourne {"status": "no_model"} et /predict retourne 503.

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger
from PIL import Image
from prometheus_client import generate_latest

from src.serving.middleware import (
    HTTP_ERRORS,
    PREDICTION_CONFIDENCE,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
    REQUESTS_TOTAL,
)
from src.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model.onnx"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Statistiques ImageNet pour la normalisation (identique au pipeline val/test)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224
MODEL_VERSION = "0.0.0"

# ---------------------------------------------------------------------------
# Etat global du modele
# ---------------------------------------------------------------------------
ort_session = None
class_names: list[str] = []


def load_model() -> bool:
    """Charge le modele ONNX et les noms de classes.

    Tente de charger le fichier ONNX et le fichier JSON des classes.
    Si un fichier est absent, le serveur demarre sans modele.

    Returns:
        True si le modele a ete charge avec succes.
    """
    global ort_session, class_names, MODEL_VERSION

    if not MODEL_PATH.exists():
        logger.warning(f"Modele ONNX introuvable : {MODEL_PATH}")
        return False

    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"Modele ONNX charge : {MODEL_PATH}")

        # Charger les noms de classes
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                class_names = json.load(f)
            logger.info(f"Classes chargees : {len(class_names)} especes")
        else:
            logger.warning(f"Fichier de classes introuvable : {CLASS_NAMES_PATH}")

        MODEL_VERSION = MODEL_PATH.stat().st_mtime_ns.__str__()[:10]
        return True

    except Exception as e:
        logger.error(f"Erreur au chargement du modele : {e}")
        return False


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesse une image pour l'inference ONNX.

    Applique le meme pipeline que les transforms val/test du Dataset :
    Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet).

    Args:
        image_bytes: Contenu brut de l'image (JPEG/PNG).

    Returns:
        Array numpy de forme (1, 3, 224, 224), float32, normalise.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Resize : cote le plus petit a 256px
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop 224x224
    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # ToTensor + Normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]  # Ajouter batch dimension


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Champy Classifier API",
    description="API de classification de champignons (30 especes) via ONNX Runtime",
    version="1.0.0",
)


@app.on_event("startup")  # type: ignore[misc]
async def startup_event() -> None:
    """Charge le modele au demarrage du serveur."""
    load_model()


_FILE_PARAM = File(description="Image JPEG ou PNG du champignon")


@app.post(  # type: ignore[misc]
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict(
    file: UploadFile = _FILE_PARAM,
    top_n: int = 5,
) -> PredictionResponse:
    """Predit l'espece d'un champignon a partir d'une image.

    Charge l'image, applique le preprocessing identique au training,
    et retourne les top-N predictions avec scores de confiance.

    Args:
        file: Fichier image uploade.
        top_n: Nombre de predictions a retourner (defaut 5).

    Returns:
        PredictionResponse avec les top-N predictions.

    Raises:
        HTTPException: 503 si le modele n'est pas charge.
    """
    REQUESTS_TOTAL.labels(method="POST", endpoint="/predict").inc()

    if ort_session is None:
        HTTP_ERRORS.labels(status_code="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible. Verifiez que best_model.onnx est present.",
        )

    start_time = time.perf_counter()

    # Lire et preprocesser l'image
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    # Inference ONNX
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    # Top-N
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    predictions = []
    for rank, idx in enumerate(top_indices, start=1):
        species_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        confidence = float(probabilities[idx])
        predictions.append(PredictionItem(species=species_name, confidence=confidence, rank=rank))

    latency = time.perf_counter() - start_time
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(predictions[0].confidence)
    PREDICTIONS_TOTAL.labels(species=predictions[0].species).inc()

    logger.info(
        f"Prediction : {predictions[0].species} "
        f"({predictions[0].confidence:.2%}) en {latency:.3f}s"
    )

    return PredictionResponse(predictions=predictions, model_version=MODEL_VERSION)


@app.get("/health", response_model=HealthResponse)  # type: ignore[misc]
async def health() -> HealthResponse:
    """Retourne l'etat de sante du service.

    Returns:
        HealthResponse avec le statut et l'etat du modele.
    """
    REQUESTS_TOTAL.labels(method="GET", endpoint="/health").inc()

    if ort_session is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=MODEL_VERSION,
        )
    return HealthResponse(
        status="no_model",
        model_loaded=False,
    )


@app.get("/metrics")  # type: ignore[misc]
async def metrics() -> Response:
    """Expose les metriques Prometheus au format text/plain.

    Returns:
        Response avec les metriques au format Prometheus.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get(  # type: ignore[misc]
    "/model/info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
)
async def model_info() -> ModelInfoResponse:
    """Retourne les metadonnees du modele charge.

    Returns:
        ModelInfoResponse avec les informations du modele ONNX.

    Raises:
        HTTPException: 503 si le modele n'est pas charge.
    """
    REQUESTS_TOTAL.labels(method="GET", endpoint="/model/info").inc()

    if ort_session is None:
        HTTP_ERRORS.labels(status_code="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible.",
        )

    input_shape = ort_session.get_inputs()[0].shape
    return ModelInfoResponse(
        model_path=str(MODEL_PATH),
        model_version=MODEL_VERSION,
        num_classes=len(class_names),
        class_names=class_names,
        input_shape=[int(s) if isinstance(s, int) else 0 for s in input_shape],
    )
