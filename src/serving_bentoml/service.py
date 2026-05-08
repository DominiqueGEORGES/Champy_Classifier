"""Service BentoML pour le classificateur de champignons.

Point d'entree du service BentoML 1.4 (style ``@bentoml.service``). Charge
le modele ONNX depuis le Model Store BentoML (alimente par
``scripts/import_model_to_bentoml.py``) et expose les endpoints :

- ``POST /predict``        : upload d'une image, top-N predictions.
- ``GET  /health``         : statut du service + version/architecture/classes.
- ``GET  /model/info``     : metadonnees completes du modele (depuis les
                              labels du Model Store).
- ``GET  /metrics``        : metriques Prometheus (natives + custom).
- ``GET  /healthz``, ``/readyz`` : exposes nativement par BentoML.

Le batching adaptatif est active sur la methode interne ``infer_batch``
(``max_batch_size=32``, ``max_latency_ms=100``). L'endpoint public
``predict`` reste mono-image pour l'ergonomie HTTP : BentoML pool
automatiquement les appels concurrents lorsqu'ils convergent vers
``infer_batch``.

Usage :
    bentoml serve src.serving_bentoml.service:ChampyService --port 8020
"""

from __future__ import annotations

import time
from typing import Any

import bentoml
import numpy as np
from loguru import logger
from PIL.Image import Image as PILImage  # noqa: TC002 (BentoML introspecte le type au runtime)
from prometheus_client import Counter, Histogram, Summary

from src.serving_bentoml.preprocessing import preprocess_pil
from src.serving_bentoml.runner import DEFAULT_MODEL_TAG, OnnxRunner
from src.serving_bentoml.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Metriques Prometheus custom (les metriques HTTP natives BentoML sont deja
# exposees sur /metrics : requests_total, request_duration_seconds, etc.).
# On n'ajoute ici que ce qui est specifique au domaine metier :
# distribution des especes predites et confiance top-1.
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "champy_predictions_total",
    "Nombre total de predictions par espece",
    ["species"],
)

PREDICTION_LATENCY = Histogram(
    "champy_prediction_latency_seconds",
    "Latence de l'inference de bout-en-bout (preprocessing + ONNX)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_CONFIDENCE = Summary(
    "champy_prediction_confidence",
    "Confiance de la prediction top-1",
)

# Configuration adaptive batching de l'inference ONNX (cf. cahier des charges).
MAX_BATCH_SIZE = 32
MAX_LATENCY_MS = 100


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Calcule un softmax numeriquement stable.

    Args:
        logits: Vecteur 1D de logits.

    Returns:
        Vecteur de probabilites de meme dimension, sommant a 1.
    """
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


@bentoml.service(
    name="champy_classifier",
    description=(
        "API d'inference du classificateur de champignons (30 especes) "
        "basee sur un modele ConvNeXt-Tiny exporte en ONNX. Migration de "
        "FastAPI vers BentoML pour aligner avec la roadmap equipe."
    ),
    labels={
        "owner": "champy-team",
        "stage": "production",
        "framework": "bentoml",
        "backbone": "convnext_tiny",
    },
    traffic={"timeout": 60},
)
class ChampyService:
    """Service BentoML qui sert le classificateur de champignons.

    Le modele ONNX est charge depuis le Model Store BentoML au demarrage
    du worker (``__init__``). Si le Model Store ne contient pas encore
    de modele tagge ``champy_classifier:latest``, le service demarre
    quand meme et expose ``/health`` avec ``status=no_model`` ; les
    appels a ``/predict`` retournent une erreur 503.

    Attributes:
        runner: Wrapper ``OnnxRunner`` qui encapsule la session
            ``onnxruntime`` et les metadonnees du Model Store.
    """

    def __init__(self) -> None:
        """Initialise le service en chargeant le modele ONNX.

        Le chargement se fait via ``OnnxRunner.load()`` qui resout le
        tag ``champy_classifier:latest`` dans le Model Store BentoML.
        """
        self.runner = OnnxRunner(model_tag=DEFAULT_MODEL_TAG)
        loaded = self.runner.load()
        if not loaded:
            logger.warning(
                "Service BentoML demarre SANS modele. "
                "Lancer 'python scripts/import_model_to_bentoml.py' pour "
                "alimenter le Model Store."
            )

    # ------------------------------------------------------------------
    # Methode batchable interne : c'est ici que le pooling adaptatif a lieu.
    # Le batch_dim=0 indique que la dimension de batching est l'axe 0 du
    # tableau d'entree (et de sortie). max_batch_size=32 et max_latency_ms=100
    # respectent le cahier des charges.
    # ------------------------------------------------------------------
    @bentoml.api(  # type: ignore[misc]
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    async def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """Inference ONNX sur un batch d'images preprocessees.

        Cette methode est exposee comme endpoint HTTP (``POST /infer_batch``)
        car BentoML 1.4 ne distingue pas methodes publiques et internes ;
        elle est cependant principalement appelee depuis ``predict`` pour
        que le pooling adaptatif fasse converger les requetes concurrentes.
        Elle est ``async`` car BentoML 1.4 route les appels intra-service
        a travers un proxy interne qui retourne une coroutine.

        Args:
            batch: Tableau ``(N, 3, 224, 224)`` float32 normalise ImageNet.

        Returns:
            Tableau ``(N, num_classes)`` float32 contenant les logits bruts.

        Raises:
            RuntimeError: Si le modele n'est pas charge dans le Model Store.
        """
        if not self.runner.is_loaded:
            raise RuntimeError(
                "Modele non disponible dans le Model Store. "
                "Lancer 'python scripts/import_model_to_bentoml.py'."
            )
        # Le proxy RPC interne de BentoML serialise les tableaux numpy en
        # float64 par defaut. ONNX Runtime exige du float32 pour ce graphe :
        # cast explicite (no-op si le batch est deja float32).
        batch_f32 = np.ascontiguousarray(batch, dtype=np.float32)
        return self.runner.predict(batch_f32)

    # ------------------------------------------------------------------
    # Endpoints HTTP publics
    # ------------------------------------------------------------------
    @bentoml.api(route="/predict")  # type: ignore[misc]
    async def predict(self, image: PILImage, top_n: int = 5) -> PredictionResponse:
        """Predit l'espece d'un champignon a partir d'une image.

        BentoML decode automatiquement le upload HTTP en ``PIL.Image`` grace
        a l'annotation de type. L'inference ONNX passe par ``infer_batch``
        pour beneficier du batching adaptatif quand plusieurs requetes
        arrivent concurrentes.

        Args:
            image: Image envoyee par le client (JPEG/PNG decodee par BentoML).
            top_n: Nombre de predictions a retourner (defaut 5).

        Returns:
            PredictionResponse avec les top-N especes triees par confiance.

        Raises:
            bentoml.exceptions.ServiceUnavailable: 503 si modele non charge.
        """
        if not self.runner.is_loaded:
            raise bentoml.exceptions.ServiceUnavailable(
                "Modele non disponible dans le Model Store BentoML."
            )

        start_time = time.perf_counter()

        # Preprocessing strict (Resize 256 -> CenterCrop 224 -> Normalize ImageNet).
        arr = preprocess_pil(image)

        # Inference via la methode batchable : BentoML poolera les appels
        # concurrents jusqu'a max_batch_size ou max_latency_ms. L'appel
        # passe par un proxy RPC interne (cf. infer_batch), d'ou le await.
        logits_batch = await self.infer_batch(arr)
        logits = logits_batch[0]

        # Softmax + top-N
        probabilities = _softmax(logits)
        top_n_clamped = max(1, min(top_n, len(self.runner.class_names)))
        top_indices = np.argsort(probabilities)[::-1][:top_n_clamped]

        predictions: list[PredictionItem] = []
        for rank, idx in enumerate(top_indices, start=1):
            species = (
                self.runner.class_names[int(idx)]
                if int(idx) < len(self.runner.class_names)
                else f"class_{int(idx)}"
            )
            predictions.append(
                PredictionItem(
                    species=species,
                    confidence=float(probabilities[int(idx)]),
                    rank=rank,
                )
            )

        latency = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.observe(predictions[0].confidence)
        PREDICTIONS_TOTAL.labels(species=predictions[0].species).inc()

        logger.info(
            f"Prediction : {predictions[0].species} "
            f"({predictions[0].confidence:.2%}) en {latency:.3f}s"
        )

        return PredictionResponse(
            predictions=predictions,
            model_version=self.runner.labels.get("version", "unknown"),
        )

    @bentoml.api(route="/health")  # type: ignore[misc]
    def health(self) -> HealthResponse:
        """Retourne l'etat de sante enrichi du service.

        Complement aux ``/healthz`` et ``/readyz`` natifs de BentoML : on
        ajoute la version du modele charge, ce qui simplifie le debug et
        la verification cote client (Streamlit / Grafana annotations).

        Returns:
            HealthResponse avec ``status``, ``model_loaded``, ``model_version``.
        """
        if self.runner.is_loaded:
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_version=self.runner.labels.get("version", "unknown"),
            )
        return HealthResponse(
            status="no_model",
            model_loaded=False,
        )

    @bentoml.api(route="/model/info")  # type: ignore[misc]
    def model_info(self) -> ModelInfoResponse:
        """Retourne les metadonnees du modele charge.

        Lit les labels et metadonnees attaches au modele dans le Model
        Store BentoML (pas de hardcoded : tout est resolu dynamiquement).

        Returns:
            ModelInfoResponse avec le tag, la version, l'architecture,
            le nombre de classes, leurs noms, et la forme d'entree attendue.

        Raises:
            bentoml.exceptions.ServiceUnavailable: 503 si modele non charge.
        """
        if not self.runner.is_loaded:
            raise bentoml.exceptions.ServiceUnavailable(
                "Modele non disponible dans le Model Store BentoML."
            )

        return ModelInfoResponse(
            model_path=self.runner.model_tag,
            model_version=self.runner.labels.get("version", "unknown"),
            architecture=self.runner.labels.get("architecture", "unknown"),
            num_classes=len(self.runner.class_names),
            class_names=self.runner.class_names,
            input_shape=self.runner.input_shape,
        )


# Permet d'importer ``service`` au niveau module (convention BentoML pour
# les CLI ``bentoml serve service:ChampyService``). Aucune logique ajoutee :
# la classe est l'unique point d'entree.
service: Any = ChampyService
