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

import asyncio
import base64
import hashlib
import io
import os
import time
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
from loguru import logger
from PIL import Image
from PIL.Image import Image as PILImage  # noqa: TC002 (BentoML introspecte le type au runtime)
from prometheus_client import Counter, Histogram, Summary

from src.serving_bentoml.preprocessing import preprocess_pil
from src.serving_bentoml.runner import DEFAULT_MODEL_TAG, OnnxRunner
from src.serving_bentoml.schemas import (
    CheckpointMetadata,
    ExplainResponse,
    HealthResponse,
    ModelFileInfo,
    ModelInfoResponse,
    ModelRegistryResponse,
    OnnxValidation,
    PredictionItem,
    PredictionResponse,
)
from src.serving_bentoml.storage import PredictionRecord, PredictionStore

# ---------------------------------------------------------------------------
# Stockage des predictions (SQLite WAL).
# Le chemin est configurable via l'env var CHAMPY_PREDICTIONS_DB (utilise par
# le compose Docker pour pointer vers le volume persistant). Defaut local :
# data/runtime/predictions.db a la racine du repo.
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "runtime" / "predictions.db"
)
PREDICTIONS_DB_PATH = Path(os.environ.get("CHAMPY_PREDICTIONS_DB", _DEFAULT_DB_PATH))

# Modele PyTorch pour Grad-CAM (l'API serve ONNX en standard, mais Grad-CAM
# necessite l'acces aux gradients d'ou un second chargement PyTorch).
_DEFAULT_PYTORCH_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "best_model.pt"
PYTORCH_MODEL_PATH = Path(os.environ.get("CHAMPY_PYTORCH_MODEL_PATH", _DEFAULT_PYTORCH_PATH))
NUM_CLASSES = 30

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

        Le chargement du modele se fait via ``OnnxRunner.load()`` (sync,
        appel direct au Model Store). Le ``PredictionStore`` SQLite est
        cree mais pas ouvert : son ``init()`` est paresseux (premiere
        prediction) car il est asynchrone et le constructeur BentoML ne
        peut pas etre ``async``.
        """
        self.runner = OnnxRunner(model_tag=DEFAULT_MODEL_TAG)
        loaded = self.runner.load()
        if not loaded:
            logger.warning(
                "Service BentoML demarre SANS modele. "
                "Lancer 'python scripts/import_model_to_bentoml.py' pour "
                "alimenter le Model Store."
            )
        self.store = PredictionStore(db_path=PREDICTIONS_DB_PATH)
        self._store_init_lock = asyncio.Lock()
        # Reference forte vers les taches fire-and-forget de persistence.
        # Sans ca, le GC peut collecter la Task avant son completion et
        # interrompre l'ecriture (Python <3.13 documente ce comportement).
        self._pending_writes: set[asyncio.Task[None]] = set()
        # Grad-CAM : initialise paresseusement au premier appel /explain
        # (import torch lourd, on evite de payer ce cout au demarrage).
        self.pytorch_model: Any = None
        self.cam: Any = None
        self._gradcam_init_lock = asyncio.Lock()

    async def _ensure_store(self) -> None:
        """Initialise le store SQLite a la premiere prediction.

        Le verrou evite que deux requetes concurrentes initialisent le
        store en double (l'init est idempotente, mais creer deux
        connexions WAL au meme fichier doublerait inutilement les
        descripteurs de fichier).
        """
        if self.store.is_open:
            return
        async with self._store_init_lock:
            if not self.store.is_open:
                try:
                    await self.store.init()
                except Exception as exc:
                    logger.error(f"Echec init PredictionStore : {exc}")

    async def _ensure_gradcam(self) -> bool:
        """Initialise PyTorch + GradCAM au premier appel /explain.

        Returns:
            True si l'initialisation a reussi (ou avait deja eu lieu), False
            en cas d'erreur (modele PyTorch absent, dependance manquante).
        """
        if self.cam is not None:
            return True
        async with self._gradcam_init_lock:
            if self.cam is not None:
                return True
            if not PYTORCH_MODEL_PATH.exists():
                logger.error(f"Modele PyTorch introuvable : {PYTORCH_MODEL_PATH}")
                return False
            try:
                import torch
                from pytorch_grad_cam import GradCAM
                from torchvision.models import convnext_tiny

                model = convnext_tiny(weights=None)
                in_features = model.classifier[2].in_features
                model.classifier[2] = torch.nn.Linear(in_features, NUM_CLASSES)
                state_dict = torch.load(PYTORCH_MODEL_PATH, map_location="cpu", weights_only=True)
                if isinstance(state_dict, dict):
                    for key in ("state_dict", "model_state_dict", "model"):
                        if key in state_dict:
                            state_dict = state_dict[key]
                            break
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                self.pytorch_model = model
                self.cam = GradCAM(model=model, target_layers=[model.features[-1]])
                logger.info(
                    f"PyTorch + GradCAM charges depuis {PYTORCH_MODEL_PATH} "
                    "(target_layer=model.features[-1])"
                )
                return True
            except Exception as exc:
                logger.error(f"Echec init GradCAM : {exc}")
                return False

    @bentoml.api(route="/explain")  # type: ignore[misc]
    async def explain(self, image: PILImage, target_class_id: int = -1) -> ExplainResponse:
        """Genere une visualisation Grad-CAM de la decision du modele.

        Args:
            image: Image a expliquer.
            target_class_id: Classe pour laquelle expliquer la decision.
                Si -1 (defaut), utilise la classe predite top-1 via ONNX.

        Returns:
            ExplainResponse avec original/heatmap/overlay en base64 PNG.

        Raises:
            bentoml.exceptions.ServiceUnavailable: 503 si PyTorch ou
                GradCAM indisponibles.
        """
        if not await self._ensure_gradcam():
            raise bentoml.exceptions.ServiceUnavailable(
                f"GradCAM indisponible. Verifier {PYTORCH_MODEL_PATH}."
            )

        # Si target_class_id non specifie : predire d'abord via ONNX (rapide)
        if target_class_id < 0:
            arr = preprocess_pil(image)
            logits_batch = await self.infer_batch(arr)
            target_class_id = int(np.argmax(logits_batch[0]))

        # Preprocessing PyTorch (224x224 + normalisation ImageNet)
        import torch
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        image_rgb = image.convert("RGB").resize((224, 224))
        image_array = np.array(image_rgb).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_array - mean) / std
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

        targets = [ClassifierOutputTarget(target_class_id)]
        grayscale_cam = self.cam(input_tensor=tensor, targets=targets)[0]

        original_uint8 = (image_array * 255).astype(np.uint8)
        heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)
        overlay_uint8 = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)

        def _to_b64(arr: np.ndarray) -> str:
            """Encode un tableau numpy en chaîne base64 PNG (helper Grad-CAM)."""
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        class_name = (
            self.runner.class_names[target_class_id]
            if target_class_id < len(self.runner.class_names)
            else f"class_{target_class_id}"
        )
        return ExplainResponse(
            target_class_id=target_class_id,
            target_class_name=class_name,
            original_b64=_to_b64(original_uint8),
            heatmap_b64=_to_b64(heatmap_uint8),
            overlay_b64=_to_b64(overlay_uint8),
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

        # Persistence asynchrone fire-and-forget : ne bloque pas la reponse.
        # Le store WAL serialise les ecritures en interne (busy_timeout=5000ms).
        # On garde une reference forte sur la Task pour eviter qu'elle soit
        # collectee par le GC avant la fin de l'ecriture (cf. RUF006).
        await self._ensure_store()
        if self.store.is_open:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            top5_dump = [p.model_dump() for p in predictions[:5]]
            task = asyncio.create_task(
                self._save_prediction_safe(
                    image_hash=image_hash,
                    predicted_class=predictions[0].species,
                    confidence=predictions[0].confidence,
                    top5=top5_dump,
                    latency_ms=latency * 1000.0,
                )
            )
            self._pending_writes.add(task)
            task.add_done_callback(self._pending_writes.discard)

        return PredictionResponse(
            predictions=predictions,
            model_version=self.runner.labels.get("version", "unknown"),
        )

    async def _save_prediction_safe(
        self,
        image_hash: str,
        predicted_class: str,
        confidence: float,
        top5: list[dict[str, Any]],
        latency_ms: float,
    ) -> None:
        """Wrapper qui logue les erreurs au lieu de les laisser silencieuses.

        ``asyncio.create_task`` qui leve une exception sans handler
        affiche un warning ``Task exception was never retrieved`` mais
        ne remonte pas l'erreur. Ce wrapper garantit qu'une defaillance
        du store (disque plein, base verrouillee > 5s, etc.) apparaisse
        dans les logs sans casser le hot path du predict.

        Args:
            image_hash: SHA256 hex de l'image.
            predicted_class: Espece du top-1.
            confidence: Confiance du top-1, [0, 1].
            top5: 5 meilleures predictions (dicts serialisables).
            latency_ms: Latence d'inference en millisecondes.
        """
        try:
            await self.store.save_prediction(
                image_hash=image_hash,
                predicted_class=predicted_class,
                confidence=confidence,
                top5=top5,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.error(f"Echec persistence prediction : {exc}")

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

    @bentoml.api(route="/predictions/recent")  # type: ignore[misc]
    async def predictions_recent(
        self, hours: int = 24, limit: int = 1000
    ) -> list[PredictionRecord]:
        """Retourne les predictions des ``hours`` dernieres heures.

        BentoML 1.4 ne mappe pas les query params : ``hours`` et ``limit``
        sont passes en JSON dans le body de la requete POST.

        Args:
            hours: Fenetre temporelle en heures (defaut 24).
            limit: Nombre maximum de lignes (defaut 1000).

        Returns:
            Liste de ``PredictionRecord`` triee par timestamp decroissant.
        """
        await self._ensure_store()
        if not self.store.is_open:
            return []
        return await self.store.get_recent(hours=hours, limit=limit)

    @bentoml.api(route="/model/registry")  # type: ignore[misc]
    async def model_registry(self) -> ModelRegistryResponse:
        """Inventaire des fichiers modèles + métadonnées (checkpoint, ONNX, classes).

        Centralise tout ce que la page Streamlit Model Registry affichait
        précédemment en accès direct au filesystem.

        Returns:
            ModelRegistryResponse avec liste des fichiers, métadonnées du
            checkpoint PyTorch, validation ONNX et liste des classes.
        """
        models_dir = Path("/app/models")
        files: list[ModelFileInfo] = []
        if models_dir.exists():
            for ext in ("*.pt", "*.onnx"):
                for f in models_dir.glob(ext):
                    files.append(
                        ModelFileInfo(
                            filename=f.name,
                            format=f.suffix.upper(),
                            size_mb=round(f.stat().st_size / 1024 / 1024, 1),
                        )
                    )

        # Métadonnées du checkpoint PyTorch (epoch, best_score)
        checkpoint: CheckpointMetadata | None = None
        ckpt_path = models_dir / "best_model.pt"
        if ckpt_path.exists():
            try:
                import torch

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict):
                    checkpoint = CheckpointMetadata(
                        epoch=ckpt.get("epoch"),
                        best_score=ckpt.get("best_score"),
                    )
            except Exception as exc:
                logger.warning(f"Lecture checkpoint impossible : {exc}")

        # Validation ONNX
        onnx_validation: OnnxValidation | None = None
        onnx_path = models_dir / "best_model.onnx"
        if onnx_path.exists():
            try:
                import onnx

                model = onnx.load(str(onnx_path))
                onnx.checker.check_model(model)
                inp_shape = (
                    [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
                    if model.graph.input
                    else None
                )
                out_shape = (
                    [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
                    if model.graph.output
                    else None
                )
                onnx_validation = OnnxValidation(
                    valid=True, input_shape=inp_shape, output_shape=out_shape
                )
            except Exception as exc:
                onnx_validation = OnnxValidation(valid=False, error=str(exc))

        # Liste des classes (depuis le runner BentoML déjà chargé)
        class_names = self.runner.class_names if self.runner.is_loaded else []

        return ModelRegistryResponse(
            models=files,
            checkpoint=checkpoint,
            onnx_validation=onnx_validation,
            num_classes=len(class_names) if class_names else None,
            class_names=class_names,
        )


# Permet d'importer ``service`` au niveau module (convention BentoML pour
# les CLI ``bentoml serve service:ChampyService``). Aucune logique ajoutee :
# la classe est l'unique point d'entree.
service: Any = ChampyService
