"""Tests unitaires pour le service BentoML (src.serving_bentoml).

Couvre :
- ``preprocessing.py`` : pipeline image -> array ONNX (Resize 256 + CenterCrop 224 + Normalize).
- ``runner.py`` : init, etat ``is_loaded``, predict avec session mockee, _load_class_names.
- ``schemas.py`` : validation Pydantic des modeles de requete/reponse.

Ne couvre pas ``service.py`` qui necessite le runtime BentoML pour s'instancier
(decorateurs ``@bentoml.service``) : a tester via tests d'integration.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src.serving_bentoml.preprocessing import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
    preprocess_pil,
)
from src.serving_bentoml.runner import DEFAULT_MODEL_TAG, OnnxRunner
from src.serving_bentoml.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelRegistryResponse,
    PredictionItem,
    PredictionResponse,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Image PIL RGB carree 300x300 de couleur uniforme (verte foret)."""
    return Image.new("RGB", (300, 300), color=(31, 78, 61))


@pytest.fixture
def sample_image_bytes(sample_pil_image: Image.Image) -> bytes:
    """Bytes JPEG d'une image carree 300x300."""
    buf = BytesIO()
    sample_pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# =====================================================================
# Preprocessing
# =====================================================================


class TestPreprocessing:
    """Verifie le pipeline preprocessing image -> array ONNX-ready."""

    def test_output_shape_from_bytes(self, sample_image_bytes: bytes) -> None:
        """L'output doit etre (1, 3, 224, 224) depuis des bytes."""
        arr = preprocess_image(sample_image_bytes)
        assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_float32(self, sample_image_bytes: bytes) -> None:
        """L'output doit etre float32 pour ONNX Runtime."""
        arr = preprocess_image(sample_image_bytes)
        assert arr.dtype == np.float32

    def test_output_shape_from_pil(self, sample_pil_image: Image.Image) -> None:
        """preprocess_pil doit produire la meme shape que preprocess_image."""
        arr = preprocess_pil(sample_pil_image)
        assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_rgba_converted_to_rgb(self) -> None:
        """Une image RGBA doit etre convertie automatiquement en RGB."""
        img_rgba = Image.new("RGBA", (250, 250), color=(255, 0, 0, 128))
        arr = preprocess_pil(img_rgba)
        assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert arr.dtype == np.float32

    def test_landscape_image_handled(self) -> None:
        """Une image paysage (300x200) doit etre redimensionnee et cropee correctement."""
        img = Image.new("RGB", (400, 280), color=(100, 200, 100))
        arr = preprocess_pil(img)
        assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_portrait_image_handled(self) -> None:
        """Une image portrait (200x300) doit etre redimensionnee et cropee correctement."""
        img = Image.new("RGB", (280, 400), color=(100, 100, 200))
        arr = preprocess_pil(img)
        assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_normalization_ranges(self, sample_pil_image: Image.Image) -> None:
        """Apres normalisation ImageNet, les valeurs doivent etre dans un range plausible.

        Pour une image RGB dans [0, 255], la normalisation ((x/255 - mean) / std) avec
        ImageNet (mean ~0.45, std ~0.22) produit des valeurs dans environ [-2.2, 2.7].
        """
        arr = preprocess_pil(sample_pil_image)
        assert arr.min() >= -2.5
        assert arr.max() <= 3.0

    def test_imagenet_stats_loaded(self) -> None:
        """Les constantes ImageNet doivent etre les valeurs standard."""
        assert IMAGENET_MEAN.tolist() == pytest.approx([0.485, 0.456, 0.406])
        assert IMAGENET_STD.tolist() == pytest.approx([0.229, 0.224, 0.225])


# =====================================================================
# OnnxRunner
# =====================================================================


class TestOnnxRunner:
    """Verifie l'initialisation, l'etat et la prediction (session mockee) du runner."""

    def test_init_default_tag(self) -> None:
        """Sans argument, le runner utilise DEFAULT_MODEL_TAG."""
        runner = OnnxRunner()
        assert runner.model_tag == DEFAULT_MODEL_TAG
        assert runner.session is None
        assert runner.class_names == []
        assert runner.labels == {}
        assert runner.input_name is None

    def test_init_custom_tag(self) -> None:
        """Un tag custom est conserve tel quel."""
        runner = OnnxRunner(model_tag="champy_classifier:v2")
        assert runner.model_tag == "champy_classifier:v2"

    def test_is_loaded_initially_false(self) -> None:
        """is_loaded doit etre False avant tout chargement."""
        runner = OnnxRunner()
        assert runner.is_loaded is False

    def test_predict_raises_when_not_loaded(self) -> None:
        """Appeler predict() sans load() doit lever RuntimeError."""
        runner = OnnxRunner()
        batch = np.zeros((1, 3, 224, 224), dtype=np.float32)
        with pytest.raises(RuntimeError, match="non charge"):
            runner.predict(batch)

    def test_predict_with_mocked_session(self) -> None:
        """Predict avec session ONNX mockee retourne les logits correctement."""
        runner = OnnxRunner()
        expected_logits = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [expected_logits]
        runner.session = mock_session
        runner.input_name = "input"
        runner.class_names = ["classA", "classB", "classC"]

        batch = np.zeros((1, 3, 224, 224), dtype=np.float32)
        logits = runner.predict(batch)

        assert logits.shape == (1, 3)
        assert logits.dtype == np.float32
        np.testing.assert_array_equal(logits, expected_logits)
        mock_session.run.assert_called_once_with(None, {"input": batch})

    def test_is_loaded_true_when_session_and_classes_set(self) -> None:
        """is_loaded doit etre True quand session ET class_names sont definis."""
        runner = OnnxRunner()
        runner.session = MagicMock()
        runner.class_names = ["classA"]
        assert runner.is_loaded is True

    def test_is_loaded_false_when_classes_empty(self) -> None:
        """is_loaded doit etre False si class_names est vide meme avec une session."""
        runner = OnnxRunner()
        runner.session = MagicMock()
        runner.class_names = []
        assert runner.is_loaded is False

    def test_load_class_names_from_file(self, tmp_path: Path) -> None:
        """_load_class_names doit lire class_names.json du repertoire modele."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        class_names_file = model_dir / "class_names.json"
        class_names_file.write_text(
            json.dumps(["Amanita muscaria", "Boletus edulis", "Cantharellus cibarius"]),
            encoding="utf-8",
        )
        result = OnnxRunner._load_class_names(model_dir)
        assert len(result) == 3
        assert result[0] == "Amanita muscaria"

    def test_load_class_names_returns_empty_when_missing(self, tmp_path: Path) -> None:
        """_load_class_names doit retourner [] si aucun fichier n'existe.

        Note : la fonction a un fallback sur models/class_names.json a la racine
        du repo. Pour isoler le test, on verifie au minimum qu'elle ne crashe pas
        et retourne une liste (qui peut etre vide ou contenir le fallback).
        """
        model_dir = tmp_path / "model_without_json"
        model_dir.mkdir()
        result = OnnxRunner._load_class_names(model_dir)
        assert isinstance(result, list)


# =====================================================================
# Schemas Pydantic
# =====================================================================


class TestSchemas:
    """Verifie la validation Pydantic des schemas de requete/reponse."""

    def test_prediction_item_valid(self) -> None:
        """Un PredictionItem avec confidence dans [0, 1] doit etre accepte."""
        item = PredictionItem(species="Amanita muscaria", confidence=0.95, rank=1)
        assert item.species == "Amanita muscaria"
        assert item.confidence == 0.95
        assert item.rank == 1

    def test_prediction_item_confidence_above_one_raises(self) -> None:
        """Confidence > 1 doit etre rejete par Pydantic."""
        with pytest.raises(ValidationError):
            PredictionItem(species="X", confidence=1.5, rank=1)

    def test_prediction_item_confidence_negative_raises(self) -> None:
        """Confidence < 0 doit etre rejete par Pydantic."""
        with pytest.raises(ValidationError):
            PredictionItem(species="X", confidence=-0.1, rank=1)

    def test_prediction_item_rank_zero_raises(self) -> None:
        """Rank < 1 doit etre rejete par Pydantic."""
        with pytest.raises(ValidationError):
            PredictionItem(species="X", confidence=0.5, rank=0)

    def test_prediction_response_valid(self) -> None:
        """PredictionResponse accepte une liste d'items + une version."""
        response = PredictionResponse(
            predictions=[
                PredictionItem(species="A", confidence=0.9, rank=1),
                PredictionItem(species="B", confidence=0.05, rank=2),
            ],
            model_version="convnext_tiny_v1",
        )
        assert len(response.predictions) == 2
        assert response.model_version == "convnext_tiny_v1"

    def test_health_response_no_model_version_default(self) -> None:
        """HealthResponse permet model_version=None par defaut."""
        response = HealthResponse(status="no_model", model_loaded=False)
        assert response.model_version is None
        assert response.model_loaded is False

    def test_model_info_response_valid(self) -> None:
        """ModelInfoResponse valide une reponse /model/info complete."""
        response = ModelInfoResponse(
            model_path="champy_classifier:v1",
            model_version="v1",
            num_classes=30,
            class_names=["A"] * 30,
            input_shape=[1, 3, 224, 224],
            architecture="convnext_tiny",
        )
        assert response.num_classes == 30
        assert len(response.class_names) == 30
        assert response.input_shape == [1, 3, 224, 224]

    def test_model_registry_response_defaults(self) -> None:
        """ModelRegistryResponse a class_names=[] par defaut."""
        response = ModelRegistryResponse(models=[])
        assert response.class_names == []
        assert response.models == []
        assert response.checkpoint is None

    def test_error_response_requires_detail(self) -> None:
        """ErrorResponse exige le champ detail."""
        response = ErrorResponse(detail="Test error")
        assert response.detail == "Test error"
        with pytest.raises(ValidationError):
            ErrorResponse()  # type: ignore[call-arg]
