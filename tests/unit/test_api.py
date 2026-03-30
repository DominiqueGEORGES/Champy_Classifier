"""Tests unitaires pour l'API FastAPI (src.serving.app)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client() -> TestClient:
    """Cree un client de test FastAPI sans modele charge.

    Returns:
        TestClient configure avec l'app FastAPI.
    """
    from src.serving.app import app

    return TestClient(app)


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    """Genere une image JPEG de test en memoire.

    Returns:
        Contenu brut d'une image JPEG 64x64 pixels.
    """
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests pour l'endpoint GET /health."""

    def test_health_no_model(self, client: TestClient) -> None:
        """Verifie que /health retourne no_model si aucun modele n'est charge."""
        import src.serving.app as app_module

        app_module.ort_session = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_model"
        assert data["model_loaded"] is False

    def test_health_with_model(self, client: TestClient) -> None:
        """Verifie que /health retourne healthy si un modele est charge."""
        import src.serving.app as app_module

        app_module.ort_session = MagicMock()
        app_module.MODEL_VERSION = "test_v1"
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "test_v1"
        # Nettoyage
        app_module.ort_session = None


class TestPredictEndpoint:
    """Tests pour l'endpoint POST /predict."""

    def test_predict_no_model_returns_503(
        self, client: TestClient, dummy_image_bytes: bytes
    ) -> None:
        """Verifie que /predict retourne 503 sans modele."""
        import src.serving.app as app_module

        app_module.ort_session = None
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 503

    def test_predict_with_mock_model(self, client: TestClient, dummy_image_bytes: bytes) -> None:
        """Verifie que /predict retourne des predictions avec un modele mock."""
        import src.serving.app as app_module

        # Mock du modele ONNX
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        # Simuler 30 logits
        fake_logits = np.random.randn(1, 30).astype(np.float32)
        mock_session.run.return_value = [fake_logits]

        app_module.ort_session = mock_session
        app_module.class_names = [f"Species_{i}" for i in range(30)]
        app_module.MODEL_VERSION = "mock_v1"

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5  # top_n defaut
        assert data["predictions"][0]["rank"] == 1
        assert 0 <= data["predictions"][0]["confidence"] <= 1
        assert data["model_version"] == "mock_v1"

        # Nettoyage
        app_module.ort_session = None


class TestMetricsEndpoint:
    """Tests pour l'endpoint GET /metrics."""

    def test_metrics_returns_prometheus_format(self, client: TestClient) -> None:
        """Verifie que /metrics retourne du texte Prometheus."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "champy_predictions_total" in response.text


class TestModelInfoEndpoint:
    """Tests pour l'endpoint GET /model/info."""

    def test_model_info_no_model(self, client: TestClient) -> None:
        """Verifie que /model/info retourne 503 sans modele."""
        import src.serving.app as app_module

        app_module.ort_session = None
        response = client.get("/model/info")
        assert response.status_code == 503

    def test_model_info_with_mock(self, client: TestClient) -> None:
        """Verifie que /model/info retourne les metadonnees du modele."""
        import src.serving.app as app_module

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 224, 224]
        mock_session.get_inputs.return_value = [mock_input]

        app_module.ort_session = mock_session
        app_module.class_names = ["A", "B", "C"]
        app_module.MODEL_VERSION = "info_v1"

        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 3
        assert data["input_shape"] == [1, 3, 224, 224]

        # Nettoyage
        app_module.ort_session = None


class TestPreprocessImage:
    """Tests pour la fonction preprocess_image."""

    def test_output_shape(self, dummy_image_bytes: bytes) -> None:
        """Verifie la forme de sortie du preprocessing."""
        from src.serving.app import preprocess_image

        result = preprocess_image(dummy_image_bytes)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_normalized_values(self, dummy_image_bytes: bytes) -> None:
        """Verifie que les valeurs sont dans une plage raisonnable apres normalisation."""
        from src.serving.app import preprocess_image

        result = preprocess_image(dummy_image_bytes)
        # Apres normalisation ImageNet, les valeurs sont typiquement entre -3 et 3
        assert result.min() > -5
        assert result.max() < 5
