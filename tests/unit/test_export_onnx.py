"""Tests unitaires pour src.models.export_onnx."""

from __future__ import annotations

from pathlib import Path

from src.models.export_onnx import (
    compare_outputs,
    export_to_onnx,
    save_class_names,
    validate_onnx,
)
from src.models.resnet import create_resnet50


class TestExportOnnx:
    """Tests pour l'export ONNX."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Verifie que l'export cree un fichier ONNX."""
        model = create_resnet50(num_classes=5, pretrained=False)
        model.eval()
        output = tmp_path / "test_model.onnx"
        result = export_to_onnx(model, output, image_size=32)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_validate_exported_model(self, tmp_path: Path) -> None:
        """Verifie que le fichier ONNX exporte est valide."""
        model = create_resnet50(num_classes=5, pretrained=False)
        model.eval()
        output = tmp_path / "test_model.onnx"
        export_to_onnx(model, output, image_size=32)
        assert validate_onnx(output) is True


class TestCompareOutputs:
    """Tests pour la comparaison PyTorch vs ONNX."""

    def test_outputs_match(self, tmp_path: Path) -> None:
        """Verifie que les sorties PyTorch et ONNX sont identiques."""
        model = create_resnet50(num_classes=5, pretrained=False)
        model.eval()
        output = tmp_path / "test_model.onnx"
        export_to_onnx(model, output, image_size=32)

        result = compare_outputs(model, output, image_size=32, n_samples=3)
        assert result["all_match"] is True
        assert result["max_abs_diff"] < 1e-4


class TestSaveClassNames:
    """Tests pour la sauvegarde des noms de classes."""

    def test_creates_json(self, tmp_path: Path) -> None:
        """Verifie la creation du fichier JSON de noms de classes."""
        import json

        output = tmp_path / "classes.json"
        save_class_names(output, num_classes=30)
        assert output.exists()
        with open(output, encoding="utf-8") as f:
            names = json.load(f)
        assert isinstance(names, list)
        assert len(names) == 30
