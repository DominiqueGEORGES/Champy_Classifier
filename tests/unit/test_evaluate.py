"""Tests unitaires pour src.training.evaluate."""

from __future__ import annotations

from pathlib import Path

from src.training.evaluate import (
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_json,
)


class TestSaveConfusionMatrix:
    """Tests pour la sauvegarde de la matrice de confusion."""

    def test_creates_png(self, tmp_path: Path) -> None:
        """Verifie qu'un fichier PNG est cree."""
        output = tmp_path / "cm.png"
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        class_names = ["A", "B", "C"]

        result = save_confusion_matrix(y_true, y_pred, class_names, output)
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_no_normalize(self, tmp_path: Path) -> None:
        """Verifie la sauvegarde sans normalisation."""
        output = tmp_path / "cm_raw.png"
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        class_names = ["A", "B"]

        result = save_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            output,
            normalize=None,  # type: ignore[arg-type]
        )
        assert result.exists()


class TestSaveLearningCurves:
    """Tests pour la sauvegarde des courbes d'apprentissage."""

    def test_creates_png(self, tmp_path: Path) -> None:
        """Verifie qu'un fichier PNG est cree."""
        output = tmp_path / "curves.png"
        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "val_acc": [0.3, 0.5, 0.7],
            "val_f1": [0.2, 0.4, 0.6],
        }

        result = save_learning_curves(history, output)
        assert result == output
        assert output.exists()

    def test_partial_history(self, tmp_path: Path) -> None:
        """Verifie avec un historique partiel (pas de val_f1)."""
        output = tmp_path / "curves_partial.png"
        history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
        }

        result = save_learning_curves(history, output)
        assert result.exists()


class TestSaveMetricsJson:
    """Tests pour la sauvegarde des metriques JSON."""

    def test_creates_json(self, tmp_path: Path) -> None:
        """Verifie qu'un fichier JSON est cree avec le bon contenu."""
        import json

        output = tmp_path / "metrics.json"
        metrics = {"accuracy": 0.95, "f1_macro": 0.93}

        result = save_metrics_json(metrics, output)
        assert result == output
        assert output.exists()

        with open(output, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.95

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Verifie la creation des repertoires parents si manquants."""
        output = tmp_path / "deep" / "nested" / "metrics.json"
        save_metrics_json({"test": True}, output)
        assert output.exists()
