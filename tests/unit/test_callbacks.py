"""Tests unitaires pour src.training.callbacks."""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.callbacks import EarlyStopping, ModelCheckpoint


class TestEarlyStopping:
    """Tests pour le callback EarlyStopping."""

    def test_no_stop_when_improving(self) -> None:
        """Verifie que l'entrainement continue tant que la metrique s'ameliore."""
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)
        assert es.counter == 0

    def test_stops_after_patience(self) -> None:
        """Verifie que l'arret se declenche apres patience epochs sans amelioration."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(0.5)  # Meilleur score initial
        es.step(0.6)  # Pas d'amelioration -> counter=1
        es.step(0.7)  # counter=2
        result = es.step(0.8)  # counter=3 -> arret
        assert result is True
        assert es.should_stop is True

    def test_resets_counter_on_improvement(self) -> None:
        """Verifie que le compteur se remet a zero apres une amelioration."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)
        es.step(1.1)  # counter=1
        es.step(1.2)  # counter=2
        es.step(0.5)  # Amelioration -> counter=0
        assert es.counter == 0
        assert es.best_score == 0.5

    def test_mode_max(self) -> None:
        """Verifie le fonctionnement en mode max (accuracy)."""
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.8)
        es.step(0.9)  # Amelioration
        assert es.best_score == 0.9
        es.step(0.85)  # Pas d'amelioration -> counter=1
        result = es.step(0.88)  # counter=2 -> arret
        assert result is True

    def test_min_delta(self) -> None:
        """Verifie que min_delta est pris en compte."""
        es = EarlyStopping(patience=2, mode="min", min_delta=0.01)
        es.step(1.0)
        # Amelioration de 0.005 < min_delta -> pas un vrai progres
        es.step(0.995)
        assert es.counter == 1


class TestModelCheckpoint:
    """Tests pour le callback ModelCheckpoint."""

    def test_saves_on_improvement(self, tmp_path: Path) -> None:
        """Verifie que le modele est sauvegarde quand la metrique s'ameliore."""
        save_path = tmp_path / "model.pt"
        ckpt = ModelCheckpoint(save_path=save_path, mode="min")
        model = torch.nn.Linear(10, 2)

        saved = ckpt.step(1.0, model, epoch=1)
        assert saved is True
        assert save_path.exists()

    def test_no_save_on_worse(self, tmp_path: Path) -> None:
        """Verifie que le modele n'est pas sauvegarde si la metrique empire."""
        save_path = tmp_path / "model.pt"
        ckpt = ModelCheckpoint(save_path=save_path, mode="min")
        model = torch.nn.Linear(10, 2)

        ckpt.step(0.5, model, epoch=1)  # Sauvegarde initiale
        saved = ckpt.step(0.8, model, epoch=2)  # Pire -> pas sauvegarde
        assert saved is False

    def test_checkpoint_content(self, tmp_path: Path) -> None:
        """Verifie le contenu du checkpoint sauvegarde."""
        save_path = tmp_path / "model.pt"
        ckpt = ModelCheckpoint(save_path=save_path, mode="min")
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        ckpt.step(0.5, model, optimizer=optimizer, epoch=5)

        loaded = torch.load(save_path, weights_only=True)
        assert loaded["epoch"] == 5
        assert loaded["best_score"] == 0.5
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_mode_max(self, tmp_path: Path) -> None:
        """Verifie le fonctionnement en mode max."""
        save_path = tmp_path / "model.pt"
        ckpt = ModelCheckpoint(save_path=save_path, mode="max")
        model = torch.nn.Linear(10, 2)

        ckpt.step(0.8, model, epoch=1)
        saved = ckpt.step(0.9, model, epoch=2)  # Meilleur -> sauvegarde
        assert saved is True
        assert ckpt.best_score == 0.9

        saved = ckpt.step(0.85, model, epoch=3)  # Pire -> pas sauvegarde
        assert saved is False
