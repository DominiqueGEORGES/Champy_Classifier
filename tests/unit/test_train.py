"""Tests unitaires pour src.training.train (fine-tuning deux phases)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.models.resnet import create_resnet50
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.train import build_optimizer, build_scheduler, run_phase


class TestTrainingConfig:
    """Tests pour les nouveaux champs 2-phase de TrainingConfig."""

    def test_default_two_phase_fields(self) -> None:
        """Verifie les valeurs par defaut des champs 2-phase."""
        config = TrainingConfig()
        assert config.freeze_backbone_epochs == 10
        assert config.total_epochs == 30
        assert config.lr_phase1 == pytest.approx(1e-3)
        assert config.lr_phase2 == pytest.approx(1e-5)

    def test_override_two_phase_fields(self) -> None:
        """Verifie la surcharge des champs 2-phase."""
        config = TrainingConfig(
            freeze_backbone_epochs=5,
            total_epochs=15,
            lr_phase1=2e-3,
            lr_phase2=5e-5,
        )
        assert config.freeze_backbone_epochs == 5
        assert config.total_epochs == 15
        assert config.lr_phase1 == pytest.approx(2e-3)
        assert config.lr_phase2 == pytest.approx(5e-5)


class TestBuildOptimizer:
    """Tests pour build_optimizer."""

    def test_adamw_uses_given_lr(self) -> None:
        """Verifie que le lr passe est applique (pas celui du config)."""
        config = TrainingConfig(optimizer="adamw", weight_decay=1e-4)
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, config, lr=5e-4)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    def test_sgd_fallback(self) -> None:
        """Verifie le fallback SGD quand optimizer != adamw."""
        config = TrainingConfig(optimizer="sgd")
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, config, lr=1e-2)
        assert isinstance(opt, torch.optim.SGD)

    def test_filters_frozen_params(self) -> None:
        """Verifie que seuls les params entrainables sont passes a l'optimizer."""
        config = TrainingConfig(optimizer="adamw")
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        # Geler la premiere couche
        for p in model[0].parameters():
            p.requires_grad = False
        opt = build_optimizer(model, config, lr=1e-3)
        # L'optimizer ne doit voir que les params de la 2e couche
        n_opt_params = sum(len(g["params"]) for g in opt.param_groups)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_opt_params == n_trainable


class TestBuildScheduler:
    """Tests pour build_scheduler."""

    def test_cosine_scheduler(self) -> None:
        """Verifie la construction d'un scheduler cosine."""
        config = TrainingConfig(scheduler="cosine")
        opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3)
        sched = build_scheduler(opt, num_epochs=10, config=config)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert sched.T_max == 10

    def test_step_scheduler_fallback(self) -> None:
        """Verifie le fallback StepLR quand scheduler != cosine."""
        config = TrainingConfig(scheduler="step")
        opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3)
        sched = build_scheduler(opt, num_epochs=10, config=config)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_num_epochs_clamped_to_one(self) -> None:
        """Verifie que T_max ne peut pas etre nul ou negatif."""
        config = TrainingConfig(scheduler="cosine")
        opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3)
        sched = build_scheduler(opt, num_epochs=0, config=config)
        assert sched.T_max >= 1


class TestRunPhase:
    """Tests d'integration pour run_phase (sans MLflow tracking externe)."""

    @pytest.fixture()
    def mini_setup(
        self,
        tmp_dataset: tuple[Path, Path, list[str]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict:
        """Construit un mini environnement d'entrainement CPU.

        Args:
            tmp_dataset: Fixture (manifest_path, data_dir, class_names).
            tmp_path: Repertoire temporaire pytest.
            monkeypatch: Fixture pour patcher MLflow en file-based.

        Returns:
            Dict avec les objets necessaires pour appeler run_phase.
        """
        import mlflow

        # MLflow local (pas d'appel reseau)
        mlflow.set_tracking_uri(f"file:///{tmp_path / 'mlruns'}")

        manifest, data_dir, classes = tmp_dataset
        config = TrainingConfig(
            num_classes=len(classes),
            batch_size=2,
            num_workers=0,
            image_size=32,
            pretrained=False,
            freeze_backbone_epochs=1,
            total_epochs=2,
            mixed_precision=False,
        )

        from src.data.dataloader import create_all_loaders

        train_loader, val_loader, _test_loader = create_all_loaders(
            config, manifest_path=manifest, data_dir=data_dir
        )

        model = create_resnet50(
            num_classes=config.num_classes,
            pretrained=False,
            freeze_backbone=True,
        )
        device = torch.device("cpu")
        model = model.to(device)

        return {
            "config": config,
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "criterion": nn.CrossEntropyLoss(),
            "device": device,
            "checkpoint": ModelCheckpoint(save_path=tmp_path / "best.pt", mode="min"),
            "history": {
                "train_loss": [],
                "val_loss": [],
                "val_acc": [],
                "val_f1": [],
                "phase": [],
            },
        }

    def test_run_phase_appends_history(self, mini_setup: dict) -> None:
        """Verifie que run_phase remplit l'historique avec le bon nombre d'epochs."""
        import mlflow

        setup = mini_setup
        opt = build_optimizer(setup["model"], setup["config"], lr=1e-3)
        sched = build_scheduler(opt, num_epochs=2, config=setup["config"])

        with mlflow.start_run():
            last_epoch = run_phase(
                phase_num=1,
                model=setup["model"],
                train_loader=setup["train_loader"],
                val_loader=setup["val_loader"],
                criterion=setup["criterion"],
                optimizer=opt,
                scheduler=sched,
                device=setup["device"],
                scaler=None,
                config=setup["config"],
                start_epoch=1,
                end_epoch=2,
                history=setup["history"],
                checkpoint=setup["checkpoint"],
                total_epochs=2,
                early_stopping=None,
            )

        assert last_epoch == 2
        assert len(setup["history"]["train_loss"]) == 2
        assert len(setup["history"]["val_loss"]) == 2
        assert setup["history"]["phase"] == [1.0, 1.0]

    def test_run_phase_early_stopping_breaks_loop(self, mini_setup: dict) -> None:
        """Verifie qu'un EarlyStopping declenche termine la phase avant end_epoch."""
        import mlflow

        setup = mini_setup
        opt = build_optimizer(setup["model"], setup["config"], lr=1e-3)
        sched = build_scheduler(opt, num_epochs=5, config=setup["config"])
        # Patience=0 -> s'arrete au premier "non-ameliorant"
        early_stop = EarlyStopping(patience=0, mode="min")

        with mlflow.start_run():
            last_epoch = run_phase(
                phase_num=2,
                model=setup["model"],
                train_loader=setup["train_loader"],
                val_loader=setup["val_loader"],
                criterion=setup["criterion"],
                optimizer=opt,
                scheduler=sched,
                device=setup["device"],
                scaler=None,
                config=setup["config"],
                start_epoch=1,
                end_epoch=5,
                history=setup["history"],
                checkpoint=setup["checkpoint"],
                total_epochs=5,
                early_stopping=early_stop,
            )

        # Avec patience=0, l'arret doit survenir avant end_epoch=5.
        assert last_epoch < 5
        assert len(setup["history"]["phase"]) == last_epoch


class TestPhaseFreezingFlow:
    """Tests pour la logique de gel/degel du backbone entre phases."""

    def test_phase1_model_has_frozen_backbone(self) -> None:
        """Verifie qu'avec freeze_backbone=True, seule la tete est entrainable."""
        model = create_resnet50(num_classes=30, pretrained=False, freeze_backbone=True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        assert n_trainable < n_total
        # Seule la tete fc doit etre entrainable
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                assert param.requires_grad, f"{name} devrait etre entrainable"
            else:
                assert not param.requires_grad, f"{name} devrait etre gele"

    def test_unfreeze_restores_full_training(self) -> None:
        """Verifie que unfreeze_backbone_layers debloque toutes les couches."""
        from src.models.resnet import unfreeze_backbone_layers

        model = create_resnet50(num_classes=30, pretrained=False, freeze_backbone=True)
        unfreeze_backbone_layers(model, unfreeze_from=0)
        for _name, param in model.named_parameters():
            assert param.requires_grad
