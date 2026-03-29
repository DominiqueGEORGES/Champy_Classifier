"""Script d'entrainement principal pour le classificateur de champignons.

Lance un entrainement ResNet50 en transfer learning avec :
- Mixed precision (AMP) pour les GPU a VRAM limitee (4 GB)
- Early stopping et checkpointing du meilleur modele
- Tracking MLflow (hyperparams, metriques par epoch, artefacts)
- Seed global pour la reproductibilite

Usage:
    python -m src.training.train --config configs/training/default.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast

from src.config import MODELS_DIR, get_mlflow_settings, get_training_config
from src.data.dataloader import create_all_loaders
from src.models.resnet import create_resnet50
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_json,
)


def set_seed(seed: int) -> None:
    """Fixe la graine aleatoire pour la reproductibilite.

    Configure les generateurs de torch, numpy et random
    pour garantir des resultats deterministes.

    Args:
        seed: Valeur de la graine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed global fixe a {seed}")


def get_device() -> torch.device:
    """Detecte et retourne le meilleur device disponible.

    Returns:
        torch.device : cuda si GPU disponible, sinon cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU detecte : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Pas de GPU, entrainement sur CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
) -> float:
    """Execute une epoch d'entrainement.

    Gere la mixed precision (AMP) et le gradient accumulation
    pour les GPU a VRAM limitee.

    Args:
        model: Modele PyTorch.
        dataloader: DataLoader d'entrainement.
        criterion: Fonction de perte.
        optimizer: Optimiseur.
        device: Device de calcul.
        scaler: GradScaler pour la mixed precision. None pour desactiver.
        accumulation_steps: Nombre de steps de gradient accumulation.

    Returns:
        Loss moyenne sur l'epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass avec mixed precision
        if scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        num_batches += 1

    return running_loss / max(num_batches, 1)


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Execute une epoch de validation.

    Args:
        model: Modele PyTorch en mode eval.
        dataloader: DataLoader de validation.
        criterion: Fonction de perte.
        device: Device de calcul.

    Returns:
        Tuple (val_loss, val_accuracy, val_f1_macro).
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / max(len(dataloader), 1)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return avg_loss, accuracy, f1_macro


def train(config_path: str | Path | None = None) -> Path:
    """Fonction principale d'entrainement.

    Execute le pipeline complet : chargement config, creation modele,
    entrainement avec early stopping, evaluation finale, sauvegarde
    des artefacts et tracking MLflow.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
            Si None, utilise les valeurs par defaut.

    Returns:
        Chemin vers le meilleur checkpoint sauvegarde.
    """
    # -- Configuration --
    config = get_training_config(config_path)
    mlflow_settings = get_mlflow_settings()
    set_seed(config.seed)
    device = get_device()

    logger.info(f"Configuration : {config.model_dump()}")

    # -- Donnees --
    train_loader, val_loader, test_loader = create_all_loaders(config)
    train_dataset = train_loader.dataset
    class_names: list[str] = train_dataset.class_names
    logger.info(
        f"Donnees : train={len(train_dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    # -- Modele --
    model = create_resnet50(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )
    model = model.to(device)

    # -- Optimiseur et scheduler --
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # -- Mixed precision --
    scaler = (
        GradScaler(device=device.type)
        if config.mixed_precision and device.type == "cuda"
        else None
    )

    # -- Callbacks --
    checkpoint_path = MODELS_DIR / "best_model.pt"
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")
    checkpoint = ModelCheckpoint(save_path=checkpoint_path, mode="min")

    # -- MLflow --
    mlflow.set_tracking_uri(mlflow_settings.mlflow_tracking_uri)
    if mlflow_settings.dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_settings.dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_settings.dagshub_token

    # -- Historique pour les courbes --
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    with mlflow.start_run(run_name=f"resnet50_{config.seed}"):
        # Log des hyperparametres
        mlflow.log_params(config.model_dump())

        # -- Boucle d'entrainement --
        start_time = time.time()

        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                accumulation_steps=config.gradient_accumulation_steps,
            )

            # Validation
            val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

            # Scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Historique
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{config.epochs} ({epoch_time:.0f}s) - "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.2e}"
            )

            # MLflow metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1_macro": val_f1,
                    "lr": current_lr,
                },
                step=epoch,
            )

            # Callbacks
            checkpoint.step(val_loss, model, optimizer, epoch)

            if early_stopping.step(val_loss):
                logger.warning(f"Early stopping a l'epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Entrainement termine en {total_time:.0f}s ({total_time/60:.1f}min)")

        # -- Evaluation finale sur test --
        logger.info("Evaluation finale sur le split test...")

        # Recharger le meilleur checkpoint
        if checkpoint_path.exists():
            best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(f"Meilleur checkpoint recharge (epoch {best_ckpt['epoch']})")

        test_metrics = evaluate_model(model, test_loader, device, class_names)

        # Log des metriques finales
        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

        # -- Artefacts --
        artifacts_dir = MODELS_DIR / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm_path = save_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            class_names,
            artifacts_dir / "confusion_matrix.png",
        )
        mlflow.log_artifact(str(cm_path))

        # Courbes d'apprentissage
        curves_path = save_learning_curves(history, artifacts_dir / "learning_curves.png")
        mlflow.log_artifact(str(curves_path))

        # Metriques JSON
        metrics_path = save_metrics_json(
            {
                "accuracy": test_metrics["accuracy"],
                "f1_macro": test_metrics["f1_macro"],
                "report": test_metrics["report_dict"],
                "history": history,
            },
            artifacts_dir / "metrics.json",
        )
        mlflow.log_artifact(str(metrics_path))

        # Log du rapport textuel
        logger.info(f"\n{test_metrics['report_text']}")

    return checkpoint_path


def main() -> None:
    """Point d'entree CLI du script d'entrainement."""
    parser = argparse.ArgumentParser(description="Entrainement ResNet50 champignons")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    logger.info(f"Demarrage de l'entrainement avec config : {args.config}")
    checkpoint_path = train(args.config)
    logger.info(f"Entrainement termine. Checkpoint : {checkpoint_path}")


if __name__ == "__main__":
    main()
