"""Script d'entrainement principal pour le classificateur de champignons.

Lance un entrainement ResNet50 en transfer learning en deux phases :

    - Phase 1 (``freeze_backbone_epochs`` premières epochs) : backbone
      gelé, seule la tête de classification est entraînée avec
      ``lr_phase1`` (typiquement 1e-3).
    - Phase 2 (epochs restantes jusqu'à ``total_epochs``) : backbone
      dégelé, fine-tuning complet avec ``lr_phase2`` (typiquement 1e-5).

Chaque phase a son propre optimizer et son propre scheduler cosine ;
l'early stopping ne s'active qu'en phase 2 (la phase 1 tourne toujours
jusqu'au bout pour amorcer la tête). Le checkpoint du meilleur modèle
est globalement partagé entre les deux phases.

Autres options :

    - Mixed precision (AMP) si GPU disponible et ``mixed_precision=True``.
    - Tracking MLflow : hyperparamètres, métriques par epoch, artefacts,
      phase loggée à la fois comme tag et comme métrique par epoch pour
      visualiser la transition.
    - Seed global pour la reproductibilité.

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

from src.config import MODELS_DIR, TrainingConfig, get_mlflow_settings, get_training_config
from src.data.dataloader import create_all_loaders
from src.models.resnet import create_resnet50, unfreeze_backbone_layers
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_json,
)


def set_seed(seed: int) -> None:
    """Fixe la graine aléatoire pour la reproductibilité.

    Configure les générateurs de torch, numpy et random pour garantir
    des résultats déterministes.

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
    logger.info(f"Seed global fixé à {seed}")


def get_device() -> torch.device:
    """Détecte et retourne le meilleur device disponible.

    Returns:
        torch.device : cuda si GPU disponible, sinon cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Pas de GPU, entraînement sur CPU")
    return device


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    lr: float,
) -> torch.optim.Optimizer:
    """Construit un optimizer pour les paramètres entraînables du modèle.

    Filtre les paramètres selon ``requires_grad`` pour ignorer automatiquement
    le backbone gelé en phase 1.

    Args:
        model: Modèle PyTorch.
        config: Configuration d'entraînement (type d'optimizer, weight_decay).
        lr: Learning rate à utiliser (spécifique à la phase courante).

    Returns:
        Instance d'optimizer PyTorch.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construit le scheduler de learning rate pour une phase.

    Args:
        optimizer: Optimizer cible.
        num_epochs: Nombre d'epochs de la phase (utilisé comme T_max en cosine).
        config: Configuration d'entraînement.

    Returns:
        Instance de scheduler PyTorch.
    """
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
) -> float:
    """Exécute une epoch d'entraînement.

    Gère la mixed precision (AMP) et le gradient accumulation pour les
    GPU à VRAM limitée.

    Args:
        model: Modèle PyTorch.
        dataloader: DataLoader d'entraînement.
        criterion: Fonction de perte.
        optimizer: Optimiseur.
        device: Device de calcul.
        scaler: GradScaler pour la mixed precision. None pour désactiver.
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
    """Exécute une epoch de validation.

    Args:
        model: Modèle PyTorch en mode eval.
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


def run_phase(
    *,
    phase_num: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    scaler: GradScaler | None,
    config: TrainingConfig,
    start_epoch: int,
    end_epoch: int,
    history: dict[str, list[float]],
    checkpoint: ModelCheckpoint,
    total_epochs: int,
    early_stopping: EarlyStopping | None = None,
) -> int:
    """Exécute une phase d'entraînement de ``start_epoch`` à ``end_epoch``.

    La phase logge les métriques par epoch dans MLflow en incluant le
    numéro de phase comme métrique pour visualiser la transition.
    Le checkpoint global tracke le meilleur val_loss à travers les phases.

    Args:
        phase_num: Numéro de la phase (1 ou 2).
        model: Modèle à entraîner.
        train_loader: DataLoader d'entraînement.
        val_loader: DataLoader de validation.
        criterion: Fonction de perte.
        optimizer: Optimizer de la phase.
        scheduler: Scheduler de la phase.
        device: Device PyTorch.
        scaler: GradScaler pour AMP (ou None).
        config: Configuration d'entraînement.
        start_epoch: Premier epoch (inclus, 1-indexé).
        end_epoch: Dernier epoch (inclus).
        history: Dict accumulateur des courbes d'apprentissage.
        checkpoint: Callback de checkpointing partagé entre phases.
        total_epochs: Nombre total d'epochs (pour l'affichage).
        early_stopping: Callback d'early stopping (None = désactivé).

    Returns:
        Le dernier epoch effectivement exécuté (utile si early stopping
        est déclenché avant ``end_epoch``).
    """
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"=== Phase {phase_num} : epochs {start_epoch}-{end_epoch}, "
        f"lr={optimizer.param_groups[0]['lr']:.2e}, "
        f"{num_trainable:,} params entraînables ==="
    )

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            accumulation_steps=config.gradient_accumulation_steps,
        )
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["phase"].append(float(phase_num))

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Phase {phase_num} - Epoch {epoch}/{total_epochs} ({epoch_time:.0f}s) - "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.2e}"
        )

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "lr": current_lr,
                "phase": phase_num,
            },
            step=epoch,
        )

        checkpoint.step(val_loss, model, optimizer, epoch)
        last_epoch = epoch

        if early_stopping is not None and early_stopping.step(val_loss):
            logger.warning(f"Early stopping déclenché phase {phase_num} epoch {epoch}")
            break

    return last_epoch


def train(
    config_path: str | Path | None = None,
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Pipeline d'entraînement complet en deux phases.

    Orchestration : chargement de la configuration, création du modèle
    et des DataLoaders, phase 1 avec backbone gelé, phase 2 avec
    fine-tuning complet, évaluation finale, sauvegarde des artefacts
    et tracking MLflow.

    Args:
        config_path: Chemin vers le fichier YAML de configuration. Si
            None, utilise les valeurs par défaut et .env.
        manifest_path: Chemin optionnel vers le manifest split. Si
            None, utilise le manifest par défaut (data/split_manifest.csv).
        data_dir: Répertoire racine des images (surcharge optionnelle
            pour les tests).

    Returns:
        Chemin vers le meilleur checkpoint sauvegardé.
    """
    # -- Configuration --
    config = get_training_config(config_path)
    mlflow_settings = get_mlflow_settings()
    set_seed(config.seed)
    device = get_device()

    logger.info(f"Configuration : {config.model_dump()}")

    # -- Données --
    train_loader, val_loader, test_loader = create_all_loaders(
        config, manifest_path=manifest_path, data_dir=data_dir
    )
    train_dataset = train_loader.dataset
    class_names: list[str] = train_dataset.class_names
    logger.info(
        f"Données : train={len(train_dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    # -- Modèle : backbone gelé si phase 1 activée --
    has_phase1 = config.freeze_backbone_epochs > 0
    model = create_resnet50(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=has_phase1,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # -- Mixed precision (uniquement si GPU) --
    scaler = (
        GradScaler(device=device.type)
        if config.mixed_precision and device.type == "cuda"
        else None
    )

    # -- Callbacks partagés entre phases --
    checkpoint_path = MODELS_DIR / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(save_path=checkpoint_path, mode="min")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "phase": [],
    }

    # -- MLflow --
    mlflow.set_tracking_uri(mlflow_settings.mlflow_tracking_uri)
    if mlflow_settings.dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_settings.dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_settings.dagshub_token

    with mlflow.start_run(run_name=f"resnet50_2phase_{config.seed}"):
        mlflow.log_params(config.model_dump())
        mlflow.set_tag("training_mode", "two_phase" if has_phase1 else "single_phase")
        mlflow.set_tag("phase2_start_epoch", config.freeze_backbone_epochs + 1)

        start_time = time.time()
        last_epoch = 0

        # -- Phase 1 : backbone gelé, entraînement de la tête --
        if has_phase1:
            optimizer_p1 = build_optimizer(model, config, lr=config.lr_phase1)
            scheduler_p1 = build_scheduler(optimizer_p1, config.freeze_backbone_epochs, config)
            last_epoch = run_phase(
                phase_num=1,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_p1,
                scheduler=scheduler_p1,
                device=device,
                scaler=scaler,
                config=config,
                start_epoch=1,
                end_epoch=config.freeze_backbone_epochs,
                history=history,
                checkpoint=checkpoint,
                total_epochs=config.total_epochs,
                early_stopping=None,  # Pas d'early stopping en phase 1
            )

        # -- Phase 2 : backbone dégelé, fine-tuning complet --
        if config.total_epochs > last_epoch:
            unfreeze_backbone_layers(model)
            phase2_epochs = config.total_epochs - last_epoch
            optimizer_p2 = build_optimizer(model, config, lr=config.lr_phase2)
            scheduler_p2 = build_scheduler(optimizer_p2, phase2_epochs, config)
            early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")
            last_epoch = run_phase(
                phase_num=2,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_p2,
                scheduler=scheduler_p2,
                device=device,
                scaler=scaler,
                config=config,
                start_epoch=last_epoch + 1,
                end_epoch=config.total_epochs,
                history=history,
                checkpoint=checkpoint,
                total_epochs=config.total_epochs,
                early_stopping=early_stopping,
            )

        total_time = time.time() - start_time
        logger.info(f"Entraînement terminé en {total_time:.0f}s ({total_time / 60:.1f}min)")

        # -- Évaluation finale sur test --
        logger.info("Évaluation finale sur le split test...")

        if checkpoint_path.exists():
            best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(f"Meilleur checkpoint rechargé (epoch {best_ckpt['epoch']})")

        test_metrics = evaluate_model(model, test_loader, device, class_names)

        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

        # -- Artefacts --
        artifacts_dir = MODELS_DIR / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        cm_path = save_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            class_names,
            artifacts_dir / "confusion_matrix.png",
        )
        mlflow.log_artifact(str(cm_path))

        curves_path = save_learning_curves(history, artifacts_dir / "learning_curves.png")
        mlflow.log_artifact(str(curves_path))

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

        logger.info(f"\n{test_metrics['report_text']}")

    return checkpoint_path


def main() -> None:
    """Point d'entrée CLI du script d'entraînement."""
    parser = argparse.ArgumentParser(description="Entraînement ResNet50 champignons (2 phases)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    logger.info(f"Démarrage de l'entraînement avec config : {args.config}")
    checkpoint_path = train(args.config)
    logger.info(f"Entraînement terminé. Checkpoint : {checkpoint_path}")


if __name__ == "__main__":
    main()
