"""Evaluation du modele : metriques, confusion matrix, artefacts.

Calcule les metriques de classification (accuracy, F1 macro, rapport
complet) et genere les artefacts visuels (confusion matrix PNG,
courbes d'apprentissage) pour MLflow et le Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

matplotlib.use("Agg")  # Backend non-interactif pour le serveur


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evalue le modele sur un DataLoader et retourne les metriques.

    Parcourt le DataLoader en mode evaluation (sans gradient),
    calcule l'accuracy, le F1 macro et le rapport de classification
    complet par classe.

    Args:
        model: Modele PyTorch en mode eval.
        dataloader: DataLoader d'evaluation (val ou test).
        device: Device (cpu ou cuda).
        class_names: Noms des classes pour le rapport. Si None,
            utilise les indices numeriques.

    Returns:
        Dictionnaire avec les cles : 'accuracy', 'f1_macro',
        'y_true', 'y_pred', 'report_dict', 'report_text'.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    target_names = class_names or [str(i) for i in range(max(y_true.max(), y_pred.max()) + 1)]
    report_text = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    logger.info(f"Evaluation : accuracy={accuracy:.4f}, F1 macro={f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "report_dict": report_dict,
        "report_text": report_text,
    }


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    normalize: str = "true",
) -> Path:
    """Genere et sauvegarde la matrice de confusion en image PNG.

    Args:
        y_true: Labels reels.
        y_pred: Labels predits.
        class_names: Noms des classes pour les axes.
        output_path: Chemin de sortie du fichier PNG.
        normalize: Mode de normalisation ('true', 'pred', 'all', ou None).

    Returns:
        Chemin du fichier PNG sauvegarde.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Vrais labels",
        xlabel="Labels predits",
        title=f"Matrice de confusion (normalisation={normalize})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annoter les cellules
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Matrice de confusion sauvegardee : {output_path}")
    return output_path


def save_learning_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> Path:
    """Genere et sauvegarde les courbes d'apprentissage en image PNG.

    Args:
        history: Dictionnaire avec les cles 'train_loss', 'val_loss',
            et optionnellement 'val_acc', 'val_f1'.
        output_path: Chemin de sortie du fichier PNG.

    Returns:
        Chemin du fichier PNG sauvegarde.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Evolution de la loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metriques
    ax = axes[1]
    if "val_acc" in history:
        ax.plot(epochs, history["val_acc"], label="Val accuracy")
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], label="Val F1 macro")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Evolution des metriques")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Courbes d'apprentissage sauvegardees : {output_path}")
    return output_path


def save_metrics_json(
    metrics: dict[str, Any],
    output_path: Path,
) -> Path:
    """Sauvegarde les metriques au format JSON pour le Streamlit.

    Args:
        metrics: Dictionnaire de metriques (accuracy, f1, report, etc.).
        output_path: Chemin de sortie du fichier JSON.

    Returns:
        Chemin du fichier JSON sauvegarde.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metriques sauvegardees : {output_path}")
    return output_path
