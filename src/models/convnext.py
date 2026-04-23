"""Modèle ConvNeXt-Tiny pour la classification de champignons.

Utilise le transfer learning depuis les poids ImageNet pré-entraînés.
La tête de classification (``model.classifier[2]``) est remplacée par
une couche Linear adaptée aux 30 classes du projet. Le backbone
(``model.features``) peut être gelé ou dégelé selon la phase
d'entraînement.

ConvNeXt-Tiny a ~28M paramètres (vs 23M pour ResNet50) mais tourne
à ~30% de la latence inference ONNX ; à qualité égale, c'est un bon
candidat pour comparer les architectures.
"""

from __future__ import annotations

import torch.nn as nn
from loguru import logger
from torchvision import models


def create_convnext_tiny(
    num_classes: int = 30,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Crée un ConvNeXt-Tiny avec tête de classification personnalisée.

    Remplace la dernière couche Linear du classifier par une tête
    adaptée au nombre de classes du projet. Le reste du classifier
    (LayerNorm2d + Flatten) est conservé tel quel.

    Args:
        num_classes: Nombre de classes de sortie.
        pretrained: Si True, charge les poids ImageNet pré-entraînés.
        freeze_backbone: Si True, gèle ``model.features`` pour n'entraîner
            que la tête en phase 1.

    Returns:
        Modèle ConvNeXt-Tiny prêt pour l'entraînement ou l'inférence.
    """
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)

    # Remplacer la couche Linear finale du classifier (classifier[2])
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        freeze_features_layers(model)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ConvNeXt-Tiny créé : {num_params:,} params total, "
        f"{num_trainable:,} entraînables, "
        f"backbone {'gelé' if freeze_backbone else 'dégelé'}"
    )

    return model


def freeze_features_layers(model: nn.Module) -> None:
    """Gèle toutes les couches du backbone (``model.features``).

    Le classifier (LayerNorm + Flatten + Linear) reste entraînable,
    utile pour la phase 1 du transfer learning où seule la tête de
    classification est ajustée.

    Args:
        model: Modèle ConvNeXt dont le backbone sera gelé.
    """
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = False
    logger.info("Backbone ConvNeXt gelé (seul le classifier est entraînable)")


def unfreeze_features_layers(model: nn.Module) -> None:
    """Dégèle toutes les couches du modèle ConvNeXt.

    Utile pour la phase 2 du fine-tuning où le backbone entier est
    réentraîné avec un learning rate plus faible.

    Args:
        model: Modèle ConvNeXt à dégeler.
    """
    for param in model.parameters():
        param.requires_grad = True
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone ConvNeXt dégelé ({num_trainable:,} params entraînables)")
