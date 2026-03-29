"""Modele ResNet50 pour la classification de champignons.

Utilise le transfer learning depuis les poids ImageNet pre-entraines.
La tete de classification est remplacee par une couche Dense adaptee
aux 30 classes du projet. Le backbone peut etre gele ou degle
selon la phase d'entrainement.
"""

from __future__ import annotations

import torch.nn as nn
from loguru import logger
from torchvision import models


def create_resnet50(
    num_classes: int = 30,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Cree un ResNet50 avec tete de classification personnalisee.

    Remplace la derniere couche fully-connected du ResNet50 par
    une tete adaptee au nombre de classes du projet.

    Args:
        num_classes: Nombre de classes de sortie.
        pretrained: Si True, charge les poids ImageNet pre-entraines.
        freeze_backbone: Si True, gele toutes les couches sauf la tete.

    Returns:
        Modele ResNet50 pret pour l'entrainement ou l'inference.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Remplacer la tete de classification
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        freeze_backbone_layers(model)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ResNet50 cree : {num_params:,} params total, "
        f"{num_trainable:,} entrainables, "
        f"backbone {'gele' if freeze_backbone else 'degle'}"
    )

    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    """Gele toutes les couches du backbone (tout sauf model.fc).

    Utile pour la phase 1 du transfer learning ou seule
    la tete de classification est entrainee.

    Args:
        model: Modele ResNet50 dont le backbone sera gele.
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    logger.info("Backbone gele (seule la tete fc est entrainable)")


def unfreeze_backbone_layers(
    model: nn.Module,
    unfreeze_from: int = 0,
) -> None:
    """Degele les couches du backbone a partir d'un certain layer.

    Pour le fine-tuning, on degele progressivement les couches
    profondes du ResNet. Les layers ResNet50 sont :
    conv1, bn1, layer1, layer2, layer3, layer4, fc.

    Args:
        model: Modele ResNet50 a degeler.
        unfreeze_from: Indice du layer a partir duquel degeler
            (0=tout, 1=layer1+, 2=layer2+, 3=layer3+, 4=layer4+fc).
    """
    layer_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    unfreeze_set = set(layer_names[max(0, unfreeze_from) :])

    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level in unfreeze_set:
            param.requires_grad = True

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Backbone degle a partir de layer {unfreeze_from} "
        f"({num_trainable:,} params entrainables)"
    )
