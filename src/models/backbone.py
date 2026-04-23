"""Factory unifiée pour les backbones de classification.

Expose une API commune ``create_backbone(model_name, ...)`` qui dispatche
vers le constructeur spécifique à chaque architecture supportée :

    - ``resnet50``       -> ``src.models.resnet.create_resnet50``
    - ``convnext_tiny``  -> ``src.models.convnext.create_convnext_tiny``

Expose aussi des helpers de gel/dégel indépendants de l'architecture
(``freeze_backbone``, ``unfreeze_backbone``). Ces helpers s'appuient
sur un attribut ``head_module_name`` posé sur le modèle par la factory
pour savoir quelle partie garder entraînable quand le backbone est gelé.

Pour ajouter un nouveau backbone :

    1. Créer ``src/models/<backbone>.py`` avec un ``create_<name>`` qui
       remplace la tête de classification par un Linear vers ``num_classes``.
    2. Ajouter l'entrée dans ``_HEAD_MODULE`` ci-dessous avec le nom
       du module parent de la tête (``fc``, ``classifier``, etc.).
    3. Ajouter le branchement dans ``create_backbone``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.convnext import create_convnext_tiny
from src.models.resnet import create_resnet50

if TYPE_CHECKING:
    import torch.nn as nn

# Nom du module contenant la tête de classification, par architecture.
# Tout paramètre dont le nom ne commence pas par ce préfixe (+ '.')
# est considéré comme appartenant au backbone et donc gelé en phase 1.
_HEAD_MODULE: dict[str, str] = {
    "resnet50": "fc",
    "convnext_tiny": "classifier",
}

SUPPORTED_MODELS: tuple[str, ...] = tuple(_HEAD_MODULE.keys())


def create_backbone(
    model_name: str,
    num_classes: int = 30,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Crée un backbone selon son nom et lui attache ses métadonnées.

    Dispatche vers le constructeur spécifique et pose l'attribut
    ``head_module_name`` sur le modèle retourné pour que les helpers
    ``freeze_backbone`` / ``unfreeze_backbone`` sachent quelle partie
    préserver.

    Args:
        model_name: Nom du modèle (``'resnet50'`` ou ``'convnext_tiny'``).
        num_classes: Nombre de classes de sortie.
        pretrained: Si True, charge les poids ImageNet.
        freeze_backbone: Si True, gèle le backbone dès la création
            (phase 1 du fine-tuning).

    Returns:
        Modèle PyTorch avec attribut ``head_module_name``.

    Raises:
        ValueError: Si ``model_name`` n'est pas dans ``SUPPORTED_MODELS``.
    """
    if model_name not in _HEAD_MODULE:
        supported = ", ".join(sorted(_HEAD_MODULE.keys()))
        msg = f"Modèle inconnu : '{model_name}'. Modèles supportés : {supported}."
        raise ValueError(msg)

    if model_name == "resnet50":
        model = create_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:  # convnext_tiny
        model = create_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    # Métadonnée lue par freeze_backbone / unfreeze_backbone.
    model.head_module_name = _HEAD_MODULE[model_name]
    return model


def get_head_module_name(model: nn.Module) -> str:
    """Retourne le nom du module tête (via l'attribut posé par la factory).

    Args:
        model: Modèle créé par ``create_backbone``.

    Returns:
        Nom du module tête (``'fc'`` ou ``'classifier'``).

    Raises:
        AttributeError: Si le modèle n'a pas été créé via ``create_backbone``.
    """
    name = getattr(model, "head_module_name", None)
    if name is None:
        msg = (
            "Le modèle n'a pas d'attribut 'head_module_name'. "
            "Utilisez create_backbone() pour le créer, pas directement "
            "les constructeurs d'architecture."
        )
        raise AttributeError(msg)
    return str(name)


def freeze_backbone(model: nn.Module) -> None:
    """Gèle toutes les couches du backbone en gardant la tête entraînable.

    S'appuie sur ``model.head_module_name`` posé par ``create_backbone``
    pour identifier la partie à préserver. Tout paramètre dont le nom
    commence par ``<head_name>.`` reste entraînable.

    Args:
        model: Modèle créé par ``create_backbone``.
    """
    head_name = get_head_module_name(model)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_name + ".")


def unfreeze_backbone(model: nn.Module) -> None:
    """Dégèle toutes les couches du modèle (phase 2 du fine-tuning).

    Rend tous les paramètres entraînables, quelle que soit l'architecture.

    Args:
        model: Modèle à dégeler.
    """
    for param in model.parameters():
        param.requires_grad = True
