"""Callbacks pour l'entrainement : early stopping et checkpointing.

Ces callbacks sont appeles a la fin de chaque epoch de validation
pour decider de sauvegarder le modele et/ou d'arreter l'entrainement.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger


class EarlyStopping:
    """Arrete l'entrainement si la metrique ne s'ameliore plus.

    Surveille une metrique de validation (par defaut val_loss) et
    declenche l'arret apres `patience` epochs sans amelioration.

    Attributes:
        patience: Nombre d'epochs sans amelioration avant l'arret.
        min_delta: Amelioration minimale pour considerer un progres.
        mode: 'min' pour minimiser (loss), 'max' pour maximiser (accuracy).
        counter: Nombre d'epochs depuis la derniere amelioration.
        best_score: Meilleure valeur observee.
        should_stop: True si l'entrainement doit s'arreter.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Initialise le callback d'early stopping.

        Args:
            patience: Nombre d'epochs sans amelioration avant l'arret.
            min_delta: Seuil minimal d'amelioration.
            mode: 'min' (val_loss) ou 'max' (val_accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        """Verifie si la valeur actuelle est meilleure que la reference.

        Args:
            current: Valeur de la metrique a l'epoch courante.

        Returns:
            True si c'est une amelioration.
        """
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, metric_value: float) -> bool:
        """Met a jour l'etat apres une epoch de validation.

        Args:
            metric_value: Valeur de la metrique surveillee.

        Returns:
            True si l'entrainement doit s'arreter.
        """
        if self._is_improvement(metric_value):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping : pas d'amelioration depuis {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(
                    f"EarlyStopping declenche apres {self.patience} epochs sans amelioration"
                )
        return self.should_stop


class ModelCheckpoint:
    """Sauvegarde le meilleur modele pendant l'entrainement.

    Compare la metrique de validation a chaque epoch et sauvegarde
    le modele si c'est la meilleure valeur observee.

    Attributes:
        save_path: Chemin de sauvegarde du modele.
        mode: 'min' (loss) ou 'max' (accuracy).
        best_score: Meilleure valeur observee.
    """

    def __init__(
        self,
        save_path: Path,
        mode: str = "min",
    ) -> None:
        """Initialise le callback de checkpointing.

        Args:
            save_path: Chemin ou sauvegarder le meilleur modele (.pt).
            mode: 'min' (val_loss) ou 'max' (val_accuracy).
        """
        self.save_path = Path(save_path)
        self.mode = mode
        self.best_score: float | None = None

    def _is_improvement(self, current: float) -> bool:
        """Verifie si la valeur actuelle est la meilleure.

        Args:
            current: Valeur de la metrique a l'epoch courante.

        Returns:
            True si c'est un nouveau meilleur score.
        """
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def step(
        self,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
    ) -> bool:
        """Sauvegarde le modele si la metrique s'est amelioree.

        Args:
            metric_value: Valeur de la metrique surveillee.
            model: Modele PyTorch a sauvegarder.
            optimizer: Optimiseur (optionnel, sauvegarde dans le checkpoint).
            epoch: Numero de l'epoch courante.

        Returns:
            True si le modele a ete sauvegarde.
        """
        if not self._is_improvement(metric_value):
            return False

        self.best_score = metric_value
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": self.best_score,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, self.save_path)
        logger.info(
            f"Checkpoint sauvegarde : {self.save_path} "
            f"(epoch={epoch}, {self.mode}={metric_value:.4f})"
        )
        return True
