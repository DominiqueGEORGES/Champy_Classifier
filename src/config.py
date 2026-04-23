"""Configuration du projet via Pydantic Settings.

Charge les parametres depuis le fichier .env, avec possibilite
de surcharge par fichier YAML pour les hyperparametres d'entrainement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Chemins du projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "split"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Configuration MLflow / DagsHub
# ---------------------------------------------------------------------------
class MLflowSettings(BaseSettings):
    """Parametres de connexion au serveur MLflow.

    Charge automatiquement depuis les variables d'environnement
    ou le fichier .env a la racine du projet.
    """

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow"
    dagshub_user: str = "LoicFocraud"
    dagshub_token: str = ""


# ---------------------------------------------------------------------------
# Hyperparametres d'entrainement
# ---------------------------------------------------------------------------
class TrainingConfig(BaseSettings):
    """Hyperparametres d'entrainement, surchargeables par YAML ou .env.

    Les variables d'environnement doivent etre prefixees par TRAIN_
    (ex: TRAIN_LR=0.001, TRAIN_BATCH_SIZE=32).
    """

    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env", extra="ignore")

    model_name: str = "resnet50"
    num_classes: int = 30
    pretrained: bool = True

    # Fine-tuning deux phases :
    # - phase 1 : backbone gele, seule la tete est entrainee (lr_phase1)
    # - phase 2 : backbone degele, fine-tuning complet (lr_phase2)
    # Si freeze_backbone_epochs == 0, la phase 1 est sautee.
    freeze_backbone_epochs: int = 10
    total_epochs: int = 30
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-5

    batch_size: int = 16
    seed: int = 42

    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"

    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"

    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 0  # Par defaut 0 sur Windows (fork non supporte)

    image_size: int = 224
    augmentation: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Charge la configuration depuis un fichier YAML.

        Les valeurs du YAML surchargent les valeurs par defaut.
        Les variables .env restent utilisees pour les champs absents du YAML.

        Args:
            path: Chemin vers le fichier YAML.

        Returns:
            Instance de TrainingConfig avec les valeurs fusionnees.

        Raises:
            FileNotFoundError: Si le fichier YAML n'existe pas.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            msg = f"Fichier de configuration introuvable : {yaml_path}"
            raise FileNotFoundError(msg)
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)


# ---------------------------------------------------------------------------
# Configuration API / Serving
# ---------------------------------------------------------------------------
class ServingSettings(BaseSettings):
    """Parametres du serveur FastAPI pour l'inference ONNX.

    Utilise les variables d'environnement sans prefixe
    (API_HOST, API_PORT, MODEL_PATH, etc.).
    """

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: Path = MODELS_DIR / "model.onnx"
    environment: str = "development"
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Fonctions de chargement
# ---------------------------------------------------------------------------
def get_mlflow_settings() -> MLflowSettings:
    """Retourne les parametres MLflow depuis l'environnement."""
    return MLflowSettings()


def get_training_config(yaml_path: str | Path | None = None) -> TrainingConfig:
    """Retourne la configuration d'entrainement.

    Args:
        yaml_path: Chemin optionnel vers un fichier YAML de surcharge.
            Si None, utilise uniquement les valeurs par defaut et .env.

    Returns:
        Instance de TrainingConfig.
    """
    if yaml_path is not None:
        return TrainingConfig.from_yaml(yaml_path)
    return TrainingConfig()


def get_serving_settings() -> ServingSettings:
    """Retourne les parametres du serveur d'inference depuis l'environnement."""
    return ServingSettings()
