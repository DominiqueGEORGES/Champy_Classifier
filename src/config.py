"""Project configuration using Pydantic Settings.

Loads from .env file, with optional YAML override for training hyperparams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "split"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# MLflow / DagsHub settings
# ---------------------------------------------------------------------------
class MLflowSettings(BaseSettings):
    """MLflow tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = (
        "https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow"
    )
    dagshub_user: str = "LoicFocraud"
    dagshub_token: str = ""


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
class TrainingConfig(BaseSettings):
    """Training hyperparameters -- defaults can be overridden by YAML."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env", extra="ignore")

    model_name: str = "resnet50"
    num_classes: int = 30
    pretrained: bool = True
    freeze_backbone: bool = False

    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 30
    seed: int = 42

    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 2

    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"

    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 0  # Windows default -- test 2 with persistent_workers

    image_size: int = 224
    augmentation: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load config from a YAML file, with .env as fallback."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            msg = f"Config file not found: {yaml_path}"
            raise FileNotFoundError(msg)
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)


# ---------------------------------------------------------------------------
# API / Serving settings
# ---------------------------------------------------------------------------
class ServingSettings(BaseSettings):
    """FastAPI serving configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: Path = MODELS_DIR / "model.onnx"
    environment: str = "development"
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
def get_mlflow_settings() -> MLflowSettings:
    """Return MLflow settings singleton."""
    return MLflowSettings()


def get_training_config(yaml_path: str | Path | None = None) -> TrainingConfig:
    """Return training config, optionally from a YAML file."""
    if yaml_path is not None:
        return TrainingConfig.from_yaml(yaml_path)
    return TrainingConfig()


def get_serving_settings() -> ServingSettings:
    """Return serving settings singleton."""
    return ServingSettings()
