"""Schemas Pydantic pour les requetes et reponses du service BentoML.

Definit les modeles de validation pour les endpoints /predict, /health
et /model/info. Reproduit a l'identique les schemas de l'API FastAPI
historique (``src/serving/schemas.py``) pour garantir la compatibilite
des clients (Streamlit, tests d'integration, scripts de benchmark).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Une prediction individuelle dans le top-N.

    Attributes:
        species: Nom scientifique de l'espece predite.
        confidence: Score de confiance entre 0 et 1.
        rank: Rang de la prediction (1 = plus probable).
    """

    species: str = Field(description="Nom scientifique de l'espece")
    confidence: float = Field(ge=0, le=1, description="Score de confiance [0, 1]")
    rank: int = Field(ge=1, description="Rang de la prediction")


class PredictionResponse(BaseModel):
    """Reponse de l'endpoint /predict.

    Attributes:
        predictions: Liste des top-N predictions triees par confiance.
        model_version: Version du modele utilise pour l'inference.
    """

    predictions: list[PredictionItem] = Field(description="Top-N predictions")
    model_version: str = Field(description="Version du modele ONNX")


class HealthResponse(BaseModel):
    """Reponse de l'endpoint /health.

    Attributes:
        status: Etat du service ('healthy', 'no_model', 'error').
        model_loaded: True si le modele ONNX est charge en memoire.
        model_version: Version du modele, ou None si non charge.
    """

    status: str = Field(description="Etat du service")
    model_loaded: bool = Field(description="Modele charge en memoire")
    model_version: str | None = Field(default=None, description="Version du modele")


class ModelInfoResponse(BaseModel):
    """Reponse de l'endpoint /model/info.

    Attributes:
        model_path: Tag du modele dans le Model Store BentoML.
        model_version: Version du modele.
        num_classes: Nombre de classes de sortie.
        class_names: Liste des noms de classes.
        input_shape: Forme attendue de l'entree (ex: [1, 3, 224, 224]).
        architecture: Backbone du modele (ex: 'convnext_tiny').
    """

    model_path: str = Field(description="Tag du modele dans le Model Store BentoML")
    model_version: str = Field(description="Version du modele")
    num_classes: int = Field(description="Nombre de classes")
    class_names: list[str] = Field(description="Noms des classes")
    input_shape: list[int] = Field(description="Forme de l'entree attendue")
    architecture: str = Field(description="Backbone du modele")


class ErrorResponse(BaseModel):
    """Reponse d'erreur standard.

    Attributes:
        detail: Message d'erreur descriptif.
    """

    detail: str = Field(description="Message d'erreur")
