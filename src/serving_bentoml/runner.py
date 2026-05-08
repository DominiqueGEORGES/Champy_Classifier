"""Runner ONNX pour le service BentoML.

Encapsule le chargement du modele ONNX depuis le Model Store BentoML et
expose une interface ``predict(batch)`` simple. Cette couche d'abstraction
permet :

- d'isoler la logique de chargement du Model Store de la definition du
  service (``service.py``) ;
- de mocker le runner dans les tests unitaires sans demarrer un serveur
  BentoML complet ;
- de centraliser la lecture des metadonnees attachees au modele (labels,
  ``class_names``, architecture) au moment de l'inference.

Note : depuis BentoML 1.4 le sous-module ``bentoml.onnx`` est marque comme
deprecated mais reste fonctionnel. Voir PLAYBOOK.md (etape 6 BentoML) pour
le plan de migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Tag par defaut pour le modele dans le Model Store BentoML.
DEFAULT_MODEL_TAG = "champy_classifier:latest"


class OnnxRunner:
    """Wrapper autour du Model Store BentoML + ONNX Runtime.

    Charge le modele ONNX referencee dans le Model Store BentoML (via
    ``bentoml.onnx.load_model``) et expose un appel ``predict`` qui rend
    les logits bruts. Le softmax et le top-N sont laisses a l'appelant
    (le service) pour rester compatible avec un eventuel batching.

    Attributes:
        model_tag: Tag du modele dans le Model Store BentoML.
        session: Session ``onnxruntime.InferenceSession`` une fois chargee.
        class_names: Liste des noms d'especes (lue depuis class_names.json).
        labels: Labels associes au modele dans le Model Store (version,
            architecture, etc.).
        input_name: Nom du tenseur d'entree du graphe ONNX.
    """

    def __init__(self, model_tag: str = DEFAULT_MODEL_TAG) -> None:
        """Initialise le runner sans charger le modele.

        Le chargement effectif est differe a ``load()`` pour que le service
        puisse demarrer meme si le Model Store est temporairement vide
        (graceful degradation : ``/health`` repond ``no_model``).

        Args:
            model_tag: Tag du modele dans le Model Store BentoML.
        """
        self.model_tag: str = model_tag
        self.session: Any | None = None
        self.class_names: list[str] = []
        self.labels: dict[str, str] = {}
        self.input_name: str | None = None
        self.input_shape: list[int] = []

    def load(self) -> bool:
        """Charge le modele ONNX depuis le Model Store BentoML.

        Recupere le modele via ``bentoml.onnx.get(tag)`` puis instancie une
        session ``onnxruntime`` (CPU). Charge en parallele le fichier
        ``class_names.json`` co-localise dans le repertoire du modele.

        Returns:
            True si le modele a ete charge avec succes, False sinon.
        """
        try:
            import bentoml
            import onnxruntime as ort
        except ImportError as exc:
            logger.error(f"Dependance manquante pour le runner ONNX : {exc}")
            return False

        try:
            bento_model = bentoml.onnx.get(self.model_tag)
        except Exception as exc:
            logger.warning(
                f"Modele introuvable dans le Model Store BentoML ({self.model_tag}) : {exc}"
            )
            return False

        # ``bentoml.onnx.save_model`` ecrit toujours le graphe dans
        # ``saved_model.onnx`` a la racine du repertoire du modele. On
        # retombe sur un glob ``*.onnx`` pour rester robuste si BentoML
        # change cette convention dans une version ulterieure.
        model_dir = Path(bento_model.path)
        onnx_path = model_dir / "saved_model.onnx"
        if not onnx_path.exists():
            candidates = sorted(model_dir.glob("*.onnx"))
            if not candidates:
                logger.error(f"Aucun fichier .onnx trouve dans {model_dir}")
                return False
            onnx_path = candidates[0]

        try:
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            logger.error(f"Erreur d'initialisation onnxruntime : {exc}")
            return False

        # Metadonnees du graphe
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.input_shape = [int(s) if isinstance(s, int) else 0 for s in inputs[0].shape]

        # Labels attaches au modele dans le Model Store
        self.labels = dict(bento_model.info.labels or {})

        # Noms de classes : priorite aux ``custom_objects`` du Model Store
        # (embarques au moment de l'import), puis fallback sur le
        # ``class_names.json`` a la racine du repo (situation source-only).
        custom_class_names = (bento_model.custom_objects or {}).get("class_names")
        if isinstance(custom_class_names, list) and custom_class_names:
            self.class_names = [str(c) for c in custom_class_names]
            logger.info(f"Classes lues depuis custom_objects : {len(self.class_names)} especes")
        else:
            self.class_names = self._load_class_names(Path(bento_model.path))

        logger.info(
            f"Modele BentoML charge : {self.model_tag} "
            f"(architecture={self.labels.get('architecture', '?')}, "
            f"version={self.labels.get('version', '?')}, "
            f"classes={len(self.class_names)})"
        )
        return True

    @staticmethod
    def _load_class_names(model_dir: Path) -> list[str]:
        """Charge la liste des noms de classes.

        Cherche d'abord ``class_names.json`` co-localise avec le modele
        (situation ideale, packagee dans le bento), puis retombe sur le
        fichier a la racine du repo (``models/class_names.json``) pour
        rester compatible avec un service lance depuis la source.

        Args:
            model_dir: Repertoire du modele dans le Model Store.

        Returns:
            Liste des noms d'especes, ou liste vide si le fichier est absent.
        """
        candidates = [
            model_dir / "class_names.json",
            Path(__file__).resolve().parent.parent.parent / "models" / "class_names.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    names: list[str] = json.load(f)
                logger.info(f"Classes chargees depuis {candidate} : {len(names)} especes")
                return names
        logger.warning("Aucun fichier class_names.json trouve.")
        return []

    @property
    def is_loaded(self) -> bool:
        """Indique si le modele est pret a recevoir des inferences.

        Returns:
            True si la session ONNX est instanciee et qu'au moins une classe
            est connue, False sinon.
        """
        return self.session is not None and len(self.class_names) > 0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Calcule les logits bruts pour un batch d'images preprocessees.

        Args:
            batch: Tableau ``(N, 3, 224, 224)`` float32 normalise ImageNet.

        Returns:
            Tableau ``(N, num_classes)`` float32 contenant les logits bruts
            (le softmax et le top-N sont la responsabilite de l'appelant).

        Raises:
            RuntimeError: Si le runner n'a pas ete charge.
        """
        if self.session is None or self.input_name is None:
            raise RuntimeError("Runner ONNX non charge. Appeler load() avant predict().")
        outputs = self.session.run(None, {self.input_name: batch})
        return np.asarray(outputs[0], dtype=np.float32)
