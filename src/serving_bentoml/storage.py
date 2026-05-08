"""Stockage SQLite des predictions du service BentoML.

Persiste chaque prediction en base SQLite locale pour alimenter le Quality
Monitor (Bloc R) et la detection de derive (Bloc M3 - Evidently). Le
schema est minimal mais suffisant pour les besoins de monitoring : id,
timestamp, hash de l'image (deduplication), classe predite, confiance,
top-5 complet (JSON), et latence d'inference.

Concurrence
-----------
Le service BentoML peut traiter plusieurs requetes en parallele
(adaptive batching, workers async). Pour eviter les ``database is
locked`` :

- ``PRAGMA journal_mode=WAL`` : permet plusieurs lecteurs simultanes
  pendant qu'un ecriveur ecrit.
- ``PRAGMA synchronous=NORMAL`` : compromis sécurité / performance
  recommande pour WAL.
- ``PRAGMA busy_timeout=5000`` : si un autre ecriveur tient le verrou,
  attendre jusqu'a 5 secondes avant de remonter une erreur.
- ``aiosqlite`` : driver asyncio, chaque connexion porte son propre
  thread interne. Les operations asynchrones sont serialisees par le
  thread, donc partager une seule connexion entre coroutines est sur.

Cela garantit qu'un debit raisonnable d'inferences (~100 req/s) peut
etre stocke sans perte ni blocage du chemin chaud du predict.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    top5_json TEXT NOT NULL,
    latency_ms REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions(timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_class
    ON predictions(predicted_class);
"""


@dataclass(frozen=True)
class PredictionRecord:
    """Une prediction telle que stockee en base.

    Attributes:
        id: UUID4 sous forme de string.
        timestamp: Horodatage UTC ISO 8601.
        image_hash: SHA256 hex digest des pixels de l'image (dedup).
        predicted_class: Espece du top-1.
        confidence: Confiance [0, 1] du top-1.
        top5: Liste de dictionnaires ``{species, confidence, rank}``.
        latency_ms: Latence d'inference end-to-end en millisecondes.
    """

    id: str
    timestamp: str
    image_hash: str
    predicted_class: str
    confidence: float
    top5: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ClassCount:
    """Resultat d'une agregation par classe.

    Attributes:
        predicted_class: Nom de l'espece.
        count: Nombre d'occurrences sur la fenetre temporelle.
    """

    predicted_class: str
    count: int


class PredictionStore:
    """Store SQLite asynchrone des predictions de l'API.

    Le store ouvre une connexion ``aiosqlite`` persistante (un thread
    interne dedie) et l'utilise pour toutes les operations. La premiere
    methode appelee declenche l'initialisation paresseuse (schema +
    PRAGMAs WAL). Le store est sur a partager entre coroutines.

    Attributes:
        db_path: Chemin du fichier SQLite.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialise le store sans ouvrir la base.

        L'ouverture effective est differee a ``init()`` pour rester
        compatible avec un constructeur synchrone (cas BentoML 1.4 ou
        ``__init__`` ne peut pas etre ``async``). Le service appellera
        ``init()`` la premiere fois qu'il aura besoin du store.

        Args:
            db_path: Chemin absolu du fichier SQLite a utiliser.
        """
        self.db_path: Path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._initialized: bool = False

    async def init(self) -> None:
        """Ouvre la base et applique le schema + les PRAGMAs WAL.

        Cree le fichier SQLite et le repertoire parent s'ils n'existent
        pas. Idempotent : peut etre rappelee sans effet apres la
        premiere initialisation reussie.
        """
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self.db_path)
        # WAL : multi-readers + 1 writer concurrent, pas de blocage des
        # lectures pendant l'ecriture.
        await conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL synchronous : compromis recommande pour WAL.
        await conn.execute("PRAGMA synchronous=NORMAL")
        # Si la base est verrouillee par un autre ecriveur, attendre
        # jusqu'a 5 secondes avant de lever une erreur.
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.executescript(_SCHEMA)
        await conn.commit()
        # Active la representation Row -> dict pour les get_recent.
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        self._initialized = True
        logger.info(f"PredictionStore initialise : {self.db_path} (WAL, busy_timeout=5000ms)")

    @property
    def is_open(self) -> bool:
        """Indique si le store est pret a recevoir des operations.

        Returns:
            True si ``init()`` a ete appelee avec succes.
        """
        return self._initialized and self._conn is not None

    async def save_prediction(
        self,
        image_hash: str,
        predicted_class: str,
        confidence: float,
        top5: Sequence[dict[str, Any]],
        latency_ms: float,
        timestamp: datetime | None = None,
    ) -> str:
        """Persiste une prediction en base.

        Cette coroutine est concue pour etre lancee en fire-and-forget
        depuis le hot path de ``/predict`` (via ``asyncio.create_task``)
        pour ne pas bloquer la reponse a l'utilisateur.

        Args:
            image_hash: SHA256 hex de l'image (dedup, traçabilite).
            predicted_class: Espece du top-1.
            confidence: Confiance du top-1, [0, 1].
            top5: Liste des 5 meilleures predictions (dicts serialisables).
            latency_ms: Latence d'inference end-to-end en ms.
            timestamp: Horodatage UTC ; ``None`` = ``datetime.now(UTC)``.

        Returns:
            UUID de la ligne inseree (forme string).

        Raises:
            RuntimeError: Si le store n'a pas ete initialise.
        """
        if not self.is_open:
            raise RuntimeError("PredictionStore non initialise. Appeler init() avant.")
        assert self._conn is not None  # narrow pour mypy
        record_id = str(uuid.uuid4())
        ts = (timestamp or datetime.now(UTC)).isoformat()
        await self._conn.execute(
            """
            INSERT INTO predictions
                (id, timestamp, image_hash, predicted_class, confidence,
                 top5_json, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                ts,
                image_hash,
                predicted_class,
                float(confidence),
                json.dumps(list(top5), ensure_ascii=False),
                float(latency_ms),
            ),
        )
        await self._conn.commit()
        return record_id

    async def get_recent(self, hours: int = 24, limit: int = 1000) -> list[PredictionRecord]:
        """Recupere les predictions des ``hours`` dernieres heures.

        Args:
            hours: Fenetre temporelle, en heures (defaut 24).
            limit: Nombre max de lignes a retourner (defaut 1000).

        Returns:
            Liste de ``PredictionRecord`` triee par timestamp decroissant.

        Raises:
            RuntimeError: Si le store n'a pas ete initialise.
        """
        if not self.is_open:
            raise RuntimeError("PredictionStore non initialise. Appeler init() avant.")
        assert self._conn is not None
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        async with self._conn.execute(
            """
            SELECT id, timestamp, image_hash, predicted_class, confidence,
                   top5_json, latency_ms
            FROM predictions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            PredictionRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                image_hash=row["image_hash"],
                predicted_class=row["predicted_class"],
                confidence=float(row["confidence"]),
                top5=json.loads(row["top5_json"]),
                latency_ms=float(row["latency_ms"]),
            )
            for row in rows
        ]

    async def get_class_distribution(self, since: datetime | None = None) -> list[ClassCount]:
        """Compte les predictions par classe depuis un instant donne.

        Args:
            since: Borne inferieure UTC (incluse). ``None`` = pas de
                borne (toute la base).

        Returns:
            Liste triee par count decroissant.

        Raises:
            RuntimeError: Si le store n'a pas ete initialise.
        """
        if not self.is_open:
            raise RuntimeError("PredictionStore non initialise. Appeler init() avant.")
        assert self._conn is not None
        if since is None:
            query = """
                SELECT predicted_class, COUNT(*) AS cnt
                FROM predictions
                GROUP BY predicted_class
                ORDER BY cnt DESC
            """
            params: tuple[Any, ...] = ()
        else:
            query = """
                SELECT predicted_class, COUNT(*) AS cnt
                FROM predictions
                WHERE timestamp >= ?
                GROUP BY predicted_class
                ORDER BY cnt DESC
            """
            params = (since.isoformat(),)
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            ClassCount(predicted_class=row["predicted_class"], count=int(row["cnt"]))
            for row in rows
        ]

    async def count(self) -> int:
        """Compte le nombre total de lignes en base.

        Utile pour les tests et les health checks.

        Returns:
            Nombre total d'enregistrements.

        Raises:
            RuntimeError: Si le store n'a pas ete initialise.
        """
        if not self.is_open:
            raise RuntimeError("PredictionStore non initialise. Appeler init() avant.")
        assert self._conn is not None
        async with self._conn.execute("SELECT COUNT(*) FROM predictions") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        """Ferme la connexion proprement.

        Idempotent : peut etre rappelee plusieurs fois sans effet.
        """
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            logger.info(f"PredictionStore ferme : {self.db_path}")
