"""Tests unitaires du ``PredictionStore`` (SQLite + WAL + concurrence).

Verifient :
- Initialisation idempotente (init() peut etre rappelee)
- Activation effective du mode WAL
- Insertion + recuperation d'un enregistrement
- ``get_recent`` filtre correctement par fenetre temporelle
- ``get_class_distribution`` agrege correctement avec et sans borne
- Concurrence : 100 ecritures simultanees sans perte ni ``database is locked``
- Cleanup propre (close idempotente)

Le module n'utilise pas ``pytest-asyncio`` (pas dans les deps du projet) :
chaque test execute son scenario via ``asyncio.run`` dans une fonction
synchrone, ce qui garde la compatibilite avec la config pytest existante.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.serving_bentoml.storage import PredictionStore


def _run(coro: object) -> object:
    """Execute une coroutine de test dans un nouvel event loop.

    Args:
        coro: Coroutine a executer.

    Returns:
        La valeur retournee par la coroutine.
    """
    return asyncio.run(coro)  # type: ignore[arg-type]


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Fournit un chemin de DB SQLite isole par test.

    Args:
        tmp_path: Repertoire temporaire pytest.

    Returns:
        Chemin vers ``predictions.db`` dans le repertoire tmp.
    """
    return tmp_path / "predictions.db"


def test_init_creates_file_and_enables_wal(db_path: Path) -> None:
    """init() cree le fichier SQLite et active le mode WAL."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        assert not store.is_open
        await store.init()
        assert store.is_open
        assert db_path.exists()
        assert store._conn is not None
        # Verifie que le mode WAL est bien actif.
        async with store._conn.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"
        await store.close()

    _run(scenario())


def test_init_is_idempotent(db_path: Path) -> None:
    """Appeler init() deux fois ne doit pas planter ni dupliquer la connexion."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        first_conn = store._conn
        await store.init()
        assert store._conn is first_conn
        await store.close()

    _run(scenario())


def test_save_and_get_recent_round_trip(db_path: Path) -> None:
    """Un enregistrement insere doit etre lu identique par get_recent."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        record_id = await store.save_prediction(
            image_hash="a" * 64,
            predicted_class="Amanita rubescens",
            confidence=0.95,
            top5=[
                {"species": "Amanita rubescens", "confidence": 0.95, "rank": 1},
                {"species": "Amanita phalloides", "confidence": 0.03, "rank": 2},
            ],
            latency_ms=42.5,
        )
        records = await store.get_recent(hours=1)
        assert len(records) == 1
        rec = records[0]
        assert rec.id == record_id
        assert rec.image_hash == "a" * 64
        assert rec.predicted_class == "Amanita rubescens"
        assert abs(rec.confidence - 0.95) < 1e-9
        assert rec.top5[0]["species"] == "Amanita rubescens"
        assert rec.latency_ms == pytest.approx(42.5)
        await store.close()

    _run(scenario())


def test_get_recent_filters_by_window(db_path: Path) -> None:
    """get_recent ne doit retourner que les lignes dans la fenetre."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        old_ts = datetime.now(UTC) - timedelta(hours=48)
        recent_ts = datetime.now(UTC) - timedelta(minutes=5)
        await store.save_prediction(
            image_hash="b" * 64,
            predicted_class="Old",
            confidence=0.9,
            top5=[],
            latency_ms=10.0,
            timestamp=old_ts,
        )
        await store.save_prediction(
            image_hash="c" * 64,
            predicted_class="Recent",
            confidence=0.9,
            top5=[],
            latency_ms=10.0,
            timestamp=recent_ts,
        )
        records = await store.get_recent(hours=1)
        assert len(records) == 1
        assert records[0].predicted_class == "Recent"
        # Fenetre large : les deux lignes
        records_all = await store.get_recent(hours=72)
        assert len(records_all) == 2
        await store.close()

    _run(scenario())


def test_get_class_distribution(db_path: Path) -> None:
    """L'agregation par classe doit compter correctement."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        for species, n in [("Amanita rubescens", 3), ("Boletus edulis", 2)]:
            for _ in range(n):
                await store.save_prediction(
                    image_hash="d" * 64,
                    predicted_class=species,
                    confidence=0.9,
                    top5=[],
                    latency_ms=10.0,
                )
        dist = await store.get_class_distribution()
        as_dict = {c.predicted_class: c.count for c in dist}
        assert as_dict == {"Amanita rubescens": 3, "Boletus edulis": 2}
        # Le tri par count decroissant est garanti
        assert dist[0].predicted_class == "Amanita rubescens"
        await store.close()

    _run(scenario())


def test_get_class_distribution_with_since(db_path: Path) -> None:
    """L'agregation avec ``since`` doit filtrer la fenetre temporelle."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        old_ts = datetime.now(UTC) - timedelta(hours=48)
        recent_ts = datetime.now(UTC) - timedelta(minutes=5)
        await store.save_prediction(
            image_hash="e" * 64,
            predicted_class="OldClass",
            confidence=0.9,
            top5=[],
            latency_ms=10.0,
            timestamp=old_ts,
        )
        await store.save_prediction(
            image_hash="f" * 64,
            predicted_class="RecentClass",
            confidence=0.9,
            top5=[],
            latency_ms=10.0,
            timestamp=recent_ts,
        )
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        dist = await store.get_class_distribution(since=cutoff)
        assert {c.predicted_class for c in dist} == {"RecentClass"}
        await store.close()

    _run(scenario())


def test_concurrent_writes_no_loss(db_path: Path) -> None:
    """100 ecritures concurrentes via gather doivent toutes aboutir.

    Le mode WAL + busy_timeout=5000ms doit absorber la contention sans
    lever ``database is locked``.
    """

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()

        async def write_one(i: int) -> str:
            return await store.save_prediction(
                image_hash=f"{i:064d}",
                predicted_class=f"Species{i % 5}",
                confidence=0.5 + (i % 50) / 100.0,
                top5=[],
                latency_ms=float(i),
            )

        results = await asyncio.gather(*(write_one(i) for i in range(100)))
        assert len(results) == 100
        assert len(set(results)) == 100  # IDs uniques
        assert await store.count() == 100
        await store.close()

    _run(scenario())


def test_count_and_close_are_idempotent(db_path: Path) -> None:
    """count() retourne 0 sur une base vide ; close() est idempotente."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        await store.init()
        assert await store.count() == 0
        await store.close()
        # close idempotente : ne plante pas si appele a nouveau
        await store.close()

    _run(scenario())


def test_save_before_init_raises(db_path: Path) -> None:
    """save_prediction sans init() doit lever RuntimeError explicite."""

    async def scenario() -> None:
        store = PredictionStore(db_path)
        with pytest.raises(RuntimeError, match="non initialise"):
            await store.save_prediction(
                image_hash="0" * 64,
                predicted_class="X",
                confidence=0.5,
                top5=[],
                latency_ms=1.0,
            )

    _run(scenario())
