"""Exporter Prometheus des metriques par container via l'API Docker.

Contexte : cAdvisor lit le filesystem interne de Docker (structure overlay2)
pour mapper les cgroups aux noms de containers. Avec le containerd image store
(snapshotter, defaut sur Docker 29+), cette structure n'existe plus au meme
endroit et cAdvisor n'attache plus les noms. Cet exporter contourne le probleme
en passant uniquement par l'API Docker (socket), independante du storage driver.

Metriques exposees, labellisees par container (name) :
  - docker_container_cpu_percent
  - docker_container_memory_usage_bytes
  - docker_container_memory_limit_bytes
  - docker_container_memory_percent
  - docker_container_network_rx_bytes
  - docker_container_network_tx_bytes
  - docker_container_block_read_bytes
  - docker_container_block_write_bytes
  - docker_container_up (1 si running, 0 sinon)

Un thread de fond collecte toutes les COLLECT_INTERVAL secondes et met a jour
les gauges ; l'endpoint /metrics sert les dernieres valeurs connues.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import docker
from prometheus_client import Gauge, start_http_server

# ---------------------------------------------------------------------
# Logging : format conforme aux conventions du projet
# (timestamp ISO 8601 avec timezone, "timestamp [LEVEL] logger.name: message")
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("docker_stats_exporter")

# ---------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9417"))
COLLECT_INTERVAL = int(os.environ.get("COLLECT_INTERVAL", "15"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

# Filtre des containers a exposer (regex applique sur le nom via re.search).
# Vide  -> tous les containers du daemon (exporter generique).
# "champy_" -> uniquement les containers du projet Champy.
# Utiliser "^champy_" pour ancrer strictement en debut de nom si besoin.
CONTAINER_FILTER = os.environ.get("CONTAINER_FILTER", "").strip()
try:
    FILTER_RE = re.compile(CONTAINER_FILTER) if CONTAINER_FILTER else None
except re.error as exc:
    logger.error("CONTAINER_FILTER invalide (%r): %s. Filtre desactive.", CONTAINER_FILTER, exc)
    FILTER_RE = None

LABELS = ["name"]

CPU_PERCENT = Gauge("docker_container_cpu_percent", "CPU usage percent", LABELS)
MEM_USAGE = Gauge("docker_container_memory_usage_bytes", "Memory usage in bytes", LABELS)
MEM_LIMIT = Gauge("docker_container_memory_limit_bytes", "Memory limit in bytes", LABELS)
MEM_PERCENT = Gauge("docker_container_memory_percent", "Memory usage percent", LABELS)
NET_RX = Gauge("docker_container_network_rx_bytes", "Network received bytes", LABELS)
NET_TX = Gauge("docker_container_network_tx_bytes", "Network transmitted bytes", LABELS)
BLK_READ = Gauge("docker_container_block_read_bytes", "Block IO read bytes", LABELS)
BLK_WRITE = Gauge("docker_container_block_write_bytes", "Block IO write bytes", LABELS)
CONTAINER_UP = Gauge("docker_container_up", "1 si le container est running", LABELS)

ALL_GAUGES = [
    CPU_PERCENT,
    MEM_USAGE,
    MEM_LIMIT,
    MEM_PERCENT,
    NET_RX,
    NET_TX,
    BLK_READ,
    BLK_WRITE,
    CONTAINER_UP,
]


def _cpu_percent(stats: dict[str, Any]) -> float:
    """Calcule le pourcentage CPU a partir des deltas (compatible cgroup v2)."""
    try:
        cpu = stats["cpu_stats"]
        precpu = stats["precpu_stats"]
        cpu_delta = cpu["cpu_usage"]["total_usage"] - precpu["cpu_usage"]["total_usage"]
        system_delta = cpu.get("system_cpu_usage", 0) - precpu.get("system_cpu_usage", 0)
        online = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage", [1]) or [1])
        if system_delta > 0 and cpu_delta >= 0:
            return float(round((cpu_delta / system_delta) * online * 100.0, 2))
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return 0.0


def _memory(stats: dict[str, Any]) -> tuple[float, float, float]:
    """Retourne (usage_reel, limite, pourcentage). cgroup v2 : on retire inactive_file."""
    try:
        mem = stats["memory_stats"]
        usage = mem.get("usage", 0)
        inactive = mem.get("stats", {}).get("inactive_file", 0)
        real = max(usage - inactive, 0)
        limit = mem.get("limit", 0)
        pct = round((real / limit) * 100.0, 2) if limit else 0.0
        return float(real), float(limit), pct
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0, 0.0, 0.0


def _network(stats: dict[str, Any]) -> tuple[float, float]:
    """Retourne (rx_total, tx_total) sommes sur toutes les interfaces."""
    nets = stats.get("networks") or {}
    rx = sum(n.get("rx_bytes", 0) for n in nets.values())
    tx = sum(n.get("tx_bytes", 0) for n in nets.values())
    return float(rx), float(tx)


def _block_io(stats: dict[str, Any]) -> tuple[float, float]:
    """Retourne (read_total, write_total). Peut etre vide en cgroup v2."""
    entries = stats.get("blkio_stats", {}).get("io_service_bytes_recursive") or []
    read = sum(e.get("value", 0) for e in entries if e.get("op", "").lower() in ("read", "rbytes"))
    write = sum(
        e.get("value", 0) for e in entries if e.get("op", "").lower() in ("write", "wbytes")
    )
    return float(read), float(write)


def _collect_one(container: Any) -> dict[str, Any] | None:
    """Recupere et calcule les metriques d'un container. Retourne un dict ou None."""
    try:
        name = container.name
        if container.status != "running":
            return {"name": name, "up": 0}
        stats = container.stats(stream=False)
        mem_usage, mem_limit, mem_pct = _memory(stats)
        net_rx, net_tx = _network(stats)
        blk_read, blk_write = _block_io(stats)
        return {
            "name": name,
            "up": 1,
            "cpu": _cpu_percent(stats),
            "mem_usage": mem_usage,
            "mem_limit": mem_limit,
            "mem_pct": mem_pct,
            "net_rx": net_rx,
            "net_tx": net_tx,
            "blk_read": blk_read,
            "blk_write": blk_write,
        }
    except Exception as exc:
        logger.warning("Echec collecte container %s: %s", getattr(container, "name", "?"), exc)
        return None


def _apply(metrics: dict[str, Any]) -> None:
    """Met a jour les gauges Prometheus pour un container."""
    name = metrics["name"]
    CONTAINER_UP.labels(name).set(metrics.get("up", 0))
    if metrics.get("up") != 1:
        # Container arrete : on remet les gauges de charge a zero.
        for g in (
            CPU_PERCENT,
            MEM_USAGE,
            MEM_LIMIT,
            MEM_PERCENT,
            NET_RX,
            NET_TX,
            BLK_READ,
            BLK_WRITE,
        ):
            g.labels(name).set(0)
        return
    CPU_PERCENT.labels(name).set(metrics["cpu"])
    MEM_USAGE.labels(name).set(metrics["mem_usage"])
    MEM_LIMIT.labels(name).set(metrics["mem_limit"])
    MEM_PERCENT.labels(name).set(metrics["mem_pct"])
    NET_RX.labels(name).set(metrics["net_rx"])
    NET_TX.labels(name).set(metrics["net_tx"])
    BLK_READ.labels(name).set(metrics["blk_read"])
    BLK_WRITE.labels(name).set(metrics["blk_write"])


def collect_loop(client: Any) -> None:
    """Boucle de collecte en thread de fond."""
    while True:
        start = time.monotonic()
        try:
            containers = client.containers.list(all=True)
            # Filtrage optionnel : ne garder que les containers dont le nom
            # matche CONTAINER_FILTER (ex: "champy_" pour ce projet uniquement).
            if FILTER_RE is not None:
                containers = [c for c in containers if FILTER_RE.search(c.name)]
            seen: set[str] = set()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(_collect_one, c): c for c in containers}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        _apply(result)
                        seen.add(result["name"])
            logger.info(
                "Collecte terminee : %d containers, %.2fs",
                len(seen),
                time.monotonic() - start,
            )
        except Exception as exc:
            logger.error("Erreur dans la boucle de collecte: %s", exc)
        elapsed = time.monotonic() - start
        time.sleep(max(COLLECT_INTERVAL - elapsed, 1))


def main() -> None:
    """Point d'entree : demarre le serveur /metrics et la boucle de collecte."""
    try:
        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
    except Exception as exc:
        logger.error("Impossible de joindre le daemon Docker: %s", exc)
        sys.exit(1)

    start_http_server(EXPORTER_PORT)
    filter_desc = CONTAINER_FILTER if FILTER_RE is not None else "(aucun, tous les containers)"
    logger.info(
        "Exporter demarre sur le port %d (intervalle %ds, filtre: %s)",
        EXPORTER_PORT,
        COLLECT_INTERVAL,
        filter_desc,
    )
    collect_loop(client)


if __name__ == "__main__":
    main()
