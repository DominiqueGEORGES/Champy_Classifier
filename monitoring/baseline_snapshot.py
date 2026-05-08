"""Calcule la baseline de reference pour la detection de drift.

Lit le manifest de split, charge le modele ONNX courant, fait tourner
l'inference sur le test set et agrege une distribution de reference
(class distribution + confidence stats par classe + histogramme global)
qui sera ensuite comparee aux predictions de production par
``run_drift_report.py`` (Evidently DataDriftPreset).

Le snapshot est sauvegarde en JSON pour etre auto-suffisant : pas besoin
d'un access au modele ou aux donnees au moment de la generation du
rapport. A relancer :

- une fois apres la mise en place (boot)
- a chaque promotion d'un nouveau modele en production (changement de
  ``best_model.onnx`` ou changement de tag ``latest`` dans le Model
  Store BentoML)
- a chaque re-curation des donnees (filtre OpenCLIP rejoue, nouveau
  manifest)

Usage :
    python monitoring/baseline_snapshot.py
    python monitoring/baseline_snapshot.py --model models/convnext_tiny_v2.0.0.onnx
    python monitoring/baseline_snapshot.py --split val --output monitoring/baseline_val.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger

# Reutilise le preprocessing strict du serving pour garantir que la
# baseline est calculee dans les memes conditions que les predictions
# servies en production (Resize 256 -> CenterCrop 224 -> Normalize ImageNet).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.serving_bentoml.preprocessing import preprocess_image  # noqa: E402

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_model.onnx"
DEFAULT_CLASS_NAMES_PATH = REPO_ROOT / "models" / "class_names.json"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "raw" / "Mushrooms_images"
DEFAULT_OUTPUT = REPO_ROOT / "monitoring" / "baseline_reference.json"
DEFAULT_SPLIT = "test"

# Buckets pour l'histogramme global des confidences.
CONFIDENCE_BUCKETS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def parse_args() -> argparse.Namespace:
    """Lit les arguments CLI.

    Returns:
        Namespace avec ``model``, ``class_names``, ``manifest``,
        ``images_dir``, ``split``, ``output``.
    """
    parser = argparse.ArgumentParser(
        description="Genere une baseline de reference pour la detection de drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Chemin du modele ONNX a utiliser pour l'inference.",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=DEFAULT_CLASS_NAMES_PATH,
        help="Chemin du JSON listant les especes (ordre = label_map).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV de split (split, path, label).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Repertoire racine ou rglob trouve les images.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test"],
        help="Split a utiliser pour la baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin de sortie du JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre d'images (debug).",
    )
    return parser.parse_args()


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Calcule un softmax numeriquement stable.

    Args:
        logits: Vecteur 1D de logits.

    Returns:
        Vecteur de probabilites de meme dimension, sommant a 1.
    """
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    """Construit un index ``nom de fichier -> chemin absolu``.

    Le repertoire ``data/raw/Mushrooms_images`` peut contenir plusieurs
    centaines de milliers de fichiers ; faire un ``rglob`` par image
    serait O(N x M) avec N images cherchees et M fichiers totaux. On
    fait un seul scan O(M) en amont, puis une lookup O(1) par image.
    Quand le meme nom apparait plusieurs fois (rare), on conserve le
    premier rencontre.

    Args:
        images_dir: Racine a explorer.

    Returns:
        Mapping ``nom de fichier -> chemin absolu``.
    """
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.is_file() and path.name not in index:
            index[path.name] = path
    return index


def compute_baseline(
    model_path: Path,
    class_names_path: Path,
    manifest: Path,
    images_dir: Path,
    split: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Calcule la baseline en faisant tourner l'inference sur le split.

    Args:
        model_path: Chemin du fichier ONNX.
        class_names_path: Chemin du JSON des classes.
        manifest: CSV de split.
        images_dir: Racine d'images.
        split: ``train``, ``val`` ou ``test``.
        limit: Si non ``None``, ne traite que les ``limit`` premieres images.

    Returns:
        Dictionnaire structure (cf. ``main``).

    Raises:
        FileNotFoundError: Si un fichier requis est absent.
        ValueError: Si le split n'est pas trouve dans le manifest.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modele ONNX introuvable : {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"class_names.json introuvable : {class_names_path}")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest introuvable : {manifest}")

    with open(class_names_path, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    logger.info(f"Classes : {len(class_names)} especes")

    df = pd.read_csv(manifest)
    df_split = df[df["split"] == split].reset_index(drop=True)
    if df_split.empty:
        raise ValueError(f"Aucune ligne pour le split '{split}' dans {manifest}")
    if limit is not None:
        df_split = df_split.head(limit)
    logger.info(f"Split '{split}' : {len(df_split)} images")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    logger.info(f"Indexation des images sous {images_dir}...")
    index = _build_image_index(images_dir)
    logger.info(f"  {len(index)} fichiers indexes")

    # Statistiques cumulees.
    per_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "conf_sum": 0.0, "conf_min": 1.0, "conf_max": 0.0}
    )
    all_confs: list[float] = []
    n_top1_correct = 0
    n_processed = 0
    skipped: list[str] = []
    start = time.perf_counter()

    for _, row in df_split.iterrows():
        path = index.get(str(row["path"]))
        if path is None:
            skipped.append(str(row["path"]))
            continue
        try:
            with open(path, "rb") as f:
                arr = preprocess_image(f.read())
        except Exception as exc:
            skipped.append(f"{row['path']} ({exc})")
            continue

        logits = session.run(None, {input_name: arr})[0][0]
        probs = _softmax(logits)
        top1_idx = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])
        top1_species = class_names[top1_idx]

        per_class[top1_species]["count"] += 1
        per_class[top1_species]["conf_sum"] += top1_conf
        per_class[top1_species]["conf_min"] = min(per_class[top1_species]["conf_min"], top1_conf)
        per_class[top1_species]["conf_max"] = max(per_class[top1_species]["conf_max"], top1_conf)
        all_confs.append(top1_conf)
        if top1_species == row["label"]:
            n_top1_correct += 1
        n_processed += 1

        if n_processed % 200 == 0:
            elapsed = time.perf_counter() - start
            rate = n_processed / elapsed if elapsed > 0 else 0
            logger.info(f"  {n_processed}/{len(df_split)} ({rate:.1f} img/s)")

    elapsed = time.perf_counter() - start
    logger.info(f"Inference terminee : {n_processed} images en {elapsed:.1f}s")
    if skipped:
        logger.warning(f"{len(skipped)} images skipped (premieres : {skipped[:5]})")

    # Histogramme global des confidences.
    confs = np.asarray(all_confs, dtype=np.float64)
    histogram: list[dict[str, Any]] = []
    for low, high in itertools.pairwise(CONFIDENCE_BUCKETS):
        mask = (confs >= low) & (confs < high)
        histogram.append(
            {
                "bucket": f"[{low:.2f}, {high:.2f})",
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "share": float(mask.sum() / len(confs)) if len(confs) else 0.0,
            }
        )
    # Le dernier bucket inclut la borne haute pour ne pas perdre les 1.0 exacts.
    histogram[-1]["count"] = int(((confs >= CONFIDENCE_BUCKETS[-2]) & (confs <= 1.0)).sum())
    histogram[-1]["share"] = float(histogram[-1]["count"] / len(confs)) if len(confs) else 0.0

    # Finalisation per-class : confidence moyenne.
    per_class_out: dict[str, dict[str, Any]] = {}
    for species, stats in sorted(per_class.items(), key=lambda kv: -kv[1]["count"]):
        count = stats["count"]
        per_class_out[species] = {
            "count": count,
            "share": count / n_processed if n_processed else 0.0,
            "confidence_mean": stats["conf_sum"] / count if count else 0.0,
            "confidence_min": stats["conf_min"] if count else 0.0,
            "confidence_max": stats["conf_max"] if count else 0.0,
        }

    return {
        "metadata": {
            "model_path": str(model_path),
            "class_names_path": str(class_names_path),
            "manifest": str(manifest),
            "split": split,
            "n_images": int(n_processed),
            "n_skipped": len(skipped),
            "duration_seconds": round(elapsed, 2),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "global": {
            "top1_accuracy": n_top1_correct / n_processed if n_processed else 0.0,
            "confidence_mean": float(confs.mean()) if confs.size else 0.0,
            "confidence_p10": float(np.percentile(confs, 10)) if confs.size else 0.0,
            "confidence_p50": float(np.percentile(confs, 50)) if confs.size else 0.0,
            "confidence_p95": float(np.percentile(confs, 95)) if confs.size else 0.0,
        },
        "per_class": per_class_out,
        "confidence_histogram": histogram,
    }


def main() -> int:
    """Point d'entree CLI.

    Returns:
        Code de sortie : 0 si succes, 1 sinon.
    """
    args = parse_args()
    try:
        baseline = compute_baseline(
            model_path=args.model,
            class_names_path=args.class_names,
            manifest=args.manifest,
            images_dir=args.images_dir,
            split=args.split,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception(f"Echec compute_baseline : {exc}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    logger.success(f"Baseline ecrite : {args.output}")
    logger.info(
        f"  Top-1 accuracy : {baseline['global']['top1_accuracy']:.2%}, "
        f"conf moyenne : {baseline['global']['confidence_mean']:.4f}, "
        f"classes vues : {len(baseline['per_class'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
