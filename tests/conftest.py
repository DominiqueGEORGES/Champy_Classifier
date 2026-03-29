"""Fixtures partagees pour les tests du projet Champy Classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Cree un mini-dataset de test avec manifest CSV.

    Genere 3 classes de 4 images chacune (12 images total),
    un manifest CSV avec un split train/val/test,
    et retourne les chemins necessaires.

    Args:
        tmp_path: Repertoire temporaire fourni par pytest.

    Returns:
        Tuple (manifest_path, data_dir, class_names).
    """
    classes = ["Amanita_muscaria", "Boletus_edulis", "Russula_emetica"]
    data_dir = tmp_path / "processed"

    # Creer les images de test (petites images JPEG 32x32)
    for cls in classes:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            img = Image.new("RGB", (32, 32), color=(i * 60, 100, 150))
            img.save(cls_dir / f"{i + 1}.jpg")

    # Creer le manifest CSV
    manifest_path = tmp_path / "split_manifest.csv"
    rows = []
    for cls in classes:
        # 2 train, 1 val, 1 test par classe
        rows.append(("train", f"{cls}/1.jpg", cls))
        rows.append(("train", f"{cls}/2.jpg", cls))
        rows.append(("val", f"{cls}/3.jpg", cls))
        rows.append(("test", f"{cls}/4.jpg", cls))

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "path", "label"])
        writer.writerows(rows)

    return manifest_path, data_dir, classes
