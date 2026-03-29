"""Tests unitaires pour src.data.dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.dataset import (
    MushroomDataset,
    build_label_map,
    get_eval_transforms,
    get_train_transforms,
    load_manifest,
)


class TestLoadManifest:
    """Tests pour la fonction load_manifest."""

    def test_loads_correct_split(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que seul le split demande est charge."""
        manifest, _, _ = tmp_dataset
        paths, labels = load_manifest(manifest, "train")
        assert len(paths) == 6  # 2 par classe * 3 classes
        assert all(lbl in labels for lbl in ["Amanita_muscaria", "Boletus_edulis"])

    def test_val_split(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie le chargement du split validation."""
        manifest, _, _ = tmp_dataset
        paths, labels = load_manifest(manifest, "val")
        assert len(paths) == 3  # 1 par classe * 3 classes

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Verifie l'erreur si le manifest n'existe pas."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv", "train")

    def test_raises_on_empty_split(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie l'erreur si le split demande est vide."""
        import pytest

        manifest, _, _ = tmp_dataset
        with pytest.raises(ValueError, match="Aucune entree"):
            load_manifest(manifest, "nonexistent_split")


class TestBuildLabelMap:
    """Tests pour la fonction build_label_map."""

    def test_sorted_and_unique(self) -> None:
        """Verifie que le mapping est trie et sans doublons."""
        labels = ["Boletus", "Amanita", "Boletus", "Russula", "Amanita"]
        label_map = build_label_map(labels)
        assert label_map == {"Amanita": 0, "Boletus": 1, "Russula": 2}

    def test_deterministic(self) -> None:
        """Verifie que le mapping est deterministe."""
        labels = ["C", "A", "B"]
        assert build_label_map(labels) == build_label_map(["B", "A", "C"])


class TestTransforms:
    """Tests pour les fonctions de transforms."""

    def test_train_transforms_output_shape(self) -> None:
        """Verifie la forme de sortie des transforms d'entrainement."""
        from PIL import Image

        t = get_train_transforms(224)
        img = Image.new("RGB", (320, 240))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_output_shape(self) -> None:
        """Verifie la forme de sortie des transforms d'evaluation."""
        from PIL import Image

        t = get_eval_transforms(224)
        img = Image.new("RGB", (320, 240))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_deterministic(self) -> None:
        """Verifie que les transforms d'evaluation sont deterministes."""
        from PIL import Image

        t = get_eval_transforms(224)
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        out1 = t(img)
        out2 = t(img)
        assert torch.equal(out1, out2)


class TestMushroomDataset:
    """Tests pour la classe MushroomDataset."""

    def test_len(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que __len__ retourne le bon nombre d'images."""
        manifest, data_dir, _ = tmp_dataset
        ds = MushroomDataset(manifest, "train", data_dir)
        assert len(ds) == 6

    def test_getitem_returns_tensor_and_int(
        self, tmp_dataset: tuple[Path, Path, list[str]]
    ) -> None:
        """Verifie que __getitem__ retourne un tensor et un entier."""
        manifest, data_dir, _ = tmp_dataset
        ds = MushroomDataset(manifest, "train", data_dir)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_targets_attribute(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que l'attribut targets est un tensor de la bonne taille."""
        manifest, data_dir, _ = tmp_dataset
        ds = MushroomDataset(manifest, "train", data_dir)
        assert ds.targets.shape == (6,)
        assert ds.targets.dtype == torch.long

    def test_class_names_sorted(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie que class_names est trie alphabetiquement."""
        manifest, data_dir, classes = tmp_dataset
        ds = MushroomDataset(manifest, "train", data_dir)
        assert ds.class_names == sorted(classes)

    def test_num_classes(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie le nombre de classes."""
        manifest, data_dir, _ = tmp_dataset
        ds = MushroomDataset(manifest, "train", data_dir)
        assert ds.num_classes == 3

    def test_shared_label_map(self, tmp_dataset: tuple[Path, Path, list[str]]) -> None:
        """Verifie qu'un label_map partage est respecte."""
        manifest, data_dir, _ = tmp_dataset
        shared_map = {"Amanita_muscaria": 0, "Boletus_edulis": 1, "Russula_emetica": 2}
        ds = MushroomDataset(manifest, "train", data_dir, label_map=shared_map)
        assert ds.label_map == shared_map
