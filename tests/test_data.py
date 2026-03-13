import os
import tempfile
import torch
import pytest
from PIL import Image

from data import ClassImages, split_indices, compute_mean_std, make_mixup_collate


@pytest.fixture
def tmp_items():
    with tempfile.TemporaryDirectory() as d:
        items = []
        for i in range(20):
            path = os.path.join(d, f"img_{i}_label{i % 5}.jpg")
            Image.new("RGB", (64, 64), color=(i * 10 % 256, 0, 0)).save(path)
            items.append((path, i % 5))
        yield items


class TestClassImages:
    def test_len(self, tmp_items):
        ds = ClassImages(tmp_items)
        assert len(ds) == 20

    def test_getitem_returns_tuple(self, tmp_items):
        ds = ClassImages(tmp_items)
        img, label = ds[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, int)

    def test_getitem_with_transform(self, tmp_items):
        from torchvision import transforms
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = ClassImages(tmp_items, transform=tfm)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3


class TestSplitIndices:
    def test_coverage(self):
        train, val = split_indices(100)
        assert set(train + val) == set(range(100))

    def test_ratio(self):
        train, val = split_indices(100, train_fraction=0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_determinism(self):
        t1, v1 = split_indices(100, seed=42)
        t2, v2 = split_indices(100, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds(self):
        t1, _ = split_indices(100, seed=1)
        t2, _ = split_indices(100, seed=2)
        assert t1 != t2

    def test_no_overlap(self):
        train, val = split_indices(100)
        assert len(set(train) & set(val)) == 0


class TestComputeMeanStd:
    def test_range(self, tmp_items):
        from torchvision import transforms
        ds = ClassImages(tmp_items, transform=transforms.ToTensor())
        mean, std = compute_mean_std(ds, batch_size=8, num_workers=0)
        assert len(mean) == 3
        assert len(std) == 3
        for m in mean:
            assert 0.0 <= m <= 1.0
        for s in std:
            assert 0.0 <= s <= 1.0


class TestMixupCollate:
    def test_soft_labels(self):
        collate = make_mixup_collate(num_classes=5)
        batch = [(torch.randn(3, 32, 32), i % 5) for i in range(8)]
        images, labels = collate(batch)
        assert images.shape[0] == 8
        assert labels.ndim == 2
        assert labels.shape == (8, 5)
