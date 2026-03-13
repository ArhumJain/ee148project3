import torch
from PIL import Image
import pytest

from transforms import (
    _resize_long_side, _pad_to_square,
    make_uniform_compose, make_final_compose, make_augment_compose,
)
from config import TARGET


@pytest.fixture
def square_image():
    return Image.new("RGB", (200, 200), color=(128, 64, 32))


@pytest.fixture
def wide_image():
    return Image.new("RGB", (400, 200), color=(128, 64, 32))


@pytest.fixture
def tall_image():
    return Image.new("RGB", (200, 400), color=(128, 64, 32))


@pytest.fixture
def grayscale_image():
    return Image.new("L", (200, 200), color=128)


class TestResizeLongSide:
    def test_square(self, square_image):
        out = _resize_long_side(square_image, TARGET)
        assert out.size == (TARGET, TARGET)

    def test_wide(self, wide_image):
        out = _resize_long_side(wide_image, TARGET)
        assert out.size[0] == TARGET
        assert out.size[1] <= TARGET

    def test_tall(self, tall_image):
        out = _resize_long_side(tall_image, TARGET)
        assert out.size[1] == TARGET
        assert out.size[0] <= TARGET


class TestPadToSquare:
    def test_already_square(self, square_image):
        resized = _resize_long_side(square_image, TARGET)
        padded = _pad_to_square(resized, TARGET)
        assert padded.size == (TARGET, TARGET)

    def test_wide_padded(self, wide_image):
        resized = _resize_long_side(wide_image, TARGET)
        padded = _pad_to_square(resized, TARGET)
        assert padded.size == (TARGET, TARGET)

    def test_tall_padded(self, tall_image):
        resized = _resize_long_side(tall_image, TARGET)
        padded = _pad_to_square(resized, TARGET)
        assert padded.size == (TARGET, TARGET)


class TestComposes:
    def test_uniform_output_shape(self, square_image):
        tfm = make_uniform_compose()
        out = tfm(square_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_uniform_wide(self, wide_image):
        tfm = make_uniform_compose()
        out = tfm(wide_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_uniform_tall(self, tall_image):
        tfm = make_uniform_compose()
        out = tfm(tall_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_uniform_grayscale_converts(self, grayscale_image):
        tfm = make_uniform_compose()
        out = tfm(grayscale_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_final_output_shape(self, square_image):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        tfm = make_final_compose(mean, std)
        out = tfm(square_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_final_normalized(self, square_image):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        tfm = make_final_compose(mean, std)
        out = tfm(square_image)
        assert out.min() < 0 or out.max() > 1

    def test_augment_output_shape(self, square_image):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        tfm = make_augment_compose(mean, std)
        out = tfm(square_image)
        assert out.shape == (3, TARGET, TARGET)

    def test_various_sizes(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        tfm = make_final_compose(mean, std)
        for w, h in [(128, 128), (256, 128), (128, 256), (512, 300)]:
            img = Image.new("RGB", (w, h))
            out = tfm(img)
            assert out.shape == (3, TARGET, TARGET), f"Failed for {w}x{h}"
