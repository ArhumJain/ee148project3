import torch
import torch.nn as nn
import tempfile
import os
import pytest
from PIL import Image

from pipeline import ResizeLongSide, PadToSquare, DigitClassifierPipeline
from config import TARGET


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 10),
    )


@pytest.fixture
def pipeline(dummy_model):
    return DigitClassifierPipeline(
        model=dummy_model,
        input_height=TARGET,
        input_width=TARGET,
        input_channels=3,
        device="cpu",
    )


class TestResizeLongSide:
    def test_square_input(self):
        layer = ResizeLongSide(224)
        x = torch.randn(3, 200, 200)
        out = layer(x)
        assert max(out.shape[-2:]) == 224

    def test_wide_input(self):
        layer = ResizeLongSide(224)
        x = torch.randn(3, 100, 400)
        out = layer(x)
        assert out.shape[-1] == 224

    def test_tall_input(self):
        layer = ResizeLongSide(224)
        x = torch.randn(3, 400, 100)
        out = layer(x)
        assert out.shape[-2] == 224


class TestPadToSquare:
    def test_already_square(self):
        layer = PadToSquare(224)
        x = torch.randn(3, 224, 224)
        out = layer(x)
        assert out.shape == (3, 224, 224)

    def test_needs_padding(self):
        layer = PadToSquare(224)
        x = torch.randn(3, 112, 224)
        out = layer(x)
        assert out.shape == (3, 224, 224)


class TestDigitClassifierPipeline:
    def test_forward_shape(self, pipeline):
        x = torch.randn(4, 3, TARGET, TARGET)
        preds = pipeline(x)
        assert preds.shape == (4,)

    def test_predictions_in_range(self, pipeline):
        x = torch.randn(4, 3, TARGET, TARGET)
        preds = pipeline(x)
        assert all(0 <= p < 10 for p in preds.tolist())

    def test_run_with_pil_images(self, pipeline):
        images = [Image.new("RGB", (128, 200)) for _ in range(3)]
        preds = pipeline.run(images)
        assert len(preds) == 3
        assert all(isinstance(p, int) for p in preds)

    def test_preprocess_uniform_output(self, pipeline):
        sizes = [(128, 128), (256, 128), (128, 256), (400, 200)]
        from torchvision import transforms
        for w, h in sizes:
            img = Image.new("RGB", (w, h))
            tensor = transforms.ToTensor()(img)
            out = pipeline.preprocess_layers(tensor)
            assert out.shape == (3, TARGET, TARGET), f"Failed for {w}x{h}"

    def test_torchscript_export(self, pipeline):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pipeline.pt")
            pipeline.save_pipeline_local(path)
            assert os.path.isfile(path)

            loaded = torch.jit.load(path)
            x = torch.randn(2, 3, TARGET, TARGET)
            preds = loaded(x)
            assert preds.shape == (2,)
