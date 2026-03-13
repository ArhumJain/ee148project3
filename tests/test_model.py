import torch
import pytest

from model import (
    SqueezeExcitation, PreNormalization, MBConv,
    RelativeMultiHeadAttention, TransformerDownsampleBlock,
    TransformerBlock, CoAtNet0,
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestSqueezeExcitation:
    def test_output_shape(self, device):
        se = SqueezeExcitation(64).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = se(x)
        assert out.shape == x.shape

    def test_different_channels(self, device):
        for ch in [32, 128, 256]:
            se = SqueezeExcitation(ch).to(device)
            x = torch.randn(1, ch, 4, 4, device=device)
            assert se(x).shape == x.shape


class TestPreNormalization:
    def test_output_shape(self, device):
        conv = torch.nn.Conv2d(64, 64, 3, padding=1)
        pn = PreNormalization(64, conv, torch.nn.BatchNorm2d).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = pn(x)
        assert out.shape == x.shape


class TestMBConv:
    def test_same_channels(self, device):
        block = MBConv(64, 64).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_channel_change(self, device):
        block = MBConv(64, 128).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_downsample(self, device):
        block = MBConv(64, 128, downsample=True).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)

    def test_no_downsample_same_spatial(self, device):
        block = MBConv(96, 96).to(device)
        x = torch.randn(1, 96, 32, 32, device=device)
        assert block(x).shape == (1, 96, 32, 32)


class TestRelativeMultiHeadAttention:
    def test_output_shape(self, device):
        feat_size = 7
        dim = 192
        attn = RelativeMultiHeadAttention(feat_size, dim).to(device)
        x = torch.randn(2, feat_size * feat_size, dim, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_different_feat_sizes(self, device):
        for fs in [4, 7, 14]:
            dim = 128
            attn = RelativeMultiHeadAttention(fs, dim).to(device)
            x = torch.randn(1, fs * fs, dim, device=device)
            assert attn(x).shape == x.shape


class TestTransformerDownsampleBlock:
    def test_output_shape(self, device):
        block = TransformerDownsampleBlock(7, 192, 384).to(device)
        x = torch.randn(2, 192, 14, 14, device=device)
        out = block(x)
        assert out.shape == (2, 7 * 7, 384)


class TestTransformerBlock:
    def test_output_shape(self, device):
        block = TransformerBlock(7, 384).to(device)
        x = torch.randn(2, 49, 384, device=device)
        out = block(x)
        assert out.shape == x.shape


class TestCoAtNet0:
    def test_output_shape(self, device):
        model = CoAtNet0(num_classes=10).to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 10)

    def test_different_num_classes(self, device):
        model = CoAtNet0(num_classes=200).to(device)
        x = torch.randn(1, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (1, 200)

    def test_gradient_flow(self, device):
        model = CoAtNet0(num_classes=10).to(device)
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_size_1(self, device):
        model = CoAtNet0(num_classes=10).to(device)
        x = torch.randn(1, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (1, 10)
