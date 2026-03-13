import os
import tempfile
import torch
import torch.nn as nn
import pytest

from training import save_checkpoint, load_checkpoint


@pytest.fixture
def simple_model():
    return nn.Linear(10, 5)


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def simple_scheduler(simple_optimizer):
    return torch.optim.lr_scheduler.StepLR(simple_optimizer, step_size=10)


class TestCheckpoint:
    def test_save_load_roundtrip(self, simple_model, simple_optimizer):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, simple_model, simple_optimizer, epoch=5, best_val_acc=0.85)

            model2 = nn.Linear(10, 5)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            start_epoch, best_acc, _ = load_checkpoint(path, model2, opt2)

            assert start_epoch == 6
            assert best_acc == 0.85

            for p1, p2 in zip(simple_model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)

    def test_with_scheduler(self, simple_model, simple_optimizer, simple_scheduler):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, simple_model, simple_optimizer,
                            scheduler=simple_scheduler, epoch=3)

            model2 = nn.Linear(10, 5)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10)
            start_epoch, _, _ = load_checkpoint(path, model2, opt2, sched2)
            assert start_epoch == 4

    def test_with_ema(self, simple_model, simple_optimizer):
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(simple_model)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, simple_model, simple_optimizer,
                            ema_model=ema, epoch=2)

            model2 = nn.Linear(10, 5)
            ema2 = AveragedModel(model2)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            load_checkpoint(path, model2, opt2, ema_model=ema2)

    def test_extra_data(self, simple_model, simple_optimizer):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            extra = {"custom_key": "custom_value", "history": {"loss": [1, 2, 3]}}
            save_checkpoint(path, simple_model, simple_optimizer,
                            epoch=1, extra=extra)

            _, _, ckpt = load_checkpoint(path, simple_model, simple_optimizer)
            assert ckpt["custom_key"] == "custom_value"
            assert ckpt["history"] == {"loss": [1, 2, 3]}

    def test_directory_creation(self, simple_model, simple_optimizer):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "dir", "ckpt.pt")
            save_checkpoint(path, simple_model, simple_optimizer, epoch=0)
            assert os.path.isfile(path)
