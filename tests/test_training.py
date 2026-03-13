import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from training import get_decay_param_groups, train_one_epoch, validate


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )


@pytest.fixture
def train_loader():
    x = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


@pytest.fixture
def val_loader():
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


class TestGetDecayParamGroups:
    def test_two_groups(self, simple_model):
        groups = get_decay_param_groups(simple_model, weight_decay=0.05)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.05
        assert groups[1]["weight_decay"] == 0.0

    def test_no_overlap(self, simple_model):
        groups = get_decay_param_groups(simple_model, weight_decay=0.05)
        decay_ids = {id(p) for p in groups[0]["params"]}
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        assert len(decay_ids & no_decay_ids) == 0

    def test_all_params_covered(self, simple_model):
        groups = get_decay_param_groups(simple_model, weight_decay=0.05)
        total_in_groups = sum(len(g["params"]) for g in groups)
        total_params = sum(1 for p in simple_model.parameters() if p.requires_grad)
        assert total_in_groups == total_params

    def test_with_lr(self, simple_model):
        groups = get_decay_param_groups(simple_model, weight_decay=0.05, lr=1e-3)
        for g in groups:
            assert g["lr"] == 1e-3

    def test_bias_in_no_decay(self, simple_model):
        groups = get_decay_param_groups(simple_model, weight_decay=0.05)
        no_decay_params = groups[1]["params"]
        bias_params = [p for n, p in simple_model.named_parameters() if "bias" in n]
        no_decay_ids = {id(p) for p in no_decay_params}
        for bp in bias_params:
            assert id(bp) in no_decay_ids


class TestTrainOneEpoch:
    def test_returns_valid_metrics(self, simple_model, train_loader):
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss, acc = train_one_epoch(simple_model, train_loader, optimizer, "cpu", "test")
        assert loss > 0
        assert 0.0 <= acc <= 1.0

    def test_with_soft_labels(self, simple_model):
        x = torch.randn(32, 10)
        y = torch.zeros(32, 5)
        for i in range(32):
            y[i, i % 5] = 0.8
            y[i, (i + 1) % 5] = 0.2
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss, acc = train_one_epoch(simple_model, loader, optimizer, "cpu", "test")
        assert loss > 0
        assert 0.0 <= acc <= 1.0

    def test_with_ema(self, simple_model, train_loader):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(simple_model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss, acc = train_one_epoch(
            simple_model, train_loader, optimizer, "cpu", "test", ema_model=ema
        )
        assert loss > 0


class TestValidate:
    def test_returns_valid_metrics(self, simple_model, val_loader):
        loss, acc = validate(simple_model, val_loader, "cpu", "test")
        assert loss > 0
        assert 0.0 <= acc <= 1.0
