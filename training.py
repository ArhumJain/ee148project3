import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def get_decay_param_groups(model, weight_decay, lr=None):
    decay = []
    no_decay = []
    norm_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param_name.endswith("bias") or isinstance(module, norm_classes):
                no_decay.append(param)
            else:
                decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if lr is not None:
        for g in groups:
            g["lr"] = lr
    return groups


def save_checkpoint(path, model, optimizer, scheduler=None, epoch=0,
                    best_val_acc=0.0, ema_model=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
    }
    if ema_model is not None:
        ckpt["ema_model"] = ema_model.state_dict()
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    ema_model=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema_model is not None and ckpt.get("ema_model") is not None:
        ema_model.load_state_dict(ckpt["ema_model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", -1) + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    return start_epoch, best_val_acc, ckpt


def train_one_epoch(model, loader, optimizer, device, epoch_label,
                    ema_model=None, grad_clip=1.0):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{epoch_label} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(model)

        loss_sum += loss.item() * images.size(0)
        # soft labels from mixup have shape (B, C) — compare argmax
        if labels.ndim == 2:
            correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
        else:
            correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, device, epoch_label):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"{epoch_label} [val]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels).item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return loss_sum / total, correct / total
