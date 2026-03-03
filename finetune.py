"""
Finetune CoAtNet-0 pretrained on Tiny ImageNet (200 classes) → target dataset (10 classes).
Loads pretrained weights, replaces the classification head, and finetunes.
Uses the same data pipeline as main.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, default_collate
import os
import json
from tqdm import tqdm

from load_dataset import items
from main import (
    ClassImages,
    CoAtNet0,
    make_uniform_compose,
    compute_mean_std,
    get_or_compute_mean_std,
    make_final_compose,
    make_augment_compose,
    TARGET,
)

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

# ── Config ──
NUM_CLASSES = 10
PRETRAINED_PATH = "checkpoints/pretrain_tiny_imagenet224/best.pt"

# ── Train / val split (same 80/20 seeded split as main.py) ──
num_samples = len(items)
num_train = int(0.8 * num_samples)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(num_samples, generator=g).tolist()
train_idx = perm[:num_train]
val_idx   = perm[num_train:]

# ── Compute mean/std on training split ──
mean = None
std  = None

if mean is not None and std is not None:
    print("Using cached mean/std.")
else:
    print("Computing mean/std ...")
    uniform_only = make_uniform_compose(TARGET)
    stats_dataset = ClassImages(items=items, transform=uniform_only)
    subset_for_stats = Subset(stats_dataset, train_idx)
    mean, std = compute_mean_std(subset_for_stats, batch_size=256,
                                  num_workers=4, device=str(DEVICE))
    print(f"  mean = {mean}")
    print(f"  std  = {std}")
    print("  (paste these back into this file to skip recomputation)")

# ── Transforms ──
final_tfms   = make_final_compose(mean, std, target=TARGET)
augment_tfms = make_augment_compose(mean, std, target=TARGET)

train_base = ClassImages(items=items, transform=augment_tfms)
val_base   = ClassImages(items=items, transform=final_tfms)

train_dataset = Subset(train_base, train_idx)
val_dataset   = Subset(val_base,   val_idx)

# ── MixUp / CutMix ──
mixup  = v2.MixUp(alpha=0.2, num_classes=NUM_CLASSES)
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

BATCH_SIZE  = 64
NUM_WORKERS = 24

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    persistent_workers=True,
    drop_last=True,
    pin_memory=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
)

# ── Model: load pretrained, swap head ──
model = CoAtNet0(num_classes=200, image_size=TARGET)  # match pretrained architecture

ckpt = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded pretrained weights from {PRETRAINED_PATH}")
print(f"  pretrain val_acc: {ckpt.get('best_val_acc', ckpt.get('val_acc', '?'))}")

# Replace classification head: 200 → 10 classes
model.head = nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, NUM_CLASSES),
)
model = model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"CoAtNet-0 parameters: {num_params:,}")

# ── Freeze backbone initially, then unfreeze ──
# Phase 1: train only the new head (warmup_head_epochs)
# Phase 2: unfreeze everything and finetune with lower LR
WARMUP_HEAD_EPOCHS = 3

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

# ── Hyperparameters ──
head_lr = 1e-3           # higher LR for randomly initialized head
finetune_lr = 2e-4       # lower LR for pretrained backbone
weight_decay = 0.05
epochs = 400
warm_up_period = 5       # LR warmup within finetune phase
patience = 100
patience_delta = 0.0

# ── Weight decay groups ──
def get_decay_param_groups(model, weight_decay, lr):
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
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

# ── Checkpointing ──
checkpoint_dir = "checkpoints/finetune224"
best_path   = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, phase, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "phase": phase,
    }
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", -1) + 1, ckpt.get("best_val_acc", 0.0), ckpt.get("phase", "finetune"), ckpt.get("history")

# ── History ──
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
    "phase": [],
}
history_path = os.path.join(checkpoint_dir, "history.json")

# ── Training helpers ──
def train_one_epoch(model, loader, optimizer, device, epoch_label):
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
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
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


# ═══════════════════════════════════════════════════════════════
# Phase 1: Head-only warmup
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 1: Head-only warmup ({WARMUP_HEAD_EPOCHS} epochs)")
print(f"{'='*60}")

freeze_backbone(model)
head_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=head_lr, weight_decay=weight_decay
)

best_val_acc = 0.0

for epoch in range(WARMUP_HEAD_EPOCHS):
    label = f"Epoch {epoch+1}/{WARMUP_HEAD_EPOCHS}"
    train_loss, train_acc = train_one_epoch(model, train_loader, head_optimizer, DEVICE, label)
    val_loss, val_acc = validate(model, val_loader, DEVICE, label)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(head_lr)
    history["phase"].append("head_warmup")

    print(f"{label} | lr {head_lr:.2e} | "
          f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(best_path, model, head_optimizer, None,
                        epoch=epoch, best_val_acc=best_val_acc, phase="head_warmup",
                        extra={"val_acc": val_acc, "history": history})
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")

# ═══════════════════════════════════════════════════════════════
# Phase 2: Full finetune
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 2: Full finetune ({epochs} epochs, patience={patience})")
print(f"{'='*60}")

unfreeze_all(model)
optimizer = torch.optim.AdamW(
    get_decay_param_groups(model, weight_decay, lr=finetune_lr),
    lr=finetune_lr
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=warm_up_period
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - warm_up_period
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warm_up_period],
)

# ── Resume from checkpoint if available ──
start_epoch = 0
failing_epochs = 0

resume_path = latest_path if os.path.isfile(latest_path) else None

if resume_path is not None:
    ckpt_data = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    saved_phase = ckpt_data.get("phase", "finetune")
    if saved_phase == "finetune":
        start_epoch, best_val_acc, _, saved_history = load_checkpoint(
            resume_path, model, optimizer, scheduler, device=DEVICE
        )
        if saved_history is not None:
            history = saved_history
        print(f"Resuming finetune from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

for epoch in range(start_epoch, epochs):
    label = f"Epoch {epoch+1}/{epochs}"
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, label)
    scheduler.step()
    val_loss, val_acc = validate(model, val_loader, DEVICE, label)
    current_lr = scheduler.get_last_lr()[0]

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    history["phase"].append("finetune")

    print(f"{label} | lr {current_lr:.2e} | "
          f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    ckpt_extra = {"val_acc": val_acc, "history": history}

    if val_acc > (best_val_acc + patience_delta):
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(best_path, model, optimizer, scheduler,
                        epoch=epoch, best_val_acc=best_val_acc, phase="finetune",
                        extra=ckpt_extra)
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")
    else:
        failing_epochs += 1

    save_checkpoint(latest_path, model, optimizer, scheduler,
                    epoch=epoch, best_val_acc=best_val_acc, phase="finetune",
                    extra=ckpt_extra)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if failing_epochs >= patience:
        print(f"Early stopping at epoch {epoch+1}, best={best_val_acc:.4f}")
        break

print(f"\nFinetuning complete. Best val_acc: {best_val_acc:.4f}")
print(f"Best weights: {best_path}")
print(f"History:      {history_path}")
