import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, default_collate
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from load_dataset import items
from config import TARGET, DATASET_MEAN, DATASET_STD, DEVICE
from model import CoAtNet0
from transforms import make_final_compose, make_augment_compose
from data import ClassImages, split_indices, make_mixup_collate
from training import get_decay_param_groups, save_checkpoint, load_checkpoint, train_one_epoch, validate

print("Using device:", DEVICE)

# --- data ---

NUM_CLASSES = 10
train_idx, val_idx = split_indices(len(items))

# mixup can be toggled off mid-training
mixup_collate = make_mixup_collate(NUM_CLASSES)
use_mixup = True
DISABLE_MIXUP_EPOCH = 90

def collate_fn(batch):
    if use_mixup:
        return mixup_collate(batch)
    return default_collate(batch)

train_dataset = Subset(ClassImages(items, transform=make_augment_compose(DATASET_MEAN, DATASET_STD)), train_idx)
val_dataset = Subset(ClassImages(items, transform=make_final_compose(DATASET_MEAN, DATASET_STD)), val_idx)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    num_workers=24, collate_fn=collate_fn,
    persistent_workers=True, drop_last=True, pin_memory=True, prefetch_factor=4,
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False,
    num_workers=24, persistent_workers=True, pin_memory=True, prefetch_factor=4,
)

# --- load pretrained model, swap head for 10 classes ---

PRETRAINED_PATH = "checkpoints/pretrain_tiny_imagenet224/best.pt"

model = CoAtNet0(num_classes=200, image_size=TARGET)
ckpt = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded pretrained weights from {PRETRAINED_PATH}")
print(f"  pretrain val_acc: {ckpt.get('best_val_acc', ckpt.get('val_acc', '?'))}")

model.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, NUM_CLASSES))
model = model.to(DEVICE)
print(f"CoAtNet-0 parameters: {sum(p.numel() for p in model.parameters()):,}")

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

# --- hyperparams ---

head_lr = 1e-3
finetune_lr = 2e-4
weight_decay = 0.05
epochs = 400
warm_up_period = 5
patience = 100
WARMUP_HEAD_EPOCHS = 3

checkpoint_dir = "checkpoints/finetune224ema_mixoff"
best_path = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")
history_path = os.path.join(checkpoint_dir, "history.json")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}
best_val_acc = 0.0

# check if phase 1 was already done
skip_phase1 = False
if os.path.isfile(latest_path):
    _ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    if _ckpt.get("phase") == "finetune":
        skip_phase1 = True
    del _ckpt

# ============================================================
# Phase 1: freeze backbone, train only the new head
# ============================================================

if skip_phase1:
    print("\nPhase 1 already completed (resuming Phase 2). Skipping head warmup.")
else:
    print(f"\n{'='*60}")
    print(f"Phase 1: Head-only warmup ({WARMUP_HEAD_EPOCHS} epochs)")
    print(f"{'='*60}")

    for name, param in model.named_parameters():
        if not name.startswith("head."):
            param.requires_grad = False

    head_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_lr, weight_decay=weight_decay,
    )

    for epoch in range(WARMUP_HEAD_EPOCHS):
        label = f"Epoch {epoch+1}/{WARMUP_HEAD_EPOCHS}"
        train_loss, train_acc = train_one_epoch(model, train_loader, head_optimizer, DEVICE, label, ema_model=ema_model)
        val_loss, val_acc = validate(ema_model, val_loader, DEVICE, label)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(head_lr)
        history["phase"].append("head_warmup")

        print(f"{label} | lr {head_lr:.2e} | "
              f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# ============================================================
# Phase 2: unfreeze everything, finetune with cosine schedule
# ============================================================

print(f"\n{'='*60}")
print(f"Phase 2: Full finetune ({epochs} epochs, patience={patience})")
print(f"{'='*60}")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    get_decay_param_groups(model, weight_decay, lr=finetune_lr), lr=finetune_lr,
)
warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm_up_period)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warm_up_period)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warm_up_period],
)

start_epoch = 0
if os.path.isfile(latest_path):
    _ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    if _ckpt.get("phase") == "finetune":
        start_epoch, best_val_acc, rckpt = load_checkpoint(
            latest_path, model, optimizer, scheduler, ema_model=ema_model, device=DEVICE,
        )
        if rckpt.get("history"):
            history = rckpt["history"]
        print(f"Resuming finetune from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    del _ckpt

failing_epochs = 0
for epoch in range(start_epoch, epochs):
    # turn off mixup late in training
    if DISABLE_MIXUP_EPOCH is not None and epoch >= DISABLE_MIXUP_EPOCH and use_mixup:
        use_mixup = False
        print(f"  MixUp/CutMix disabled at epoch {epoch+1}")

    label = f"Epoch {epoch+1}/{epochs}"
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, label, ema_model=ema_model)
    scheduler.step()
    val_loss, val_acc = validate(ema_model, val_loader, DEVICE, label)
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

    extra = {"val_acc": val_acc, "history": history, "phase": "finetune"}

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(best_path, model, optimizer, scheduler, epoch=epoch,
                        best_val_acc=best_val_acc, ema_model=ema_model, extra=extra)
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")
    else:
        failing_epochs += 1

    save_checkpoint(latest_path, model, optimizer, scheduler, epoch=epoch,
                    best_val_acc=best_val_acc, ema_model=ema_model, extra=extra)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if failing_epochs >= patience:
        print(f"Early stopping at epoch {epoch+1}, best={best_val_acc:.4f}")
        break

print(f"\nFinetuning complete. Best val_acc: {best_val_acc:.4f}")
