import os
import json
import torch
from torch.utils.data import DataLoader, Subset

from load_tiny_imagenet import items_tiny_imagenet
from config import TARGET, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD, DEVICE
from model import CoAtNet0
from transforms import make_final_compose, make_augment_compose
from data import ClassImages, split_indices, make_mixup_collate
from training import get_decay_param_groups, save_checkpoint, load_checkpoint, train_one_epoch, validate

print("Using device:", DEVICE)

# --- data (same split as pretrain.py) ---

NUM_CLASSES = 200
train_idx, val_idx = split_indices(len(items_tiny_imagenet))

train_dataset = Subset(
    ClassImages(items_tiny_imagenet, transform=make_augment_compose(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)),
    train_idx,
)
val_dataset = Subset(
    ClassImages(items_tiny_imagenet, transform=make_final_compose(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)),
    val_idx,
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=4, collate_fn=make_mixup_collate(NUM_CLASSES),
    persistent_workers=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False,
    num_workers=4, persistent_workers=True,
)

# --- load best pretrain checkpoint and continue at lower lr ---

model = CoAtNet0(num_classes=NUM_CLASSES, image_size=TARGET).to(DEVICE)

source_ckpt = torch.load("checkpoints/pretrain_tiny_imagenet224/best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(source_ckpt["model"])
prev_val_acc = source_ckpt.get("best_val_acc", 0.0)
print(f"Loaded pretrain checkpoint, val_acc={prev_val_acc:.4f}")

lr = 1e-4
extra_epochs = 100
optimizer = torch.optim.AdamW(get_decay_param_groups(model, weight_decay=0.05), lr=lr)

# --- resume if we already started extra training ---

checkpoint_dir = "checkpoints/pretrain_tiny_imagenet224_extra"
best_path = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")
history_path = os.path.join(checkpoint_dir, "history.json")

start_epoch = 0
best_val_acc = prev_val_acc
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

if os.path.isfile(latest_path):
    start_epoch, best_val_acc, ckpt = load_checkpoint(latest_path, model, optimizer, device=DEVICE)
    if ckpt.get("history"):
        history = ckpt["history"]
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

# --- train (no scheduler, constant lr) ---

failing_epochs = 0
for epoch in range(start_epoch, extra_epochs):
    label = f"Epoch {epoch+1}/{extra_epochs}"
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, label)
    val_loss, val_acc = validate(model, val_loader, DEVICE, label)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(lr)

    print(f"{label} | lr {lr:.2e} | "
          f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    extra = {"val_acc": val_acc, "history": history}

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(best_path, model, optimizer, epoch=epoch, best_val_acc=best_val_acc, extra=extra)
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")
    else:
        failing_epochs += 1

    save_checkpoint(latest_path, model, optimizer, epoch=epoch, best_val_acc=best_val_acc, extra=extra)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if failing_epochs >= 70:
        print(f"Early stopping at epoch {epoch+1}, best={best_val_acc:.4f}")
        break

print(f"\nExtra pretraining complete. Best val_acc: {best_val_acc:.4f}")
