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

# --- data ---

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
    num_workers=24, collate_fn=make_mixup_collate(NUM_CLASSES),
    persistent_workers=True, drop_last=True, pin_memory=True, prefetch_factor=4,
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False,
    num_workers=24, persistent_workers=True, pin_memory=True, prefetch_factor=4,
)

# --- model + optimizer ---

model = CoAtNet0(num_classes=NUM_CLASSES, image_size=TARGET).to(DEVICE)
print(f"CoAtNet-0 parameters: {sum(p.numel() for p in model.parameters()):,}")

lr = 1e-3
epochs = 200
warm_up_period = 5

optimizer = torch.optim.AdamW(get_decay_param_groups(model, weight_decay=0.05), lr=lr)

warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=warm_up_period)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warm_up_period)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warm_up_period],
)

# --- resume if checkpoint exists ---

checkpoint_dir = "checkpoints/pretrain_tiny_imagenet224"
best_path = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")
history_path = os.path.join(checkpoint_dir, "history.json")

start_epoch = 0
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

if os.path.isfile(latest_path):
    start_epoch, best_val_acc, ckpt = load_checkpoint(latest_path, model, optimizer, scheduler, device=DEVICE)
    if ckpt.get("history"):
        history = ckpt["history"]
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

# --- train ---

failing_epochs = 0
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

    print(f"{label} | lr {current_lr:.2e} | "
          f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    extra = {"val_acc": val_acc, "history": history}

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(best_path, model, optimizer, scheduler, epoch=epoch, best_val_acc=best_val_acc, extra=extra)
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")
    else:
        failing_epochs += 1

    save_checkpoint(latest_path, model, optimizer, scheduler, epoch=epoch, best_val_acc=best_val_acc, extra=extra)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if failing_epochs >= 70:
        print(f"Early stopping at epoch {epoch+1}, best={best_val_acc:.4f}")
        break

print(f"\nPretraining complete. Best val_acc: {best_val_acc:.4f}")
