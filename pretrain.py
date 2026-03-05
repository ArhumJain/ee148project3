import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, default_collate
import os
import json
from tqdm import tqdm

from load_tiny_imagenet import items_tiny_imagenet
from main import (
    ClassImages,
    CoAtNet0,
    make_uniform_compose,
    compute_mean_std,
    make_final_compose,
    make_augment_compose,
    TARGET,
)

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

num_samples = len(items_tiny_imagenet)
num_train = int(0.8 * num_samples)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(num_samples, generator=g).tolist()
train_idx = perm[:num_train]
val_idx   = perm[num_train:]

mean = None
std  = None

if mean is not None and std is not None:
    print("Using cached Tiny ImageNet mean/std.")
else:
    print("Computing Tiny ImageNet mean/std ...")
    uniform_only = make_uniform_compose(TARGET)
    stats_dataset = ClassImages(items=items_tiny_imagenet, transform=uniform_only)
    subset_for_stats = Subset(stats_dataset, train_idx)
    mean, std = compute_mean_std(subset_for_stats, batch_size=256,
                                  num_workers=4, device=str(DEVICE))
    print(f"  mean = {mean}")
    print(f"  std  = {std}")
    print("  (paste these back into this file to skip recomputation)")

final_tfms   = make_final_compose(mean, std, target=TARGET)
augment_tfms = make_augment_compose(mean, std, target=TARGET)

train_base = ClassImages(items=items_tiny_imagenet, transform=augment_tfms)
val_base   = ClassImages(items=items_tiny_imagenet, transform=final_tfms)

train_dataset = Subset(train_base, train_idx)
val_dataset   = Subset(val_base,   val_idx)

NUM_CLASSES = 200
mixup  = v2.MixUp(alpha=0.2, num_classes=NUM_CLASSES)
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

BATCH_SIZE  = 128
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
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
)

model = CoAtNet0(num_classes=NUM_CLASSES, image_size=TARGET).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"CoAtNet-0 parameters: {num_params:,}")

lr = 1e-3
weight_decay = 0.05
epochs = 200
warm_up_period = 5
patience = 70
patience_delta = 0.0

def get_decay_param_groups(model, weight_decay):
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
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(get_decay_param_groups(model, weight_decay), lr=lr)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.05, total_iters=warm_up_period
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - warm_up_period
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warm_up_period],
)

checkpoint_dir = "checkpoints/pretrain_tiny_imagenet224"
best_path   = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
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
    return ckpt.get("epoch", -1) + 1, ckpt.get("best_val_acc", 0.0), ckpt.get("history")

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
}
history_path = os.path.join(checkpoint_dir, "history.json")

start_epoch = 0
best_val_acc = 0.0
failing_epochs = 0

resume_path = latest_path if os.path.isfile(latest_path) else None

if resume_path is not None:
    start_epoch, best_val_acc, saved_history = load_checkpoint(
        resume_path, model, optimizer, scheduler, device=DEVICE
    )
    if saved_history is not None:
        history = saved_history
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

for epoch in range(start_epoch, epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        # MixUp/CutMix labels are soft — compare argmax
        train_correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
        train_total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    train_acc = train_correct / train_total
    avg_train_loss = train_loss_sum / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            val_loss_sum += F.cross_entropy(logits, labels).item() * images.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss_sum / val_total
    current_lr = scheduler.get_last_lr()[0]

    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"Epoch {epoch+1:3d}/{epochs} | "
          f"lr {current_lr:.2e} | "
          f"train_loss {avg_train_loss:.4f} | train_acc {train_acc:.4f} | "
          f"val_loss {avg_val_loss:.4f} | val_acc {val_acc:.4f}")

    ckpt_extra = {"val_acc": val_acc, "history": history}

    if val_acc > (best_val_acc + patience_delta):
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(best_path, model, optimizer, scheduler,
                        epoch=epoch, best_val_acc=best_val_acc,
                        extra=ckpt_extra)
        print(f"  -> Saved new best (val_acc={val_acc:.4f})")
    else:
        failing_epochs += 1

    save_checkpoint(latest_path, model, optimizer, scheduler,
                    epoch=epoch, best_val_acc=best_val_acc,
                    extra=ckpt_extra)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if failing_epochs >= patience:
        print(f"Early stopping at epoch {epoch+1}, best={best_val_acc:.4f}")
        break

print(f"\nPretraining complete. Best val_acc: {best_val_acc:.4f}")
print(f"Best weights: {best_path}")
print(f"History:      {history_path}")
