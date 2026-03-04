"""
Evaluate finetuned CoAtNet-0 on the validation set:
  1. Standard (no augmentation) forward pass
  2. Test-Time Augmentation (TTA) — average logits over multiple augmented views

TTA augmentations are milder than training augmentations:
  - Horizontal flip
  - Small affine perturbations (rotation, translate, scale)
  - Light color jitter
  No RandomErasing or GaussianBlur — those destroy info and hurt at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os
from tqdm import tqdm

from load_dataset import items
from main import (
    ClassImages,
    CoAtNet0,
    make_uniform_compose,
    compute_mean_std,
    make_final_compose,
    TARGET,
)

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

# ── Config ──
NUM_CLASSES = 10
CHECKPOINT_PATH = "checkpoints/finetune224ema/best.pt"
TTA_RUNS = 10  # number of augmented views per image

# ── Train / val split (same seed as training) ──
num_samples = len(items)
num_train = int(0.8 * num_samples)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(num_samples, generator=g).tolist()
train_idx = perm[:num_train]
val_idx   = perm[num_train:]

# ── Mean / std ──
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

# ── Standard transform (no augmentation) ──
final_tfms = make_final_compose(mean, std, target=TARGET)

# ── TTA transform (mild augmentations, always applied) ──
def make_tta_compose(mean, std, target=TARGET, fill=0):
    def resize_long_side(img):
        w, h = img.size
        if w >= h:
            new_w = target
            new_h = int(round(h * (target / w)))
        else:
            new_h = target
            new_w = int(round(w * (target / h)))
        return TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)

    def pad_to_square(img):
        w, h = img.size
        pad_w = target - w
        pad_h = target - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return TF.pad(img, padding, fill=fill)

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(resize_long_side),
        T.Lambda(pad_to_square),

        # Mild affine: small rotation, slight translate/scale
        T.RandomApply([
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05),
                scale=(0.9, 1.1), shear=5,
                interpolation=InterpolationMode.BILINEAR
            )
        ], p=0.8),

        # Light color jitter
        T.RandomApply([
            T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.01)
        ], p=0.4),

        # Random horizontal flip
        T.RandomHorizontalFlip(p=0.5),

        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

tta_tfms = make_tta_compose(mean, std, target=TARGET)

# ── Load model (use EMA weights if available) ──
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

model = CoAtNet0(num_classes=NUM_CLASSES, image_size=TARGET).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

if "ema_model" in ckpt:
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    ema_model.load_state_dict(ckpt["ema_model"])
    model = ema_model
    print(f"Loaded EMA weights from {CHECKPOINT_PATH}")
else:
    model.load_state_dict(ckpt["model"])
    print(f"Loaded standard weights from {CHECKPOINT_PATH} (no EMA found)")

model.eval()
print(f"  best_val_acc from training: {ckpt.get('best_val_acc', '?')}")

BATCH_SIZE = 128
NUM_WORKERS = 8

# ═══════════════════════════════════════════════════════════════
# 1. Standard evaluation (no augmentation)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Standard evaluation (no augmentation)")
print(f"{'='*60}")

val_dataset = ClassImages(items=items, transform=final_tfms)
val_subset = Subset(val_dataset, val_idx)
val_loader = DataLoader(
    val_subset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Standard eval"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

standard_acc = correct / total
print(f"Standard val accuracy: {standard_acc:.4f} ({correct}/{total})")

# Per-class accuracy
print("\nPer-class accuracy:")
for c in range(NUM_CLASSES):
    mask = [i for i, l in enumerate(all_labels) if l == c]
    if len(mask) == 0:
        continue
    class_correct = sum(1 for i in mask if all_preds[i] == c)
    print(f"  Class {c}: {class_correct}/{len(mask)} = {class_correct/len(mask):.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. TTA evaluation
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"TTA evaluation ({TTA_RUNS} augmented views + 1 clean)")
print(f"{'='*60}")

# Accumulate logits: one clean pass + TTA_RUNS augmented passes
num_val = len(val_idx)
accumulated_logits = torch.zeros(num_val, NUM_CLASSES)
true_labels = torch.zeros(num_val, dtype=torch.long)

# Clean pass (weight = 1)
val_dataset_clean = ClassImages(items=items, transform=final_tfms)
val_subset_clean = Subset(val_dataset_clean, val_idx)
clean_loader = DataLoader(
    val_subset_clean, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

offset = 0
with torch.no_grad():
    for images, labels in tqdm(clean_loader, desc="TTA clean pass"):
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images).cpu()
        bs = logits.size(0)
        accumulated_logits[offset:offset+bs] += logits
        true_labels[offset:offset+bs] = labels
        offset += bs

# Augmented passes
for run in range(TTA_RUNS):
    val_dataset_tta = ClassImages(items=items, transform=tta_tfms)
    val_subset_tta = Subset(val_dataset_tta, val_idx)
    tta_loader = DataLoader(
        val_subset_tta, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    offset = 0
    with torch.no_grad():
        for images, labels in tqdm(tta_loader, desc=f"TTA run {run+1}/{TTA_RUNS}"):
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images).cpu()
            bs = logits.size(0)
            accumulated_logits[offset:offset+bs] += logits
            offset += bs

# Final predictions from averaged logits
tta_preds = accumulated_logits.argmax(1)
tta_correct = (tta_preds == true_labels).sum().item()
tta_acc = tta_correct / num_val

print(f"\nTTA val accuracy: {tta_acc:.4f} ({tta_correct}/{num_val})")
print(f"Standard:         {standard_acc:.4f}")
print(f"Improvement:      {tta_acc - standard_acc:+.4f}")

# Per-class TTA accuracy
print("\nPer-class TTA accuracy:")
for c in range(NUM_CLASSES):
    mask = (true_labels == c)
    if mask.sum() == 0:
        continue
    class_correct = (tta_preds[mask] == c).sum().item()
    class_total = mask.sum().item()
    print(f"  Class {c}: {class_correct}/{class_total} = {class_correct/class_total:.4f}")
