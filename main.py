import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Subset
import os
import json
from load_dataset import items
from PIL import Image

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()  # macOS ARM chips
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

class ClassImages(Dataset):
    def __init__(
        self,
        items, # (filepath, label)
        transform: transforms.transforms.Compose = None
    ):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label


TARGET = 128

def make_uniform_compose(target=TARGET, fill=0):
    # make all dimensions and aspect ratios uniform (1:1 aspect ratio, 128x128)
    def resize_long_side(img):
        w, h = img.size
        if w >= h:
            new_w = target
            new_h = int(round(h * (target / w)))
        else:
            new_h = target
            new_w = int(round(w * (target / h)))
        return TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)

    # for inputs not initially 1:1, add padding as needed to make 1:1
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
        T.ToTensor(),
    ])

# compute the mean and std of
@torch.no_grad()
def compute_mean_std(train_dataset, batch_size=64, num_workers=8, device="cuda"):
    """
    Computes channel-wise mean/std on the *training* set after uniformization.
    Assumes train_dataset returns (tensor_image, label) and its transform is set.
    """

    # At this point train_dataset employs the transform pipeline in the above function which makes all inputs uniform
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False, persistent_workers=True, prefetch_factor=4)

    channel_sum = torch.zeros(3, device=device)
    channel_sumsq = torch.zeros(3, device=device)
    total_pixels = 0

    for x, label in loader:
        x = x.to(device, non_blocking=True)  # [B,C,H,W]
        b, c, h, w = x.shape
        pixels = b * h * w
        total_pixels += pixels

        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sumsq += (x * x).sum(dim=(0, 2, 3))

    mean = channel_sum / total_pixels
    var = channel_sumsq / total_pixels - mean * mean
    std = torch.sqrt(var.clamp_min(1e-12))

    return mean.cpu().tolist(), std.cpu().tolist()

# this function just serves to consider cached values which I input by hand
# if you want to calculate mean and std again (something about uniformization process changes,
# datset changes, target dimension size changes, etc.) just set mean and std to None and the computation
# will commence. This caching is done becuse it takes a long time for the mean and std calculation to happen
def get_or_compute_mean_std(train_subset,
                            batch_size=256, num_workers=8, device="cuda"):

    mean = [0.5467587113380432, 0.5017048120498657, 0.45586690306663513]
    std = [0.25268396735191345, 0.24393820762634277, 0.24675340950489044]
    if mean != None and std != None:
        print("Using cached mean/std.")
        return mean, std

    print("No cached stats found. Computing mean/std...")
    mean, std = compute_mean_std(train_subset, batch_size=batch_size,
                                 num_workers=num_workers, device=device)
    return mean, std

# returns the transform composition of the uniformization process from above + normalization with datset mean and std
def make_final_compose(mean, std, target=TARGET, fill=0):
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
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

# generate transformation pipeline just for train dataloader
# this does uniformization + normalization as above but also adds random augmentation transforms
# this processing transform is critical for improve generalizability and validation accuracy
def make_augment_compose(mean, std, target=TARGET, fill=0):
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

        # 90% chance of applying an affine augmentaiton that rotates, translates, scales, and shears all at once to some degree
        T.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=8, interpolation=InterpolationMode.BILINEAR)
        ], p=0.9),

        # 50% chance of applying color jitter
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02)
        ], p=0.5),

        # Reduce information in data and blur features
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.15),

        T.ToTensor(),

        # Keep erased blocks small as this can lead to digit being unrecognizable
        T.RandomErasing(0.2, scale=(0.01, 0.05)),


        T.Normalize(mean=mean, std=std),
    ])

dataset = ClassImages(
    items=items,
    transform=None
)

train_fraction = 0.8 # ... # how much of your data do you want to use for training, and how much do you want to save for validation?
num_samples = len(dataset)

num_train = int(train_fraction * num_samples)
num_val = num_samples - num_train

g = torch.Generator().manual_seed(42)
permutations = torch.randperm(num_samples, generator=g).tolist()
train_idx = permutations[:num_train]
val_idx   = permutations[num_train:]

uniform_only = make_uniform_compose(TARGET)
stats_dataset = ClassImages(items=items, transform=uniform_only)
subset_for_stats = Subset(stats_dataset, train_idx)

mean, std = get_or_compute_mean_std(subset_for_stats, batch_size=256, device=str(DEVICE))
print(f"Mean: {mean}")
print(f"Std: {std}")

final_tfms   = make_final_compose(mean, std, target=TARGET)
augment_tfms = make_augment_compose(mean, std, target=TARGET)

train_base_dataset = ClassImages(items=items, transform=augment_tfms)
val_base_dataset   = ClassImages(items=items, transform=final_tfms)

train_dataset = Subset(train_base_dataset, train_idx)
val_dataset   = Subset(val_base_dataset, val_idx)

from torchvision.transforms import v2
from torch.utils.data import default_collate

mixup = v2.MixUp(alpha=0.2, num_classes=10)
cutmix = v2.CutMix(num_classes=10)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

BATCH_SIZE = 128     # Consider adjusting
NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    persistent_workers=True,
    drop_last=True,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    pin_memory=False
)

x, y = next(iter(train_loader))
print(x.shape, y.shape)

