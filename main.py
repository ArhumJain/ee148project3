from math import exp
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

    # mean = [0.5467587113380432, 0.5017048120498657, 0.45586690306663513]
    # std = [0.25268396735191345, 0.24393820762634277, 0.24675340950489044]
    mean = None
    std = None
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

# Model
# ---------------------------------

class SqueezeExcitation(nn.Module):
                       # input_features, hidden_dim
    def __init__(self, input_features, expansion=0.25):
        super().__init__()
        self.hidden_size = max(1, int(input_features * expansion))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.fc = nn.Sequential(
                nn.Linear(input_features, self.hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(self.hidden_size, input_features, bias=False),
                nn.Sigmoid()
                )

    # x: (B, C, H, W)
    def forward(self, x):
        squeeze = self.pool(x)
        # (B, C)
        scale: torch.Tensor = self.fc(squeeze)
        output = x * scale[:, :, None, None] # channel weighting
        return output

class PreNormalization(nn.Module):
    def __init__(
            self, 
            num_features,
            module: nn.Module,
            norm: nn.Module,
    ):
        super().__init__()
        self.norm = norm(num_features)
        self.module = module

    def forward(self, x, **kwargs):
        return self.module(self.norm(x), **kwargs)

class MBConv(nn.Module):
    def __init__(
            self,
            input_features,
            output_features,
            expansion_rate=4,
            shrinkage_rate=0.25,
            downsample = False,
    ):
        super().__init__()
        
        self.downsample = downsample
        self.hidden_dim = int(input_features * expansion_rate)
        self.stride = 2 if downsample else 1
        self.project = not (input_features == output_features)

        if downsample:
            self.identity_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.project or downsample:
            self.identity_proj = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv = nn.Sequential(
                nn.Conv2d(input_features, self.hidden_dim, kernel_size=1, stride=self.stride, padding=0, bias=False), # Expansion
                nn.BatchNorm2d(self.hidden_dim),
                nn.GELU(),

                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, groups=self.hidden_dim, bias=False), # depthwise convolution
                nn.BatchNorm2d(self.hidden_dim),
                nn.GELU(),
                SqueezeExcitation(self.hidden_dim, expansion=shrinkage_rate),

                nn.Conv2d(self.hidden_dim, output_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output_features),
        )

        self.conv = PreNormalization(input_features, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        identity = x
        if (self.downsample):
            identity = self.identity_proj(self.identity_pool(x))
        elif (self.project):
            identity = self.identity_proj(x)
        return identity + self.conv(x)
        
class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, feat_size, input_features):
        super().__init__()
        self.d_model = input_features          # channel dimension, not spatial size
        self.num_heads = self.d_model // 32    # head size = 32 (Table 3)
        self.d_k = 32

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

        # Per-head relative position bias table P of size [(2H-1)*(2W-1)] (Appendix A.1)
        # For square feature maps H=W=feat_size
        self.feat_size = feat_size
        num_rel = (2 * feat_size - 1) ** 2
        self.relative_bias_table = nn.Parameter(torch.zeros(self.num_heads, num_rel))
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

        # Precompute flat index into bias table for every (query_pos, key_pos) pair.
        # Relative offset (i-i', j-j') is shifted by feat_size-1 to be non-negative,
        # then encoded as a single integer: row_offset*(2*feat_size-1) + col_offset
        coords_h = torch.arange(feat_size)
        coords_w = torch.arange(feat_size)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
        coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)  # (N, 2)

        rel = coords[:, None, :] - coords[None, :, :]          # (N, N, 2)
        rel[:, :, 0] += feat_size - 1                           # shift to [0, 2H-2]
        rel[:, :, 1] += feat_size - 1                           # shift to [0, 2W-2]
        rel_index = rel[:, :, 0] * (2 * feat_size - 1) + rel[:, :, 1]  # (N, N)
        self.register_buffer('relative_position_index', rel_index.long())

    def forward(self, x):
        # x: (B, N, C) where N = feat_size^2, C = d_model
        B, N, C = x.shape

        q = self.W_q(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # Relative position bias: (num_heads, N, N) → (1, num_heads, N, N)
        rel_bias = self.relative_bias_table[:, self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(1, self.num_heads, N, N)

        # F.scaled_dot_product_attention handles scaling, softmax, and matmul
        # attn_mask is added to QK^T/sqrt(d_k) before softmax — exactly what we need
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
        out = out.transpose(1, 2).reshape(B, N, C)              # (B, N, C)
        return self.W_o(out)

class TransformerDownsampleBlock(nn.Module):
    """
    First Transformer block of a stage — handles spatial downsampling + channel projection.
    Input:  (B, C_in, H, W)   from MBConv or previous Transformer stage
    Output: (B, N, C_out)      flattened for subsequent TransformerBlocks

    Implements Eq. 4: x ← Proj(Pool(x)) + Attention(Pool(Norm(x)))
    Then FFN:         x ← x + FFN(Norm(x))
    """
    def __init__(self, feat_size, input_features, output_features, expansion_rate=4):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.identity_proj = nn.Conv2d(input_features, output_features,
                                       kernel_size=1, bias=False)

        self.attn_norm = nn.LayerNorm(input_features)
        self.attn = RelativeMultiHeadAttention(feat_size, input_features)
        self.attn_proj = nn.Linear(input_features, output_features)

        self.ffn_norm = nn.LayerNorm(output_features)
        self.ffn = nn.Sequential(
                nn.Linear(output_features, int(output_features * expansion_rate)),
                nn.GELU(),
                nn.Linear(int(output_features * expansion_rate), output_features),
        )

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, C, H, W = x.shape

        # Identity: Proj(Pool(x))
        identity = self.identity_proj(self.pool(x))     # (B, C_out, H', W')
        identity = identity.flatten(2).transpose(1, 2)  # (B, N, C_out)

        # Residual: Norm → Pool → Attention → Proj
        res = x.flatten(2).transpose(1, 2)              # (B, H*W, C_in)
        res = self.attn_norm(res)
        res = res.transpose(1, 2).reshape(B, C, H, W)
        res = self.pool(res).flatten(2).transpose(1, 2) # (B, N, C_in)
        res = self.attn_proj(self.attn(res))             # (B, N, C_out)

        x = identity + res

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x                                         # (B, N, C_out)


class TransformerBlock(nn.Module):
    """
    Non-downsampling Transformer block. Operates entirely on (B, N, C) tensors
    to avoid reshape overhead between consecutive blocks.

    x ← x + Attention(Norm(x))
    x ← x + FFN(Norm(x))
    """
    def __init__(self, feat_size, dim, expansion_rate=4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(feat_size, dim)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
                nn.Linear(dim, int(dim * expansion_rate)),
                nn.GELU(),
                nn.Linear(int(dim * expansion_rate), dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CoAtNet0(nn.Module):
    """
    CoAtNet-0 (Table 3): C-C-T-T layout.
    S0: 2-layer conv stem          (L=2, D=64)
    S1: MBConv                     (L=2, D=96)
    S2: MBConv                     (L=3, D=192)
    S3: Transformer w/ rel-attn    (L=5, D=384)
    S4: Transformer w/ rel-attn    (L=2, D=768)
    ~25M params
    """
    def __init__(self, num_classes=10, input_channels=3, image_size=128):
        super().__init__()

        dims    = [64, 96, 192, 384, 768]
        depths  = [2,  2,  3,   5,   2]

        # Spatial sizes at each stage output (each stage halves resolution)
        s3_size = image_size // 16   # 8  for 128
        s4_size = image_size // 32   # 4  for 128

        # S0: 2-layer convolutional stem (stride-2 on first conv → ½ spatial)
        self.s0 = nn.Sequential(
                nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU(),
                nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU(),
        )

        # S1: MBConv (L=2, D=96)
        self.s1 = nn.Sequential(
                MBConv(dims[0], dims[1], downsample=True),
                *[MBConv(dims[1], dims[1]) for _ in range(depths[1] - 1)],
        )

        # S2: MBConv (L=3, D=192)
        self.s2 = nn.Sequential(
                MBConv(dims[1], dims[2], downsample=True),
                *[MBConv(dims[2], dims[2]) for _ in range(depths[2] - 1)],
        )

        # S3: Transformer (L=5, D=384)
        # Downsample block: (B,C,H,W) → (B,N,C), then remaining blocks stay in (B,N,C)
        self.s3_down = TransformerDownsampleBlock(s3_size, dims[2], dims[3])
        self.s3 = nn.Sequential(
                *[TransformerBlock(s3_size, dims[3]) for _ in range(depths[3] - 1)],
        )

        # S4: Transformer (L=2, D=768)
        self.s4_down = TransformerDownsampleBlock(s4_size, dims[3], dims[4])
        self.s4 = nn.Sequential(
                *[TransformerBlock(s4_size, dims[4]) for _ in range(depths[4] - 1)],
        )

        # Classification head (Appendix A.1: global avg pool, no CLS token)
        self.s3_size = s3_size
        self.head = nn.Sequential(
                nn.LayerNorm(dims[4]),
                nn.Linear(dims[4], num_classes),
        )

    def forward(self, x):
        # S0–S2: convolution stages, (B, C, H, W) throughout
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)                                  # (B, 192, 16, 16)

        # S3: downsample converts to (B, N, C), remaining blocks stay there
        x = self.s3_down(x)                              # (B, 64, 384)
        x = self.s3(x)

        # S4: reshape back to (B, C, H, W) for the downsample block's Pool
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.s3_size, self.s3_size)
        x = self.s4_down(x)                              # (B, 16, 768)
        x = self.s4(x)

        # Head: mean-pool over tokens, then classify
        x = x.mean(dim=1)                               # (B, 768)
        return self.head(x)


LOAD_BEST_MODEL = False

model = CoAtNet0(num_classes=10) # ...) # what should num_classes be?

if (LOAD_BEST_MODEL):
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()
    print("Loaded saved best model!")

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)# ... # what might be a good loss function?
val_criterion = nn.CrossEntropyLoss()

# lr = 0.001
# lr = 0.002
# weight_decay = 0.003
# epochs = 450
# warm_up_period = 7
# patience = 75
# patience_delta = 0.0

lr = 0.001
weight_decay = 0.05
epochs = 450
warm_up_period = 7
patience = 75
patience_delta = 0.0


# optimizer = torch.optim.Adam(model.parameters(), lr= # ...) # what might be a good learning rate?
#                              # feel free to change the optimizer around.
def get_decay_param_groups(model, weight_decay):
    decay = []
    disable_decay = []

    normalization_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name.endswith("bias"):
                disable_decay.append(param)
                continue

            if isinstance(module, normalization_classes):
                disable_decay.append(param)
                continue

            decay.append(param)
    assert len(set(map(id, decay)).intersection(set(map(id, disable_decay)))) == 0

    return [{"params": decay, "weight_decay": weight_decay},
          {"params": disable_decay, "weight_decay": 0.0}]

# optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
optimizer = torch.optim.AdamW(get_decay_param_groups(model, weight_decay=weight_decay), lr)

# Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=warm_up_period)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warm_up_period)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warm_up_period],
)

train_losses = [] # Loss for each batch
train_accuracies = [] # Accuracy for each epoch
val_losses = [] # Loss for each batch
val_accuracies = [] # Accuracy for each epoch

checkpoint_dir = "checkpoints/sequence_scheduler_300"
best_acc_path = os.path.join(checkpoint_dir, "best.pt")
latest_path = os.path.join(checkpoint_dir, "last.pt")

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    return start_epoch, best_val_acc, checkpoint


start_epoch = 0
best_val_acc = 0.0
failing_epochs = 0

# resume_path = "checkpoints/sequence_scheduler/best.pt"
resume_path = None

if resume_path is not None:
    start_epoch, best_val_acc, _ = load_checkpoint(
        resume_path, model, optimizer, scheduler, device=DEVICE
    )
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

for epoch in range(start_epoch, epochs):
    print(f"Training Epoch {epoch}/{epochs}")
    final_loss = None
    total_count = 0
    correct_count = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if (labels.shape == (BATCH_SIZE,)):
            labels = F.one_hot(labels, 10).float()
        optimizer.zero_grad() # necessary for training
        if len(train_losses) < 2:
            print(f"Labels shape {labels.shape}")
        outputs = model(images)
        # loss = criterion(outputs.squeeze(), labels)
        # loss = soft_target_cross_entropy(outputs, labels)
        loss = F.cross_entropy(outputs, labels)

        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        correct_count += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        total_count += len(labels)
    train_accuracies.append(correct_count/total_count)
    print(f"Train loss: {final_loss}")
    print(f"Train accuracy: {correct_count/total_count}")

    final_loss = None
    total_count = 0
    total_loss_sum = 0
    correct_count = 0
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            # loss = val_criterion(outputs.squeeze(), labels)
            # loss = soft_target_cross_entropy(outputs, labels)
            loss = F.cross_entropy(outputs, labels)

            total_loss_sum += loss.item() * len(labels)
            correct_count += (outputs.argmax(dim=1) == labels).sum().item()
            total_count += len(labels)

    val_accuracies.append(correct_count/total_count)
    final_loss = total_loss_sum/total_count
    val_losses.append(final_loss)
    print(f"Validation loss: {final_loss}")
    print(f"Validation accuracy: {correct_count/total_count}")


    if scheduler is not None:
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()}")

    val_acc = val_accuracies[-1]

    if val_acc > (best_val_acc + patience_delta):
        best_val_acc = val_acc
        failing_epochs = 0
        save_checkpoint(
            best_acc_path,
            model, optimizer, scheduler,
            epoch=epoch,
            best_val_acc=best_val_acc,
            extra={"val_acc": val_acc}
        )
        print(f"Saved new best validation accuracy: {best_val_acc:.4f} at epoch {epoch}")
    else:
        failing_epochs += 1

    save_checkpoint(
        latest_path,
        model, optimizer, scheduler,
        epoch=epoch,
        best_val_acc=best_val_acc,
        extra={"val_acc": val_acc}
    )

    if failing_epochs >= patience:
        print(f"Early stopping at epoch {epoch}, best={best_val_acc:.4f}")
        break

