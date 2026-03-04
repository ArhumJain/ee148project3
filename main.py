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


TARGET = 224

def make_uniform_compose(target=TARGET, fill=0):
    # make all dimensions and aspect ratios uniform (1:1 aspect ratio)
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

@torch.no_grad()
def compute_mean_std(train_dataset, batch_size=64, num_workers=8, device="cuda"):
    """
    Computes channel-wise mean/std on the *training* set after uniformization.
    Assumes train_dataset returns (tensor_image, label) and its transform is set.
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False, persistent_workers=(num_workers > 0), prefetch_factor=( 4 if num_workers > 0 else None))

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


# ── Model ──

class SqueezeExcitation(nn.Module):
                       # input_features, hidden_dim
    def __init__(self, input_features, expansion=0.25):
        super().__init__()
        self.hidden_size = max(1, int(input_features * expansion))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) # Average pooling entire channel down to size 1, one scalar value
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
        output = x * scale[:, :, None, None] # per channel weighting
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

    def forward(self, x):
        return self.module(self.norm(x))

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
        else:
            self.identity_pool = nn.Identity()
        if self.project or downsample:
            self.identity_proj = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.identity_proj = nn.Identity()

        self.conv = nn.Sequential(
                nn.Conv2d(input_features, self.hidden_dim, kernel_size=1, stride=self.stride, padding=0, bias=False), # Expansion
                nn.BatchNorm2d(self.hidden_dim),
                nn.GELU(),

                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, groups=self.hidden_dim, bias=False), # depthwise convolution, setting groups 
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
