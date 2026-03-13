import torch
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import v2
from PIL import Image

from config import SPLIT_SEED, TRAIN_FRACTION


class ClassImages(Dataset):
    def __init__(self, items, transform=None):
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


def split_indices(num_samples, train_fraction=TRAIN_FRACTION, seed=SPLIT_SEED):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g).tolist()
    num_train = int(train_fraction * num_samples)
    return perm[:num_train], perm[num_train:]


@torch.no_grad()
def compute_mean_std(dataset, batch_size=256, num_workers=4, device="cpu"):
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    channel_sum = torch.zeros(3, device=device)
    channel_sumsq = torch.zeros(3, device=device)
    total_pixels = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b, c, h, w = x.shape
        pixels = b * h * w
        total_pixels += pixels
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sumsq += (x * x).sum(dim=(0, 2, 3))

    mean = channel_sum / total_pixels
    var = channel_sumsq / total_pixels - mean * mean
    std = torch.sqrt(var.clamp_min(1e-12))
    return mean.cpu().tolist(), std.cpu().tolist()


def make_mixup_collate(num_classes, alpha=0.2):
    mixup = v2.MixUp(alpha=alpha, num_classes=num_classes)
    cutmix = v2.CutMix(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))
    return collate_fn
