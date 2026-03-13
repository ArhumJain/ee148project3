import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from config import TARGET


def _resize_long_side(img, target):
    w, h = img.size
    if w >= h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)


def _pad_to_square(img, target, fill=0):
    w, h = img.size
    pad_w = target - w
    pad_h = target - h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(img, padding, fill=fill)


def make_uniform_compose(target=TARGET, fill=0):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(lambda img: _resize_long_side(img, target)),
        T.Lambda(lambda img: _pad_to_square(img, target, fill)),
        T.ToTensor(),
    ])


def make_final_compose(mean, std, target=TARGET, fill=0):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(lambda img: _resize_long_side(img, target)),
        T.Lambda(lambda img: _pad_to_square(img, target, fill)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def make_augment_compose(mean, std, target=TARGET, fill=0):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(lambda img: _resize_long_side(img, target)),
        T.Lambda(lambda img: _pad_to_square(img, target, fill)),
        T.RandomApply([
            transforms.RandomAffine(
                degrees=15, translate=(0.10, 0.10), scale=(0.85, 1.15),
                shear=8, interpolation=InterpolationMode.BILINEAR,
            )
        ], p=0.9),
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02)
        ], p=0.5),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.15),
        T.ToTensor(),
        T.RandomErasing(0.2, scale=(0.01, 0.05)),
        T.Normalize(mean=mean, std=std),
    ])
