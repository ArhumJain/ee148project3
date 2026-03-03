"""
Download Tiny ImageNet (100K images, 200 classes, 64x64) and build (filepath, label)
items in the same format as load_dataset.py.

Structure after extraction:
    data/tiny-imagenet-200/
        train/  → 200 folders (n01443537, …), each with 500 images
        val/    → images/ folder + val_annotations.txt
        test/   → images/ folder (no labels, unused)
"""

import os
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TINYIMAGENET_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download():
    """Download the zip if not already present."""
    if os.path.isdir(TINYIMAGENET_DIR) and os.listdir(TINYIMAGENET_DIR):
        print("[tiny-imagenet] Already extracted, skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isfile(ZIP_PATH):
        print("[tiny-imagenet] Downloading (~237 MB) ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("[tiny-imagenet] Download complete.")

    print("[tiny-imagenet] Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(ZIP_PATH)
    print("[tiny-imagenet] Ready.")


def _build_class_to_label(root):
    """Map WordNet IDs (n01443537, ...) -> integer labels 0-199."""
    wnids_path = os.path.join(root, "wnids.txt")
    with open(wnids_path) as f:
        wnids = sorted(f.read().strip().split("\n"))
    return {wnid: i for i, wnid in enumerate(wnids)}


def _collect_train(root, class_to_label):
    """Collect training items: each class has its own subfolder with images/."""
    items = []
    train_dir = os.path.join(root, "train")
    for wnid, label in class_to_label.items():
        img_dir = os.path.join(train_dir, wnid, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                items.append((os.path.join(img_dir, fname), label))
    return items


def _collect_val(root, class_to_label):
    """Collect validation items using val_annotations.txt."""
    val_dir = os.path.join(root, "val")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    items = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, wnid = parts[0], parts[1]
            label = class_to_label[wnid]
            path = os.path.join(val_dir, "images", fname)
            if os.path.isfile(path):
                items.append((path, label))
    return items


def load_tiny_imagenet_items():
    """
    Download (if needed) and return all (filepath, label) items combined.
    Labels are integers 0-199. Returns a single list for consistency with
    load_dataset.py — the caller handles train/val splitting.
    """
    _download()
    class_to_label = _build_class_to_label(TINYIMAGENET_DIR)
    train_items = _collect_train(TINYIMAGENET_DIR, class_to_label)
    val_items = _collect_val(TINYIMAGENET_DIR, class_to_label)
    all_items = train_items + val_items
    return all_items


# When imported, produce the items list
items_tiny_imagenet = load_tiny_imagenet_items()
print(f"[tiny-imagenet] Loaded {len(items_tiny_imagenet)} items, 200 classes.")
