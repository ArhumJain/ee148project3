from PIL import Image
from numpy import transpose
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomOrder, v2
from torch.utils.data import DataLoader, Subset, default_collate
import os
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.image as image
import types


from main import (
    ClassImages,
    CoAtNet0,
    TransformerDownsampleBlock,
    load_checkpoint,
    make_uniform_compose,
    compute_mean_std,
    get_or_compute_mean_std,
    make_final_compose,
    make_augment_compose,
    TARGET,
)

from load_dataset import items

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

checkpoint_path = "checkpoints/finetune224ema/best.pt"

model = CoAtNet0(num_classes=10, image_size=TARGET)
load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None)
model = model.to(DEVICE)
model.eval()

# ── Train / val split (same 80/20 seeded split as main.py) ──
num_samples = len(items)
num_train = int(0.8 * num_samples)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(num_samples, generator=g).tolist()
ids = perm
train_idx = perm[:num_train]
# val_idx   = perm[num_train:]
random_image_id = perm[random.randint(0, len(perm))]

print(items[random_image_id])

random_image_path, random_image_label = items[random_image_id]

random_image = Image.open(random_image_path)

mean = [0.5462899804115295, 0.5005961060523987, 0.45557186007499695]
std  = [0.25494423508644104, 0.24657973647117615, 0.24912121891975403]

final_tfms = make_final_compose(mean, std, target=TARGET)
transformed_image = final_tfms(random_image)

transformed_image = transformed_image[None, :, :, :].to(DEVICE) # batch size 1
prediction = torch.argmax(model(transformed_image), dim=1)

plt.imshow(random_image)
plt.title(f"True: {random_image_label}, Pred: {prediction[0]}")

plt.show()


def valaccPlots():
    # ── Training history plots ──
    checkpoint_dirs = [
        ("Pretrain Tiny ImageNet (128)", "checkpoints/pretrain_tiny_imagenet"),
        ("Pretrain Tiny ImageNet (224)", "checkpoints/pretrain_tiny_imagenet224"),
        ("Finetune 224", "checkpoints/finetune224"),
        ("Finetune 224 + EMA", "checkpoints/finetune224ema"),
    ]

    for name, ckpt_dir in checkpoint_dirs:
        history_path = os.path.join(ckpt_dir, "history.json")
        if not os.path.isfile(history_path):
            print(f"Skipping {name}: no history.json found")
            continue

        with open(history_path) as f:
            h = json.load(f)

        epochs = list(range(1, len(h["train_loss"]) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(name, fontsize=14, fontweight="bold")

        # Loss plot
        ax1.plot(epochs, h["train_loss"], label="Train Loss")
        ax1.plot(epochs, h["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, h["train_acc"], label="Train Acc")
        ax2.plot(epochs, h["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        best_val_idx = max(range(len(h["val_acc"])), key=lambda i: h["val_acc"][i])
        ax2.axvline(x=best_val_idx + 1, color="red", linestyle="--", alpha=0.5,
                    label=f"Best: {h['val_acc'][best_val_idx]:.4f} @ epoch {best_val_idx + 1}")
        ax2.legend()

        fig.tight_layout()

    plt.show()


@torch.no_grad()
def show_failures(model, items, transform, device=DEVICE, num_show=4):
    """Go through all data, collect misclassified images, and display four of them."""
    model.eval()
    failures = []  # (image_path, true_label, predicted_label, confidence)

    for path, label in tqdm(items, desc="Scanning for failures"):
        img = Image.open(path)
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        conf = probs[0, pred].item()

        if pred != label:
            failures.append((path, label, pred, conf))
        img.close()

    print(f"\nTotal: {len(items)}, Failures: {len(failures)}, "
          f"Accuracy: {1 - len(failures)/len(items):.4f}")

    # Write all failures to JSON
    failures_json = [
        {"path": p, "true": t, "pred": pr}
        for p, t, pr, _ in failures
    ]
    with open("failures.json", "w") as f:
        json.dump(failures_json, f, indent=2)
    print(f"Wrote {len(failures)} failures to failures.json")

    if len(failures) == 0:
        print("No failures found!")
        return

    shown = random.sample(failures, min(num_show, len(failures)))

    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]
    fig.suptitle(f"Misclassified Samples ({len(failures)}/{len(items)} total failures)",
                 fontsize=14, fontweight="bold")

    for ax, (path, true, pred, conf) in zip(axes, shown):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"True: {true}, Pred: {pred}\nConf: {conf:.2%}", fontsize=11)
        ax.axis("off")
        img.close()

    fig.tight_layout()
    plt.show()

def show_failures_from_json(path="failures.json", num_show=100, per_page=20):
    """Load failures from JSON and display in a paginated, scrollable viewer.
    Navigate with arrow keys (left/right), scroll wheel, or 'q' to quit."""
    with open(path) as f:
        failures = json.load(f)

    shown = random.sample(failures, min(num_show, len(failures)))
    cols = 5
    rows = per_page // cols
    num_pages = (len(shown) + per_page - 1) // per_page
    state = {"page": 0}

    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
    axes = axes.flatten()

    def draw_page(page):
        for ax in axes:
            ax.clear()
            ax.axis("off")
        start = page * per_page
        end = min(start + per_page, len(shown))
        for ax, entry in zip(axes, shown[start:end]):
            img = Image.open(entry["path"])
            ax.imshow(img)
            ax.set_title(f"True: {entry['true']}, Pred: {entry['pred']}", fontsize=10)
            ax.axis("off")
            img.close()
        fig.suptitle(f"Failures (page {page+1}/{num_pages}, {len(failures)} total)  "
                     f"[scroll / arrow keys to navigate, 'q' to quit]",
                     fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right" and state["page"] < num_pages - 1:
            state["page"] += 1
            draw_page(state["page"])
        elif event.key == "left" and state["page"] > 0:
            state["page"] -= 1
            draw_page(state["page"])
        elif event.key == "q":
            plt.close(fig)

    def on_scroll(event):
        if event.button == "up" and state["page"] > 0:
            state["page"] -= 1
            draw_page(state["page"])
        elif event.button == "down" and state["page"] < num_pages - 1:
            state["page"] += 1
            draw_page(state["page"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    draw_page(0)
    fig.tight_layout()
    plt.show()

@torch.no_grad()
def attention_heat_map(model, image_tensor, layer, upscale=False):
    def patched_forward(self, x):
        B, N, C = x.shape
        q = self.W_q(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)

        rel_bias = self.relative_bias_table[:, self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(1, self.num_heads, N, N)

        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale + rel_bias
        attn = F.softmax(attn, dim=-1)

        self._saved_attn = attn.detach()

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.W_o(out)

    original_forward = layer.attn.forward
    layer.attn.forward = types.MethodType(patched_forward, layer.attn)
    model(image_tensor)
    layer.attn.forward = original_forward

    matrix = layer.attn._saved_attn
    attention_averages_map = torch.mean(matrix, dim=-2).reshape((matrix.shape[0], matrix.shape[1], int(matrix.shape[-1]**0.5), int(matrix.shape[-1]**0.5)))

    avg_map = attention_averages_map[0].mean(dim=0).cpu().numpy()

    img_display = image_tensor[0].cpu()
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img_display = (img_display * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img_display)
    ax1.set_title("Input Image")
    ax1.axis("off")

    avg_map[0, 0] = avg_map.mean()
    if upscale:
        from PIL import Image as PILImage
        import numpy as np
        avg_map = np.array(PILImage.fromarray(
            (avg_map / avg_map.max() * 255).astype('uint8')
        ).resize((TARGET, TARGET), PILImage.BICUBIC))
    ax2.imshow(avg_map, cmap="jet", interpolation="nearest")
    ax2.set_title("Attention Map")
    ax2.axis("off")

    fig.tight_layout()
    plt.show()

    return attention_averages_map

attention_heat_map(model, transformed_image, model.s3[0], True)
# show_failures(model, items, final_tfms)
# show_failures_from_json()

