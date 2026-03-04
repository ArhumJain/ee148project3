from PIL import Image
from numpy import transpose
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomOrder, v2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import transforms
import os
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.image as image

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

# ── Train / val split (same 80/20 seeded split as main.py) ──
num_samples = len(items)
num_train = int(0.8 * num_samples)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(num_samples, generator=g).tolist()
ids = perm
train_idx = perm[:num_train]
val_idx   = perm[num_train:]

hf_info = {
    'username': 'Isukali',
    'token': 'REDACTED',
    'repo_name': 'ee148a-project',   # DON'T CHANGE
    'filename': 'pipeline-vit.pt'    # DON'T CHANGE
}

class ResizeLongSide(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
    def forward(self, x):
        w = x.shape[-1]
        h = x.shape[-2]
        if w >= h:
            new_w = self.target
            new_h = int(round(h * (self.target / w)))
        else:
            new_h = self.target
            new_w = int(round(w * (self.target / h)))

        return TF.resize(x, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)

class PadToSquare(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
    def forward(self, x):
        w, h = x.shape[-1], x.shape[-2]
        pad_w = self.target - w
        pad_h = self.target - h

        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return TF.pad(x, padding, fill=0)


class DigitClassifierPipeline(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_height: int,
        input_width: int,
        input_channels: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.mean = [0.5462899804115295, 0.5005961060523987, 0.45557186007499695]
        self.std  = [0.25494423508644104, 0.24657973647117615, 0.24912121891975403]

        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.preprocess_layers = nn.Sequential(
            ResizeLongSide(TARGET),
            PadToSquare(TARGET),
            transforms.Normalize(mean=self.mean, std=self.std)
        )


        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        The main entry point for the saved model.
        Args:
            images: Tensor of shape (B, C, H, W) or (B, H, W)
        Returns:
            Tensor of class indices (B,)
        """

        logits = self.model(images)
        predictions = torch.argmax(logits, dim=1)

        return predictions

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        """
        Compiles the ENTIRE pipeline (transforms + model + post)
        and saves it to a file.
        """
        self.cpu()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        self.to(self.device)

    @torch.jit.ignore
    def push_to_hub(
        self,
        token: str,
        repo_id: str = 'ee148a-project',
        filename: str = "pipeline-cnn.pt",
    ):
        """
        Saves the pipeline to a local file and pushes it to the Hugging Face Hub.

        Args:
            token (str): HF token.
            repo_id (str): The ID of your repo,
                           e.g., "{username}/ee148a-project"
            filename (str): The name the file will have on the Hub,
                            e.g. 'pipeline-cnn.pt'
        """
        # 1. Save locally first
        local_path = f"temp_{filename}"
        self.save_pipeline_local(local_path)

        # 2. Initialize API
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)

        # 3. Upload the file
        print(f"Uploading {filename} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload compiled pipeline: {filename}"
        )

        # 4. Cleanup local temp file
        import os
        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Success! Upload available at https://huggingface.co/{repo_id}/blob/main/{filename}")
        return True


    @torch.jit.ignore
    def run(self, pil_images: list):
        """Run pipeline on PIL images."""


        if self.input_channels == 3:
            convert_to = 'RGB'
        elif self.input_channels == 1:
            convert_to = 'L'

        tensor_list = [
            transforms.ToTensor()(img.convert(convert_to))
            for img in pil_images
        ]
        processed_tensor_list = [
            self.preprocess_layers(x)
            for x in tensor_list
        ]
        # processed_tensor_list = [
        #     uniform_normalize_transform(img) for img in pil_images
        # ]

        batch = torch.stack(processed_tensor_list).to(self.device)
        print(batch.shape)
        predictions = self.forward(batch).tolist()

        return predictions

# Test pipeline fuctionality (by running sample images)
def predict_sample(
    pipeline: DigitClassifierPipeline,
    seed: int = None
):
    # Assumes images and labels still exist
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    random_idxs = np.random.choice(len(items), size=15)
    sample_images=[Image.open(items[idx][0]) for idx in random_idxs]
    sample_labels = [items[idx][1] for idx in random_idxs]
    predictions = pipeline.run(sample_images)
    for img, pred, true in zip(sample_images, predictions, sample_labels):
        if isinstance(pred, (list, tuple, np.ndarray, torch.Tensor)):
            print(f"WARNING! type(pred): {type(pred)}. Ensure a scalar value for each prediction per image")
        print(f"Predicted: {pred}, True: {true}")
        plt.imshow(img.resize((224, 224)))
        plt.show()
        # display(img.resize((128, 128)))
        print('='*100)

checkpoint_path = "checkpoints/finetune224ema/best.pt"


cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
print("Using device:", DEVICE)

model = CoAtNet0(num_classes=10, image_size=TARGET)
load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None)
model = model.to(DEVICE)
model.eval()

pipeline = DigitClassifierPipeline(
    model=model,
    input_height=TARGET,#..,
    input_width=TARGET, #..,
    input_channels=3, #..,
)

def save_and_export(
    pipeline: DigitClassifierPipeline,
    hf_info: dict,
):
    try:
        success = pipeline.push_to_hub(
            token=hf_info['token'],
            repo_id=f"{hf_info['username']}/{hf_info['repo_name']}",
            filename=hf_info['filename']
        )
        if success:
            import json
            with open('submission.json', 'w') as f:
                json.dump(hf_info, f, indent=4)
            print("Saved json to submission.json")
            return hf_info
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    save_and_export(pipeline=pipeline, hf_info=hf_info)
