import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from config import TARGET, DATASET_MEAN, DATASET_STD


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
    def __init__(self, model: nn.Module, input_height: int, input_width: int,
                 input_channels: int = 3,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.mean = DATASET_MEAN
        self.std = DATASET_STD
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.preprocess_layers = nn.Sequential(
            ResizeLongSide(TARGET),
            PadToSquare(TARGET),
            transforms.Normalize(mean=self.mean, std=self.std),
        )

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)
        return torch.argmax(logits, dim=1)

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        self.cpu()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        self.to(self.device)

    @torch.jit.ignore
    def push_to_hub(self, token: str, repo_id: str = "ee148a-project",
                    filename: str = "pipeline-vit.pt"):
        local_path = f"temp_{filename}"
        self.save_pipeline_local(local_path)

        from huggingface_hub import HfApi
        api = HfApi(token=token)

        print(f"Uploading {filename} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload compiled pipeline: {filename}",
        )

        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Success! Upload available at https://huggingface.co/{repo_id}/blob/main/{filename}")
        return True

    @torch.jit.ignore
    def run(self, pil_images: list):
        convert_to = "RGB" if self.input_channels == 3 else "L"
        tensor_list = [
            transforms.ToTensor()(img.convert(convert_to))
            for img in pil_images
        ]
        processed = [self.preprocess_layers(x) for x in tensor_list]
        batch = torch.stack(processed).to(self.device)
        return self.forward(batch).tolist()
