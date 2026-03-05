import os
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from create_pipeline import DigitClassifierPipeline
from load_dataset import items
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

load_hf_info = {
    'username': os.environ['HF_USERNAME'],
    'token': os.environ['HF_TOKEN'],
    'repo_name': os.environ['HF_REPO_NAME'],
    'filename': os.environ['HF_FILENAME'],
}

model_path = hf_hub_download(
    repo_id=f"{load_hf_info['username']}/{load_hf_info['repo_name']}",
    filename=load_hf_info['filename'],
    token=load_hf_info['token']
)

loaded_pipeline = torch.jit.load(model_path)

def load_predict_sample(
    pipeline: DigitClassifierPipeline,
    seed: int = None
):
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    random_idxs = np.random.choice(len(items), size=15)
    sample_images = [Image.open(items[idx][0]) for idx in random_idxs]
    sample_labels = [items[idx][1] for idx in random_idxs]
    processed_images = [pipeline.preprocess_layers((transforms.ToTensor()) (img.convert("RGB"))) for img in sample_images]
    batch = torch.stack(processed_images)
    predictions = pipeline.forward(batch).tolist()
    for img, pred, true in zip(sample_images, predictions, sample_labels):
        if isinstance(pred, (list, tuple, np.ndarray, torch.Tensor)):
            print(f"WARNING! type(pred): {type(pred)}. Ensure a scalar value for each prediction per image")
        print(f"Predicted: {pred}, True: {true}")
        plt.imshow(img.resize((224, 224)))
        plt.show()
        print('='*100)

load_predict_sample(loaded_pipeline)

@torch.no_grad()
def test_full_dataset(pipeline, items, batch_size=64):
    """Run pipeline on all items in batches and report accuracy."""
    correct = 0
    total = len(items)

    for i in tqdm(range(0, total, batch_size), desc="Testing pipeline"):
        batch_items = items[i:i+batch_size]
        tensors = []
        labels = []
        for path, label in batch_items:
            img = Image.open(path)
            tensor = transforms.ToTensor()(img.convert("RGB"))
            tensors.append(pipeline.preprocess_layers(tensor).detach())
            labels.append(label)
            img.close()
        batch = torch.stack(tensors)
        preds = pipeline.forward(batch)
        correct += (preds == torch.tensor(labels)).sum().item()
        del batch, preds, tensors
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    acc = correct / total
    print(f"Full dataset accuracy: {acc:.4f} ({correct}/{total})")
    return acc

test_full_dataset(loaded_pipeline, items)
