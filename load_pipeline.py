import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from create_pipeline import DigitClassifierPipeline
from load_dataset import items
import matplotlib.pyplot as plt

load_hf_info = {
    'username': 'Isukali',
    'token': 'REDACTED',
    'repo_name': 'ee148a-project',
    'filename': 'pipeline-cnn.pt'
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
    # Assumes images and labels still exist
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
