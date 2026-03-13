import os
import json
import torch
from dotenv import load_dotenv

from config import TARGET, DEVICE
from model import CoAtNet0
from pipeline import DigitClassifierPipeline
from training import load_checkpoint

load_dotenv()
print("Using device:", DEVICE)

hf_info = {
    'username': os.environ['HF_USERNAME'],
    'token': os.environ['HF_TOKEN'],
    'repo_name': os.environ['HF_REPO_NAME'],
    'filename': os.environ['HF_FILENAME'],
}

checkpoint_path = "checkpoints/finetune224ema/best.pt"
model = CoAtNet0(num_classes=10, image_size=TARGET)
load_checkpoint(checkpoint_path, model)
model = model.to(DEVICE)
model.eval()

pipeline = DigitClassifierPipeline(
    model=model,
    input_height=TARGET,
    input_width=TARGET,
    input_channels=3,
)


def predict_sample(pipeline, seed=None):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    from load_dataset import items

    if seed is not None:
        np.random.seed(seed)
    random_idxs = np.random.choice(len(items), size=15)
    sample_images = [Image.open(items[idx][0]) for idx in random_idxs]
    sample_labels = [items[idx][1] for idx in random_idxs]
    predictions = pipeline.run(sample_images)
    for img, pred, true in zip(sample_images, predictions, sample_labels):
        print(f"Predicted: {pred}, True: {true}")
        plt.imshow(img.resize((224, 224)))
        plt.show()
        print('=' * 100)


def save_and_export(pipeline, hf_info):
    try:
        success = pipeline.push_to_hub(
            token=hf_info['token'],
            repo_id=f"{hf_info['username']}/{hf_info['repo_name']}",
            filename=hf_info['filename']
        )
        if success:
            with open('submission.json', 'w') as f:
                json.dump(hf_info, f, indent=4)
            print("Saved json to submission.json")
            return hf_info
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    save_and_export(pipeline=pipeline, hf_info=hf_info)
