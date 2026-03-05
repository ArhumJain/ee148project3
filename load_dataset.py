import gdown
import os
import zipfile
from PIL import Image

DATASET_URLS = [
    # Copies with resized images to max (512, _) or (_, 512) dimension
    'https://drive.google.com/uc?export=download&id=1otCXQ5BfPeyGJgK3XS-kv5iAJphhCRvJ',
    'https://drive.google.com/uc?export=download&id=1ogM03UInKocJSD219s5peyqj4vMY94ob',
    'https://drive.google.com/uc?export=download&id=1ngRkIXsf4rZ-NFG1Q1cQB61X1qUKEYRx',
    'https://drive.google.com/uc?export=download&id=1n53ZL8tUvAf3PW7qLAwFh8yRHKy0CXX9',
    'https://drive.google.com/uc?export=download&id=1n3axJY9LmhZQxLsZUN6gcdIkTwuf5Ofg',
    'https://drive.google.com/uc?export=download&id=1g4kUzGMtUgXlPlB-StZ-rkxV9_3S4ReY',
    'https://drive.google.com/uc?export=download&id=1aR2-zsys1zuERGtpaqBqI1uV0XjGzYsc',
    'https://drive.google.com/uc?export=download&id=1NESAyQ7QEegcpbrKuN0EsIsoynEIsdP9',
    'https://drive.google.com/uc?export=download&id=1CwDEoSrxcNw_U4DX6JBy4ed4W1wYbSRp',
    'https://drive.google.com/uc?export=download&id=1ov6EFLfXfPAXW2BSXHaXdN2cpl9Tsg23',

    # Old url (full image sizes)
    'https://drive.google.com/uc?export=download&id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ',
]

HUGGINGFACE_DATASET = 'EE148-project/MNIST-in-the-world'

def download_and_extract(
    url: str = 'https://drive.google.com/uc?id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ',
    output_zip_path: str = 'data/dataset.zip',
    force_download: bool = False
) -> str:
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    if not os.path.exists(output_zip_path) or force_download:
        gdown.download(url, output_zip_path, quiet=False)
    data_dir = output_zip_path.replace('.zip', '')
    with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extracted to {data_dir}")
    return data_dir

def load_data_from_huggingface(dataset_name: str = HUGGINGFACE_DATASET) -> list[dict]:
    from datasets import load_dataset
    print(f"Downloading dataset from HuggingFace: {dataset_name}")
    ds = load_dataset(dataset_name, split='train', streaming=True)
    dataset = []
    for item in ds:
        dataset.append({
            'img': item['image'],
            'label': item['label'],
            'path': None,
        })
    print(f"Loaded {len(dataset)} images from HuggingFace")
    return dataset

def load_data(data_dir: str) -> list[dict]:
    items = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    for f in filenames:
        path = os.path.join(data_dir, f)
        label = int(path.split('_')[-1].replace('.jpg', '').replace('label', ''))
        items.append((path, label))
    return items

def download_and_load_data(
    urls: list[str] = DATASET_URLS,
    output_zip_path: str = 'data/dataset.zip',
) -> list[dict]:
    for url in urls:
        try:
            data_dir = download_and_extract(url, output_zip_path, force_download=False)
            return load_data(data_dir)
        except Exception as e:
            print(f"Google Drive download failed ({type(e).__name__}): {e}")

    print("All Google Drive links failed. Falling back to HuggingFace...")
    return load_data_from_huggingface()

items: list[dict] = download_and_load_data()
