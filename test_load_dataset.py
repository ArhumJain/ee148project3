from load_dataset import items
import random
from PIL import Image
import matplotlib.pyplot as plt
import os

print(len(items))

i = random.randrange(len(items))
path, label = items[i]
print(f"idx = {i}, label = {label}")
os.makedirs("output_images", exist_ok=True)
with Image.open(path) as img:
    img.save("output_images/test_load_dataset.png")

min_res = None
min_area = None
max_res = None
max_area = None

for path, _ in items:
    with Image.open(path) as img:
        w, h = img.size
    area = w * h

    if min_area is None or area < min_area:
        min_area = area
        min_res = (w, h)

    if max_area is None or area > max_area:
        max_area = area
        max_res = (w, h)

if min_res is None:
    print("No images provided.")
else:
    print(f"Smallest resolution found: {min_res[0]}x{min_res[1]}")
    print(f"Largest  resolution found: {max_res[0]}x{max_res[1]}")


