import os
import numpy as np
from utils.io import load_sentinel_image, load_mask
from utils.patching import make_patches

IMG_DIR = "data/raw/sentinel"
MASK_DIR = "data/processed"
PATCHES_DIR = "data_patches"

CITIES = [
    "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"
]

def process_city(city):
    print(f"Processing {city}")

    img_path = f"{IMG_DIR}/{city}/openEO.tif"
    mask_path = f"{MASK_DIR}/{city}/mask.tif"

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Path is non-existent for {city}")
        return

    img = load_sentinel_image(img_path)
    mask = load_mask(mask_path)

    patches_img, patches_mask = make_patches(img, mask, patch_size=PATCH_SIZE)

    # Create city folders inside data_patches
    city_patches = f"{PATCHES_DIR}/{city}"
    os.makedirs(city_patches, exist_ok=True)

    # Save patches into the folders
    np.save(f"{city_out}/images.npy", patches_img)
    np.save(f"{city_out}/masks.npy", patches_mask)

    print(f"{len(patches_img)} patches saved for {city}")

for city in CITIES:
    process_city(city)
