import os
import numpy as np
from utils.io import load_sentinel_image, load_mask
from utils.patching import make_patches

# Iterates over the raw-data folder and saves all the patches as .npy arrays or directly as lists of arrays

IMG_DIR = "data/raw/sentinel"
MASK_DIR = "data/processed"
PATCHES_DIR = "data_patches"

CITIES = [
    "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"
]

def process_city(c, patch_size):
    print(f"Processing {c}")

    img_path = f"{IMG_DIR}/{c}/openEO.tif"
    mask_path = f"{MASK_DIR}/{c}/mask.tif"

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Path is non-existent for {c}")
        return

    img = load_sentinel_image(img_path)
    mask = load_mask(mask_path)

    patches_img, patches_mask = make_patches(img, mask, patch_size=patch_size)
    patches_img = patches_img / 1000.0
    patches_img = np.clip(patches_img, 0, 1)

    # Create city folders inside data_patches
    city_patches = f"{PATCHES_DIR}/{c}"
    os.makedirs(city_patches, exist_ok=True)

    # Save patches into the folders
    np.save(f"{city_patches}/images.npy", patches_img)
    np.save(f"{city_patches}/masks.npy", patches_mask)

    print(f"{len(patches_img)} patches saved for {c}")
    return patches_img, patches_mask
