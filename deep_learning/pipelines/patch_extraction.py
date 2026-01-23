import os
import numpy as np
from utils.io import load_sentinel_image, load_mask
from utils.patches import make_patches

'''Iterates over the raw-data folder and saves all the patches as .npy arrays or directly as lists of arrays'''

# Get absolute path to project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
# Set the paths to images, masks and patches
IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "sentinel")
MASK_DIR = os.path.join(BASE_DIR, "data", "processed")
PATCHES_DIR = os.path.join(BASE_DIR, "data_patches")

def process_city(city, patch_size):
    print(f"Processing {city}")

    img_path = f"{IMG_DIR}/{city}/openEO.tif"
    mask_path = f"{MASK_DIR}/{city}/mask.tif"

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Path is non-existent for {city}")
        return

    img = load_sentinel_image(img_path)
    mask = load_mask(mask_path)

    patches_img, patches_mask = make_patches(img, mask, patch_size=patch_size)
 
    # Create city folders inside data_patches
    city_patches = f"{PATCHES_DIR}/{city}"
    os.makedirs(city_patches, exist_ok=True)

    # Save patches into the folders
    np.save(f"{city_patches}/images.npy", patches_img)
    np.save(f"{city_patches}/masks.npy", patches_mask)

    print(f"{len(patches_img)} patches saved for {city}")
    

