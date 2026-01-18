import os
import numpy as np
from deep_learning.utils.io import load_sentinel_image, load_mask
from deep_learning.utils.patches import make_patches

IMG_DIR = "data/raw/sentinel"
MASK_DIR = "data/processed"
PATCH_DIR = "data_patches"
PATCH_SIZE = 64

CITIES = [ "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"]

for city in CITIES:
    print(f"Processing {city}")

    img_path = f"{IMG_DIR}/{city}/openEO.tif"
    mask_path = f"{MASK_DIR}/{city}/mask.tif"

    img = load_sentinel_image(img_path)   # (4, H, W)
    mask = load_mask(mask_path)            # (H, W)

    patches_img, patches_mask = make_patches(img, mask, PATCH_SIZE)

    # Normalisierung (Sentinel-typisch)
    patches_img = patches_img / 1000.0
    patches_img = np.clip(patches_img, 0, 1)

    out_dir = f"{PATCH_DIR}/{city}"
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"{out_dir}/images.npy", patches_img)
    np.save(f"{out_dir}/masks.npy", patches_mask)

    print(f"Saved {patches_img.shape[0]} patches")
