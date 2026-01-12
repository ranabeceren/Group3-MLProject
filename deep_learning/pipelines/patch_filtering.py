import os
import numpy as np
from deep_learning.utils.patch_selection import patch_selection

#uses patch_selection to delete the patches we don't want

def patch_filtering(data_dir="data_patches", threshold=0.6):
    print("DEBUG: patch_filtering threshold: ", threshold)

    img_patches = []
    mask_patches = []

    for city in os.listdir(data_dir):
        city_path = os.path.join(data_dir, city)
        if not os.path.isdir(city_path):
            continue

        imgs = np.load (f"{city_path}/images.npy")
        masks = np.load(f"{city_path}/masks.npy")

        # Print out number of patches before selection
        print(f"{city}: before = {len(imgs)}")

        # Select patches
        masks, imgs = patch_selection(masks, imgs, threshold)

        # Print out number of patches after selection
        print(f"{city}: after = {len(imgs)}")

        img_patches.append(imgs)
        mask_patches.append(masks)

    # Concatenate all image patches and all mask patches sapetrately into one single NumPy array
    img_patches = np.concatenate(img_patches, axis=0)
    mask_patches = np.concatenate(mask_patches, axis=0)

    # Print out total number of patches
    print(f"Total number of image patches: ", len(img_patches))
    print(f"Total number of mask patches: ", len(mask_patches))

    return img_patches, mask_patches

