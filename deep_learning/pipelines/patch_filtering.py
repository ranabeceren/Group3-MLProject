import os
import numpy as np
from utils.patches import patch_selection
#uses patch_selection to delete the patches with less building coverage

def patch_filtering(patches_dir, threshold):

    # Get absolute path to project root
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    # Build the data dir
    PATCHES_DIR = os.path.join(BASE_DIR, patches_dir)

    img_patches = []
    mask_patches = []
    print("\n_____________________________________\n")
    print("!!PATCH FILTRATION STARTED!!")
    print("_____________________________________\n")
    for city in os.listdir(PATCHES_DIR):
        city_path = os.path.join(PATCHES_DIR, city)
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
        print("------------------------")

        img_patches.append(imgs)
        mask_patches.append(masks)

    # Concatenate all image patches and all mask patches sapetrately into one single NumPy array
    img_patches = np.concatenate(img_patches, axis=0)
    mask_patches = np.concatenate(mask_patches, axis=0)

    # Print out total number of patches
    print(f"Total number of image patches: ", len(img_patches))
    print(f"Total number of mask patches: ", len(mask_patches))

    print("\n_____________________________________\n")
    print("!!PATCH FILTRATION COMPLETED!!")
    print("_____________________________________")

    return img_patches, mask_patches

