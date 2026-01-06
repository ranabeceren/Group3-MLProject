import numpy as np

def make_patches(img, mask, patch_size=128):
    """
    Cuts a large image and mask into smaller aligned patches.

    Args:
        img (np.ndarray): (C, H, W)
        mask (np.ndarray): (H, W)
        patch_size (int): size of square patches
    Returns:
        patches_img (np.ndarray): (number of patches, C, patch_size, patch_size)
        patches_mask (np.ndarray): (number of patches, patch_size, patch_size)
    """ 
    H, W = mask.shape
    patches_img = []
    patches_mask = []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):

            patch_img = img[:, i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]

            if patch_img.shape[1] != patch_size or patch_img.shape[2] != patch_size:
                continue

            patches_img.append(patch_img)
            patches_mask.append(patch_mask)

    return np.stack(patches_img), np.stack(patches_mask)
