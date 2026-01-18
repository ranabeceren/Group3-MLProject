import numpy as np

def make_patches(img, mask, patch_size):
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

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):

            patch_img = img[:, i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]

            if patch_img.shape[1] != patch_size or patch_img.shape[2] != patch_size:
                continue

            patches_img.append(patch_img)
            patches_mask.append(patch_mask)

    return np.stack(patches_img), np.stack(patches_mask)

def patch_selection(patches_mask, patches_img, threshold):

    """
    Removes patches with less building coverage than the set threshold.
    e.g. threshold=0.7, patches with less than 0.7 building coverage will be removed.
    """
    
    # Initiate an empty list to track the indicies to delete patch and mask
    indices = []
    for i, patch_mask in enumerate(patches_mask):
        count = np.sum(patch_mask == 0)
        non_building_percentage = count / patch_mask.size

        if non_building_percentage > threshold:
            indices.append(i)
    
    # Delete the detected mask and it's corresponding image
    patches_mask = np.delete(patches_mask, indices, axis=0)
    patches_img = np.delete(patches_img, indices, axis=0)

    return patches_mask, patches_img
