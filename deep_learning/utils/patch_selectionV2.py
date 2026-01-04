import numpy as np

def patch_selectionV2(patches_mask, patches_img, threshold=0.7):
    """
    Removes patches with minimum building coverage.
    
    patches_mask: numpy array of shape (N, H, W) or (N, H, W, 1)
    patches_img: numpy array of shape (N, H, W, C)
    threshold: minimum fraction of building pixels to keep
    """
    # Assuming building pixels are > 0
    building_fraction = np.sum(patches_mask > 0, axis=(1, 2)) / patches_mask.shape[1] / patches_mask.shape[2]
    
    # Keep only patches with building fraction >= (1 - threshold)
    keep_indices = building_fraction >= (1 - threshold)
    
    return patches_mask[keep_indices], patches_img[keep_indices]