import numpy as np

def patch_selection(patches_mask, patches_img, threshold=0.3):
    """
    Removes patches with minimum building coverage.
    """
    
    # Iniciate an empty list to track the indicies to delete patch and mask
    indices = []
    for i, patch_mask in enumerate(patches_mask):
        building_pixles = np.sum(patch_mask > 0)
        building_percentage = building_pixles / patch_mask.size

        if building_percentage < threshold:
            indices.append(i)
        #count = np.sum(patch_mask == 0)
        #non_building_percentage = count / patch_mask.size

        #if non_building_percentage > threshold:
        #    indices.append(i)
    
    # Delete the detected mask and it's corresponding image
    patches_mask = np.delete(patches_mask, indices, axis=0)
    patches_img = np.delete(patches_img, indices, axis=0)

    return patches_mask, patches_img
