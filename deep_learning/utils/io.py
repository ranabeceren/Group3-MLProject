import rasterio
import numpy as np

def load_sentinel_image(path):
    """
    Loads a multi-band Sentinel-2 GeoTIFF.

    Args:
        path (str): Path to Sentinel image.
    Returns:
        np.ndarray: float32 array of shape (C, H, W)
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    #img = img[:3, :, :]
    return img

    

def load_mask(path):
    """
    Loads a binary building mask.
    
    Args:
        path: Path to mask GeoTIFF.
    Returns:
        np.ndarray: uint8 array of shape (H,W)
    """
    with rasterio.open(path) as src:
        mask = src.read(1) # one band
        mask = mask.astype(np.uint8)
    return mask