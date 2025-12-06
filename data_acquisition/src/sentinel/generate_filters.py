"""
Generate Visualization Filters (RGB, IRB, NIR, RED)

This module produces a set of visualization filters for the full-band Sentinel-2
GeoTIFFs downloaded by `download_fullbands_job.py`. The OpenEO job outputs a 
multi-band GeoTIFF (one file per city), and this script loads that file, extracts
the individual spectral bands by name (B02, B03, B04, B08), applies a local 
2–98% histogram stretch, and saves several color composites as PNG images.

The generated filters include:
 - RGB (True Color):        B04, B03, B02
 - IRB (False Color):       B08, B04, B03
 - NIR only:                B08
 - RED only:                B04

"""

import os
import rasterio
import numpy as np
from PIL import Image

CITIES = [
    "darmstadt", "porto", "amsterdam", "lisbon", "brisbane",
    "madrid", "melbourne", "barcelona", "mexico_city", "cairo", "berlin"
]

def stretch(band):
    """
    Apply a 2–98% percentile-based contrast stretch to a single band.

    This improves visualization by removing extreme pixel values and 
    enhancing contrast, without altering the scientific meaning of the band.
    """
    p2, p98 = np.percentile(band, (2, 98)) #find the clipping threshold
    return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1) #normalize to 0-1 range

def save_png(path, arr):
    """
    Convert a float32 array (0–1 range) into 8-bit PNG format and save it.
    """
    arr = (arr * 255).clip(0, 255).astype("uint8") #scale to 0-255
    Image.fromarray(arr).save(path)
    print(f"[INFO] Saved → {path}")

def process_city(city):
    """
    Load the city's full-band GeoTIFF (exported from OpenEO), extract relevant bands,
    apply histogram stretching, and save several visualization products.

    Expected input file: data/raw/sentinel/<city>/openEO.tif
    """
    print(f"\n=== Processing filters for {city} ===")

    full_path = f"data/raw/sentinel/{city}/openEO.tif"
    if not os.path.exists(full_path):
        print(f"[WARN] Missing openEO.tif for {city} — skipping.")
        return
    #if no full-band geotiff was downloaded, skip this city

    with rasterio.open(full_path) as src: #read the geotiff using rasterio
        arr = src.read()      # shape = (bands, H, W)
        band_names = src.descriptions  # band names from TIFF metadata

    band_map = {name: arr[i] for i, name in enumerate(band_names)}
    #convert band names into a dictionary for easier access

    # Extract true bands by name
    B02 = band_map["B02"]  # blue
    B03 = band_map["B03"]  # green
    B04 = band_map["B04"]  # red
    B08 = band_map["B08"]  # nir

    city_dir = f"data/raw/sentinel/{city}"

    # RGB
    rgb = np.stack([stretch(B04), stretch(B03), stretch(B02)], axis=-1)
    save_png(f"{city_dir}/rgb.png", rgb)

    # IRB (False Color)
    irb = np.stack([stretch(B08), stretch(B04), stretch(B03)], axis=-1)
    save_png(f"{city_dir}/irb.png", irb)

    # NIR-only
    save_png(f"{city_dir}/nir.png", stretch(B08))

    # RED-only
    save_png(f"{city_dir}/red.png", stretch(B04))

    print(f"[DONE] All filters generated for {city}")

if __name__ == "__main__":
#apply the filter generation process to all cities
    for city in CITIES:
        process_city(city)
