"""
Visual Overlap Analysis Between OSM Buildings and Satellite Imagery:

This module generates two important quality-control products for each city:
1. overlapping_mask.png  
   - A rasterized representation of the OSM building polygons.
   - Overlapping geometry pixels accumulate values (MergeAlg.add), allowing us
     to detect inconsistencies such as duplicated or intersecting polygons.

2. overlay.png  
   - A semi-transparent overlay showing the building footprints on top of the 
     Sentinel RGB image. This makes it visually obvious whether the OSM buildings
     are correctly aligned with the satellite imagery after reprojection and 
     rasterization.

"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from PIL import Image

def create_overlapping_and_overlay(city):
    """
    For a given city, generate two visual products:
      1) overlapping_mask.png  : counts overlapping OSM polygons per pixel  
      2) overlay.png           : OSM buildings blended with Sentinel RGB  
    """
    print(f"\n=== Creating overlapping + overlay for {city} ===")

    #file paths
    raster_path = f"data/raw/sentinel/{city}/openEO.tif" #sentinel rgb/nir raster
    buildings_path = f"data/osm/{city}_buildings_utm.geojson" #cleaned, utm-projected osm buildings
    out_dir = f"data/processed/{city}"
    os.makedirs(out_dir, exist_ok=True)

    overlap_png  = f"{out_dir}/overlapping_mask.png"
    overlay_png  = f"{out_dir}/overlay.png"

    #load sentinel raster (rgb only)
    if not os.path.exists(raster_path):
        print(f"[WARN] No raster for {city}, skipping.")
        return

    with rasterio.open(raster_path) as src:
        rgb = src.read([1,2,3])  # B04,B03,B02 → red, green, blue
        raster_crs = src.crs
        raster_transform = src.transform
        height, width = src.height, src.width

    rgb = rgb.astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)
    rgb = (rgb * 255).astype(np.uint8)
    # Normalize RGB for PNG

    rgb_img = np.transpose(rgb, (1, 2, 0))   
    # Rasterio reads as [3, H, W], convert to [H,W,3]

    # load osm buildings and reproject
    if not os.path.exists(buildings_path):
        print(f"[WARN] No buildings for {city}, skipping.")
        return

    #load building polygons and convert to same crs as sentinel raster
    gdf = gpd.read_file(buildings_path).to_crs(raster_crs)

    #CREATE OVERLAPPING MASK
    overlapping_mask = rasterize(
        #this raster tracks how man polygons cover each pixel:
        #1: normal building footprint 
        #2+: overlapping or duplicated polygons
        [(geom, 1) for geom in gdf.geometry], #each polygon contributes +1
        out_shape=(height, width),
        transform=raster_transform,
        fill=0,
        dtype="uint16",   # prevent overflow while adding
        merge_alg=rasterio.enums.MergeAlg.add   # if polygons overlap, values accumulate
    )

    # Normalize overlapping mask for visualization
    if overlapping_mask.max() > 0:
        overlap_norm = overlapping_mask / overlapping_mask.max()
    else:
        overlap_norm = overlapping_mask

    #save overlapping mask png
    overlap_png_img = (overlap_norm * 255).astype(np.uint8)
    Image.fromarray(overlap_png_img).save(overlap_png)
    print(f"[INFO] Saved → {overlap_png}")

    #CREATE OVERLAY

    # Binary mask for overlay
    binary_mask = (overlapping_mask > 0).astype(np.uint8) #1 where buildings exist

    # red overlay for building pixels
    red_layer = np.zeros_like(rgb_img)
    red_layer[..., 0] = 255  # Pure red channel

    # Alpha blending
    alpha = 0.35
    overlay_img = rgb_img.copy()
    overlay_img[binary_mask == 1] = (
        (1 - alpha) * rgb_img[binary_mask == 1] + alpha * red_layer[binary_mask == 1]
    ).astype(np.uint8)

    Image.fromarray(overlay_img).save(overlay_png)
    print(f"[INFO] Saved → {overlay_png}")

    print("[DONE]")


#RUN FOR ALL CITIES

CITIES = [
    "berlin", "darmstadt", "amsterdam", "mexico_city", "cairo",
    "madrid", "porto", "lisbon", "melbourne", "barcelona", "brisbane"
]

def run_all():
    for city in CITIES:
        create_overlapping_and_overlay(city)

if __name__ == "__main__":
    run_all()
