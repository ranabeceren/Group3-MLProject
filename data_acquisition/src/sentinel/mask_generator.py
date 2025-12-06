"""
Rasterizing OSM Buildings into Binary Masks:

This module creates per-city building masks by rasterizing the cleaned,
UTM-aligned OSM building polygons into the same spatial resolution,
coordinate system, and grid layout as the processed Sentinel-2 imagery.

For each city, the script loads:
  - The full-band Sentinel raster (openEO.tif)
  - The cleaned building polygons (<city>_buildings_utm.geojson)

It then reprojects the OSM geometries into the raster's CRS (if needed),
performs rasterization using the raster's transform and resolution,
and outputs both:

  (1) mask.tif  → binary GeoTIFF aligned perfectly with the Sentinel grid  
  (2) mask.png  → simple 0–1 visualization for quick inspection  

In these masks:
  - 1 = building footprint  
  - 0 = non-building  

Producing aligned raster masks ensures that every pixel in
the Sentinel image corresponds exactly to the same location in the building mask.
"""

import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from PIL import Image
import numpy as np

CITIES = [
    "berlin", "darmstadt", "amsterdam", "mexico_city", "cairo",
    "madrid", "porto", "lisbon", "melbourne", "barcelona", "brisbane"
]

def save_mask_png(mask_array, png_path):
    """
    Convert a binary (0/1) mask array into an 8-bit grayscale PNG image.
    - 0 → black (non-building)
    - 1 → white (building)

    """
    img = (mask_array * 255).astype(np.uint8)
    Image.fromarray(img).save(png_path)
    print(f"[INFO] Saved PNG → {png_path}")

def create_building_mask_for_city(city):
    """
    Generate a raster-aligned building mask for a single city:
        1. Load Sentinel raster (openEO.tif)
        2. Load OSM building polygons (UTM-processed)
        3. Reproject buildings to match raster CRS
        4. Rasterize polygons into binary mask
        5. Save the mask as both GeoTIFF and PNG 
    """
    print(f"\n=== Processing mask for {city} ===")

    #construct required file paths
    raster_path = f"data/raw/sentinel/{city}/openEO.tif" #sentinel raster
    buildings_path = f"data/osm/{city}_buildings_utm.geojson" #clean osm buildings
    out_dir = f"data/processed/{city}" #output directory
    os.makedirs(out_dir, exist_ok=True)
    mask_path = f"{out_dir}/mask.tif"
    png_path = f"{out_dir}/mask.png"

    #load sentinel raster metadata
    if not os.path.exists(raster_path):
        print(f"[WARN] No raster for {city}, skipping.")
        return

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs #coordinate reference system
        raster_transform = src.transform #affine transform (pixel grid definition)
        width = src.width
        height = src.height

    #load osm building polygons
    if not os.path.exists(buildings_path):
        print(f"[WARN] No OSM buildings for {city}, skipping.")
        return

    gdf = gpd.read_file(buildings_path)

    # !!!!! ensure osm buildings are projected into the SAME CRS as the sentinel raster!!!!!!!
    gdf = gdf.to_crs(raster_crs)

    #rasterize building footprints (convert vectors to raster mask)
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry], #each polygon gets value 1
        out_shape=(height, width), #same pixel size & shape as sentinel
        transform=raster_transform, #ensures perfect spatial alignment
        fill=0, #areas without buildings get 0
        dtype="uint8"
    )

    #save the binary mask as geotiff
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1, #single band mask
        dtype="uint8",
        crs=raster_crs,
        transform=raster_transform
    ) as dst:
        dst.write(mask, 1)

    print(f"[DONE] Saved mask → {mask_path}")

    #also save a png preview (for quick inspection)
    save_mask_png(mask, png_path)


def create_masks_for_all_cities():
    """
    Run the mask creation process for every city in CITIES
    """
    for city in CITIES:
        create_building_mask_for_city(city)


if __name__ == "__main__":
    create_masks_for_all_cities()
