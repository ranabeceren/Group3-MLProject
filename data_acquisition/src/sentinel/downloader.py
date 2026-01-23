# UTILITY MODULS-7

"""
7-Full Download Pipeline for a Single City:

This module implements the complete Sentinel-2 preprocessing pipeline for a 
given city. Starting from a bounding box file (<city>_bbox.json), it constructs 
the required GeoJSON polygon, loads the Sentinel-2 L2A collection through the 
OpenEO backend, applies pixel-level cloud masking using the SCL band, computes 
a median temporal composite, extracts the RGB bands, downloads the result as a 
GeoTIFF, and finally applies a local histogram stretch to improve visualization.

The pipeline also converts the stretched GeoTIFF into an 8-bit PNG file so that 
the output can be easily inspected. This module essentially  serves as the orchestrator 
that chains together all individual sub-modules (connection handling, cloud masking, 
composites, etc.).

ps: it is in utility moduls because it is created to test the pipeline, to download a 
single city quicly and to debug etc. it is like a mini version of the main downloader
'download_fullbands_job.py'

"""
import json
import os
import rasterio
import numpy as np
from PIL import Image


from connection import get_connection
from band_loader import load_sentinel_collection
from cloud_mask import apply_cloud_mask
from composites import reduce_to_median, create_rgb
from bbox_utils import bbox_to_geojson_polygon
#internal modules handling connection, loading, masking, composites, bbox conversion

def load_bbox(json_path):
    """
    load the bbox dictionary from a <city>_bbox.json file
    expected format: {min_lon, min_lat, max_lon, max_lat}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def local_stretch(input_path, output_path):
    """
    apply a 2-98% local histogram stretch to improve RGB visualization
    (by strectching each band based on the 2nd and 98th percentile,
    we enhance contrast while avoiding saturation) 
    """
    print("[INFO] Applying local histogram stretch (2–98%)...")

    with rasterio.open(input_path) as src:
        img = src.read().astype(float) #shape: (3, H, W)
        profile = src.profile #original metadata
    #read the raw RGB geotiff

    def stretch(band): #define a stretching function applied band-wise
        p2, p98 = np.percentile(band, (2, 98)) #clip violent extremes
        return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)

    stretched = np.stack([
        stretch(img[0]), #red
        stretch(img[1]), #green
        stretch(img[2])  #blue
    ])
    #apply stretching to rgb channels

    profile.update(dtype="float32")
    #keep geotiff metadata but change dtype to float32

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stretched.astype("float32"))
    #save the stretched rgb to new path

    print(f"[INFO] Saved stretched RGB → {output_path}")


def download_sentinel_for_city(city_name, bbox_path):
    """
    full sentinel pipeline for ONE city:
        1-load bbox JSON
        2-convert to GeoJSON polygon
        3-load sentinel collection
        4-apply cloud mask
        5-compute median composite
        6-extract rgb
        7-download as geotiff
        8-apply local contrast stretching 
        9-export png preview
    """
    print("\n==============================")
    print(f"[INFO] Starting Sentinel pipeline for: {city_name}")
    print("==============================\n")

    bbox = load_bbox(bbox_path)
    polygon = bbox_to_geojson_polygon(bbox)
    #load bbox JSON convert and convert to openeo-ready geojson polygon

    con = get_connection()
    #ensure backend connection exists

    cube = load_sentinel_collection(polygon)
    #step1- load sentinel2 cube

    masked = apply_cloud_mask(cube)
    #step2- apply cloud masking based on scl band

    median = reduce_to_median(masked)
    #step3- median temporal composite to remove cloudy gaps

    rgb = create_rgb(median)
    #step4- extract rgb (B04,B03,B02)

    out_dir = f"data/raw/sentinel/{city_name}"
    os.makedirs(out_dir, exist_ok=True)
    #prepare output directory for this city

    raw_rgb = f"{out_dir}/sentinel_rgb_raw.tif"
    final_rgb = f"{out_dir}/sentinel_rgb.tif"
    #file paths

    print(f"[INFO] Downloading raw RGB → {raw_rgb}")
    rgb.download(raw_rgb, format="GTIFF")
    #download from openeo backend

    local_stretch(raw_rgb, final_rgb)
    #step5- local contrast stretch 

    print("[INFO] Pipeline completed successfully.\n")

    png_path = f"{out_dir}/sentinel_rgb.png"
    save_png_from_tif(final_rgb, png_path)
    #step6- create png preview for inspection

def save_png_from_tif(tif_path, png_path):
    """
   convert a stretched float32 rgb geotiff (0-1 range) into an 8-bit png (0-255)
   ps: we made this because geotiff images cannot be viewed easily (requires GIS software)
    """
    with rasterio.open(tif_path) as src:
        arr = src.read()  # shape(3, H, W), float32 0–1

    arr_png = (arr * 255).clip(0, 255).astype("uint8")
    arr_png = np.transpose(arr_png, (1, 2, 0))  # reshape: (H, W, 3)
    #scale 0-1 floats to uint8 (0-255)

    Image.fromarray(arr_png).save(png_path)
    print(f"[INFO] Saved PNG → {png_path}")

    