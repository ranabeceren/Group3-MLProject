# SENTINEL MAIN PIPELINE

"""
Full Band Download:

This module implements the production-level Sentinel-2 download workflow for all
cities in our dataset. Unlike the smaller utility pipeline (downloader.py), which
downloads only RGB for quick inspection, this script requests the *full-band*
Sentinel-2 data (B02, B03, B04, B08) after cloud masking and median compositing.

The pipeline runs fully on the OpenEO backend using asynchronous jobs. For each
city, the steps are:

1. Load the bounding box from JSON.
2. Convert it to a valid GeoJSON Polygon (required by OpenEO).
3. Load the Sentinel-2 collection inside the backend.
4. Apply SCL-based cloud masking.
5. Compute a median composite across the entire time range.
6. Extract the complete set of spectral bands needed for our project.
7. Submit an asynchronous processing job to OpenEO.
8. Download the job results into the city's output directory.
"""

from sentinel.connection import get_connection
#reuse cached openeo connection
from sentinel.downloader import load_bbox
#reads <city>_bbox.json
from sentinel.band_loader import load_sentinel_collection
#load sentinel2 cubes
from sentinel.cloud_mask import apply_cloud_mask
from sentinel.composites import reduce_to_median, create_full_band_cube
from sentinel.bbox_utils import bbox_to_geojson_polygon

import os

cities = [
    "darmstadt", "porto", "amsterdam", "lisbon", "brisbane",
    "madrid", "melbourne", "barcelona", "mexico_city", "cairo",
    "berlin"
]

con = get_connection()
#establish or reuse a connection to the openeo backend
 
for city in cities: # *MAIN LOOP* -> process each city one by one

    print("\n=====================================")
    print(f"[INFO] Processing city: {city}")
    print("=====================================\n")

    bbox_path = f"data/osm/{city}_bbox.json"
    bbox = load_bbox(bbox_path)
    polygon = bbox_to_geojson_polygon(bbox)
    # 1-load and convert the bounding box into GeoJSON polygon

    cube = load_sentinel_collection(polygon)
    # 2-load sentinel2 collection for this city

    masked = apply_cloud_mask(cube)
    # 3-apply cloud mask

    median = reduce_to_median(masked)
    # 4-create median composite over the whole summer time range

    full = create_full_band_cube(median)
    # 5-extract all needed spectral bands (rgb and B08)

    out_dir = f"data/raw/sentinel/{city}"
    os.makedirs(out_dir, exist_ok=True)
    # 6-prepare output directory for this city

    print("[INFO] Creating asynchronous job...")

    job = full.save_result(format="GTIFF").create_job()
    # 7-save result as geotiff via openeo job system

    print("[INFO] Starting job...")
    job.start_and_wait()
    #waits until backend finishes processing

    print("[INFO] Downloading job results...")
    job.download_results(out_dir)

    print(f"[DONE] Full-band saved in: {out_dir}")

