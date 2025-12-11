"""
This script analyzes the predicted building mask for each city.
It loads the mask raster, counts how many pixels are labeled as
building vs. non-building, and reports their ratios.
"""

import rasterio
import numpy as np
import os

CITIES = [
    "berlin", "darmstadt", "amsterdam", "mexico_city", "cairo",
    "madrid", "porto", "lisbon", "melbourne", "barcelona", "brisbane"
]

results = []

for city in CITIES:
    mask_path = f"data/processed/{city}/mask.tif"
    
    # Skip city if mask file does not exist
    if not os.path.exists(mask_path):
        print(f"[WARN] Missing mask for {city}")
        continue
    
    # Load raster mask (1 = building, 0 = non-building)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    
    # Compute basic pixel statistics
    total = mask.size
    buildings = np.sum(mask == 1)
    nonbuildings = np.sum(mask == 0)

    results.append({
        "city": city,
        "total_pixels": total,
        "building_pixels": buildings,
        "non_building_pixels": nonbuildings,
        "building_ratio": buildings / total,
        "non_building_ratio": nonbuildings / total
    })

for r in results:
    print(f"\nCity: {r['city']}")
    print(f"Total pixels: {r['total_pixels']}")
    print(f"Building pixels: {r['building_pixels']} ({r['building_ratio']*100:.2f}%)")
    print(f"Non-building pixels: {r['non_building_pixels']} ({r['non_building_ratio']*100:.2f}%)")
