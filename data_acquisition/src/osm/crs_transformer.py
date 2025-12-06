"""
Step4: Reprojecting OSM buildings to UTM

In this script, we implement the fourth step of our data acquisition and alignment
pipeline. After extracting building polygons and computing each city's bounding box,
we now convert the cleaned OSM building data into an appropriate UTM projection.
Working in UTM is important because Sentinel-2 data and many geospatial operations
require metric coordinates rather than lat/lon degrees, especially for rasterization,
distance calculations, and spatial alignment.

For each city, we load the cleaned <city>_buildings.geojson file, estimate the correct
UTM zone based on the centroid of its bounding box, and then reproject all building
geometries into that UTM CRS. The resulting <city>_buildings_utm.geojson files provide
a uniform and metric-based representation of the data, which is essential for the next
steps in the pipeline, such as creating raster building masks and aligning OSM data
with Sentinel-2 imagery.
"""

import os
import geopandas as gpd
import math
from osm_processing import load_buildings
#instead of defining tje load_buildings function again, we import it from 
#osm_processing.py to avoid code duplication and keep the pipeline consistent

def guess_utm_epsg(gdf):
    """
    Compute UTM CRS from bounding-box center 
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    #total_bounds returns (min_lon, min_lat, max_lon, max_lat)
    lon = (minx + maxx) / 2
    #approximate longitude of the city's center area
    lat = (miny + maxy) / 2
    #approximate latitude of the city's center area

    zone_number = int(math.floor((lon + 180) / 6) + 1)
    #convert the longtitude to a UTM zone index (1-60)

    epsg = f"326{zone_number:02d}" if lat >= 0 else f"327{zone_number:02d}"
    #EPSG code: 326XX for northern hemisphere, 327XX for southern hemisphere

    print(f"[INFO] Selected UTM zone: EPSG:{epsg}")
    #print which UTM CRS will be used
    return epsg
    #return the epsg code so we can reproject the GeoDataFrame

def reproject_to_utm(gdf, epsg_code):
#reproject buildings from lat/lon to UTM coordinates
    print(f"[INFO] Reprojecting to EPSG:{epsg_code} ...")
    #print which crs we are converting the data into
    gdf_utm = gdf.to_crs(f"EPSG:{epsg_code}")
    #convert the building polygons into the metric UTM coordinate system
    print("[INFO] Reprojection complete.")
    #give notification that the transformation finished successfully
    return gdf_utm
    #return the reprojected GeoDataFrame

def save_utm(gdf_utm, city_name, output_dir="data/osm"):
#save the utm-projected building layer
    os.makedirs(output_dir, exist_ok=True)
    #make sure the output directory exists and create it if not
    output_path = os.path.join(output_dir, f"{city_name}_buildings_utm.geojson")
    #build the file path for saving the UTM version
    gdf_utm.to_file(output_path, driver="GeoJSON")
    #save the utm-projected buildings to a GeoJson file
    print(f"[INFO] Saved UTM buildings → {output_path}")
    #inform the user that the file has been successfully saveed

#MAIN EXECUTION!!!
if __name__ == "__main__":
#this block runs only if this script directly executed

    cities = [
        "Berlin", "Darmstadt", "Amsterdam", "Mexico City", "Cairo",
        "Madrid", "Porto", "Lisbon", "Melbourne", "Barcelona", "Brisbane"
    ]

    for city in cities:
    #loop through every city

        city_filename = city.lower().replace(" ", "_")
        #convert the city name into safe, lowercase filename

        input_path = f"data/osm/{city_filename}_buildings.geojson"
        #path to the cleaned building file generated in step2

        gdf = load_buildings(input_path)
        #load the building polygons for the current city
        if gdf is None:
        #if loading fails, skip this city and continue
            continue

        epsg = guess_utm_epsg(gdf)
        #determine the correct UTM zone based on the building extent

        gdf_utm = reproject_to_utm(gdf, epsg)
        #convert the building polygons into the selected UTM projection

        save_utm(gdf_utm, city_filename)
        #save the final utm version of the vuildings as geojson