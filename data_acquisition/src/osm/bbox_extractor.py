"""
Step3: Bounding Box extraction

In this script, we implement the third step of our data acquisition pipeline. 
After cleaning the OSM building data in Step 2, we now load the processed 
<city>_buildings.geojson files and compute a geographic bounding box for each city. 
The bounding box represents the smallest rectangle that fully contains all building 
polygons of that city.

The script loads each city's building polygons, extracts their total bounds, and
saves the result as a simple JSON file (<city>_bbox.json). These bounding box files
are then used in the next stages of Task 1, where we perform CRS transformations
and prepare the area for OpenEO-based Sentinel-2 downloads.

"""


import os
import json
import geopandas as gpd
from osm_processing import load_buildings
#instead of defining tje load_buildings function again, we import it from 
#osm_processing.py to avoid code duplication and keep the pipeline consistent

def compute_bbox(gdf):
    """
    Compute bounding box from GeoDataFrame.
    Returns (min_lon, min_lat, max_lon, max_lat).

    """
    minx, miny, maxx, maxy = gdf.total_bounds
    #extract bounding box from GeoPandas

    bbox = { #organize bounding box values into a JSON-friendly dict
        "min_lon": float(minx),
        "min_lat": float(miny),
        "max_lon": float(maxx),
        "max_lat": float(maxy),
    }
    return bbox #return the bounding box dictionary


def save_bbox(bbox, city_name, output_dir="data/osm"):
    """
    Save bounding box into a json file.
    Output: data/osm/<city>_bbox.json

    """
    os.makedirs(output_dir, exist_ok=True)
    #ensure the output directory exists
    path = os.path.join(output_dir, f"{city_name}_bbox.json")
    #build the path for saving the bbox file

    with open(path, "w", encoding="utf-8") as f:
    #open the file for writing
        json.dump(bbox, f, indent=4)
        #save the bbox dict as a formatted JSON file

    print(f"[INFO] Saved bounding box → {path}")
    #confirm that the file was saved successfully


if __name__ == "__main__":
#execute the following only when running the script directly
    cities = [
        "Berlin", "Darmstadt", "Amsterdam", "Mexico City", "Cairo",
        "Madrid", "Porto", "Lisbon", "Melbourne", "Barcelona", "Brisbane"
    ]

    for city in cities: #loop through all cities
        city_filename = city.lower().replace(" ", "_")
        #convert city name to lowercase format

        buildings_file = f"data/osm/{city_filename}_buildings.geojson"
        #construct expected building file path

        gdf = load_buildings(buildings_file)
        #load the building polygons for the city
        if gdf is None: #if loading failed, skip this city
            continue

        bbox = compute_bbox(gdf)
        #compute bounding box from the city's building polygons
        print(f"[INFO] BBOX for {city}: {bbox}")
        #print the computed bounding box

        save_bbox(bbox, city_filename)
        #save tha bounding box as JSON file
