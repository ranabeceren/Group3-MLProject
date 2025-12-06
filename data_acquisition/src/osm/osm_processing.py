"""
Step2: Cleaning and filtering OSM building data

In this script, we perform the second step of our data acquisition pipeline. 
After downloading the raw OSM building files in Step 1, our goal here is to clean 
those files and extract only the valid building polygons. The raw OSM downloads 
often contain many different feature types and some geometries that are not useful 
for our project, so we filter the data to keep only Polygon and MultiPolygon 
shapes that actually represent buildings.

"""

import os
import geopandas as gpd

def load_buildings(osm_geojson_path):
    """
    Load building polygons from a GeoJSON file.
    """
    print(f"\n[INFO] Loading: {osm_geojson_path}")

    if not os.path.exists(osm_geojson_path):
    #check if the input file actually exists
        print(f"[ERROR] File not found: {osm_geojson_path}")
        #warn the user
        return None
        #return none so the main loop knows to skip this city

    gdf = gpd.read_file(osm_geojson_path)
    #read the GeoJSON file into a GeoDataFrame (GeoPandas structure that stores geometries)

    # Keep only Polygon & MultiPolygon
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    #OSM files may contain other geometry types so here we filter to keep
    #ONLY actual building shapes

    print(f"[INFO] Found {len(gdf)} building polygons.")
    #print how many building polygons we ended up with
    return gdf
    #return the cleaned GeoDataFraem so it can be saved later


def save_buildings(buildings_gdf, city_name, output_dir="data/osm"):
    """
    Save cleaned building polygons to data/osm/<city>_buildings.geojson
    """
    os.makedirs(output_dir, exist_ok=True)
    #create the output directory if it does not exist
    #ps: "exist_ok=True" avoids errors if it already exists

    output_path = os.path.join(output_dir, f"{city_name}_buildings.geojson")
    #build the final output path for the cleaned GeoJSON file

    buildings_gdf.to_file(output_path, driver="GeoJSON")
    #save the cleaned GeoDataFrame as a GeoJSON file 

    print(f"[INFO] Saved → {output_path}")
    #confirm to the user that the cleaned file saved successfully


if __name__ == "__main__":
#this block only runs if we execute this script directly, 
#not when we import from another file

    cities = [
        "Berlin", "Darmstadt", "Amsterdam", "Mexico City", "Cairo",
        "Madrid", "Porto", "Lisbon", "Melbourne", "Barcelona", "Brisbane"
    ]

    for city in cities:
    #loop through every city in the list
        
        city_filename = city.lower().replace(" ", "_")
        #convert the city name to safe filename format(lowercase+no spaces)

        input_path = f"data/osm/{city_filename}.geojson"
        #construct the expected path for the raw OSM download from step1

        buildings = load_buildings(input_path)
        #load and filter the raw building dataset

        if buildings is not None:
        #only save if the file was successfully loaded
            save_buildings(buildings, city_filename)
            #save the cleaned building polygons as <city>_buildings.geojson
