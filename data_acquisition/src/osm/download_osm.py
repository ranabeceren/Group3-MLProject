"""
Step1: OSM Building data download

In this script, we implement the first step of our data acquisition pipeline. 
Our goal is to automatically download building footprints from OpenStreetMap 
for all selected cities. 

The script loops over a predefined list of major cities, downloads each city’s 
OSM map data using Pyrosm, extracts only the features tagged as buildings, and 
saves them as GeoJSON files. These GeoJSON files serve as the input for the 
next steps of the pipeline, where we clean the data further and derive the 
geometric area we will use for the satellite image download.

""" 

from pyrosm import get_data, OSM 
#allows automatic download and processing of OSM data

import geopandas as gpd 
#required to process geographic data and it allows saving GeoJSON files

import os
#for operations such as folder creation, file path merging etc.

CITIES = [
   "Berlin", "Darmstadt", "Amsterdam",  "Mexico City", "Cairo",
   "Madrid", "Porto", "Lisbon", "Melbourne", "Barcelona", "Brisbane"
]
#list of cities to download: these cities were chosen because of their OSM and Sentinel2 data quality
#for instance new york's OSM data was huge, and Prague's OSM data was not available
    #{Downloading OSM buildings for Prague...
    #Failed for Prague: The dataset 'prague' is not available.}

OUTPUT_DIR = "data/osm"
#this line sets up the output directory, all building GeoJSON files will be stored inside 'data/osm'

os.makedirs(OUTPUT_DIR, exist_ok=True)
#if the directory does not already exist, then create it!

#MAIN DOWNLOAD FUNCTION!!!
def download_osm_buildings():
    #This function 'download_osm_buildings()' will loop through each city, download its OSM file, 
    # extract the building data and finally gonna save it as GeoJSON
    for city in CITIES:
        print(f"\n Downloading OSM buildings for {city}...")

        try:
            fp = get_data(city)  
          # 'get_data(city)' automatically downloads the .pbf file containing
          # all map information for the selected city

            osm = OSM(fp)
          # convert the downloaded protobuf file into a readable OSM object

            buildings = osm.get_buildings()
          # OSM contains many features that we don't really need for this project
          # so with osm.get_buildings() we are keeping only polygons tagged as buildings

            path = os.path.join(OUTPUT_DIR, f"{city.lower().replace(' ', '_')}.geojson")
          # we are building the output file path, each city will be saved as its own GeoJSON file
          # city names will be in lowercase to ensure consistent file names

            buildings.to_file(path, driver="GeoJSON")
          # save the building data as GeoJSON


            print(f"Saved: {path}")
            print(f"Total buildings: {len(buildings)}")
          # success messages

        except Exception as e:
            print(f"   Failed for {city}: {e}")
          # if a city fails (big file, connection error, corrupted data)
          # the script continues instead of stopping

#EXECUTION
#This block ensures the function runs ONLY if the script is executed directly (not when imported as a module)
if __name__ == "__main__": # when the file is executed directly, __name__ == "__main__"
                           # when the file is imported ; __name__ becomes the module name
    download_osm_buildings()
