"""
OSM Data Pipeline
Full Automation

This script orchestrates all steps of the OSM data preperation pipeline:
1-download raw OSM data
2-clean and filter building polygons
3-extract bounding boxes for each city
4-reproject the buildings into the correct UTM CRS

Running this file will automatically execute all steps for all cities
"""

from download_osm import download_osm_buildings
from osm_processing import load_buildings, save_buildings
from bbox_extractor import compute_bbox, save_bbox
from crs_transformer import guess_utm_epsg, reproject_to_utm, save_utm

import os
import geopandas as gpd

CITIES = [
    "Berlin", "Darmstadt", "Amsterdam", "Mexico City", "Cairo",
    "Madrid", "Porto", "Lisbon", "Melbourne", "Barcelona", "Brisbane"
]

#STEP1-download raw osm data
def step1_download_osm():
    print("\n!!Downloading OSM Data!!")
    download_osm_buildings()
    #this function already loops over all cities

#STEP2-clean&filter buildings(polygon only)
def step2_clean_osm():
    print("\n!!Cleaning OSM Data!!")
    for city in CITIES:
        city_file = city.lower().replace(" ", "_")
        input_path = f"data/osm/{city_file}.geojson" #step1 output
        gdf = load_buildings(input_path)

        if gdf is None:
            print(f"[WARNING] Skipping {city} due to missing file")
            continue 

        save_buildings(gdf, city_file)
        #output will be: data/osm/<city>_buildings.geojson

#STEP3-extract bounding boxes
def step3_extract_bbox():
    print("\n!!Extracting Bounding Boxes!!")
    for city in CITIES:
        city_file = city.lower().replace(" ", "_")
        buildings_path = f"data/osm/{city_file}_buildings.geojson" #step1 output
        gdf = load_buildings(buildings_path)

        if gdf is None:
            print(f"[WARNING] Skipping BBOX for {city}")
            continue

        bbox= compute_bbox(gdf)
        save_bbox(bbox, city_file)

#STEP4-reproject buildings to utm
def step4_reproject_utm():
    print("\n!!Reprojecting Buildings to UTM!!")
    for city in CITIES:
        city_file = city.lower().replace(" ", "_")
        buildings_path = f"data/osm/{city_file}_buildings.geojson" #step1 output
        gdf = load_buildings(buildings_path)

        if gdf is None:
            print(f"[WARNING] Skipping UTM step for {city}")
            continue

        epsg= guess_utm_epsg(gdf)
        gdf_utm = reproject_to_utm(gdf, epsg)
        save_utm(gdf_utm, city_file)

#MAIN RUNNER
if __name__ == "__main__":
    print("\n___________________________\n")
    print("!!OSM PIPELINE STARTED!!")
    print("\n___________________________\n")

    step1_download_osm()
    step2_clean_osm()
    step3_extract_bbox()
    step4_reproject_utm()

    print("\n___________________________\n")
    print("!!OSM PIPELINE COMPLETED!!")
    print("\n___________________________\n")


