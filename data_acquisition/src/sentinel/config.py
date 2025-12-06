# UTILITY MODULS-1

"""
1- Configuration Module:
In this file, we define all global configuration parameters required for the
Sentinel-2 data acquisition part of our pipeline. Instead of hardcoding URLs,
collection names, band selections, and time ranges in different scripts, this
central configuration module keeps everything consistent and easy to modify.

"""

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
#URL of the OpenEO backend we will use for Sentinel-2 requests
#initially, we worked with VITO's backend, but due to slow performance
#and limited credit quotas, we switched to the CDSE backend for faster,
#more stable, and unrestricted processing.

S2_COLLECTION = "SENTINEL2_L2A"
#name of the sentinel-2 collection we want to access

BANDS = ["B02", "B03", "B04", "B08", "SCL"]
#list of sentinel-2 bands we plan to download. 
#B02-B03-B04 are RGB, B08 is NIR, SCL is the Scene Classification Layer
#SCL provides a pixel-wise mask including clouds-very useful for filtering 
#out unusable pixels

DEFAULT_TEMPORAL_EXTENT = ("2023-06-01", "2023-09-30")
#default time range
#we select the summer months because they typically have fewer clouds in most regions
#additionally, using year 2023 ensures that the L2A products are fully processed and 
#reliably available for cities like Cairo, where newer years may have incomplete datasets.

RAW_OUTPUT_DIR = "data/raw/sentinel/"
#directory where all raw sentinel outputs will be stored
