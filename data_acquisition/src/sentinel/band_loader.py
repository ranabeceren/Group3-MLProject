
# UTILITY MODULS-4

"""
4-Sentinel-2 Band Loader via OpenEO:
This module provides a helper function for loading Sentinel-2 data through the
OpenEO backend. Given a valid GeoJSON Polygon, the function requests the Sentinel-2 L2A 
collection and fetches only the selected spectral bands that we defined in the config.py

Because the backend we are using (CDSE OpenEO) does not support scene-level
cloud filtering, we rely solely on the SCL pixel masks instead of the
traditional cloudCoverage filter. All filtering is done after loading the cube.

PS: 
earlier in the project we mistakenly passed the raw bounding-box dictionary 
(min/max lat/lon values) directly into `load_collection()`. This resulted 
in an empty data cube because OpenEO requires the spatial extent to be expressed 
as a *GeoJSON Polygon*. After debugging this, we introduced the bbox-to-GeoJSON 
conversion function and now ensure that the spatial extent is always in the 
correct format.
"""

from connection import get_connection
#reuse the cached OpenEO connection!
from config import (
    S2_COLLECTION,
    BANDS, #default band list(RGB, NIR, SCL)
    DEFAULT_TEMPORAL_EXTENT,
)
import pprint


def load_sentinel_collection(openeo_bbox, #this should come from bbox_to_geojson_polygon(),
                            #not from raw bbox.json
                             temporal_extent=DEFAULT_TEMPORAL_EXTENT,
                             bands=BANDS):
    """
    Load a Sentinel-2 data cube from OpenEO.
    """

    con = get_connection()
    #get or reuse the OpenEO backend connection
    print("\n[INFO] Loading Sentinel-2 collection...")
    print("[INFO] Spatial extent:")
    pprint.pprint(openeo_bbox)
    print(f"[INFO] Temporal extent: {temporal_extent[0]} → {temporal_extent[1]}")
    print(f"[INFO] Bands: {bands}")
    print("[INFO] Scene cloud filter: DISABLED (not supported by backend)\n")
    #CDSE backend does not support the simple cloud coverage filter, so cloud
    #removal must be handled manually using SCL mask

    cube = con.load_collection(
        S2_COLLECTION,
        spatial_extent=openeo_bbox,
        temporal_extent=list(temporal_extent), #convert the tuple to a list (openeo requirement)
        bands=bands
    )

    print("[INFO] Sentinel-2 collection loaded successfully.")
    return cube
