# UTILITY MODULS-3

"""
3-Converting Bounding Boxes to GeoJSON Polygons:
Converts a bounding box dictionary into a GeoJSON polygon.

PS:
Earlier in the project, we encountered an issue where we passed the raw bbox.json 
directly into `load_collection()`, assuming the backend would accept the bounding 
box values as-is. However, OpenEO requires the spatial extent to be expressed as 
a valid GeoJSON Polygon. Passing anything else (such as a simple min/max dictionary) 
causes the backend to load an empty data cube without throwing an explicit error.
This utility function was added to prevent that issue by ensuring that every bbox 
is converted into the correct GeoJSON Polygon format before being used in any 
Sentinel-2 download requests.
"""
def bbox_to_geojson_polygon(bbox):
    """
    The input bbox is expected to have the following structure:
    bbox: {
        'min_lon': float,
        'min_lat': float,
        'max_lon': float,
        'max_lat': float
    }

    This function returns a GeoJSON Polygon as output:
    {
      "type": "Polygon",
      "coordinates": [...]
    }
    """
    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]
    #extract numeritic boundaries from the bbox dictionary

    return { #construct and return a GeoJSON polygon that outlines the bbox
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], #bottom-left
            [max_lon, min_lat], #bottom-right
            [max_lon, max_lat], #top-right
            [min_lon, max_lat], #top-left
            [min_lon, min_lat] #close polygon by repeating the first coordinate
        ]]
    }
