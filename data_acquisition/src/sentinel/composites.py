# UTILITY MODULS-6

"""
6-Temporal Composites (Median) and RGB Extraction:
This module contains helper functions for producing temporal composites from 
Sentinel-2 data cubes. Since a single acquisition date may contain clouds 
or missing pixels, we combine all images within the selected time range 
(e.g., the entire summer) and compute a 'median composite'. 

The median reducer collapses the time dimension into a single cloud-free 
image by selecting the median pixel value across all dates. (Cloud-covered pixels 
are masked out earlier in the pipeline, so they do not influence the median.) 
This effectively fills gaps that occur on cloudy days and generates a 
clean, representative surface reflectance image.

The module also provides convenience functions for extracting RGB bands  
(B04-B03-B02) and a full-band cube including the NIR band (B08).

"""

def reduce_to_median(cube):
    """
    reduce the time dimension of a sentinel-2 data cube using median reducer
    this func will produce a cloud free stable composite image that represents 
    the entire data range (which is all summer in our project)
    """
    print("[INFO] Reducing time dimension using median...")

    reduced = cube.reduce_dimension(
        dimension="t", #collapse the time dimension
        reducer="median" #use median as the statistical reducer
    )
    print("[INFO] Median composite created.")
    return reduced


def create_rgb(cube):
    """
    extract a rgb composite from the data cube
    sentinel-2 standard rgb=B04,B03,B02
    we wrote this func to prepare a three-band rgb cube that can be visualized 
    or exported
    ps: no stretching is done here
    """
    print("[INFO] Creating RGB composite (B04-B03-B02)...")
    return cube.filter_bands(["B04", "B03", "B02"])#standard sentinel2 rgb order, important!!

def create_full_band_cube(cube):
    """
    extract 4 key bands from the data cube
    """
    return cube.filter_bands(["B04", "B03", "B02", "B08"])
