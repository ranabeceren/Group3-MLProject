# UTILITY MODULS-5

"""
5-Cloud Masking Using the SCL Band:

This module generates a cloud mask for Sentinel-2 imagery based on the SCL 
band. The SCL band classifies every pixel into categories(such as vegetation, 
water, cloud shadow, medium/high cloud probability, thin cirrus, etc.). 
Since the backend we are using (CDSE) does not allow scene-level cloud filtering, 
we must apply pixel-level cloud masking manually.

"""

from openeo.processes import eq, or_
#openeo process-graph operations

def apply_cloud_mask(cube):
    """
    The function `apply_cloud_mask()` identifies all SCL classes that correspond to 
    cloud contamination (values 3, 8, 9, and 10) and constructs a boolean mask where:
    - True  → pixel should be removed (cloudy)
    - False → pixel should be kept (clear)

    The mask is then applied to the spectral bands (B02, B03, B04, B08), returning a 
    cleaned data cube that excludes cloud-covered pixels.
    """

    print("[INFO] Applying SCL-based cloud mask...")

    scl = cube.filter_bands(["SCL"])
    #extract the scl band from the full data cube

    cloud_mask = scl.apply(
        lambda x:
            or_(
                or_(eq(x, 3), eq(x, 8)),
                or_(eq(x, 9), eq(x, 10))
            )
    )
    #build a boolean mask using OpenEO's functional process graph
    # eq(x, value): returns True where pixel equals that SCL class
    # or_(a, b): logical OR, combining multiple conditions
    #the mask is true on cloud-like pixels:
    #- class 3: cloud shadow
    #- class 8: medium cloud probability
    #- class 9: high cloud probability
    #- class 10: thin cirrus

    spectral = cube.filter_bands(["B02", "B03", "B04", "B08"])
    #select the spectral bands we want to keep

    masked = spectral.mask(cloud_mask)
    #apply the cloud mask to remove clouded pixels
    #if mask(true) then drop the pixel
    #if mask(false) then keep the pixel

    print("[INFO] Cloud mask applied.")
    return masked
    #return the 'cloud free' data cube
