# UTILITY MODULS-2

"""
2-OpenEO Backend Connection Manager:
This module is responsible for creating and managing the connection to the OpenEO 
backend that we use to download Sentinel-2 data. Instead of creating a new backend 
connection every time we submit a job or request data, this module initializes the 
connection once and then reuses it throughout the entire pipeline. The module also 
handles OIDC authentication, meaning that if the user already has a valid refresh 
token stored locally, the login will happen automatically without requiring 
manual input.

"""

import openeo
from sentinel.config import BACKEND_URL

_cached_connection = None
#cache variable to store the connection object
#initially set to None, meaning no connection has been created yet
#once initialized, the same connection will be reused for all subsequent calls.

def get_connection():
    """
    Establish or return an existing OpenEO backend connection.
    
    This function follows this approach:
    - On the first call, it connects to the backend and performs OIDC authentication.
    - On subsequent calls, it reuses the same connection stored in _cached_connection.
    
    If the user already has an OIDC refresh token stored locally, the login process
    will be automatic.
    """
    global _cached_connection
    #do not use a local variable, use the module-level one

    if _cached_connection is None:
    #if the connection has not been created yet, initialize it
        con = openeo.connect(BACKEND_URL)
        #connect to the backend defined in the config.py
        con.authenticate_oidc()     
        #automatically handles login via OIDC tokens
        _cached_connection = con
        #store the connection for reuse
    return _cached_connection
    #return the cached connection object
    #always the same object after first call
