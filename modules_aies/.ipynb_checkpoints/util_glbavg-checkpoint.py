import numpy as np
import xarray as xr
from numpy import meshgrid, deg2rad, gradient, sin, cos
from xarray import DataArray

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84 
    
    Source: https://gist.github.com/lgloege/6377b0d418982d2ec1c19d17c251f90e
    
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    # import numpy as np
    # from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon, mask=None):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Source: https://gist.github.com/lgloege/6377b0d418982d2ec1c19d17c251f90e
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """

    if mask is None:
        mask = np.ones((len(lat),
                        len(lon)))  
    
    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * mask * dx    
    
    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_weighted_avg(ds,
                      lat_name='lat',
                      lon_name='lon',
                      mask=None,
                      integral=False):
    """
    Calculate area weighted average over masked region 
    - grid cells outside mask have zero area (don't count to global area)
    - if integral = True, then calculate global total over masked region
    - if mask is None, then masked region is globe
    """
    
    da_area = area_grid(ds[lat_name],
                        ds[lon_name],
                        mask)
    
    total_area = da_area.sum([lat_name,
                              lon_name])
    
    if integral:
        ds_weighted = ds*da_area
    else:
        ds_weighted = (ds*da_area) / total_area
        
    ds_avg = ds_weighted.sum([lat_name,
                              lon_name])
    
    return ds_avg.where(ds_avg[list(ds.data_vars)[0]] != 0) # to keep nan as in mask



def spatial_mask(ds,
                 dataset=True):
    if not dataset:
        ds = ds.to_dataset()
    ds_mask = ds.where(ds.to_array().isnull,1)
    ds_mask = ds_mask.where(ds_mask == 1,0)
    return ds_mask.to_array().squeeze()