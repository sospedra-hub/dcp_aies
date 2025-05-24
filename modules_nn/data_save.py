import numpy as np
import xarray as xr
from pathlib import Path

def save_nan_mask(lat,
                  lon,
                  nanremover,
                  name='mask',
                  dir_name=None,
                  file_name=None):
    ds = xr.DataArray(
                      np.ones((len(lat),
                               len(lon))),
                      dims=("lat",
                            "lon"),
                      coords={
                               "lat": ("lat", lat),
                               "lon": ("lon", lon),
                              },
                      name=f'{name}'
                      )
    nanremover.to_map(nanremover.sample(ds)).to_netcdf(path=Path(f'{dir_name}',
                                                                 f'{file_name}.nc',
                                                                 mode='w'))
