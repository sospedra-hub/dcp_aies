import numpy as np
import xarray as xr
from pathlib import Path


def load_nan_mask(dir_in,
                  file_in,
                  xarray=True):
    print("====")
    print(f"loading nan mask..")
    for f in sorted(Path(dir_in).glob(f"{file_in}.nc")):
        ds = xr.open_dataset(f)
        if xarray:
            ds = ds.to_array()
        print("done")    
    return ds



def load_verification(dir_in,
                      two_dim=False):
    
    print("====")
    print("loading verification..")
    
    datasets = []
    years = []
    
    for f in sorted(Path(dir_in).glob("*.nc")):
        
        fname_split = f.stem.split("_")
        
        years.append(fname_split[-2])
        
        # ds = xr.open_dataset(f)
        ds = xr.open_mfdataset(f)
  
        var = list(ds.keys())[0]
        
        if two_dim:
            ds = ds.rename_dims({'time': 'month'})
            ds = ds.expand_dims('year', axis=0)
            ds['year'] = np.array([ds.time.dt.year[0]])
            ds = ds.drop_vars('time')
            ds['month'] = np.arange(len(ds['month'])) + 1
            
        datasets.append(ds[var])
        
    da = xr.concat(datasets,
                   dim=datasets[0].dims[0])
    
    print("done")
    
    return da


def load_forecasts(dir_in,
                   ensemble_mean=True,
                   ensemble_list=False):
    
    print("====")
    print("loading forecasts..")
        
    if ensemble_list:
        
        ds = xr.open_mfdataset(str(Path(dir_in,"*.nc")),
                               combine='nested',
                               concat_dim='year').sel(ensembles=ensemble_list)
        if ensemble_mean:
            ds = ds.mean('ensembles') 
        else:
            print(f'Warning: ensemble mean is {ensemble_mean}. Training for ensemble ...')
            
    else:
        
        if ensemble_mean:
            ds = xr.open_mfdataset(str(Path(dir_in,"*.nc")),
                                   combine='nested',
                                   concat_dim='year').mean('ensembles')#.load()
        else:
            ds = xr.open_mfdataset(str(Path(dir_in,"*.nc")),
                                   combine='nested',
                                   concat_dim='year')
            ds = ds.transpose('year',
                              'lead_time',
                              'ensembles',
                              ...)
        
    print("done")
        
    return ds[list(ds.keys())[0]]



