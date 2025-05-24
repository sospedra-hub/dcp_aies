import numpy as np
import xarray as xr
from pathlib import Path


def load_nadj_forecasts(dir_in,
                        file_key='*',
                        var='fgco2',
                        verbose=True):
    if verbose:
        print("loading data..")
        
    ds = xr.open_mfdataset(f'{dir_in}/{file_key}.nc',
                           combine='nested',
                           concat_dim='year').squeeze()['nn_adjusted'].to_dataset(name=var)   
    ds = ds.rename({'lead_time':'time'})
    ds['time'] = np.arange(ds.time.size)
    if verbose:
        print("done")
    return ds


def load_nadj_ensemble(dir_in,
                       file_key='*',
                       var='fgco2',
                       EE=10,
                       emean=False,
                       model_test=None,
                       verbose=True):
    
    if verbose:
        print("loading ensemble..")

    datasets = []
    for iE in np.arange(EE):
        iE1 = iE + 1
       
        if model_test is None:
            dir_in_suffix = f'E{iE1}'
        else:
            dir_in_suffix = f'E{iE1}/OutOfSource/{model_test}'
        
        ds = load_nadj_forecasts(f'{dir_in}/{dir_in_suffix}',
                                 file_key,
                                 var,
                                 verbose=False)
        ds = ds.assign_coords(ensembles=iE1)
        ds = ds.expand_dims('ensembles',axis=1)
        datasets.append(ds)
    ds_combined = xr.concat(datasets,
                            dim='ensembles').sortby('ensembles')
    
    if emean:
        ds_combined = ds_combined.mean('ensembles')
    
            
    if verbose:
        print("done")
        
    return ds_combined


def load_xadj_forecasts(dir_in=None,
                        file_key=None,
                        adj_type='Bias_Adjusted',
                        sample_type='Out-of-Sample'):
    
    for f in sorted(Path(f"{dir_in}/{adj_type}/{sample_type}").glob(f"{file_key}*.nc")):
        return xr.open_mfdataset(f)    
    
