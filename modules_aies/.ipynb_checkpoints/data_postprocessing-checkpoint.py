import numpy as np
import xarray as xr
from xarray import DataArray
from pathlib import Path

import matplotlib.pyplot as plt


def create_train_mask(ds,
                      exclude_idx=0):
    
    xds = ds[list(ds.data_vars)[0]]
    
    mask = np.full((xds.shape[0],
                    xds.shape[1]),
                   False,
                   dtype=bool)
    x = np.arange(0,
                  12*xds.shape[0],
                  12)   
    y = np.arange(1,
                  xds.shape[1] + 1)
    idx_array = x[..., None] + y
    mask[idx_array >= idx_array[-1,
                                exclude_idx + 12]] = True
    return mask


def get_badj(obs,
             hnd,
             fct=None,
             exclude_idx=0,
             dims_to_broadcast=4,
             show_train_mask=False):
    '''
    Function Get Bias Adjusted Forecast:
    '''
    var = list(hnd.data_vars)[0]
    
    train_mask = create_train_mask(hnd,
                                  exclude_idx=exclude_idx)
    
    if dims_to_broadcast == 2:
        train_mask_template = train_mask
    if dims_to_broadcast == 3:
        train_mask_template = train_mask[...,None]
    if dims_to_broadcast == 4:
        train_mask_template = train_mask[...,None,None]
        
    preprocessing_mask = np.broadcast_to(train_mask_template,
                                         hnd[var].shape)
    
    preprocessing_mask = xr.DataArray(preprocessing_mask,
                                      dims=hnd[var].dims,
                                      coords=hnd[var].coords)        
    
    obs_clim = (obs.where(~preprocessing_mask)).mean('year')

    hnd_clim = (hnd.where(~preprocessing_mask)).mean('year')
    
    bias_hnd = hnd_clim - obs_clim # climatological bias
    
    hnd_badj = hnd - bias_hnd # bias corrected (in-sample) 
    
    if fct is not None:
        fct_badj = fct - bias_hnd # bias corrected (out-of-sample)
    else:
        fct_badj = None
        
    if show_train_mask:
        data = preprocessing_mask.sel(lon=0.).sel(lat=slice(1,2))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data,
                  aspect='auto',
                  cmap=plt.cm.gray,
                  interpolation='nearest')
        ax.set_yticks(np.arange(len(data.year.values)))
        ax.set_yticklabels(data.year.values)

    
    return fct_badj, hnd_badj, bias_hnd



def sequential_bias_adjustment(obs,        # dataset (one realization)
                               fct_ens,    # dataset (generally ensemble but can be emean)
                               y0_train=None,
                               y0_test=None,
                               test_years=None,
                               exclude_idx=0,
                               use_ensemble=False,
                               # ripf='i1p1f1',
                               # show_ts=False,
                               show_train_mask=False,
                               # do_metadata=False,
                               # metadata_type='CMIP6',
                               # dir_out=None,
                               # file_out=None,
                               # save_forecasts=False,
                               # save_hindcasts=False,
                               verbose=True):
    '''
     Function Sequential Bias Adjustment: 
     It computes climatology of raw forecast and observations over training period
     at each grid cell as a function of lead time, then compute bias and remove it 
     from forecast over test period
     --training period: (y0_train, test_year-1)
     --test_year      : (y0_test, y1_test)
     Correction over test period is done out-of-sample
     Correction over training period is done in-sample
     ---------     
    '''
    if use_ensemble:
        fct = fct_ens.mean(dim="realization")
    else:
        fct = fct_ens

    var = list(fct.keys())[0]       

    if test_years is None:
        test_years = fct.year.values[list(fct.year.values).index(y0_test):]
    if y0_train is None:
        y0_train = test_years[0] - 1
    if y0_test is None:
        y0_test = test_years[0] 

    ls = []
    
    for y_idx, test_year in enumerate(test_years):
        
        obs_seq = obs.sel(year=slice(y0_train,test_year-1))
        hnd_seq = fct.sel(year=slice(y0_train,test_year-1))
        fct_seq = fct.sel(year=test_year)

        if verbose:
            print(f"bias correction: train period: {y0_train}-{obs_seq.year.values[-1]}, test year: {test_year}")
            
        (fct_badj_seq,                      
         hnd_badj_seq,_) = get_badj(obs_seq, # obs
                                    hnd_seq, # hnd  # in-sample
                                    fct_seq, # fct  # out-of-sample
                                    exclude_idx=exclude_idx,
                                    show_train_mask=show_train_mask) 
        
        ls.append(fct_badj_seq)

    hnd_badj = hnd_badj_seq           # picks in-sample corrections 4-last test year
    fct_badj = xr.concat(ls,          # picks all out-of-sample corrections 4-all test years 
                         dim='year')  
    
    fct_ens_badj = fct_ens - fct + fct_badj # bias corrected forecast (out-of-sample)
    hnd_ens_badj = fct_ens - fct + hnd_badj # bias corrected hindcast (in-sample)

    
    return fct_ens_badj, hnd_ens_badj



def write_monthly_to_annual(ds,
                            time='time',
                            dataset=True):
    if dataset:
        dim_year = False
        if 'year' in ds.dims:
            dim_year = True
            ds = ds.rename({'year' : 'year_tmp'})
        ds.coords['month'] = np.ceil(ds[time] % 12).astype('int') 
        ds.coords['year'] = (ds[time] // 12).astype('int')
        ds_am = ds.groupby('year').mean(dim=time)
        ds_am = ds_am.rename({'year':'time'})
        if dim_year:
            ds_am = ds_am.rename({'year_tmp':'year'})
    if not dataset:
        ds_am = []
        for iyr in np.arange(ds.shape[0] // 12):
            ds_time = ds[iyr*12:(iyr+1)*12-1].mean()
            ds_am.append(ds_time)
    return ds_am


