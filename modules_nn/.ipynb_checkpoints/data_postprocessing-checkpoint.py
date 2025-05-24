import numpy as np
import xarray as xr
from xarray import DataArray
from pathlib import Path

import matplotlib.pyplot as plt

from modules.util_glbavg import *
from modules.plot_lines import quick_ts
from modules.info_meta import *



def save_data_to_file(fct_ens,
                      ripf='i1p1f1',
                      sample_type='Out-of-Sample',                      
                      use_ensemble=True,
                      do_metadata=False,
                      metadata_type='CMIP5',
                      comment=None,                       
                      dir_out=None,
                      file_out=None,
                      verbose=True):
    '''
    Function Save Data to File:
    Copy metadata if called and (if save = True) it saves data to NetCDF file
    -----
     input:
      fct_ens(year,time,realization,lat,lon) -- ensemble forecast (xr dataset)
      hnd_ens(year,time,lat,lon)             -- ensemble hindcast (xr dataset)
     dimensions:
      year        -- initial years
      time        -- lead time in months
      realization -- ensemble members
      lat         -- latitude
      lon         -- longitude
    -----
    '''
    var = list(fct_ens.data_vars)[0]
    
    years_arr    = fct_ens['year'].values 
    times_arr    = fct_ens['time'].values
    
    if do_metadata: # copy metatada --if called
        
        for key in list(fct_ens.coords):
            fct_ens[key].attrs = {}    # clean coord attrs
            if key in meta_module().coord_dict.keys():  
                fct_ens[key].attrs = meta_module().coord_dict[key] # add coord attrs
        fct_ens[var].attrs = meta_module().var_dict[var]  # add var attrs
        for key in meta_module().var_dict[var].keys():    # remove attrs of the form *attr
            if key[0] == "*":
                del fct_ens[var].attrs[key]
    
    if verbose:
        print(f"copy {sample_type} corrections over period: {years_arr[0]}-{years_arr[-1]}")
        
    if use_ensemble:
        ensemble_arr = fct_ens['realization'].values    
        for y_idx, y_dx in enumerate(years_arr):
            dir_copy = f'{dir_out}/{sample_type}/{y_dx}'
            Path(f'{dir_copy}').mkdir(parents=True,exist_ok=True)
            # fct_ens_iyr = fct_ens.sel(year=y_dx)
            fct_ens_iyr = fct_ens.sel(year=y_dx).copy() # perhaps implement this..
            dates = xr.cftime_range(f"1850",                             # reference to 1850
                                    f"{y_dx+times_arr.size//12}",
                                    freq="M",
                                    calendar="365_day")
            fct_ens_iyr['time'] = dates[np.where(dates[:].year >= y_dx)] # choose relevant period (but reference is lost!)
                                                                         # how to account for the reference?
                                                                         # make the object dates an arr, then read back to object?
            fct_ens_iyr['time'] = fct_ens_iyr['time'].assign_attrs(meta_module().coord_dict['time'])
            for iens, ens in enumerate(ensemble_arr):
                fct_iens_iyr = fct_ens_iyr.sel(realization=ens)              
                if do_metadata:
                    metadata = meta_module().global_attributes(
                                                                   var=var,
                                                                   year=y_dx,
                                                                   realization=ens,
                                                                   comment=comment                                                                   
                                                                  )
                    fct_iens_iyr.attrs = metadata[metadata_type]
                file_out_iens_iyr = f'{file_out}_{y_dx}0101_r{ens}{ripf}'                
                fct_iens_iyr.to_netcdf(f"{dir_copy}/{file_out_iens_iyr}.nc",
                                       encoding={'time': {'dtype': 'i4'}})  #  if server supports int32 only
    else:
        dir_copy = f'{dir_out}/{sample_type}'
        Path(f'{dir_copy}').mkdir(parents=True,exist_ok=True)        
        fct_ens.to_netcdf(f"{dir_copy}/{file_out}_{years_arr[0]}-{years_arr[-1]}.nc",
                          encoding={'time': {'dtype': 'i4'}})  #  if server supports int32 only
        
        

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


class Detrender:
    '''
     class Detrender: 
     functions description:
     fit --
     transform -- 
     inverse_transform --
     get_trend --
     get_trend_coeffs --
     _make_timesteps --
     _compute_trend --
    '''
    
    def __init__(self, trend_dim='year', deg=1, remove_intercept=True) -> None:
        self.trend_dim = trend_dim
        self.deg = deg
        self.slope = None
        self.intercept = None
        self.trend_axis = None
        self.remove_intercept = remove_intercept

    def fit(self, data, mask=None):
        data[self.trend_dim] = np.arange(len(data[self.trend_dim]))
        if mask is not None:
            trend_coefs = data.where(~mask).polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)
        else:
            trend_coefs = data.polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)
        slope = trend_coefs['polyfit_coefficients'][0].to_numpy()
        intercept = trend_coefs['polyfit_coefficients'][1].to_numpy()
        self.trend_axis = int(np.where(np.array(data.dims) == self.trend_dim)[0])
        self.slope = np.expand_dims(slope, axis=self.trend_axis)
        self.intercept = np.expand_dims(intercept, axis=self.trend_axis)
        return self

    def transform(self, data, start_timestep=0, remove_intercept=None):
        if remove_intercept is None:
            remove_intercept = self.remove_intercept        
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if data.shape[1] > 12 and self.slope.shape[1] <= 12:
            lead_years = int(data.shape[1] / 12)
            trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=remove_intercept) for i in range(lead_years)], axis=1)        
        else:
            trend = self._compute_trend(timesteps, with_intercept=remove_intercept)
        data_detrended = data - trend
        return data_detrended

    def inverse_transform(self, data, start_timestep=0, add_intercept=None):
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if add_intercept is None:
            add_intercept = self.remove_intercept
        if data.shape[1] > 12 and self.slope.shape[1] <= 12:
            lead_years = int(data.shape[1] / 12)
            trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=add_intercept) for i in range(lead_years)], axis=1)
        else:
            trend = self._compute_trend(timesteps, with_intercept=add_intercept)
        data_trended = data + trend
        return data_trended

    def get_trend(self, sequence_length, start_timestep=0, with_intercept=True):
        timesteps = self._make_timesteps(sequence_length, self.slope.ndim, start_timestep=start_timestep)
        trend = self._compute_trend(timesteps, with_intercept=with_intercept)
        return trend
    
    def get_trend_coeffs(self):
        return self.slope, self.intercept

    def _make_timesteps(self, sequence_length, ndims, start_timestep=0):
        timesteps = np.expand_dims(np.arange(sequence_length) + start_timestep, axis=[i for i in range(ndims) if i != self.trend_axis])
        return timesteps

    def _compute_trend(self, timesteps, with_intercept=True):
        if with_intercept:
            trend = timesteps * self.slope + self.intercept
        else:
            trend = timesteps * self.slope
        return trend
    

    
def get_badj(obs,
             hnd,
             fct=None,
             dims_to_broadcast=4,
             show_train_mask=False):
    '''
    Function Get Bias Adjusted Forecast:
    '''
    var = list(hnd.data_vars)[0]
    
    train_mask = create_train_mask(hnd)
    
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
                               use_ensemble=True,
                               ripf='i1p1f1',
                               show_ts=False,
                               show_train_mask=False,
                               do_metadata=False,
                               metadata_type='CMIP6',
                               dir_out=None,
                               file_out=None,
                               save_forecasts=False,
                               save_hindcasts=False,
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
     If called, it writes metadata to forecast/hindcast data
     If called, it copies adjusted forecast and/or hindcast data 
       --forecast: 4-each test year
       --hindcast: 4-each training year prior the last test year allowed
     ---------     
     input:
      fct_ens(year,time,realization,lat,lon) -- ensemble forecast to be corrected (xr dataset)
      obs(year,time,lat,lon)                 -- reference dataset to correction (xr dataset)
     dimensions:
      year        -- initial years
      time        -- lead time in months
      realization -- ensemble members
      lat         -- latitude
      lon         -- longitude
     output:
      fct_ens_badj(year,time,realization,lat,lon) -- fct_ens bias corrected relative to obs (xr dataset)
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
                                    show_train_mask=show_train_mask) 
        
        if show_ts: # show plots for diagnostics --if called
            
            quick_ts(['obs', 'hnd', 'hnd_badj'],
                     {
                      'obs' : area_weighted_avg(obs_seq.sel(time=0))[var],
                      'hnd' : area_weighted_avg(hnd_seq.sel(time=0))[var],
                      'hnd_badj' : area_weighted_avg(hnd_badj_seq.sel(time=0))[var],
                     },
                     linestyle_dict = {
                          'obs' : '-',
                          'hnd' : '-',
                          'hnd_badj' : '-',
                                      },
                     color_dict = {
                          'obs' : 'k',
                          'hnd' : 'b',
                          'hnd_badj' : 'r',
                     },
                     title=f'Test Year ({test_year})')
        
        
        ls.append(fct_badj_seq)

    hnd_badj = hnd_badj_seq           # picks in-sample corrections 4-last test year
    fct_badj = xr.concat(ls,          # picks all out-of-sample corrections 4-all test years 
                         dim='year')  
    
    fct_ens_badj = fct_ens - fct + fct_badj # bias corrected forecast (out-of-sample)
    hnd_ens_badj = fct_ens - fct + hnd_badj # bias corrected hindcast (in-sample)

    if save_forecasts:
        save_data_to_file(fct_ens_badj.copy(),
                          ripf=ripf,
                          sample_type='Out-of-Sample',                      
                          use_ensemble=use_ensemble,
                          do_metadata=do_metadata,
                          metadata_type=metadata_type,
                          comment='bias corrected monthly values',                       
                          dir_out=dir_out,
                          file_out=file_out,
                          verbose=verbose)    
    
    if save_hindcasts:
        save_data_to_file(hnd_ens_badj.copy(),
                          ripf=ripf,
                          sample_type='In-Sample',                      
                          use_ensemble=use_ensemble,
                          do_metadata=do_metadata,
                          metadata_type=metadata_type,
                          comment='bias corrected monthly values',                       
                          dir_out=dir_out,
                          file_out=file_out,
                          verbose=verbose)    
    
    return fct_ens_badj, hnd_ens_badj


def get_tadj_by_scaling_slopes(obs,
                               hnd,
                               fct=None,
                               dims_to_broadcast=4,
                               show_train_mask=False):
    '''
    Function Get Trend Adjusted Forecast:
    Method: ...
    '''
    
    var = list(hnd.data_vars)[0]
    
    obs_glbavg = area_weighted_avg(obs)
    hnd_glbavg = area_weighted_avg(hnd)
    
    train_mask        = create_train_mask(hnd)
    train_mask_glbavg = create_train_mask(hnd_glbavg)
    
    if dims_to_broadcast == 3:
        train_mask_template = train_mask[...,None]
    if dims_to_broadcast == 4:
        train_mask_template = train_mask[...,None,None]    
            
    preprocessing_mask = np.broadcast_to(train_mask_template,
                                         hnd[var].shape)
    preprocessing_mask = xr.DataArray(preprocessing_mask,
                                      dims=hnd[var].dims,
                                      coords=hnd[var].coords)
    
    preprocessing_mask_glbavg = np.broadcast_to(train_mask_glbavg,
                                                hnd_glbavg[var].shape)
    preprocessing_mask_glbavg = xr.DataArray(preprocessing_mask_glbavg,
                                             dims=hnd_glbavg[var].dims,
                                             coords=hnd_glbavg[var].coords)
    
    # global parameters: obs and fct slope for the global mean      
    obs_glbavg_slope, _ = Detrender(trend_dim='year',
                                    deg=1).fit(obs_glbavg[var].where(~preprocessing_mask_glbavg)).get_trend_coeffs()
    hnd_glbavg_slope, _ = Detrender(trend_dim='year',
                                    deg=1).fit(hnd_glbavg[var].where(~preprocessing_mask_glbavg)).get_trend_coeffs()
            
    # scale factor alpha = obs_glbavg_slope / hnd_glbavg_slope (for slope adjustment)
    alpha_slope = obs_glbavg_slope / hnd_glbavg_slope
      
    # local trends (with intercepts!)
    trnd = Detrender(trend_dim='year',
                     deg=1).fit(hnd[var].where(~preprocessing_mask)).get_trend(hnd[var].year.size+1, 
                                                                               with_intercept=True) 
    
    # local climatologies (broadcasted)    
    obs_clim = np.broadcast_to((obs.where(~preprocessing_mask)).mean('year').to_array(),
                              trnd.shape)
    hnd_clim = np.broadcast_to((hnd.where(~preprocessing_mask)).mean('year').to_array(),
                              trnd.shape)
    
    # broadcast alpha to trend shape
    alpha = np.broadcast_to(alpha_slope[...,
                                        None,
                                        None],
                            trnd.shape)
    
    # get residuals from local trends (with intercepts!)
    if fct is not None: # includes out-of-sample correction for fct
        delta_zz = xr.concat((hnd,fct),dim='year') - trnd  
    else:               # in-sample correction only
        delta_zz = hnd - trnd                            
    
    # superimpose residuals to corrected trends (with corrected intercepts!)
    zz = alpha*(trnd - hnd_clim) + obs_clim + delta_zz  

    # select hindcast and forecast from concatenated zz        
    hnd_tadj = zz.sel(year=slice(hnd['year'].values[0],
                                 hnd['year'].values[-1]))
    if fct is not None:
        fct_tadj = zz.sel(year=hnd['year'].values[-1]+1)
    else:
        fct_tadj = None
        
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
    
    return fct_tadj, hnd_tadj


def get_tadj_by_replacing_slopes(obs,
                                 hnd,
                                 fct=None,
                                 dims_to_broadcast=4,
                                 show_train_mask=False):
    '''
    Function Get Trend Adjusted Forecast:
    Method: ...
    '''
    var = list(hnd.data_vars)[0]
    
    train_mask = create_train_mask(hnd)

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
                                      coords=hnd[var].coords )
        
    hnd_detrender = Detrender(trend_dim='year',
                              deg=1).fit(hnd[var].where(~preprocessing_mask))
    obs_detrender = Detrender(trend_dim='year',
                              deg=1).fit(obs[var].where(~preprocessing_mask))
                                
    # get detrended hindcasts
    hnd_detrend = hnd_detrender.transform(hnd[var],
                                          remove_intercept=True)

    # adds obs trend to detrended hindcasts -- in-sample
    hnd_tadj = obs_detrender.inverse_transform(hnd_detrend,
                                               add_intercept=True).to_dataset() 
    
    if fct is not None:
        # get detrended hindcasts/forecasts combined
        fct_detrend = hnd_detrender.transform(xr.concat((hnd,fct),
                                                        dim='year')[var],
                                              remove_intercept=True)
        # adds obs trend to detrended forecasts -- out-of-sample        
        fct_tadj = obs_detrender.inverse_transform(fct_detrend,
                                        add_intercept=True).to_dataset().sel(year=fct.year.values) 
    else:
        fct_tadj = None
        
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
        
    
    return fct_tadj, hnd_tadj



def sequential_trend_adjustment(obs,        # dataset (one realization)
                                fct_ens,    # dataset (generally ensemble but can be emean)
                                do_glbavg_based_trend=False,
                                y0_train=None,
                                y0_test=None,
                                test_years=None,
                                exclude_idx=0,
                                use_ensemble=True,
                                ripf='i1p1f1',
                                show_ts=False,
                                show_train_mask=False,
                                do_metadata=False,
                                metadata_type='CMIP5',
                                dir_out=None,
                                file_out=None,
                                save_forecasts=False,
                                save_hindcasts=False,
                                verbose=True):
    '''
     Function Sequential Trend Adjustment: 
     It computes climatology of raw forecast and observations over training period
     at each grid cell as a function of lead time, then compute bias and remove it 
     from forecast over test period
     It computes climatological trends of raw forecast and observations over training period
     at each grid cell as a function of lead time, then either: (1) compute trend bias and remove 
     it from forecast over test period, or (2) compute global mean trend for the forecast and
     observations to scale local trends with the ratio of the two such that the global mean of the
     corrected forecast follows the observed trend
     if do_glbavg_based_trend:
        then case (2) 
     else:
        then case (1)
     --training period: (y0_train, test_year-1)
     --test_year      : (y0_test, y1_test)
     Correction over test period is done out-of-sample
     Correction over training period is done in-sample
     If called, it writes metadata to forecast/hindcast data
     If called, it copies adjusted forecast and/or hindcast data 
       --forecast: 4-each test year
       --hindcast: 4-each training year prior the last test year allowed
     ---------     
     input:
      fct_ens(year,time,realization,lat,lon) -- ensemble forecast to be corrected (xr dataset)
      obs(year,time,lat,lon)                 -- reference dataset to correction (xr dataset)
     dimensions:
      year        -- initial years
      time        -- lead time in months
      realization -- ensemble members
      lat         -- latitude
      lon         -- longitude
     output:
      fct_ens_tadj(year,time,realization,lat,lon) --  fct_ens trend corrected relative to obs (xr dataset)
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
            print(f"trend correction: train period: {y0_train}-{obs_seq.year.values[-1]}, test year: {test_year}")

        
        if do_glbavg_based_trend:
            (fct_tadj_seq,                      
             hnd_tadj_seq) = get_tadj_by_scaling_slopes(obs_seq, # obs
                                                        hnd_seq, # hnd
                                                        fct_seq, # fct
                                                        show_train_mask=show_train_mask) # fct
        else:
            (fct_tadj_seq,                      
             hnd_tadj_seq) = get_tadj_by_replacing_slopes(obs_seq, # obs
                                                          hnd_seq, # hnd
                                                          fct_seq, # fct
                                                          show_train_mask=show_train_mask) # fct

        if show_ts: # 
            obs_seq_glbavg      = area_weighted_avg(obs_seq)
            hnd_seq_glbavg      = area_weighted_avg(hnd_seq)
            hnd_tadj_seq_glbavg = area_weighted_avg(hnd_tadj_seq)
            
            obs_seq_glbavg_trnd = Detrender(trend_dim='year',
                                            deg=1).fit(obs_seq_glbavg[var].sel(time=0)).get_trend(obs_seq_glbavg[var].year.size)
            hnd_seq_glbavg_trnd = Detrender(trend_dim='year',
                                            deg=1).fit(hnd_seq_glbavg[var].sel(time=0)).get_trend(hnd_seq_glbavg[var].year.size)
            hnd_tadj_seq_glbavg_trnd = Detrender(trend_dim='year',
                                                 deg=1).fit(hnd_tadj_seq_glbavg[var].sel(time=0)).get_trend(hnd_tadj_seq_glbavg[var].year.size)
            quick_ts(['obs', 'hnd', 'hnd_tadj',
                      'obs_trnd', 'hnd_trnd', 'hnd_tadj_trnd'],
                     {
                      'obs' : obs_seq_glbavg.sel(time=0)[var],
                      'hnd' : hnd_seq_glbavg.sel(time=0)[var],
                      'hnd_tadj' : hnd_tadj_seq_glbavg.sel(time=0)[var],
                      'obs_trnd' : obs_seq_glbavg_trnd,
                      'hnd_trnd' : hnd_seq_glbavg_trnd,
                      'hnd_tadj_trnd' : hnd_tadj_seq_glbavg_trnd,
                     },
                     linestyle_dict = {
                          'obs' : '-',
                          'hnd' : '-',
                          'hnd_tadj' : '-',
                          'obs_trnd' : '--',
                          'hnd_trnd' : '--',
                          'hnd_tadj_trnd' : '--',
                                      },
                     color_dict = {
                          'obs' : 'k',
                          'hnd' : 'b',
                          'hnd_tadj' : 'r',
                          'obs_trnd' : 'k',
                          'hnd_trnd' : 'b',
                          'hnd_tadj_trnd' : 'r',
                     },
                     title=f'Test Year ({test_year})')
            
        ls.append(fct_tadj_seq)

    hnd_tadj = hnd_tadj_seq           # picks last in-sample fct 
    fct_tadj = xr.concat(ls,          # picks all out-of-sample fct 
                         dim='year')  
    
    fct_ens_tadj = fct_ens - fct + fct_tadj 
    hnd_ens_tadj = fct_ens - fct + hnd_tadj 
    
    if save_forecasts:
        save_data_to_file(fct_ens_tadj,#.copy(),
                          ripf=ripf,
                          sample_type='Out-of-Sample',                      
                          use_ensemble=use_ensemble,
                          do_metadata=do_metadata,
                          metadata_type=metadata_type,
                          comment='trend corrected monthly values',                       
                          dir_out=dir_out,
                          file_out=file_out,
                          verbose=verbose)    
    
    if save_hindcasts:
        save_data_to_file(hnd_ens_tadj,#.copy(),
                          ripf=ripf,
                          sample_type='In-Sample',                      
                          use_ensemble=use_ensemble,
                          do_metadata=do_metadata,
                          metadata_type=metadata_type,
                          comment='trend corrected monthly values',                       
                          dir_out=dir_out,
                          file_out=file_out,
                          verbose=verbose)    
    
    return fct_ens_tadj, hnd_ens_tadj

