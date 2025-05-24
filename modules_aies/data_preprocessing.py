import numpy as np
import xarray as xr

def align_data_to_common_base(data,
                              nldyr,
                              time_scale=12):
    '''
    rewrites data from initial years to target years
    on the same base period for each lead year
    -- it outputs NaN for the first few years of 
       aligned data without available obs 
    '''
 
    datasets = []
 
    for ildyr in np.arange(nldyr):
       
        ildyr1 = ildyr + 1
           
        inds = np.arange(ildyr*time_scale,
                         (ildyr1)*time_scale)
           
        years_ild = data.year.values - ildyr # common period per lead year
 
        years_ild = years_ild[years_ild >= data.year.values.min()]    
 
        ds = data.sel(time=inds).sel(year=years_ild)
           
        ds['year'] = years_ild + ildyr       # target years
 
        datasets.append(ds)
 
    data_aligned = xr.concat(datasets,
                             dim='time').sortby('time')
   
    return data_aligned



def merge_datasets(ds1,
                   ds2,
                   ds1_time='time',
                   ds2_time='time',
                   ds1_year='year',
                   ds2_year='year',
                   time0=0,
                   time1=11,
                   year0=None,
                   year1=None,
                   year2=None):
    '''
      merges datasets ds1 and ds2 along year dimension
      ds|_(year0,year2) = ds1|_(year0,year1-1) + ds2|_(year1,year2) 
    '''
    
    ds1 = ds1.rename({ ds1_time : 'time',
                       ds1_year : 'year'})
    ds2 = ds2.rename({ ds2_time : 'time',
                       ds2_year : 'year'})
    
    if year0 is None:
        year0 = ds1.year.values[0]
    if year1 is None:
        year1 = ds2.year.values[0]
    if year2 is None:
        year2 = ds2.year.values[-1]

    ds1x = ds1.sel(time=slice(time0,
                              time1)).sel(year=slice(year0,
                                                     year1-1))
    ds2x = ds2.sel(time=slice(time0,
                              time1)).sel(year=slice(year1,
                                                     year2))
    
    return ds1x.merge(ds2x).squeeze()



