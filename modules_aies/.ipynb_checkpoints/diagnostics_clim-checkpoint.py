'''
Wrapper to produce anomaly plots
'''

import numpy as np
import xarray as xr

from modules_aies.util_glbavg import *
from modules_aies.plot_maps   import plot_maps_wmo


def diagnostics_climatology_maps(list_ds,
                                 dict_data=None,
                                 dict_plt=None,
                                 freq='ann',
                                 season='Annual',
                                 mask=None,
                                 time_to_show=[0,1,2],
                                 years_period=None,
                                 y0_clim=None,
                                 y1_clim=None,
                                 units=None,
                                 cmap='RdBu_r',
                                 vmin=-1.5,
                                 vmax=1.5,
                                 vals=None,
                                 nvals=12,
                                 titles=None,
                                 show_bias=True,
                                 dir_name=None,
                                 file_name=None,
                                 show=False,
                                 save=False):

    verification = list(dict_data['obs'][freq].keys())[0]
    
    for ind_ds, ds in enumerate(list_ds):
        
        for ind_mdl, model in enumerate(dict_data[ds][freq].keys()): 

            map_type = 'clim'
            cbar     = False
            
            var = list(dict_data[ds][freq][model].data_vars)[0]
            
            data = []
            title = []

            for ii in time_to_show:

                if season == 'Annual':
                    ds_data_ref = dict_data['obs'][freq][verification].sel(time=int(ii))        
                    ds_data     = dict_data[ds][freq][model].sel(time=int(ii))
                    if model != verification and show_bias == True:
                        map_type = 'bias'
                        ds_data = ds_data - ds_data_ref        

                else: # seasons (not in paper)
                    ds_data_ref = dict_data['obs'][freq][verification].sel(time=int(ii)).sel(season=season)    
                    ds_data = dict_data[ds][freq][model].sel(time=int(ii)).sel(season=season)
                    if model != verification and show_bias == True:
                        map_type = 'bias'
                        ds_data = ds_data - ds_data_ref        
                        
                ds_data_glbavg = np.round(area_weighted_avg(ds_data,
                                                            mask=mask)[var].values,2)

                data.append(ds_data)

                if titles is not None:
                    title.append(f"{dict_plt[map_type]['title'][ds]}")
                else:
                    title.append(f'{var}, Year {int(ii)+1}, {ds.replace("_", " ")}, {ds_data_glbavg}')

            ds_combined = xr.concat(data,
                                    dim='panels')

            if ind_ds + ind_mdl == (len(list_ds) - 1) + (len(dict_data[ds][freq].keys()) - 1) or model == verification:
                cbar = True

            plot_maps_wmo(ds_combined[var],
                      central_longitude=0, #
                      gridlines=False,
                      cmap=dict_plt[map_type]['cmap'],
                      vmin=dict_plt[map_type]['ymin'],
                      vmax=dict_plt[map_type]['ymax'],
                      nvals=nvals,
                      vals=dict_plt[map_type]['vals'],
                      cbar=cbar,
                      cbar_label=f"{dict_plt[map_type]['cbar_label']} ({dict_plt[map_type]['units']})",
                      ncols=len(ds_combined.panels),
                      titles=title,
                      figsize=(12,3),
                      fig_dir=dir_name,
                      fig_name=f'{file_name}_{model.replace(".", "-")}_{season}_{ds}',
                      show=show,
                      save=save)