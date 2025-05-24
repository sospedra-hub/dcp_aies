'''
Wrapper to produce GCB plots
'''

import numpy as np
import xarray as xr

from modules_aies.util_glbavg import *
from modules_aies.plot_maps   import plot_single_map_wmo

def diagnostics_maps_gcb(list_data,
                         dict_data,
                         freq='ann',
                         mask=None,
                         time_to_show=0,
                         year_to_show=None,
                         cmap='RdYlBu_r',
                         vmin=None,
                         vmax=None,
                         vals=None,
                         cbar_label='',
                         cbar_extend='both',
                         cbar_all=False,
                         imshow=False,
                         titles=None,
                         dir_name=None,
                         file_name=None,
                         show=False,
                         save=False):

    count = 0
    
    cbar = cbar_all
    
    for ind_ds, ds in enumerate(list_data):
        
        for ind_src, source in enumerate(np.flip(list(dict_data[ds][freq].keys()))): 

            count += 1
            
            if bool(dict_data[ds][freq][source]):

                var = list(dict_data[ds][freq][source].data_vars)[0]
                
                if year_to_show is None:
                    year_to_show = dict_data[ds][freq][source].year.values[-1]
                    
                ds_data = dict_data[ds][freq][source].sel(time=time_to_show).sel(year=year_to_show)
                
                ds_data_glbavg = np.round(area_weighted_avg(ds_data,
                                                            mask=mask)[var].values,2)

                if count == len(list_data)*(len(dict_data[ds][freq].keys())):
                    cbar = True

                
                if titles is None:
                    title = f'{var}, {year_to_show}, Year {time_to_show+1}, {ds.replace("_", " ")}, {ds_data_glbavg}'
                else:
                    title = titles[ds]
                
                plot_single_map_wmo(ds_data[var], # as in Fig 6 of GCB2022/2023 
                                    title=title,
                                    cbar=cbar,
                                    cbar_label=cbar_label,
                                    cmap=cmap,   
                                    central_longitude=0,
                                    vals=vals,
                                    vmin=vmin,
                                    vmax=vmax,
                                    vals_signfig=4,              
                                    ticks_rotation=45,
                                    cbar_extend=cbar_extend,
                                    imshow=imshow,
                                    fig_dir=dir_name,
                                    fig_name=f'{file_name}_{ds}_gcb_yr-{year_to_show}',
                                    show=show,
                                    save=save)                            

                
