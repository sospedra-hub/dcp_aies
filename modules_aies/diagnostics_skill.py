'''
Wrapper to produce skill plots
'''

import numpy as np
import xarray as xr

from modules_aies.plot_lines   import plot_ts
from modules_aies.plot_maps    import plot_maps_wmo
from modules_aies.util_unitchg import UnitChange
from modules_aies.util_glbavg  import *


def diagnostics_skill_ts(list_ds,
                         dict_data,
                         error_bars=False,
                         freq='ann',
                         season='Annual',
                         xlabel='Forecast year',
                         title=None,
                         ncol_labels=1,
                         bbox=(.88,.5,.5,.5),
                         dict_plt_dat=None,
                         dir_name=None,
                         file_name=None,
                         show=False,
                         save=False):
    
    for meas in dict_data.keys():
        
        
        show_leg = False
        if meas == 'rmse' or meas == 'mse':
            show_leg = True
        
        datalist   = []
        datalabel  = []
        datacolor  = []
        datamarker = []
        dataset    = []
            
        for ds in list_ds:
            
            for model in dict_data[meas][ds][freq].keys():  
                
                var = list(dict_data[meas][ds][freq][model].data_vars)[0]
                
                ts = dict_data[meas][ds][freq][model][var]
                
                if freq == 'sea':
                    ts = ts.sel(season=season)
                    
                if meas == 'mse': # give as square root of global mse
                    ts = UnitChange(ts).toPgCyr()**(1/2)                 
                
                dataset.append(ts)
                datalist.append(f'{model}_{ds}')
                datalabel.append(dict_plt_dat[model][ds]['label'])                    
                datacolor.append(dict_plt_dat[model][ds]['color'])
                datamarker.append(dict_plt_dat[model][ds]['marker'])
                
        data = xr.concat(dataset,
                         dim='legend')

        plot_ts(datalist,
                data,
                error_bars=error_bars,
                title=dict_plt_dat[meas]['title'],
                time_dim='time',
                label_list=datalabel,
                color_list=datacolor,
                marker_list=['o']*len(datalist),                
                xlabel=xlabel,
                ylabel=dict_plt_dat[meas]['ylabel'],
                xmin=0,
                xstep=1,
                xticks_labels=data['time'].values + 1,
                ymin=dict_plt_dat[meas]['ymin'],
                ymax=dict_plt_dat[meas]['ymax'],
                ystep=dict_plt_dat[meas]['ystep'],
                hline=True,                
                ncol_labels=ncol_labels,
                show_leg=show_leg,
                bbox=dict_plt_dat[meas]['bbox'],
                dir_name=dir_name,
                file_name=f'{file_name}_{meas}_{season}',
                show=show,
                save=save)    


def diagnostics_skill_maps(list_ds,
                           dict_data=None,
                           dict_to_plot=None,
                           time_to_show=[0,1,2],
                           freq='ann',
                           season='Annual',
                           mask=None,
                           y0_clim=None,
                           y1_clim=None,
                           titles=None,
                           dir_name=None,
                           file_name=None,
                           show=False,
                           save=False):
                               
    
    for imeas,meas in enumerate(dict_data.keys()): 
        
        cbar = False
    
        for ind_ds, ds in enumerate(list_ds):
        
            for ind_mdl, model in enumerate(dict_data[meas][ds][freq].keys()):  
        
                var = list(dict_data[meas][ds][freq][model].data_vars)[0]
  
                data = []
                title = []
    
                for ii in time_to_show:
                    
                    if season == 'Annual':
                        ds_data = dict_data[meas][ds][freq][model].sel(time=int(ii))        
                    else:
                        ds_data = dict_data[meas][ds][freq][model].sel(time=int(ii)).sel(season=season)        
                    
                    ds_data_glbavg = np.round(area_weighted_avg(ds_data,mask=mask)[var].values,2)

                    data.append(ds_data)

                    if titles is None:
                        title.append(f'{var}, Year {int(ii)+1}, {ds.replace("_", " ")}, {ds_data_glbavg}')
                    else:
                        title.append(dict_to_plot[meas]['title'][ds])

                        
                ds_combined = xr.concat(data,
                                        dim='panels')
    
                if ind_ds + ind_mdl == (len(list_ds) - 1) + (len(dict_data[meas][ds][freq].keys()) - 1):
                    cbar = True
    
                plot_maps_wmo(ds_combined[var],
                          central_longitude=0, #
                          gridlines=False,
                          cmap=dict_to_plot[meas]['cmap'],
                          vmin=dict_to_plot[meas]['ymin'],
                          vmax=dict_to_plot[meas]['ymax'],
                          nvals=dict_to_plot[meas]['nvals'],
                          vals=None,
                          cbar=cbar,
                          cbar_label=dict_to_plot[meas]['cbar_label'],
                          cbar_extend=dict_to_plot[meas]['cbar_extend'],
                          ncols=len(ds_combined.panels),
                          titles=title,
                          figsize=(12,3),
                          fig_dir=dir_name,
                          fig_name=f'{file_name}_{meas}_{model.replace(".", "-")}_{season}_{ds}',
                          show=show,
                          save=save)