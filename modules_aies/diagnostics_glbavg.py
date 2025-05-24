'''
Wrapper to produce cascade plots
'''

import numpy as np
import xarray as xr

from modules_aies.plot_cascade import *


def diagnostics_cascade(list_ds,
                        dict_data,
                        units=None,
                        ymin=-1.,
                        ymax=1.,
                        y0=None,
                        xx_step=10,
                        dict_to_plot_data=None,                        
                        valtype='full', 
                        show_samples=False,                     
                        shadesize_left=None,
                        shadesize_right=None,
                        add_fct=False,
                        titles=None,
                        label_ts1='Observed',
                        label_ts2='Uninitialized',
                        label_ts3='Initialized',
                        dir_name=None,
                        file_name=None,
                        save=False):
    
    var = list(dict_data['obs'].data_vars)[0]
 
    a1 = np.array(dict_data['obs'][var])
    
    for ds in list_ds:
    
        xx_max = dict_data[ds].year.values[-1]+2*dict_data[ds].time.values.size // 12

        a2 = None # historical
        a3 = np.array(dict_data[ds][var]) # hindcast    

        # pad array with nan's for size consistency --if needed
        if len(a1[:,0]) > len(a3[:,0]):
            ifct = [1 if add_fct else 0][0]
            a3 = np.pad(a3,
                       ((len(a1[:,0])-len(a3[:,0])+ifct,0),(0,0)),
                       'constant',
                       constant_values=np.nan)     
            
        (ts1,
         ts2,
         ts3) = get_time_series_to_plot_cascade(a1,
                                                a2,
                                                a3,
                                                add_fct=add_fct)
        if a2 is None:
            ts2 = None

        if titles is None:
            title = f"{ds}"
        else:
            title = f"{titles[ds]}"
            
        plot_cascade(ts1,
                     ts2,
                     ts3,
                     label_ts1=label_ts1,
                     label_ts2=label_ts2,
                     label_ts3=label_ts3,
                     title=title,
                     units=units,
                     xx_max=xx_max, 
                     xx_i=y0-1,
                     xx_step=xx_step,
                     ymin=ymin,
                     ymax=ymax,
                     show_samples=show_samples,                     
                     shadesize_left=shadesize_left,
                     shadesize_right=shadesize_right,
                     dir_name=dir_name,
                     file_name=f'{file_name}_{ds}',
                     save=save)            
        