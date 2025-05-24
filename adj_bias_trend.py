#!/usr/bin/env python
# coding: utf-8

import os
import yaml 
import gc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def badj_and_tadj(dict_yaml,
                  verbose=True):
    """ 
    Reads in decadal forecasts and observations to produce bias and/or 
    trend corrected forecasts. It adjusts bias and trend at each grid cell,
    with two options to adjust trends as indicated in comment #4.

    STATUS  : Complete 
    COMMENTS: (1) reads in relevant information/switches from yaml file
              (2) reads data from models as indicated in the yaml file, e.g., CanESM5.0p2-ems
              (3) corrections done sequentially over test years,
                  increasing one year over training period for every consecutive test year
              (4) trend adjustment at grid cells can be done using two different methods: 
                  (i) scaling local slopes using adjusted globally averaged trend, or 
                  (ii) replacing raw forecast with observed slopes locally
    """
    
    var                   = dict_yaml['variable'][0]
    model                 = dict_yaml['models_to_train'][0]
    verification          = dict_yaml['verification'][0]
    nyrs_lead             = dict_yaml['number_lead_years']

    y0_trn                = dict_yaml['specify_first_train_year'] 
    y0_tst                = dict_yaml['specify_first_test_year']  
    y1_tst                = dict_yaml['specify_last_test_year']   

    dir_frnt_src          = dict_yaml['dir_source']
    dir_frnt_obs          = dict_yaml['dir_frnt_obs']
    dir_frnt_mdl          = dict_yaml['dir_frnt_mdl']
    dir_frnt_out          = dict_yaml['dir_frnt_adj']

    do_badj               = dict_yaml['do_badj'] 
    use_ensemble_badj     = dict_yaml['use_ensemble_badj'] 
    do_metadata_badj      = dict_yaml['do_metadata_badj']
    save_forecasts_badj   = dict_yaml['save_forecasts_badj']
    save_hindcasts_badj   = dict_yaml['save_hindcasts_badj']

    do_tadj               = dict_yaml['do_tadj']
    do_glbavg_based_trend = dict_yaml['do_glbavg_based_trend']
    use_ensemble_tadj     = dict_yaml['use_ensemble_tadj']
    do_metadata_tadj      = dict_yaml['do_metadata_tadj']
    save_forecasts_tadj   = dict_yaml['save_forecasts_tadj']
    save_hindcasts_tadj   = dict_yaml['save_hindcasts_tadj']
    
    ####################
    ### Save Details ###
    ####################
    
    dir_out = f'{dir_frnt_out}/{verification}/{model}' 
    
    print("======")
    print(f"output directory:")
    print(f"{dir_out}")    
        
    Path(dir_out).mkdir(parents=True,exist_ok=True)
    
    ##############
    ### Set Up ###
    ##############
    
    realm = meta_module().var_dict[var]['*realm']
    grid  = meta_module().var_dict[var]['*grid']
    units = meta_module().var_dict[var]['units']

    mdl_dict = mdl_module(dir_frnt_mdl).mdl_dict[model] 
    obs_dict = obs_module(dir_frnt_obs).obs_dict[verification][var] 
    
    if y0_trn is False:
        y0_trn = max([obs_dict['y0'],
                      mdl_dict['y0']])       # first initial year training sample
    if y0_tst is False:
        y0_tst = y0_trn + 1                  # first initial year test sample
    if y1_tst is False:
        y1_tst = min([obs_dict['y1'],
                      mdl_dict['y1']])+1     # last initial year test sample

    if verbose:
        print("======")
        print(f"y0 {verification}: {obs_dict['y0']}")
        print(f"y1 {verification}: {obs_dict['y1']}")
        print("======")
        print(f"y0 {model}: {mdl_dict['y0']}")
        print(f"y1 {model}: {mdl_dict['y1']}")
        print("======")
        print(f"y0_trn: {y0_trn}")
        print(f"y1_trn: {y0_tst-1}")
        print("======")
        print(f"y0_tst: {y0_tst}")
        print(f"y1_tst: {y1_tst}")
        print("======")

    #################################
    ### Read and Pre-Process Data ###
    #################################
    
    obs_read = load_verification(obs_dict['dir'],
                                 two_dim=True)
    hnd_read = load_forecasts(mdl_dict['dir'])
   
    print("----")
    print(f'obs_read: {obs_read.shape}')  
    print(f'hnd_read: {hnd_read.shape}')      

    obs_in, hnd_in = preprocess_data(obs_read,
                                     hnd_read)
    fct_in = hnd_read.sel(year=hnd_in.year.values[-1]+1).expand_dims('channels',
                                                                      axis=2)
    
    hnd_in = xr.concat([hnd_in,
                        fct_in],
                       "year")
    hnd_in = UnitChange(hnd_in).Int2Met(var)

    print("----")
    print(f'obs_in: {obs_in.shape}')  
    print(f'hnd_in: {hnd_in.shape}')  
    
    obs_in = obs_in.rename({'lead_time' : 'time'}).squeeze()
    hnd_in = hnd_in.rename({'lead_time' : 'time'}).squeeze()    
    
    obs = obs_in.sel(time=slice(nyrs_lead*12))
    hnd = hnd_in.sel(time=slice(nyrs_lead*12)) 
    
    print("----")
    print(f'obs: {obs.shape}')  
    print(f'hnd: {hnd.shape}')  
    
    # clean
    del (obs_read,
         hnd_read,
         obs_in,
         hnd_in)  
    gc.collect()       

    ####################################################
    ### Adjust Forecasts: sequential bias correction ###
    ####################################################
    
    print("======")
    if do_badj:
        (fct_badj,
         hnd_badj) = sequential_bias_adjustment(obs.to_dataset(),  # ref dataset
                                                hnd.to_dataset(),  # raw ensemble 
                                                y0_train=y0_trn,  
                                                y0_test=y0_tst,
                                                # exclude_idx=-1,
                                                use_ensemble=use_ensemble_badj,
                                                ripf=mdl_dict['ripf'],
                                                do_metadata=do_metadata_badj,
                                                dir_out=f"{dir_out}/Bias_Adjusted",
                                                file_out=f"{var}_{realm}_{model.replace('.','-')}",
                                                save_forecasts=save_forecasts_badj,
                                                save_hindcasts=save_hindcasts_badj)
    
    #####################################################
    ### Adjust Forecasts: sequential trend correction ###
    #####################################################

    print("======")
    if do_tadj:
        (fct_tadj,
         hnd_tadj) = sequential_trend_adjustment(obs.to_dataset(),  # ref dataset
                                                 hnd.to_dataset(),  # raw ensemble  
                                                 do_glbavg_based_trend=do_glbavg_based_trend,
                                                 y0_train=y0_trn,  
                                                 y0_test=y0_tst,
                                                 use_ensemble=use_ensemble_tadj,
                                                 ripf=mdl_dict['ripf'],
                                                 do_metadata=do_metadata_tadj,
                                                 dir_out=f"{dir_out}/Trend_Adjusted",
                                                 file_out=f"{var}_{realm}_{model.replace('.','-')}",
                                                 save_forecasts=save_forecasts_tadj,
                                                 save_hindcasts=save_hindcasts_tadj)                                          print("done")
    print("======")
    


if __name__ == '__main__':
    
    filename = f'{os.getcwd()}/config.yaml'
    with open(filename) as f:
        dict_yaml = yaml.load(f, Loader=yaml.loader.SafeLoader)

    print("======")
    print(f"variable:")
    print(f"{dict_yaml['variable'][0]}")
    print("======")
    print(f"working directory:")
    print(f"{dict_yaml['dir_source']}")

    from modules_nn.info_meta import meta_module
    from modules_nn.info_obs  import obs_module
    from modules_nn.info_mdl  import mdl_module

    from modules_nn.data_load           import *
    from modules_nn.data_preprocessing  import * 
    from modules_nn.data_postprocessing import * 

    from modules_nn.util_unitchg import *
    
    badj_and_tadj(dict_yaml)


