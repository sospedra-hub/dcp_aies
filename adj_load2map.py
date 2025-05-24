#!/usr/bin/env python
# coding: utf-8

import os
import yaml 
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import glob
import tqdm
import gc
import dask
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

def extract_params(model_dir):
    
    params = {}
    
    path = glob.glob(f'{model_dir}/*.txt')[0]
    file = open(path)
    
    content=file.readlines()
    for line in content:
        key = line.split('\t')[0]
        try:
            value = line.split('\t')[1].split('\n')[0]
        except:
            value = line.split('\t')[1]
        try:    
            params[key] = eval(value)
        except:
            params[key] = value
    return params



def nn_load2map(hnd_in,
                obs_in,
                dict_yaml,
                params,
                test_years,
                model_year=None,
                model_dir=None,
                save=False,
                save_outof_source=False):
    
    ##############
    ### Set Up ###
    ##############    

    climate_mdl_tst    = dict_yaml['models_to_test'][0]  
    verification       = dict_yaml['verification'][0]    

    model                 = params["model"]
    hidden_dims           = params["hidden_dims"]
    time_features         = params["time_features"]
    append_mode           = params['append_mode']
    batch_normalization   = params["batch_normalization"]
    dropout_rate          = params["dropout_rate"]
    extra_predictors      = params["extra_predictors"] 
    add_feature_dim       = params["add_feature_dim"]
    lead_time_mask        = params["lead_time_mask"]   
    lead_months_to_adjust = params["lead_months_to_adjust"]

    forecast_preprocessing_steps     = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]
    
    #####################################
    ### Make data consistent          ### 
    ### --remove non-overlapping nans ###
    ##################################### 
    
    ds_raw, obs_raw, nanremover = remove_nans(hnd_in,   
                                              obs_in) 
    
    # clean
    del (hnd_in,
         obs_in)  
    gc.collect()       

    print("----")
    print(f'obs_raw: {obs_raw.shape}')  
    print(f'ds_raw: {ds_raw.shape}')  
    
    ##################
    ### Set device ###
    ##################     
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ################################################
    ### Preprocessing input data (get image dim) ###
    ################################################     
    
    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year
        
    train_years = ds_raw.year[ds_raw.year <= model_year_].to_numpy()    
    n_train     = len(train_years)
    
    ds_baseline  = ds_raw[:n_train,...]
    obs_baseline = obs_raw[:n_train,...]  ##### to transform back model output
    
    print("----")
    print(f'obs_baseline: {obs_baseline.shape}')  
    print(f'ds_baseline: {ds_baseline.shape}')  
    
    train_mask   = create_train_mask(ds_baseline)
            
    preprocessing_mask_hnd = np.broadcast_to(train_mask[...,
                                                        None,
                                                        None,
                                                        None],
                                             ds_baseline.shape)
    preprocessing_mask_obs = np.broadcast_to(train_mask[...,
                                                        None,
                                                        None,
                                                        None],
                                             obs_baseline.shape) 
    
    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline,
                                                                          mask=preprocessing_mask_hnd)
    obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline,
                                                                               mask=preprocessing_mask_obs)  
    if 'standardize' in ds_pipeline.steps:
        obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'),
                                             'standardize')
    
    ds  = ds_pipeline.transform(ds_raw)
    ds_train  = ds[:n_train,...]
    ds_train  = nanremover.sample(ds_train)  # flatten and sample training data at those locations

    year_max = ds[:n_train].year[-1].values         
    
    img_dim   = ds_train.shape[-1] 

    #############################
    ### Save Clean Ocean Mask ###
    #############################    
    
    save_nan_mask(ds.lat.values,
                  ds.lon.values,
                  nanremover,
                  dir_name=model_dir,
                  file_name=f'nan_mask_{verification}_{climate_mdl_tst}') 
    
    ##################
    ### Load model ###
    ##################
    
    print("====")
    print(f"loading model..")
    print(f"model last year of training: {model_year_}")
    print(img_dim)
    
    net = model(img_dim,
                hidden_dims[0],
                hidden_dims[1],
                added_features_dim=add_feature_dim,
                append_mode=append_mode,
                batch_normalization=batch_normalization,
                dropout_rate=dropout_rate,
                device=device)
    # print(net)

    net.load_state_dict(torch.load(glob.glob(model_dir + f'/*-{model_year_}*.pth')[0],
                                   map_location=torch.device('cpu'))) # train up to to model_year_
    net.to(device)
    net.eval()
    print(f"done")
    
    
    ######################################    
    ### Set input (forecast) test data ###
    ###################################### 
    
    ds_test = ds[n_train:,...]
    ds_test = nanremover.sample(ds_test,
                                mode='Eval')
    
    del ds
    
    print("----")
    print("ds_test", ds_test.shape)
    
    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
    
    test_set = XArrayDataset(ds_test,
                             xr.ones_like(ds_test), 
                             extra_predictors=extra_predictors,
                             lead_time=None,
                             lead_time_mask=lead_time_mask,
                             time_features=time_features,
                             in_memory=False,
                             aligned=True,
                             year_max=year_max) 
                
                
    ##############################################             
    ### Map input test data (adjust forecasts) ###
    ##############################################             

    print("====")
    print(f"adjusting forecasts..")
    
    test_results = np.zeros_like(ds_test)

    for i, (x, target) in enumerate(test_set):
        
        year_idx, lead_time_list_idx = np.divmod(i,
                                                 len(test_lead_time_list))
        
        lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
            
        with torch.no_grad():
            
            if (type(x) == list) or (type(x) == tuple):
                test_raw = (x[0].unsqueeze(0).to(device),
                            x[1].unsqueeze(0).to(device))
            else:
                test_raw = x.unsqueeze(0).to(device)
                    
            test_adjusted = net(test_raw) # raw --> adj
                
            test_results[year_idx,lead_time_idx,] = test_adjusted.to(torch.device('cpu')).numpy()
    print(f"done")


    ###############################################             
    ### Reshape output data (adjusted forecast) ###
    ############################################### 
                
    print("====")
    print(f"reshaping forecast..")        
    test_results_upsampled = nanremover.to_map(test_results) 
    test_results_untransformed = obs_pipeline.inverse_transform(test_results_upsampled.values) 
    result = xr.DataArray(test_results_untransformed,
                          test_results_upsampled.coords,
                          test_results_upsampled.dims,
                          name='nn_adjusted')
    result = result.isel(lead_time=np.arange(lead_months_to_adjust)) 
    print(f"done")        

    
    ####################             
    ### Save results ###
    #################### 
    
    print("====")
    print(f"saving adjusted forecast..")        
    if save:
        
        if np.min(test_years) != np.max(test_years):
            save_name = f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}.nc'
        else:
            save_name = f'saved_model_nn_adjusted_{np.min(test_years)}.nc'
            
        if save_outof_source:
            dir_out = f"{model_dir}/OutOfSource/{dict_yaml['models_to_test'][0]}"
        else:
            dir_out = model_dir  
        
        Path(dir_out).mkdir(parents=True,exist_ok=True)
        result.to_netcdf(path=Path(dir_out,
                                   save_name,
                                   mode='w'))  
        
        print("====")                      
        print(f"Output dir: {dir_out}")
        print("====")                      
        
            
    print(f"done")        

                                   
    
    return result


                
                


if __name__ == "__main__":
    
    
    from modules_nn.info_obs              import obs_module
    from modules_nn.info_mdl              import mdl_module    
    from modules_nn.data_load             import (
                                                  load_verification,
                                                  load_forecasts,
                                                  load_nan_mask
                                                 )
    from modules_nn.data_save             import save_nan_mask
    from modules_nn.nn_models.autoencoder import Autoencoder
    from modules_nn.util_losses           import WeightedMSE
    from modules_nn.data_preprocessing    import (
                                               preprocess_data,
                                               remove_nans,
                                               create_train_mask,      
                                               AnomaliesScaler_v1,      
                                               AnomaliesScaler_v2,      
                                               Standardizer,           
                                               PreprocessingPipeline,  
                                                  )   
    from modules_nn.util_torchds          import XArrayDataset
    
    from modules_nn.util_unitchg          import UnitChange
    
    filename = f'{os.getcwd()}/config.yaml'
    with open(filename) as f:
        dict_yaml = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
    print("======")
    print(f"working directory:")
    print(f"{dict_yaml['dir_source']}")
    
    
    #############################
    ### Get config parameters ###
    #############################
    
    var                = dict_yaml['variable'][0]
    n_ens              = dict_yaml['n_ens']
    nlead_years        = dict_yaml['number_lead_years']
    test_years         = dict_yaml['test_years']
    save_output        = dict_yaml['save_output']
    
    climate_mdl_trn    = dict_yaml['models_to_train'][0]
    climate_mdl_tst    = dict_yaml['models_to_test'][0]
    verification       = dict_yaml['verification'][0]
    
    save_outof_source  = False
    if climate_mdl_tst != climate_mdl_trn:
        save_outof_source = True
    
    if save_outof_source:
        dir_frnt_fct   = dict_yaml['dir_frnt_l2m'] 
    else:
        dir_frnt_fct   = dict_yaml['dir_frnt_mdl']
        
    dir_frnt_obs       = dict_yaml['dir_frnt_obs']
    dir_frnt_out       = dict_yaml['dir_frnt_adj']
    
    dir_out = f'{dir_frnt_out}/{verification}/{climate_mdl_trn}/Autoencoder'    
    
    ################################
    ### Load and preprocess data ###
    ################################ 
    
    mdl_dict = mdl_module(dir_frnt_fct).mdl_dict[climate_mdl_tst] 
    obs_dict = obs_module(dir_frnt_obs).obs_dict[verification][var]  
    
    obs_read = load_verification(obs_dict['dir'],
                                 two_dim=True)
    hnd_read = load_forecasts(mdl_dict['dir'])
    
    print("----")
    print(f'obs_read: {obs_read.shape}')  
    print(f'hnd_read: {hnd_read.shape}')      
    
    nlead_years_to_adjust = min([nlead_years,
                                 len(hnd_read.lead_time.values)//12]) # min train/test data
    
    obs_in, hnd_in = preprocess_data(obs_read.sel(year=slice(obs_read.year.to_numpy()[0],
                                                             test_years[0]-1)),
                                     hnd_read.sel(year=slice(hnd_read.year.to_numpy()[0],
                                                             test_years[0]-1)),
                                     nlead_years_to_adjust)
    fct_in = hnd_read.sel(year=test_years).expand_dims('channels',
                                                       axis=2)

    hnd_in = xr.concat([hnd_in,
                        fct_in],
                       "year")
    hnd_in = UnitChange(hnd_in).Int2Met(var)
    
    print("----")
    print(f'obs: {obs_in.shape}')  
    print(f'hnd: {hnd_in.shape}')  
    
    # clean
    del (hnd_read,
         obs_read)  
    gc.collect()       
    
    ##########################
    ### Loop over ensemble ###
    ########################## 

    for ens in range(n_ens):
        
        print("====")
        print(f'clean data')
        nan_mask = load_nan_mask(f'{dir_out}/E{ens+1}',
                                'nan_mask').squeeze().to_numpy()
        obs_in = obs_in*nan_mask # mask out data consistent with trained model
        hnd_in = hnd_in*nan_mask # mask out data consistent with trained model
        print("----")
        print(f'mask: {nan_mask.shape}')
        print(f'obs: {obs_in.shape}')  
        print(f'hnd: {hnd_in.shape}')  
        
        params = extract_params(f'{dir_out}/E{ens+1}') # get nn parameters
        
        if params['version'] == 1:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer()) ]
            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0))  ]
        elif params['version'] == 2:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0)), 
            ('standardize', Standardizer()) ]
            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0))  ]
        else:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer()) ]
            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0))  ]
                
        params["lead_months_to_adjust"] = nlead_years_to_adjust*12     
            
        obs_in = obs_in.isel(lead_time=slice(0,params['lead_months_to_adjust']))
        hnd_in = hnd_in.isel(lead_time=slice(0,params['lead_months_to_adjust']))
        
        nn_load2map(hnd_in,
                    obs_in,
                    dict_yaml,
                    params,
                    test_years,
                    model_dir=f"{dir_out}/E{ens+1}",
                    save=save_output,
                    save_outof_source=save_outof_source)

        
        
