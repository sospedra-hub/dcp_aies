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
import tqdm
import gc
import dask
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt



def nn_train2map(hnd_in,
                 obs_in,
                 dict_yaml,
                 params,
                 results_dir=None,
                 numpy_seed=None,
                 torch_seed=None,
                 save=False):
    
    n_runs = dict_yaml['n_runs']

    y0_trn = dict_yaml['specify_first_train_year'] 
    y0_tst = dict_yaml['specify_first_test_year']  
    y1_tst = dict_yaml['specify_last_test_year']   
    
    if y0_trn is False:
        y0_trn = max([obs_in.year.values[0], 
                      hnd_in.year.values[0]])   # first initial year training sample
    if y0_tst is False:
        y0_tst = y0_trn + 1                     # first initial year test sample
    if y1_tst is False:
        y1_tst = min([obs_in.year.values[-1],
                      hnd_in.year.values[-1]])  # last initial year test sample
    
    n_years = y1_tst - y0_tst + 1

    print("====")
    print(f"y0 obs: {obs_in.year.values[0]}")
    print(f"y1 obs: {obs_in.year.values[-1]}")
    print("====")
    print(f"y0 mdl: {hnd_in.year.values[0]}")
    print(f"y1 mdl: {hnd_in.year.values[-1]}")
    print("====")
    print(f"y0_trn: {y0_trn}")
    print(f"y1_trn: {y0_tst-1}")
    print("====")
    print(f"y0_tst: {y0_tst}")
    print(f"y1_tst: {y1_tst}")
    
    ##############
    ### Set Up ###
    ##############
    
    nlead_years          = len(hnd_in.lead_time.values)//12

    version              = params["version"]  
    hyperparam           = params["hyperparam"]
    model                = params["model"]
    hidden_dims          = params["hidden_dims"]
    time_features        = params["time_features"]
    epochs               = params["epochs"]
    batch_size           = params["batch_size"]
    batch_normalization  = params["batch_normalization"]
    dropout_rate         = params["dropout_rate"]
    optimizer            = params["optimizer"]
    lr                   = params["lr"]
    l2_reg               = params["L2_reg"]
    loss_region          = params["loss_region"]
    subset_dimensions    = params["subset_dimensions"]
    extra_predictors     = params["extra_predictors"] 
    add_feature_dim      = params["add_feature_dim"]
    append_mode          = params['append_mode']    
    lead_time_mask       = params['lead_time_mask']
    
    if params["lr_scheduler"]:
        start_factor = params["start_factor"]
        end_factor = params["end_factor"]
        total_iters = params["total_iters"]
    else:
        start_factor = end_factor = total_iters = None
    
    forecast_preprocessing_steps     = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    #####################################
    ### Make data consistent          ### 
    ### --remove non-overlapping nans ###
    ##################################### 
    
    ds_raw, obs_raw, nanremover = remove_nans(hnd_in,   
                                              obs_in)
    
    save_nan_mask(hnd_in.lat.values,
                  hnd_in.lon.values,
                  nanremover,
                  dir_name=results_dir,
                  file_name='nan_mask')

    # clean
    del (hnd_in,
         obs_in)  
    gc.collect()       

    ######################
    ### Get test years ###
    ###################### 
        
    test_years = ds_raw.year[-n_years:].to_numpy()
    test_years = [*test_years,
                  test_years[-1] + 1]
    
    print("====")
    print(f"test years: {test_years[0]}..{test_years[-1]}")
    
    if n_runs > 1:
        numpy_seed = None  
        torch_seed = None  
    
    ######################################
    ### Save setup information to file ###
    ###################################### 
    
    print("====")
    print(f"saving information..")
    
    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"model\t{model.__name__}\n" +
            f"version\t{version}\n" +
            f"hidden_dims\t{hidden_dims}\n" +
            f"time_features\t{time_features}\n" +
            f"extra_predictors\t{params['extra_predictors']}\n" +
            f"append_mode\t{params['append_mode']}\n" +
            f"hyperparam\t{hyperparam}\n" +        
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"batch_normalization\t{batch_normalization}\n" +
            f"dropout_rate\t{dropout_rate}\n" +
            f"L2_reg\t{l2_reg}\n" + 
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{lr}\n" +
            f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n" + 
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"loss_region\t{loss_region}\n" +
            f"subset_dimensions\t{subset_dimensions}\n" + 
            f"lead_time_mask\t{params['lead_time_mask']}\n"+
            f"nlead_years\t{nlead_years}\n"+
            f"add_feature_dim\t{add_feature_dim}\n" 
        )

    print("done")

    gc.collect()
    
    ##################
    ### Set device ###
    ##################     
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    #######################################
    #######################################
    #######################################
    ### Sequential training and testing ###
    #######################################   
    #######################################
    #######################################
    
    print("====")
    print(f"sequential training and testing..")

    for run in range(n_runs):
        
        print(f"run {run + 1} of {n_runs}")
        
        for y_idx, test_year in enumerate(test_years):
            
            print(f"test year: {test_year}")
            
            #######################             
            #######################             
            ### Training module ###
            #######################             
            #######################             

            train_years = ds_raw.year[ds_raw.year<test_year].to_numpy()
            
            n_train     = len(train_years)
            
            ds_baseline  = ds_raw[:n_train,...]
            obs_baseline = obs_raw[:n_train,...]
            
            train_mask   = create_train_mask(ds_baseline)
            
            ##############################    
            ### Mask for available obs ###
            ##############################              

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

            if numpy_seed is not None:
                np.random.seed(numpy_seed)
            if torch_seed is not None:
                torch.manual_seed(torch_seed)

            ###################################    
            ### Preprocessing training data ###
            ###################################    
            
            ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline,
                                                                            mask=preprocessing_mask_hnd)
            obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline,
                                                                            mask=preprocessing_mask_obs)
            if 'standardize' in ds_pipeline.steps:
                obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'),
                                                     'standardize')
                
            ds  = ds_pipeline.transform(ds_raw)
            obs = obs_pipeline.transform(obs_raw)

            year_max = ds[:n_train + 1].year[-1].values 
            
            del (ds_baseline,
                 obs_baseline,
                 preprocessing_mask_obs,
                 preprocessing_mask_hnd)
            gc.collect()
            
            
            #########################    
            ### Set training data ###
            #########################   
            
            ds_train   = ds[:n_train,...]
            obs_train  = obs[:n_train,...]
            
            weights = np.cos(np.ones_like(ds_train.lon)*(np.deg2rad(ds_train.lat.to_numpy()))[..., None])  
            weights = xr.DataArray(weights,
                                   dims=ds_train.dims[-2:],
                                   name = 'weights').assign_coords({'lat':ds_train.lat, 
                                                                    'lon':ds_train.lon}) 
            weights_ = weights.copy()

                
            ###########################    
            ### Clean training data ###
            ###########################
            
            ds_train  = nanremover.sample(ds_train)  # flatten and sample training data
            obs_train = nanremover.sample(obs_train) # flatten and sample obs      data   

            weights   = nanremover.sample(weights)   # flatten and sample weights
            weights_  = nanremover.sample(weights_)
                
            weights = weights.values
            weights_ = weights_.values
                
            img_dim   = ds_train.shape[-1] 
            
            gc.collect()
            torch.cuda.empty_cache() 
            torch.cuda.synchronize() 

            if loss_region is not None:
                loss_region_indices = nanremover.extract_indices(subregions[loss_region]) 
            else:
                loss_region_indices = None

                
            #################    
            ### Set model ###
            #################  

            net = model(img_dim,
                        hidden_dims[0],
                        hidden_dims[1],
                        added_features_dim=add_feature_dim,
                        append_mode=append_mode,
                        batch_normalization=batch_normalization,
                        dropout_rate=dropout_rate,
                        device=device)
            
            net.to(device)

    
            #####################    
            ### Set optimizer ###
            ##################### 
            
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay = l2_reg)
            
            ################################    
            ### Set scheduler -if called ###
            ################################ 

            if params['lr_scheduler']:
                scheduler = lr_scheduler.LinearLR(optimizer,
                                                  start_factor=params['start_factor'],
                                                  end_factor=params['end_factor'],
                                                  total_iters=params['total_iters'])

            #########################    
            ### Set training data ###
            #########################  
                
            train_set = XArrayDataset(ds_train,
                                      obs_train,
                                      mask=train_mask,
                                      extra_predictors=extra_predictors,
                                      lead_time_mask=lead_time_mask,
                                      in_memory=True,
                                      lead_time=None, 
                                      time_features=time_features,
                                      aligned=True,
                                      year_max=year_max) 

            dataloader = DataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True)
            
            #########################    
            ### Set loss function ###
            #########################  
            
            criterion = WeightedMSE(weights=weights,
                                    device=device,
                                    hyperparam=hyperparam,
                                    reduction='mean',
                                    loss_area=loss_region_indices)
            
            ####################             
            ### Training bit ###
            ####################             

            epoch_loss = []
            net.train()
            num_batches = len(dataloader)
            
            for epoch in tqdm.tqdm(range(epochs)):

                batch_loss = 0

                for batch, (x, y) in enumerate(dataloader):

                    if (type(x) == list) or (type(x) == tuple):
                        x = (x[0].to(device),
                             x[1].to(device))
                    else:
                        x = x.to(device)

                    if (type(y) == list) or (type(y) == tuple):
                        y, m = (y[0],
                                y[1].to(device))    # m stands for lead-time dependent mask
                    else:
                        y = y
                        m  = None

                    y = y.to(device)
                    
                    optimizer.zero_grad()
                        
                    adjusted_forecast = net(x)  # raw --> adj
                    
                    loss = criterion(adjusted_forecast,
                                     y,
                                     mask=m)
                        
                    batch_loss += loss.item()

                    loss.backward()

                    optimizer.step()

                epoch_loss.append(batch_loss/num_batches)
 
                if params['lr_scheduler']:
                    scheduler.step()

            del (train_set,
                 dataloader,
                 ds_train,
                 obs_train,
                 adjusted_forecast,
                 x,
                 y,
                 m)
            gc.collect()

            
            ######################             
            ######################             
            ### Testing module ###
            ######################             
            ######################             
             
            
            if test_year < test_years[-1]:
                
                ds_test  = ds[n_train:n_train + 1,...]
                obs_test = obs[n_train:n_train + 1,...]
                
                del (ds,obs)
                
                #######################    
                ### Clean test data ###
                #######################
            
                ds_test = nanremover.sample(ds_test, 
                                            mode='Eval')   
                obs_test = nanremover.sample(obs_test)
                
                
                #####################    
                ### Set test data ###
                ##################### 
                
                test_set = XArrayDataset(ds_test,
                                         obs_test,
                                         extra_predictors=extra_predictors,
                                         lead_time=None,
                                         lead_time_mask=lead_time_mask,
                                         time_features=time_features,
                                         in_memory=False,
                                         aligned=True,
                                         year_max=year_max)
                
                dataloader_test = DataLoader(test_set,
                                             batch_size=ds_test.shape[1], # forecast range in months
                                             shuffle=False)
                
                
                
                ############################    
                ### Set accuracy measure ###
                ############################
            
                criterion_test =  WeightedMSE(weights=weights_, 
                                              device=device,
                                              hyperparam=1,
                                              reduction='mean',
                                              loss_area=loss_region_indices)
                
                
                ###################             
                ### Testing bit ###
                ###################             
                
                test_results = np.zeros_like(ds_test)

                test_loss = np.zeros(shape=(ds_test.shape[0],
                                            ds_test.shape[1]))
                    
                
                for i, (x, target) in enumerate(dataloader_test):
                    
                    year_idx = i
                    
                    net.eval()
                    
                    
                    with torch.no_grad():
                        
                        if (type(x) == list) or (type(x) == tuple):
                            test_raw = (x[0].to(device),
                                        x[1].to(device))
                        else:
                            test_raw = x.to(device)
                            
                        if (type(target) == list) or (type(target) == tuple):
                            test_obs, m = (target[0].to(device),
                                           target[1].to(device))
                            
                        else:
                            test_obs = target.to(device)
                            m = None
                                    
                        test_adjusted = net(test_raw)    # raw --> adj
                                
                        loss = criterion_test(test_adjusted,
                                              test_obs) 

                        test_results[year_idx,:,] = test_adjusted.to(torch.device('cpu')).numpy()
                        
                        test_loss[year_idx,:] = loss.item()
                        

                del  (test_set,
                      test_raw,
                      test_obs,
                      x,
                      target,
                      m,
                      test_adjusted,
                      ds_test,
                      obs_test)
                gc.collect()

                
                ######################             
                ### Gather results ###
                ###################### 
                
                test_results_upsampled = nanremover.to_map(test_results) 
                test_results_untransformed = obs_pipeline.inverse_transform(test_results_upsampled.values) 
                result = xr.DataArray(test_results_untransformed,
                                      test_results_upsampled.coords,
                                      test_results_upsampled.dims,
                                      name='nn_adjusted')
                
                
                ########################################    
                ### Save results and diagnotic plots ###
                ########################################  
                
                result.to_netcdf(path=Path(results_dir,
                                           f'nn_adjusted_{test_year}_{run+1}.nc',
                                           mode='w'))
                
                # clean
                del (test_results,
                     test_results_untransformed)
                gc.collect()
                
                fig, ax = plt.subplots(1,1,figsize=(8,5))
                ax.plot(np.arange(1,
                                  epochs+1),
                        epoch_loss)
                ax.set_title(f'Train loss \n test loss: {np.mean(test_loss)}') 
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                plt.show()
                plt.savefig(results_dir+f'/Figures/train_loss_{train_years[0]}-{test_year-1}.png')
                plt.close()

                if save:
                    if test_year >= 2016:
                        nameSave = f"MODEL_V{params['version']}_{train_years[0]}-{test_year-1}.pth"
                        torch.save( net.state_dict(),results_dir + '/' + nameSave)
                        
                del (result,
                     net,
                     optimizer)
                gc.collect()
                torch.cuda.empty_cache() 
                torch.cuda.synchronize() 

            else:

                nameSave = f"MODEL_final_V{params['version']}_{train_years[0]}-{test_years[-2]}.pth"
                
                torch.save(net.state_dict(),f'{results_dir}/{nameSave}')
                
        print("====")    
        print(f"done {run + 1} of {n_runs}")
        
    print("====")    
    print(f"done")
                
                

if __name__ == "__main__":

    from modules_nn.info_obs              import obs_module
    from modules_nn.info_mdl              import mdl_module    
    from modules_nn.info_parameters       import parameters    
    from modules_nn.data_load             import (
                                                  load_verification,
                                                  load_forecasts,
                                                  load_nan_mask
                                                  )
    from modules_nn.data_save             import save_nan_mask
    from modules_nn.data_preprocessing    import (
                                               preprocess_data,
                                               remove_nans,
                                               create_train_mask,      
                                               AnomaliesScaler_v1,      
                                               AnomaliesScaler_v2,      
                                               Standardizer,           
                                               PreprocessingPipeline,  
                                                  )   
    from modules_nn.nn_models.autoencoder import Autoencoder
    from modules_nn.util_losses           import WeightedMSE
    from modules_nn.util_subregions       import subregions
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
    save_model         = dict_yaml['save_model']
    
    climate_mdl        = dict_yaml['models_to_train'][0]
    verification       = dict_yaml['verification'][0]
    
    dir_frnt_obs       = dict_yaml['dir_frnt_obs']
    dir_frnt_fct       = dict_yaml['dir_frnt_mdl']
    dir_frnt_out       = dict_yaml['dir_frnt_adj']

    load_space_mask    = dict_yaml['load_space_mask']
    space_mask         = dict_yaml['space_mask_for_training'][0]
    
    dir_mask = f'{dir_frnt_out}/{verification}/{climate_mdl}/Masks'           
    dir_out  = f'{dir_frnt_out}/{verification}/{climate_mdl}/Autoencoder'    
        
    ######################
    ### Set parameters ###
    ######################     
    
    params = parameters(Autoencoder).params  
    
    assert params['version'] in [1,2,3]

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
    
    params['add_feature_dim'] = 0
    if params['time_features'] is not None:
        params['add_feature_dim'] = len(params['time_features'])
    if params['extra_predictors'] is not None:
        params['add_feature_dim'] = params['add_feature_dim'] + len(params['extra_predictors'])
    
    
    ################################
    ### Load and preprocess data ###
    ################################ 
    
    mdl_dict = mdl_module(dir_frnt_fct).mdl_dict[climate_mdl] 
    obs_dict = obs_module(dir_frnt_obs).obs_dict[verification][var]     
    
    obs_read = load_verification(obs_dict['dir'],
                                 two_dim=True)
    hnd_read = load_forecasts(mdl_dict['dir'])
    
    obs_in, hnd_in = preprocess_data(obs_read,
                                     hnd_read,
                                     nlead_years=nlead_years)
    hnd_in = UnitChange(hnd_in).Int2Met(var)

    print(f'obs: {obs_in.shape}')  
    print(f'hnd: {hnd_in.shape}')  

    # clean
    del (hnd_read,
         obs_read)  
    gc.collect()       

    ################################
    ### Load mask and clean data ###   
    ################################ 
    
    if load_space_mask:
        if os.path.isfile(os.path.join(dir_mask,f'{space_mask}.nc')):
            print(f"{space_mask} is in directory.")
            print("====")
            print(f'clean data')
            print(f'uses: {space_mask}')
            nan_mask = load_nan_mask(f'{dir_mask}',
                                     f'{space_mask}').squeeze().to_numpy()
            obs_in = obs_in*nan_mask # mask out data consistent with trained model
            hnd_in = hnd_in*nan_mask # mask out data consistent with trained model
            print("----")
            print(f'mask: {nan_mask.shape}')
            print(f'obs: {obs_in.shape}')  
            print(f'hnd: {hnd_in.shape}')  
        else:
            print("----")
            print(f"No {space_mask} in directory -- use default data space mask.")
    
    
    #############################
    ### Loop over NN ensemble ###
    ############################# 
    

    for ens in range(n_ens):
        print("====")
        print( f'Ensemble member: {ens+1} of {n_ens}')
        Path(f"{dir_out}/E{ens+1}").mkdir(parents=True,
                                          exist_ok=True)
        Path(f"{dir_out}/E{ens+1}/Figures").mkdir(parents=True,
                                                  exist_ok=True)

        nn_train2map(hnd_in,
                     obs_in,
                     dict_yaml,
                     params,
                     results_dir=f"{dir_out}/E{ens+1}",
                     numpy_seed=ens,
                     torch_seed=ens,
                     save=save_model)
        
        print("====")                      
        print(f"Output dir: {dir_out}/E{ens+1}")
        print("====")                      
        
