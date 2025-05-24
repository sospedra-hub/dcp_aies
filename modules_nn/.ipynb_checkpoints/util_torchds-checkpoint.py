import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import random

class XArrayDataset(Dataset):

    def __init__(self,
                 data: xr.DataArray,
                 target: xr.DataArray,
                 extra_predictors: xr.DataArray=None,
                 mask=None,
                 lead_time=None,
                 lead_time_mask = None,
                 time_features=None,
                 in_memory=True,
                 to_device=None,
                 aligned=False,
                 year_max=None,
                 model='Autoencoder'): 
        
        self.mask = mask
        self.atm_co2_features = None

        self.data = data

        self.data = self.data.stack(flattened=('year',
                                               'lead_time')).transpose('flattened',...)
        self.target = target.stack(flattened=('year',
                                              'lead_time')).transpose('flattened',...)

        if self.mask is not None:
            self.data = self.data[~self.mask.flatten()]
            if aligned:
                self.target = self.target[~self.mask.flatten()]
                
        if len(self.data.year) > 1:
            years_in_months = (self.data.year - self.data.year[0]) * 12
        else:
            years_in_months = 0
            
        if aligned:
            target_idx = np.arange(len(self.data.flattened))
        else:
            target_idx = (years_in_months + self.data.lead_time - 1).to_numpy()
        self.target = self.target[target_idx,...]

        if lead_time_mask is not None:
            self.lead_time_mask = xr.ones_like(self.target) 
            self.lead_time_mask  = self.lead_time_mask.where(self.lead_time_mask.lead_time <= lead_time_mask*12,0.25)
        else:
            self.lead_time_mask = None
        
        if lead_time is not None:
            self.data = self.data.where((self.data.lead_time>=(lead_time - 1)*12+1) & (self.data.lead_time<(lead_time *12 )+1),
                                        drop=True)
            self.target = self.target.where((self.target.month>=(lead_time - 1) * 12 + 1) & (self.target.month < (lead_time *12 )+1),
                                            drop = True)

        if extra_predictors is not None:
                
                self.use_time_features = True
                self.extra_predictors = extra_predictors.stack(flattened=('year',
                                                                          'lead_time')).transpose('flattened',...)
                try:
                    self.extra_predictors = self.extra_predictors.sel(flattened = self.data.flattened)
                except:
                    raise ValueError("Extra predictors not available at the same time points as the predictors.") 
                self.extra_predictors  = (self.extra_predictors-self.extra_predictors.min())/(self.extra_predictors.max()-self.extra_predictors.min()).values
        else:
            self.extra_predictors = None

        if time_features is not None:

            self.use_time_features = True
            
            self.time_features_list = np.array([time_features]).flatten()
            
            feature_indices = {'year': 0,
                               'lead_time': 1,
                               'month_sin': 2,
                               'month_cos': 3}

            lead_time_to_numpy = np.arange(1,len(self.data.lead_time.values)+1)
            
            y = (self.data.year.to_numpy() + np.floor(lead_time_to_numpy/12)) / year_max
            lt = lead_time_to_numpy / np.max(lead_time_to_numpy)

            msin = np.sin(2 * np.pi * lead_time_to_numpy/12.0)
            mcos = np.cos(2 * np.pi * lead_time_to_numpy/12.0)
            
            self.time_features = np.stack([y,
                                           lt,
                                           msin,
                                           mcos],
                                          axis=1)
            self.time_features = self.time_features[...,
                                                    [feature_indices[k] for k in self.time_features_list]]

        else:
            if self.extra_predictors is None:
                self.use_time_features = False
                
        if in_memory:
            
            self.data = torch.from_numpy(self.data.to_numpy()).float()
            self.target = torch.from_numpy(self.target.to_numpy()).float()
            if self.lead_time_mask is not None:
                self.lead_time_mask = torch.from_numpy(self.lead_time_mask.to_numpy()).float()

            if self.use_time_features:
                self.time_features = torch.from_numpy(self.time_features).float()

            if to_device:
                self.data.to(to_device)
                self.target.to(to_device)
                if self.lead_time_mask is not None:
                    self.lead_time_mask = self.lead_time_mask.to(to_device)
                if self.use_time_features:
                    self.time_features = self.time_features.to(to_device)
                    
                    
            
    def __getitem__(self, index):
        
        x = self.data[index,...]
        y = self.target[index,...]
        
        if self.lead_time_mask is not None:
            m = self.lead_time_mask[index,...]

        if torch.is_tensor(x):

            if self.lead_time_mask is not None:
                y_ = (y, m)
            else:
                y_ = y

            if self.use_time_features: 
                t = self.time_features[index,...]
                x_ = (x,t)
            else: 
                x_ = x

            return x_, y_
        
        else:
            
            x = torch.from_numpy(x.to_numpy()).float()
            y = torch.from_numpy(y.to_numpy()).float()

            if self.lead_time_mask is not None:
                m = torch.from_numpy(m.to_numpy()).float()
                y_ = (y, m)
            else:
                y_ = y
            
            if self.use_time_features:
                t = self.time_features[index,...]
                t = torch.from_numpy(t).float()
                x_ = (x,t)
            else:  
                x_ = x
            
            return x_, y_
        

    def __len__(self):
        return len(self.data)
    
    




