import numpy as np
from xarray import DataArray
import xarray as xr
import math


def preprocess_data(obs_in,
                    ds_in,
                    nlead_years=10):
    
    print("====")
    print("preprocessing data..")    

    obs_in = obs_in.expand_dims('channels',
                                axis=2)   # irrelevant for AE
    
    ds_in = ds_in.expand_dims('channels',
                              axis=2) 
    
    ds_out, obs_out = align_data_and_targets(ds_in,
                                             obs_in,
                                             nlead_years) 
    
    ds_out = ds_out.transpose('year',
                              'lead_time',
                              ...)
    
    obs_out = reshape_obs_to_data(obs_out,
                                  ds_out,
                                  return_xarray=True).rename({obs_out.dims[1]:ds_out.dims[1]})


    if not ds_out.year.equals(obs_out.year):
        ds_out = ds_out.sel(year=obs_out.year)
        
    print("done")
    
    return obs_out, ds_out



def align_data_and_targets(data,
                           targets,
                           nlead_years):
    
    if nlead_years*12 > data.shape[1]:
        raise ValueError(f'Maximum lead year available: {int(data.shape[1] / 12)}')
    
    last_target_year = data.year[-1].item() + nlead_years - 1
    last_years_diff = data.year[-1].item() - targets.year[-1].item()
    year_diff = targets.year[-1].item() - last_target_year
    
    if year_diff >= 0:
        ds = data[:,:(nlead_years*12),...]
        obs = targets[:len(targets.year) - year_diff,...]

    else: 
        if (last_years_diff <= 0):
            ds = data[:,:(nlead_years*12),...]
            obs = targets
        else:
            ds = data[:-last_years_diff,:(nlead_years*12),...]
            obs = targets

    return ds, obs


def reshape_obs_to_data(obs,
                        data,
                        return_xarray=False):

    nlead_years = int(data.shape[1]/12)
    ls = [obs[y :y + nlead_years,
              ...].stack(flat = ['year',
                                 'month']).reset_index('flat',
                                                       drop=True) for y in range(len(obs.year))]
    obs_reshaped = xr.concat([ds.assign_coords(flat = np.arange(1,
                                                                len(ds.flat)+1)) for ds in ls], dim = 'year').assign_coords(year=obs.year)

    return obs_reshaped.rename({'flat':'month'}).transpose('year',
                                                           'month',
                                                           ...).where((obs_reshaped.year >= data.year.min()) & (obs_reshaped.year <= data.year.max()),drop = True)



def create_train_mask(dataset,
                      exclude_idx=0):
    mask = np.full((dataset.shape[0],
                    dataset.shape[1]),
                   False,
                   dtype=bool)
    x = np.arange(0, 12*dataset.shape[0], 12)   
    y = np.arange(1, dataset.shape[1] + 1)
    idx_array = x[..., None] + y
    # mask[idx_array >= idx_array[-1, exclude_idx + 12]] = True
    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True
    return mask



def remove_nans(data,     
                trgt):  
    nanremover = Spatialnanremove()
    nanremover.fit(data[:,:12,...],
                   trgt[:,:12,...])
    data = nanremover.to_map(nanremover.sample(data))
    trgt = nanremover.to_map(nanremover.sample(trgt))
        
    return data, trgt, nanremover



class Spatialnanremove: 

    def __init__(self):
        pass

    def fit(self,
            data,
            target): 
        '''
        Extract common grid points based on trainig and target data
        '''
        self.reference_shape = xr.full_like(target[0,0,0,...],
                                            fill_value=np.nan)
        try: 
            self.reference_shape = self.reference_shape.drop(['month',
                                                              'year']) 
        except: # extract initial spatial shape
            self.reference_shape = self.reference_shape.drop(['lead_time',
                                                              'year']) 
            
        # flatten target in space and choose space points where data is not NaN.            
        temp = target.stack(ref = ['lat',
                                   'lon']).sel(ref=data.stack(ref=['lat',
                                                                   'lon']).dropna(dim='ref').ref)  
        
        #extract locations common to target and training data by dropping the remaining NaN values
        self.final_locations = temp.dropna('ref').ref 
        
        return self

    def sample(self,
               data,
               mode=None,
               loss_area=None): 
        '''
        Pass a DataArray and sample at the extracted locations
        '''
        
        conditions = ['lat' in data.dims,
                      'lon' in data.dims]

        if all(conditions): # if a map get passeed
            sampled = data.stack(ref = ['lat',
                                        'lon']).sel(ref=self.final_locations)  
        else:               #  if a flattened dataset is passed (in space)
            sampled = data.sel(ref=self.final_locations)
    
        if mode == 'Eval':  # if we are sampling the test_set, remmeber the shape of the test Dataset in a template
            self.shape = xr.full_like(sampled,
                                      fill_value = np.nan)

        return sampled
    

    def extract_indices(self,
                        loss_area): 
        
        '''
        Extract indices of the flattened dimention over a specific region
        '''

        lat_min, lat_max, lon_min, lon_max = loss_area
        subregion_indices = self.final_locations.where((self.final_locations.lat < lat_max) & (self.final_locations.lat > lat_min))
        subregion_indice = subregion_indices.where((self.final_locations.lon < lon_max) & (self.final_locations.lon > lon_min))

        return ~ subregion_indices.isnull().values

    
    def to_map(self,
               data): 
        '''
        Write flattened data to maps
        '''
        
        #  if you pass a numpy array (the output of the network)        
        if not isinstance(data,
                          np.ndarray): #
             # Unstack the flattened spatial dim and write back to the initial format 
             # as saved in self.reference_shape using NaN as fill value            
            return data.unstack().combine_first(self.reference_shape) 
        else:  
             # if you pass a numpy array (the output of the network), 
             # we use the test_set template that we saved to create a Datset.
            output = self.shape
            output[:] = data[:]
            return output.unstack().combine_first(self.reference_shape)


        
class PreprocessingPipeline:

    def __init__(self,
                 pipeline):
        self.pipeline = pipeline
        self.steps = []
        self.fitted_preprocessors = []

    def fit(self,
            data,
            mask=None):
        data_processed = data
        for step_name, preprocessor in self.pipeline:
            preprocessor.fit(data_processed,
                             mask=mask)
            data_processed = preprocessor.transform(data_processed)
            self.steps.append(step_name)
            self.fitted_preprocessors.append(preprocessor)
        return self

    def transform(self,
                  data,
                  step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(self.steps,
                                      self.fitted_preprocessors):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.transform(data_processed,
                                                    **args)
        return data_processed

    def inverse_transform(self,
                          data,
                          step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(reversed(self.steps),
                                      reversed(self.fitted_preprocessors)):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.inverse_transform(data_processed,
                                                            **args)
        return data_processed

    def get_preprocessors(self,
                          name=None):
        if name is None:
            return self.fitted_preprocessors
        else:
            idx = np.argwhere(np.array(self.steps) == name).flatten()
            if idx.size == 0:
                raise ValueError(f"{name} not in preprocessing steps!")
            return self.fitted_preprocessors[int(idx)]
    
    def add_fitted_preprocessor(self,
                                preprocessor,
                                name,
                                index=None):
        if index is None:
            self.fitted_preprocessors.append(preprocessor)
            self.steps.append(name)
        else:
            self.fitted_preprocessors.insert(index,
                                             preprocessor)
            self.steps.insert(index,
                              name)
            
            
class AnomaliesScaler:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self,
            data,
            mask=None):
        if mask is not None:
            self.mean = np.ma.array(data.to_numpy(),
                                    mask=mask).mean(axis=self.axis).data
        else:
            self.mean = data.to_numpy().mean(axis=self.axis)
        return self

    def transform(self, data):
        data_anomalies = data - self.mean
        return data_anomalies
    
    def inverse_transform(self, data):
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            nlead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(nlead_years)], axis=0)
            data_raw = data + mean
        else:
            data_raw = data + self.mean
        return data_raw
    

class AnomaliesScaler_v1:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self,
            data,
            mask=None):
        '''
        Always pass a map-like data to this function. 
        Even if it is flattened, we first write back to maps.    
        '''
        
        axis = self.axis

        if mask is not None:
            self.mean = np.ma.array(data.to_numpy(),
                                    mask=mask).mean(axis=axis).data
        else:
            self.mean = data.to_numpy().mean(axis=axis) 
        
        return self
    

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        
        data_anomalies = data_anomalies - self.mean
        
        return data_anomalies.transpose(*shape) 

    
    def inverse_transform(self, data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            nlead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(nlead_years)], axis=0)

            try:
                data_raw = data + mean
            except: # if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
        else:
            try:
                data_raw = data + self.mean
            except: # if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + self.mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
            
        return data_raw


class AnomaliesScaler_v2:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self,
            data,
            mask=None):
        '''
        We always pass a map-like data to this function. Even if it is flattened, we first write back to maps.        
        '''
        
        axis = self.axis

        if mask is not None:
            self.mean = np.ma.array(data.to_numpy(),
                                    mask=mask).mean(axis=axis).data[0:12,...] 
        else:
            self.mean = data.to_numpy().mean(axis=axis)[0:12,...] 
        
        nly = int(data.shape[1]/12) 
        self.mean = np.concatenate([self.mean for _ in range(nly)],
                                   axis = 0) 
        
        return self
    
    
    def transform(self,
                  data):

        shape = data.dims
        data_anomalies = data.copy()
        
        data_anomalies = data_anomalies - self.mean

        return data_anomalies.transpose(*shape)  # Move ensemble back to the original axis.
    
    
    def inverse_transform(self, 
                          data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            nlead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(nlead_years)], axis=0)

            try:
                data_raw = data + mean
            except: # if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
        else:
            try:
                data_raw = data + self.mean
            except: # if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + self.mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
            
        return data_raw 
    

class Standardizer:

    def __init__(self,
                 axis=None) -> None:
        self.mean = None
        self.std = None
        self.axis = axis

    def fit(self,
            data,
            mask=None):

        if mask is not None:
            marray = np.ma.array(data,
                                 mask=mask)
        else:
            marray = data.to_numpy()
        
        if self.axis is None:

            if np.isnan(marray.mean()):
                self.mean = np.ma.masked_invalid(marray).mean()
                self.std = np.ma.masked_invalid(marray).std()
            else:            
                self.mean = marray.mean()
                self.std = marray.std()
                
        else:

                self.mean = marray.mean(self.axis).data
                self.std = marray.std(self.axis).data + 1e-4

        return self

    def transform(self,
                  data):
        data_standardized = (data - self.mean) / self.std
        if self.axis is None:
            data_standardized = data_standardized.where(~np.isnan(self.mean) & (self.std != 0), 0).where(~np.isnan(self.mean)) 
        return data_standardized

    def inverse_transform(self, data):
        data_raw = data * self.std + self.mean
        return data_raw

        
