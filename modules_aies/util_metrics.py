import numpy as np
import xarray as xr
from xarray import DataArray
import gc

from modules_aies.util_glbavg import *


class measure:
    '''
     class measure 
    '''
    def __init__(self, ds1, ds2, dim_t='year', dim_x='lon', dim_y='lat') -> None:
        self.ds1 = ds1      # forecast
        self.ds2 = ds2      # observations
        self.dim_t = dim_t  # temporal dimension t (or equivalent)
        self.dim_x = dim_x  # spatial  dimension x (or equivalent)
        self.dim_y = dim_y  # spatial  dimension y (or equivalent)
        
    def bias(self):
        '''
        climatological bias
        '''
        return (self.ds1 - self.ds2).mean(self.dim_t)

    def mse(self):
        '''
        mean square error
        '''
        return ((self.ds1 - self.ds2)**2).mean(self.dim_t)

    def rmse(self):
        '''
        root mean square error
        '''
        return self.mse()**(1/2)

    def msss(self,ds_ref=None,cross_validated=False,ratio_of_averages=False):
        '''
        mean square skill score [https://www.wmolc.org/contents/index/Verification%2BHindcast]    
        '''
        fctr = 1.
        if ds_ref is None:
            ds_ref = self.ds2.mean(self.dim_t) # climatology
            if cross_validated: 
                n = len(self.ds2[dim_t].values)
                fctr = (n/(n-1))**2
        mse_num =      ((self.ds1 - self.ds2)**2).mean(self.dim_t)
        mse_den = fctr*((ds_ref   - self.ds2)**2).mean(self.dim_t)
        ratio = mse_num/mse_den
        if ratio_of_averages:
            ratio = area_weighted_avg(mse_num)/area_weighted_avg(mse_den)
        return 1. - ratio

    def corr(self,lin_detrend=False):
        '''
        anomaly correlation coefficient
        '''
        ds1 = self.ds1
        ds2 = self.ds2
        if lin_detrend:
            ds1 = self._detrend_dim(da=self.ds1.to_array(),
                                    deg=1).to_dataset(name=list(self.ds1.keys())[0]).squeeze()
            ds2 = self._detrend_dim(da=self.ds2.to_array(),
                                    deg=1).to_dataset(name=list(self.ds2.keys())[0]).squeeze()
        covariance = ((ds1 - ds1.mean(self.dim_t)) * (ds2 - ds2.mean(self.dim_t))).mean(self.dim_t)
        
        return covariance/(ds1.std(self.dim_t)*ds2.std(self.dim_t))

    
    def corr_patt(self,mask=None,area_weighted=False): # centered
        '''
        pattern correlation -- area weighted, if called
        '''
        if mask is None:
            mask = np.ones((len(self.ds1[self.dim_y]),
                            len(self.ds1[self.dim_x])))
        
        da_area = area_grid(self.ds1[self.dim_y],
                            self.ds1[self.dim_x],
                            mask)
    
        total_area = da_area.sum([self.dim_y,
                                  self.dim_x])
        
        weights = (da_area/total_area)**(1/2)

        ds1 = self.ds1
        ds2 = self.ds2
        if area_weighted:
            ds1 = self.ds1*weights
            ds2 = self.ds2*weights

        dim_xy = [self.dim_x,self.dim_y]
        covariance = ((ds1 - ds1.mean(dim_xy))*(ds2 - ds2.mean(dim_xy))).mean(dim_xy)
        return (covariance/(ds1.std(dim_xy)*ds2.std(dim_xy))).mean(self.dim_t)
    
    
    def _detrend_dim(self,da=None,deg=1):
        '''
        detrend along dimension
        '''
        p = da.polyfit(dim=self.dim_t,deg=deg)
        fit = xr.polyval(da[self.dim_t],p.polyfit_coefficients)
        return da - fit



def get_skill_measures(dict_data,
                       list_2D_measures,
                       list_1D_measures=None,
                       mask=None,
                       glbavg=False,
                       integral=False,
                       area_weighted=False):
    '''
    Get measures of model performance
    input:
     -list_measures: list of measures to harvest
     -dict_data    : data dictionary
     -mask         : mask gridcells for glbavg
     -glbavg       : if true, then compute global average
     -integral     : if true, then compute global total
    output:
     -dict_meas    : skill measures dictionary
    '''
    
    list_measures = list_2D_measures
    if list_1D_measures is not None:
        list_measures = list_measures + list_1D_measures
    
    dict_meas = {}
    for meas in list_measures:
        dict_meas[meas] = {}        
        for key in [key for key in dict_data.keys() if key[:3] != 'fct' and key[:3] != 'obs']:
            dict_meas[meas][key] = {}    
            for freq in [freq for freq in dict_data[key].keys() if freq != 'mon']:
                dict_meas[meas][key][freq]  = {}    
                verification = list(dict_data['obs'][freq].keys())[0]
                for model in dict_data[key][freq].keys():
                    if bool(dict_data[key][freq][model]):
                        ds     = dict_data[key][freq][model]
                        ds_ref = dict_data['obs'][freq][verification]
                        sm     = measure(ds,ds_ref)
                        if meas == 'bias':
                            sm = sm.bias()
                        if meas == 'mse':
                            sm = sm.mse()
                        if meas == 'rmse':
                            sm = sm.rmse()
                        if meas == 'msss':
                            sm = sm.msss()
                        if meas == 'corr':
                            sm = sm.corr()
                        if meas == 'cord':
                            sm = sm.corr(lin_detrend=True)  
                        if meas == 'msrs': # last argument = ref fct (first entry in model list)                 
                            sm = sm.msss(dict_data[key][freq][list(dict_data[key][freq].keys())[0]]) 
                        if meas == 'patc': # pattern correlation 
                            sm = sm.corr_patt(mask=mask,
                                              area_weighted=area_weighted)  
                            
                        if glbavg:
                            if meas in list_2D_measures:
                                sm = area_weighted_avg(sm,
                                                       mask=mask,
                                                       integral=True if meas == 'mse' else False)                  
                            
                        dict_meas[meas][key][freq][model] = sm
                                
    return dict_meas




