import numpy as np
import xarray as xr


class UnitChange:
    
    def __init__(self, ds) -> None:
        self.ds = ds
        
    def Kelvin2Celcius(self):
        return self.ds-273.15

    def Celcius2Kelvin(self):
        return self.ds+273.15        
        
    def Int2Met(self,var=None):
        if var == 'tas': 
            units_out = self.ds-273.15
        if var == 'ts':
            units_out = self.ds-273.15
        if var == 'pr':
            units_out = self.ds*60.*60.*24.
        if var == 'psl':
            units_out = self.ds/100.
        if var == 'fgco2':
            units_out = self.ds*60*60*24*365*1000/12*-1      
        return units_out

    def Met2Int(self,var=None):
        if var == 'tas':
            units_out = self.ds+273.15   
        if var == 'ts':
            units_out = self.ds+273.15   
        if var == 'pr':
            units_out = self.ds/60./60./24.
        if var == 'psl':
            units_out = self.ds*100.
        if var == 'fgco2':
            units_out = self.ds/(60*60*24*365*1000/12*-1)            
        return units_out
    

    
def change_units(dict_data,
                 dict_inf_data,
                 dataset_list,
                 units_out,
                 units_in_by_hand=None,
                 freq='mon'):
    
    list_IntUnits = ['K', 'kg m-2 s-1', 'Pa']
    
    for dataset in dataset_list:
        
        for key in dict_data.keys():
            
            if bool(dict_data[key][freq][dataset]):
                
                var = list(dict_data[key][freq][dataset].keys())[0]
                
                if 'units' not in dict_data[key][freq][dataset].attrs:
                    
                    if key == 'obs':
                        dict_data[key][freq][dataset].attrs['units'] = dict_inf_data[dataset]['units']           
                    else:
                        if units_in_by_hand is not None:
                            dict_data[key][freq][dataset].attrs['units'] = units_in_by_hand 
                        else:
                            print("Dataset doesn't have units attributes --must specify units by hand")
                        
                if dict_data[key][freq][dataset].attrs['units'] in list_IntUnits:
                    
                    dict_data[key][freq][dataset] = UnitChange(dict_data[key][freq][dataset]).Int2Met(var)
                    
                    dict_data[key][freq][dataset].attrs['units'] = units_out   
    
    return dict_data



