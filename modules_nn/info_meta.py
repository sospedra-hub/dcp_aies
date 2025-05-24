class meta_module(object):
    '''
    Class Metadata
    Dictionaries for variables, coordinates, and global attributes
    Note: keys starting with * are not included in var/dim attrs of copied files
    '''
    

    def __init__(self):
        
        
        self.var_dict = { 
                         'fgco2' : {
                                    '*realm' : 'Omon',
                                    '*grid'  : 'gn',
                                    'standard_name' : '...',
                                    'long_name' : '...',
                                    'comment' : '...',
                                    'units' : '...',
                                    'original_name' : '...',
                                    'cell_methods' : 'area: time: mean',
                                    'cell_measures' : 'area: areacella',
                                    'missing_value' : 1.e+20,
                                     '_FillValue' : 1.e+20,
                                  },     
                        }
        
        self.coord_dict = {              
                           'time' : {
                                      'long_name'     : 'Time',
                                      'standard_name' : 'time',
                                    },
                           'lat' : {
                                      'long_name'     : 'Latitude',
                                      'standard_name' : 'latitude',
                                      'units'         : 'degrees_north',
                                    },
                           'lon' : {
                                      'long_name'     : 'Longitude',
                                      'standard_name' : 'longitude',
                                      'units'         : 'degrees_east',
                                    },
                           'realization' : {
                                      'standard_name' : 'realization',
                                      'units'         : '1',
                                    },
                         }
        
        
    def global_attributes(self,
                          var=None,
                          year=None,
                          realization=None,
                          comment='unadjusted monthly values'):       
        
        if year < 2020:
            dcppx = 'dcppA-hindcast' 
        else:
            dcppx = 'dcppB-forecast'
            
        if realization is None:
            srealization = None
            xrealization = None
        else:
            if realization < 10:
                srealization = f'00{realization}'
            else:
                if realization < 100:
                    srealization = f'0{realization}'
                else:
                    srealization = f'{realization}'
            xrealization = srealization[1:]
        
        global_attr = {
  'CMIP6' : 'empty',    
                       }
        
        return global_attr
        
                                     
                                     
                            
