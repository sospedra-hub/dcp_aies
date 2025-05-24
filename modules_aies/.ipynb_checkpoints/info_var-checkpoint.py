class var_module(object):
    '''
    store information of variables 
        --units are for the output plots
    '''

    def __init__(self):

        self.var_dict = { 
                         'fgco2'   : {
                                    'realm' : 'Omon',
                                    'grid'  : 'gn',
                                    'units' : 'mol m-2 yr-1',                             
                                   },     
                        }
        
    def PrintDict(self,var=None):
        if var is not None:
            print(self.var_dict[var])
        else:
            print(self.var_dict)
            
                                     
                                     
                                     
