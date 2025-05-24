class obs_module(object):
    '''
    class to store information for observational data
    '''    

    def __init__(self, loc):

        self.loc = loc
        
        self.obs_dict = { 
                    "SOMFFN_v2022"   : {
                                         "fgco2"  :
                                           {
                                               "var"  : "fgco2",
                                               "units" : "mol m-2 yr-1",
                                               "y0"   : 1982, 
                                               "y1"   : 2021, 
                                               "dir"  : f"{loc}/SOMFFN_v2022",
                                               "file" : f"SOMFFN_mon_1x1deg_*",                                                   
                                           },
                                        },
                        }
        
    def PrintLoc(self):
        print(self.loc)
    
        
        
