class mdl_module(object):
    '''
    class to store information for decadal forecast models
    '''    

    def __init__(self, loc):

        self.loc = loc
        
        self.mdl_dict = { 
                         "CanESM5.0-ems" : {
                                                 "version"  : "v20190429",
                                                 "y0"       : 1981, 
                                                 "y1"       : 2023,
                                                 "ystep"    : 1,
                                                 "nens"     : 10, #10, #40
                                                 "ripf"     : f'i1p2f1',
                                                 "ripf_loc" : -3,
                                                 "dir"      : f"{loc}/canesm5-ems",
                                                 "label"    : f"CanESM5.0p2-ems",
                                            },
                         "CanESM5.0-gcb-v2023.1" : {
                                                 "version"  : "v20190429",
                                                 "y0"       : 2016, 
                                                 "y1"       : 2023,
                                                 "ystep"    : 1,
                                                 "nens"     : 10, #10, #40
                                                 "ripf"     : f'i1p2f1',
                                                 "ripf_loc" : -3,
                                                 "dir"      : f"{loc}/canesm5-ems-gcb-v2023.1",
                                                 "label"    : f"CanESM5.0p2-gcb-v2023.1",
                                            },
                    }
        
    def PrintLoc(self):
        print(self.loc)
                                     
                                     
                                     
